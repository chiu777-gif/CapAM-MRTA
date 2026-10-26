"""Regression checks for units, data isolation, simulation, sampling and resume."""
import contextlib
from dataclasses import replace
import hashlib
import heapq
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from options import get_options
from problems.mrta.config import DEFAULT_CONFIG, MRTAConfig, split_seed
from problems.mrta.data import (RandomMRTADataset, collate_mrta, load_file_dataset,
                                normalize_instance, save_training_dataset)
from problems.mrta.problem_mrta import MRTA
from nets.attention_model import AttentionModel
from utils import load_model, torch_load_cpu
from utils.functions import sample_many

REPO = Path(__file__).resolve().parents[1]


def physical_case(speed=1., capacity=2., deadlines=(60., 100.)):
    return {'loc': [[50., 0.], [100., 0.]], 'deadline': list(deadlines),
            'workload': [20., 20.], 'n_agents': 1,
            'robots_start_location': [[0., 0.]], 'robots_work_capacity': [capacity],
            'max_speed': speed}


class DataAndPhysicsTests(unittest.TestCase):
    def test_unit_scaling_preserves_times_and_cost(self):
        raw = physical_case()
        normalized = normalize_instance(raw, DEFAULT_CONFIG)
        unscaled = normalize_instance(raw, replace(DEFAULT_CONFIG, coordinate_scale=1., workload_scale=1.))
        states = [MRTA.replay(collate_mrta([item]), torch.tensor([[1, 2]]))
                  for item in (normalized, unscaled)]
        for state in states:
            torch.testing.assert_close(state.tasks_finish_time, torch.tensor([[60., 120.]]))
            torch.testing.assert_close(state.get_final_cost(), torch.tensor([[1.2]]))
            self.assertEqual(state.tasks_done_success.item(), 1)  # Exactly at deadline is on time.
        torch.testing.assert_close(normalized['max_speed'], torch.tensor(.01))

    def test_per_scenario_speed_and_robot_padding(self):
        first = normalize_instance(physical_case(speed=1.), DEFAULT_CONFIG)
        second = normalize_instance(physical_case(speed=2.), DEFAULT_CONFIG)
        batch = collate_mrta([first, second])
        state = MRTA.replay(batch, torch.tensor([[1, 2], [1, 2]]))
        torch.testing.assert_close(state.tasks_finish_time, torch.tensor([[60., 120.], [35., 70.]]))
        slow = physical_case(speed=.001, deadlines=(1e9, 1e9))
        padded = normalize_instance(slow, DEFAULT_CONFIG)
        padded['robots_start_location'] = torch.zeros(2, 2)
        padded['robots_work_capacity'] = torch.tensor([.02, 1.])
        batch = collate_mrta([padded])
        state = MRTA.make_state(batch).update(torch.tensor([1]))
        self.assertGreater(state.next_decision_time.item(), 10000)
        self.assertEqual(state.robot_taking_decision.item(), 0)

    def test_state_update_does_not_mutate_inputs_or_previous_state(self):
        batch = collate_mrta([normalize_instance(physical_case(), DEFAULT_CONFIG)])
        before = {key: value.clone() for key, value in batch.items()}
        initial = MRTA.make_state(batch)
        first = initial.update(torch.tensor([1]))
        other = initial.update(torch.tensor([2]))
        self.assertFalse(initial.visited.any())
        self.assertEqual(first.tasks_visited.item(), 1)
        self.assertEqual(other.tasks_visited.item(), 1)
        for key in batch:
            torch.testing.assert_close(batch[key], before[key])
        with self.assertRaises(ValueError):
            first.update(torch.tensor([1]))

    def test_small_graphs_and_uniform_physical_configuration(self):
        config = MRTAConfig(graph_size=10, robot_counts=(4,), map_size=200., agent_max_speed=2.,
                            workload_min=12., workload_max=12., work_capacity_min=3., work_capacity_max=3.,
                            deadline_min=77., deadline_max=77.)
        dataset = RandomMRTADataset(config, 4, 9)
        for instance in dataset:
            self.assertEqual(instance['loc'].shape, (10, 2))
            self.assertEqual(instance['n_agents'].item(), 4)
            torch.testing.assert_close(instance['max_speed'], torch.tensor(.02))
            torch.testing.assert_close(instance['workload'], torch.full((10,), .12))
            torch.testing.assert_close(instance['robots_work_capacity'], torch.full((4,), .03))
            torch.testing.assert_close(instance['deadline'], torch.full((10,), 77.))
        self.assertEqual(len(RandomMRTADataset(DEFAULT_CONFIG, 500000, 1)), 500000)

    def test_data_streams_are_repeatable_and_do_not_advance_model_rng(self):
        dataset = RandomMRTADataset(DEFAULT_CONFIG, 2, split_seed(1, 'train', 0))
        rng = torch.get_rng_state().clone()
        first = dataset[0]
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key in first:
            torch.testing.assert_close(first[key], dataset[0][key])
        next_epoch = RandomMRTADataset(DEFAULT_CONFIG, 2, split_seed(1, 'train', 1))
        self.assertFalse(torch.equal(first['loc'], next_epoch[0]['loc']))
        self.assertNotEqual(split_seed(1, 'validation'), split_seed(1, 'baseline'))

    def test_all_training_robot_counts_are_present(self):
        dataset = RandomMRTADataset(DEFAULT_CONFIG, 100, 4)
        self.assertEqual({item['n_agents'].item() for item in dataset}, {2, 3, 4, 5, 6, 7})

    def test_versioned_roundtrip_offset_and_no_double_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            config = MRTAConfig(graph_size=10, agent_max_speed=2.)
            generated = RandomMRTADataset(config, 4, 18)
            path = save_training_dataset(generated, Path(directory) / 'data.pkl')
            loaded = load_file_dataset(path, config=replace(config, agent_max_speed=8.), offset=1, num_samples=2)
            self.assertEqual(len(loaded), 2)
            for key in generated[1]:
                torch.testing.assert_close(generated[1][key], loaded[0][key])
            with self.assertRaises(FileExistsError):
                save_training_dataset(generated, path)
            with self.assertRaisesRegex(ValueError, 'unit scales'):
                load_file_dataset(path, config=replace(config, coordinate_scale=1.))

    def test_against_independent_heap_simulator_in_physical_units(self):
        dataset = RandomMRTADataset(MRTAConfig(graph_size=10), 30, 92)
        rng = np.random.RandomState(7)
        for instance in dataset:
            n_agents = instance['n_agents'].item()
            locations = instance['loc'].numpy() * DEFAULT_CONFIG.coordinate_scale
            positions = instance['robots_start_location'][:n_agents].numpy().copy() * DEFAULT_CONFIG.coordinate_scale
            workloads = instance['workload'].numpy() * DEFAULT_CONFIG.workload_scale
            capacities = instance['robots_work_capacity'][:n_agents].numpy() * DEFAULT_CONFIG.workload_scale
            deadlines = instance['deadline'].numpy()
            sequence = rng.permutation(10)
            available = [(0., robot) for robot in range(n_agents)]
            heapq.heapify(available)
            finish_times = np.zeros(10)
            for task in sequence:
                available_time, robot = heapq.heappop(available)
                travel = np.linalg.norm(locations[task] - positions[robot]) / DEFAULT_CONFIG.agent_max_speed
                finish_times[task] = available_time + travel + workloads[task] / capacities[robot]
                positions[robot] = locations[task]
                heapq.heappush(available, (finish_times[task], robot))
            state = MRTA.replay(collate_mrta([instance]), torch.tensor(sequence[None] + 1))
            np.testing.assert_allclose(state.tasks_finish_time.numpy()[0], finish_times, rtol=2e-6, atol=2e-5)
            expected_cost = np.where(finish_times > deadlines, finish_times / deadlines, 0.).sum()
            self.assertAlmostEqual(state.get_final_cost().item(), expected_cost, places=5)
            self.assertEqual(state.tasks_done_success.item(), int((finish_times <= deadlines).sum()))

    def test_legacy_units_must_be_explicit(self):
        case = {'loc_data': {'loc': [[.5, 0.]], 'deadline': [100.], 'workload': [.2]},
                'robot_data': {'robots_loc': [[0., 0.]], 'robots_capacity': [.02]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pkl'
            with path.open('wb') as handle:
                pickle.dump(case, handle)
            with self.assertRaisesRegex(ValueError, 'ambiguous units'):
                load_file_dataset(path)
            loaded = load_file_dataset(path, data_format='legacy_normalized')
            torch.testing.assert_close(loaded[0]['loc'], torch.tensor([[.5, 0.]]))
            torch.testing.assert_close(loaded[0]['max_speed'], torch.tensor(.01))

    def test_original_taptc_cases_keep_deadlines_and_work_time(self):
        path = REPO / 'data/mrta_gini/group1/r11a2i00.txt'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        one = load_file_dataset(path)[0]
        all_cases = load_file_dataset(REPO / 'data/mrta_gini')
        self.assertEqual(len(all_cases), 96)
        self.assertEqual(one['deadline'][0].item(), 255.)
        torch.testing.assert_close(one['loc'][0], torch.tensor([.69, 0.]))
        torch.testing.assert_close(one['workload'][0], torch.tensor(.2))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertTrue(all(item['loc'].size(0) == 100 for item in all_cases))

    def test_invalid_configuration_and_inputs_fail_early(self):
        for kwargs in ({'agent_max_speed': 0.}, {'robot_counts': ()},
                       {'deadline_min': 600., 'deadline_max': 50.}, {'coordinate_scale': -1.}):
            with self.assertRaises(ValueError):
                MRTAConfig(**kwargs)
        with self.assertRaises(ValueError):
            normalize_instance(physical_case(capacity=0.), DEFAULT_CONFIG)
        batch = collate_mrta([normalize_instance(physical_case(), DEFAULT_CONFIG)])
        with self.assertRaises(ValueError):
            MRTA.get_costs(batch, torch.tensor([[1, 1]]))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = AttentionModel(32, 32, MRTA)
        self.model.eval()
        self.model.set_decode_type('greedy')
        dataset = RandomMRTADataset(MRTAConfig(graph_size=10), 2, 27)
        self.batch = collate_mrta([dataset[0], dataset[1]])

    def test_forward_sampling_and_replay_use_same_cost(self):
        with torch.no_grad():
            cost, _, sequence = self.model(self.batch, return_pi=True)
            sampled, sampled_cost, done = self.model.sample_many(self.batch, batch_rep=3, iter_rep=2)
        state = MRTA.replay(self.batch, sequence)
        torch.testing.assert_close(cost, state.get_final_cost()[:, 0])
        torch.testing.assert_close(sampled_cost, cost)
        torch.testing.assert_close(sampled, sequence)
        self.assertEqual(done, state.tasks_done_success.tolist())

    def test_sampling_selects_success_count_from_best_candidate(self):
        # Repetitions are ordered [candidate0/batch0, candidate0/batch1, ...].
        def inner(_):
            return (None, torch.tensor([[1, 2], [1, 2], [2, 1], [2, 1]]),
                    torch.tensor([5., 1., 2., 7.]), torch.tensor([[0], [2], [1], [0]]))
        sequence, cost, done = sample_many(inner, None, torch.zeros(2, 1), batch_rep=2)
        torch.testing.assert_close(cost, torch.tensor([2., 1.]))
        torch.testing.assert_close(sequence, torch.tensor([[2, 1], [1, 2]]))
        self.assertEqual(done, [[1], [2]])

    def test_eval_is_independent_of_batch_companions_and_padding(self):
        with torch.no_grad():
            together, _, sequences = self.model(self.batch, return_pi=True)
            for row in range(2):
                single = {key: value[row:row + 1] for key, value in self.batch.items()}
                cost, _, seq = self.model(single, return_pi=True)
                torch.testing.assert_close(cost[0], together[row])
                torch.testing.assert_close(seq[0], sequences[row])
            padded = dict(self.batch)
            padded['robots_start_location'] = torch.cat((self.batch['robots_start_location'], torch.zeros(2, 5, 2)), 1)
            padded['robots_work_capacity'] = torch.cat((self.batch['robots_work_capacity'], torch.ones(2, 5)), 1)
            cost, _, sequence = self.model(padded, return_pi=True)
            torch.testing.assert_close(cost, together)
            torch.testing.assert_close(sequence, sequences)

    def test_one_node_encoder_is_finite(self):
        dataset = RandomMRTADataset(MRTAConfig(graph_size=1, robot_counts=(1,)), 1, 8)
        with torch.no_grad():
            cost, _ = self.model(collate_mrta([dataset[0]]))
        self.assertTrue(torch.isfinite(cost).all())


class TrainingIntegrationTests(unittest.TestCase):
    def train(self, directory, name, extra):
        from run import run
        opts = get_options(['--no_cuda', '--graph_size', '10', '--robot_counts', '2', '4',
                            '--batch_size', '2', '--epoch_size', '4', '--val_size', '4',
                            '--eval_batch_size', '2', '--embedding_dim', '32', '--no_progress_bar',
                            '--output_dir', str(directory), '--run_name', name, *extra])
        with contextlib.redirect_stdout(io.StringIO()):
            run(opts)
        return Path(opts.save_dir)

    def test_resume_matches_uninterrupted_training_and_saved_configuration(self):
        for baseline, warmup in [('exponential', 0), ('rollout', 0), ('rollout', 2)]:
            with self.subTest(baseline=baseline, warmup=warmup), tempfile.TemporaryDirectory() as directory:
                settings = ['--baseline', baseline, '--bl_warmup_epochs', str(warmup), '--lr_decay', '.9']
                full = self.train(directory, 'full', [*settings, '--n_epochs', '2'])
                first = self.train(directory, 'first', [*settings, '--n_epochs', '1'])
                # Only runtime/epoch flags are supplied. Physical/model/optimizer
                # settings must be recovered from the checkpoint's args.json.
                from run import run
                opts = get_options(['--resume', str(first / 'epoch-0.pt'), '--n_epochs', '1',
                                    '--output_dir', directory, '--run_name', 'resumed', '--no_progress_bar'])
                with contextlib.redirect_stdout(io.StringIO()):
                    run(opts)
                uninterrupted = torch_load_cpu(full / 'epoch-1.pt')
                resumed = torch_load_cpu(Path(opts.save_dir) / 'epoch-1.pt')
                for name in uninterrupted['model']:
                    torch.testing.assert_close(uninterrupted['model'][name], resumed['model'][name], rtol=0, atol=0)
                self.assertEqual(uninterrupted['lr_scheduler'], resumed['lr_scheduler'])
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    get_options(['--resume', str(first / 'epoch-0.pt'), '--agent_max_speed', '2'])

    def test_rollout_training_eval_only_and_file_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixed_validation = save_training_dataset(RandomMRTADataset(
                MRTAConfig(graph_size=10, deadline_min=123., deadline_max=123.), 1, 7),
                Path(directory) / 'validation.pkl')
            output = self.train(directory, 'rollout', ['--baseline', 'rollout', '--n_epochs', '1',
                                                       '--agent_max_speed', '2', '--deadline_min', '80',
                                                       '--deadline_max', '90', '--workload_min', '15',
                                                       '--workload_max', '16',
                                                       '--val_dataset', str(fixed_validation)])
            checkpoint = output / 'epoch-0.pt'
            state = torch_load_cpu(checkpoint)
            config = state['baseline']['dataset'].config
            self.assertEqual(config.agent_max_speed, 2.)
            self.assertEqual(config.deadline_min, 80.)
            self.assertIsInstance(state['baseline']['dataset'], RandomMRTADataset)
            self.assertEqual(len(state['baseline']['dataset']), 4)
            from run import run
            with patch('run.RolloutBaseline', side_effect=AssertionError('eval_only must not construct a baseline')):
                opts = get_options(['--eval_only', '--load_path', str(checkpoint), '--embedding_dim', '32',
                                    '--graph_size', '10', '--val_size', '2', '--no_cuda',
                                    '--output_dir', directory])
                with contextlib.redirect_stdout(io.StringIO()):
                    run(opts)
            config = replace(config, graph_size=10)
            data = save_training_dataset(RandomMRTADataset(config, 3, 10), Path(directory) / 'fixed.pkl')
            from eval import main
            with contextlib.redirect_stdout(io.StringIO()):
                results = main(['--model', str(checkpoint), '--datasets', str(data), '--no_cuda',
                                '--decode_strategy', 'sample', '--width', '3', '--eval_batch_size', '2',
                                '--max_calc_batch_size', '2', '--no_progress_bar',
                                '-o', str(Path(directory) / 'results.json')])
            self.assertEqual(len(results), 3)
            self.assertTrue(all(len(row['sequence']) == 10 for row in results))
            self.assertTrue(all(0 <= row['tasks_done'] <= 10 for row in results))
            self.assertTrue((Path(directory) / 'results.json').exists())

    def test_warmup_and_disabled_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.train(directory, 'warmup', ['--baseline', 'rollout', '--bl_warmup_epochs', '2',
                                                      '--n_epochs', '2', '--checkpoint_epochs', '0'])
            self.assertFalse(list(output.glob('epoch-*.pt')))
            self.assertTrue((output / 'convergence.csv').exists())

    def test_warmup_mixture_is_not_bypassed_by_cached_rollout_values(self):
        from reinforce_baselines import WarmupBaseline, NoBaseline
        baseline = WarmupBaseline(NoBaseline(), n_epochs=2)
        baseline.alpha = .5
        dataset = RandomMRTADataset(MRTAConfig(graph_size=10), 2, 4)
        batch = baseline.collate_fn([baseline.wrap_dataset(dataset)[i] for i in range(2)])
        inputs, cached_value = baseline.unwrap_batch(batch)
        self.assertIsNone(cached_value)
        value, loss = baseline.eval(inputs, torch.tensor([4., 8.]))
        self.assertEqual(value.item(), 3.)  # 0.5 * no-baseline(0) + 0.5 * EMA(6).
        self.assertEqual(loss, 0.)

    def test_training_updates_finite_weights_and_load_path_resets_optimizer(self):
        torch.manual_seed(1234)
        original = AttentionModel(32, 32, MRTA).project_node_embeddings.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            output = self.train(directory, 'gradient', ['--baseline', 'exponential', '--n_epochs', '1',
                                                       '--deadline_min', '10', '--deadline_max', '20'])
            checkpoint = output / 'epoch-0.pt'
            trained = torch_load_cpu(checkpoint)['model']
            self.assertFalse(torch.equal(original, trained['project_node_embeddings.weight']))
            self.assertTrue(all(torch.isfinite(value).all() for value in trained.values()))
            restarted = self.train(directory, 'weights_only', ['--load_path', str(checkpoint),
                                                              '--baseline', 'none', '--lr_model', '.001',
                                                              '--n_epochs', '1'])
            saved = torch_load_cpu(restarted / 'epoch-0.pt')
            self.assertEqual(saved['optimizer']['param_groups'][0]['lr'], .001)
            self.assertEqual(saved['baseline'], {})


if __name__ == '__main__':
    unittest.main()

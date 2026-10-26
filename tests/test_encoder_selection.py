"""Encoder selection, checkpoint compatibility and policy training integration."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from nets.attention_model import AttentionModel
from options import get_options
from problems.mrta.config import DATA_VERSION, MRTAConfig
from problems.mrta.data import RandomMRTADataset, collate_mrta, save_training_dataset
from problems.mrta.problem_mrta import MRTA
from run import run
from utils import load_model, torch_load_cpu


class EncoderSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_encoder_defaults_custom_depth_and_invalid_options(self):
        for encoder, layers in [('GCAPCN', 1), ('AM', 3), ('MAMBA', 3)]:
            opts = get_options(['--encoder', encoder, '--no_cuda'])
            self.assertEqual(opts.encoder, encoder.lower())
            self.assertEqual(opts.n_encode_layers, layers)
        self.assertEqual(get_options(['--encoder', 'am', '--n_encode_layers', '2']).n_encode_layers, 2)
        self.assertEqual(get_options(['--encoder', 'mamba', '--n_encode_layers', '2']).n_encode_layers, 2)
        for args in (['--encoder', 'unknown'], ['--n_encode_layers', '2'],
                     ['--encoder', 'am', '--n_encode_layers', '0'],
                     ['--encoder', 'mamba', '--n_encode_layers', '0']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                get_options(args)

    def test_am_encoding_preserves_task_order_equivariance_and_depot_index(self):
        torch.manual_seed(23)
        model = AttentionModel(32, 32, MRTA, encoder='am').eval()
        data = RandomMRTADataset(MRTAConfig(graph_size=5), 2, 19)
        batch = collate_mrta([data[0], data[1]])
        permutation = torch.tensor([3, 0, 4, 1, 2])
        reordered = dict(batch)
        for field in ('loc', 'deadline', 'workload'):
            reordered[field] = batch[field][:, permutation]
        with torch.no_grad():
            embeddings, graph = model.embedder(batch)
            other, other_graph = model.embedder(reordered)
        self.assertEqual(embeddings.shape, (2, 6, 32))
        torch.testing.assert_close(other[:, 0], embeddings[:, 0])
        torch.testing.assert_close(other[:, 1:], embeddings[:, 1:][:, permutation])
        torch.testing.assert_close(other_graph, graph)

    def test_am_policy_backward_sampling_and_replay(self):
        torch.manual_seed(29)
        model = AttentionModel(32, 32, MRTA, encoder='am')
        data = RandomMRTADataset(MRTAConfig(graph_size=5, robot_counts=(2,),
                                          deadline_min=1., deadline_max=2.), 2, 31)
        batch = collate_mrta([data[0], data[1]])
        model.set_decode_type('sampling')
        cost, likelihood = model(batch)
        (cost.detach() * likelihood).mean().backward()
        gradient = model.embedder.init_embed.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0.)
        model.eval().set_decode_type('greedy')
        with torch.no_grad():
            cost, _, sequence = model(batch, return_pi=True)
            sampled, sampled_cost, done = model.sample_many(batch, batch_rep=3, iter_rep=2)
        replay = MRTA.replay(batch, sequence)
        torch.testing.assert_close(cost, replay.get_final_cost()[:, 0])
        torch.testing.assert_close(sampled, sequence)
        torch.testing.assert_close(sampled_cost, cost)
        self.assertEqual(done, replay.tasks_done_success.tolist())

    def test_old_gcapcn_metadata_and_custom_am_checkpoints_load_strictly(self):
        for encoder, version, recorded_layers in [('gcapcn', 1, 3), ('gcapcn', DATA_VERSION, 1),
                                                   ('am', DATA_VERSION, 2), ('mamba', DATA_VERSION, 2)]:
            with self.subTest(encoder=encoder, version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = AttentionModel(32, 32, MRTA, encoder=encoder,
                                       n_encode_layers=1 if encoder == 'gcapcn' else 2,
                                       legacy_mode=version < DATA_VERSION)
                metadata = dict(problem='mrta', embedding_dim=32, normalization='batch',
                                tanh_clipping=10., n_encode_layers=recorded_layers)
                if version == DATA_VERSION:
                    metadata['network_version'] = version
                if encoder != 'gcapcn':
                    metadata['encoder'] = encoder
                (root / 'args.json').write_text(json.dumps(metadata))
                torch.save({'model': model.state_dict()}, root / 'epoch-0.pt')
                with contextlib.redirect_stdout(io.StringIO()):
                    restored, saved = load_model(root)
                self.assertEqual(restored.encoder, encoder)
                self.assertEqual(restored.legacy_mode, version < DATA_VERSION)
                self.assertEqual(saved['n_encode_layers'], 1 if encoder == 'gcapcn' else 2)
                self.assertEqual(list(restored.state_dict()), list(model.state_dict()))
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def train(self, root, name, epochs, encoder='am'):
        opts = get_options(['--encoder', encoder, '--no_cuda', '--graph_size', '5', '--robot_counts', '2', '4',
                            '--embedding_dim', '32', '--batch_size', '2', '--epoch_size', '4',
                            '--val_size', '2', '--eval_batch_size', '2', '--n_epochs', str(epochs),
                            '--deadline_min', '10', '--deadline_max', '20', '--lr_decay', '.9',
                            '--no_progress_bar', '--output_dir', str(root), '--run_name', name])
        with contextlib.redirect_stdout(io.StringIO()):
            run(opts)
        return Path(opts.save_dir)

    def test_am_training_resume_weight_initialization_and_standard_evaluation(self):
        self.check_training_resume_and_evaluation('am')

    def test_mamba_training_resume_weight_initialization_and_standard_evaluation(self):
        self.check_training_resume_and_evaluation('mamba')

    def check_training_resume_and_evaluation(self, encoder):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = self.train(root, 'full', 2, encoder=encoder)
            first = self.train(root, 'first', 1, encoder=encoder)
            saved = json.loads((first / 'args.json').read_text())
            self.assertEqual((saved['encoder'], saved['n_encode_layers']), (encoder, 3))
            opts = get_options(['--resume', str(first / 'epoch-0.pt'), '--n_epochs', '1',
                                '--output_dir', directory, '--run_name', 'resumed'])
            with contextlib.redirect_stdout(io.StringIO()):
                run(opts)
            expected = torch_load_cpu(full / 'epoch-1.pt')
            actual = torch_load_cpu(Path(opts.save_dir) / 'epoch-1.pt')
            for key in expected['model']:
                torch.testing.assert_close(actual['model'][key], expected['model'][key], rtol=0, atol=0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                get_options(['--resume', str(first / 'epoch-0.pt'), '--encoder', 'gcapcn'])

            # Weight-only initialization recovers architecture, not optimizer settings.
            opts = get_options(['--load_path', str(first / 'epoch-0.pt'), '--eval_only', '--no_cuda',
                                '--graph_size', '5', '--val_size', '2', '--output_dir', directory])
            self.assertEqual((opts.encoder, opts.n_encode_layers, opts.embedding_dim), (encoder, 3, 32))
            self.assertEqual(opts.lr_decay, 1.)
            with contextlib.redirect_stdout(io.StringIO()):
                run(opts)

            from eval import main as evaluate
            dataset = save_training_dataset(RandomMRTADataset(MRTAConfig(graph_size=5), 3, 10), root / 'data.pkl')
            with contextlib.redirect_stdout(io.StringIO()):
                rows = evaluate(['--model', str(first), '--datasets', str(dataset), '--no_cuda',
                                 '--decode_strategy', 'sample', '--width', '3', '--eval_batch_size', '2',
                                 '--max_calc_batch_size', '4', '--no_progress_bar', '-o', str(root / 'eval.json')])
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(len(row['sequence']) == 5 for row in rows))

    def test_resume_pre_encoder_metadata_keeps_original_gcapcn(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.train(Path(directory), 'old_gcapcn', 1, encoder='gcapcn')
            metadata = json.loads((first / 'args.json').read_text())
            del metadata['encoder']
            (first / 'args.json').write_text(json.dumps(metadata))
            opts = get_options(['--resume', str(first / 'epoch-0.pt'), '--n_epochs', '1',
                                '--output_dir', directory, '--run_name', 'resumed'])
            self.assertEqual((opts.encoder, opts.n_encode_layers), ('gcapcn', 1))
            with contextlib.redirect_stdout(io.StringIO()):
                run(opts)
            self.assertTrue((Path(opts.save_dir) / 'epoch-1.pt').is_file())


if __name__ == '__main__':
    unittest.main()

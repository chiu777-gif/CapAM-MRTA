"""Mamba feature mapping, task-ID independence and policy replay checks."""
import unittest

import torch

from nets.attention_model import AttentionModel
from nets.mamba_encoder import Mamba_encoder, ModelArgs, RMSNorm
from problems.mrta.config import MRTAConfig
from problems.mrta.data import RandomMRTADataset, collate_mrta
from problems.mrta.problem_mrta import MRTA


class MambaEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_raw_features_are_projected_and_single_direction_is_causal(self):
        torch.manual_seed(51)
        model = Mamba_encoder(ModelArgs(d_model=32, n_layer=2, vocab_size=3)).double()
        features = torch.randn(2, 6, 3, dtype=torch.float64, requires_grad=True)
        encoded, pooled = model(features)
        self.assertEqual(encoded.shape, (2, 6, 32))
        self.assertEqual(encoded.dtype, torch.float64)
        self.assertEqual(pooled.shape, (2, 32))
        encoded.square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(model.init_embed.weight.grad.norm().item(), 0.)
        changed = features.detach().clone()
        changed[:, -1] += 10.
        with torch.no_grad():
            other, _ = model(changed)
        torch.testing.assert_close(other[:, :-1], encoded[:, :-1], rtol=0, atol=0)
        self.assertFalse(torch.allclose(other[:, -1], encoded[:, -1]))
        with self.assertRaisesRegex(ValueError, 'input features'):
            model(torch.randn(2, 6, 32, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, 'masks'):
            model(features, mask=torch.zeros(2, 6, dtype=torch.bool))

    def test_task_embeddings_restore_ids_and_ignore_permutations_including_ties(self):
        torch.manual_seed(53)
        model = AttentionModel(32, 32, MRTA, encoder='mamba').eval()
        data = RandomMRTADataset(MRTAConfig(graph_size=5), 2, 19)
        batch = collate_mrta([data[0], data[1]])
        # Two equal-feature tasks in one scene and all equal in another scene.
        for field in ('loc', 'deadline'):
            batch[field][0, 4] = batch[field][0, 1]
            batch[field][1] = batch[field][1, :1].expand_as(batch[field][1]).clone()
        permutation = torch.tensor([4, 0, 3, 1, 2])
        reordered = dict(batch)
        for field in ('loc', 'deadline', 'workload'):
            reordered[field] = batch[field][:, permutation]
        with torch.no_grad():
            embeddings, graph = model.embedder(batch)
            other, other_graph = model.embedder(reordered)
        self.assertEqual(embeddings.shape, (2, 6, 32))
        torch.testing.assert_close(other[:, 0], embeddings[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(other[:, 1:], embeddings[:, 1:][:, permutation], rtol=0, atol=0)
        torch.testing.assert_close(other_graph, graph)
        torch.testing.assert_close(embeddings[0, 2], embeddings[0, 5], rtol=0, atol=0)
        torch.testing.assert_close(embeddings[1, 1:], embeddings[1, 1:2].expand_as(embeddings[1, 1:]))
        # A separate backward pass covers duplicate pooling.
        model.train()
        encoded, _ = model.embedder(batch)
        encoded.square().sum().backward()
        self.assertTrue(torch.isfinite(model.embedder.init_embed.weight.grad).all())

    def test_policy_training_sampling_replay_and_batch_independence(self):
        torch.manual_seed(59)
        model = AttentionModel(32, 32, MRTA, encoder='mamba')
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
            single = {key: value[:1] for key, value in batch.items()}
            single_cost, _, single_sequence = model(single, return_pi=True)
        replay = MRTA.replay(batch, sequence)
        torch.testing.assert_close(cost, replay.get_final_cost()[:, 0])
        torch.testing.assert_close(sampled, sequence)
        torch.testing.assert_close(sampled_cost, cost)
        self.assertEqual(done, replay.tasks_done_success.tolist())
        torch.testing.assert_close(single_cost, cost[:1])
        torch.testing.assert_close(single_sequence, sequence[:1])
        one = RandomMRTADataset(MRTAConfig(graph_size=1, robot_counts=(1,)), 1, 9)
        with torch.no_grad():
            single_node_cost, _ = model(collate_mrta([one[0]]))
        self.assertTrue(torch.isfinite(single_node_cost).all())

    def test_rmsnorm_low_precision_squares_stay_finite(self):
        norm = RMSNorm(8)
        value = torch.full((2, 4, 8), 10000., dtype=torch.float16)
        self.assertTrue(torch.isfinite(norm(value)).all())
        torch.testing.assert_close(norm(value), torch.ones(2, 4, 8))


if __name__ == '__main__':
    unittest.main()

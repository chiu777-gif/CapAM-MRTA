"""Shared simulation interface; generation and file loading are separate APIs."""
import torch
from .config import DEFAULT_CONFIG
from .data import FileMRTADataset, RandomMRTADataset, load_file_dataset
from .state_mrta import StateMRTA

# Historical name remains importable for trusted old checkpoint pickles.
MRTADataset = FileMRTADataset


class MRTA:
    NAME = 'mrta'

    @staticmethod
    def make_training_dataset(config, num_samples, seed):
        return RandomMRTADataset(config, num_samples, seed)

    @staticmethod
    def load_dataset(paths, config=DEFAULT_CONFIG, **kwargs):
        return load_file_dataset(paths, config=config, **kwargs)

    @staticmethod
    def make_dataset(filename=None, config=None, num_samples=None, seed=None, **kwargs):
        """Compatibility facade. New callers should use the explicit APIs."""
        if filename is not None:
            return MRTA.load_dataset(filename, config=config or DEFAULT_CONFIG,
                                     num_samples=num_samples, **kwargs)
        if config is None or seed is None or kwargs:
            raise ValueError('Random data requires config, num_samples and seed. '
                             'Use MRTA.make_training_dataset(config, num_samples, seed).')
        return MRTA.make_training_dataset(config, num_samples, seed)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMRTA.initialize(*args, **kwargs)

    @staticmethod
    def replay(dataset, sequences):
        if sequences.ndim != 2 or sequences.size(1) != dataset['loc'].size(1):
            raise ValueError('A solution must contain exactly one action per task')
        expected = torch.arange(1, sequences.size(1) + 1, device=sequences.device)
        if not (sequences.sort(-1).values == expected).all():
            raise ValueError('Each task must occur exactly once, numbered 1..N')
        state = MRTA.make_state(dataset)
        for step in range(sequences.size(1)):
            state = state.update(sequences[:, step])
        return state

    @staticmethod
    def get_costs(dataset, pi):
        return MRTA.replay(dataset, pi).get_final_cost().squeeze(-1), None

    @staticmethod
    def beam_search(*args, **kwargs):
        raise NotImplementedError('MRTA beam-state expansion is not supported; use greedy or sampling')

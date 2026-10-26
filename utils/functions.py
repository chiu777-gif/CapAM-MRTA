
import torch
import numpy as np
import os
import json
import inspect
from pathlib import Path
# from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F
import time


def load_problem(name):
    from problems import  MRTA
    problem = {
        'mrta': MRTA,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    kwargs = {'map_location': 'cpu'}
    # Trusted project checkpoints also contain optimizer/RNG/dataset objects.
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(load_path, **kwargs)


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None):
    from nets.attention_model import AttentionModel
    path = Path(path)
    if path.is_file():
        model_filename = path
        directory = path.parent
    elif path.is_dir():
        directory = path
        if epoch is None:
            epochs = [int(item.stem.split('-')[1]) for item in path.glob('epoch-*.pt')]
            if not epochs:
                raise FileNotFoundError('No epoch checkpoints in {}'.format(path))
            epoch = max(epochs)
        model_filename = path / 'epoch-{}.pt'.format(epoch)
    else:
        raise FileNotFoundError('Model does not exist: {}'.format(path))
    args = load_args(str(directory / 'args.json'))
    if args.get('model', 'attention') != 'attention':
        raise ValueError('Only the MRTA attention model is supported')
    from problems.mrta.config import DATA_VERSION
    legacy = args.get('network_version', 1) < DATA_VERSION
    encoder = args.get('encoder', 'gcapcn').lower()
    # Old GCAPCN runs may record a layer count that was never used.
    layers = 1 if encoder == 'gcapcn' else args.get('n_encode_layers')
    model = AttentionModel(
        args['embedding_dim'], args.get('hidden_dim', args['embedding_dim']),
        load_problem(args['problem']), n_encode_layers=layers,
        normalization=args['normalization'], tanh_clipping=args['tanh_clipping'],
        legacy_mode=legacy, encoder=encoder)
    args['encoder'] = model.encoder
    args['n_encode_layers'] = model.n_encode_layers
    checkpoint = torch_load_cpu(str(model_filename))
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    model.eval()
    print('  [*] Loaded model from {}'.format(model_filename))
    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval)

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """Select cost, sequence AND completion count from the same sampled solution."""
    if batch_rep < 1 or iter_rep < 1:
        raise ValueError('Sampling repetition counts must be positive')
    repeated = do_batch_rep(input, batch_rep)
    costs, sequences, successes = [], [], []
    for _ in range(iter_rep):
        _, pi, cost, done = inner_func(repeated)
        costs.append(cost.reshape(batch_rep, -1).t())
        sequences.append(pi.reshape(batch_rep, -1, pi.size(-1)).transpose(0, 1))
        successes.append(torch.as_tensor(done, device=cost.device).reshape(batch_rep, -1).t())
    costs = torch.cat(costs, dim=1)
    sequences = torch.cat(sequences, dim=1)
    successes = torch.cat(successes, dim=1)
    best_cost, best = costs.min(dim=1)
    rows = torch.arange(costs.size(0), device=costs.device)
    return sequences[rows, best], best_cost, successes[rows, best, None].tolist()

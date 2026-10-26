#!/usr/bin/env python
"""Train/validate with a physical configuration; file-based testing is eval.py."""
import json
import pprint
import random
from pathlib import Path
import numpy as np
import torch
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from problems.mrta.config import DATA_VERSION, MRTAConfig, split_seed
from utils import torch_load_cpu, load_problem
from utils.metrics_logger import MetricsLogger


def run(opts):
    pprint.pprint(vars(opts))
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.autograd.set_detect_anomaly(opts.detect_anomaly)
    config = MRTAConfig.from_options(opts)
    problem = load_problem(opts.problem)
    opts.device = torch.device('cuda:0' if opts.use_cuda else 'cpu')
    load_data = torch_load_cpu(opts.resume or opts.load_path) if (opts.resume or opts.load_path) else {}
    if opts.resume and load_data.get('schema_version') != DATA_VERSION:
        raise ValueError('Only new-format checkpoints can be resumed; use --load_path for old weights')
    if opts.resume:
        opts.epoch_start = load_data['epoch'] + 1
    model = AttentionModel(opts.embedding_dim, opts.hidden_dim, problem,
                           n_encode_layers=opts.n_encode_layers, normalization=opts.normalization,
                           tanh_clipping=opts.tanh_clipping, encoder=opts.encoder).to(opts.device)
    if load_data:
        model.load_state_dict(load_data.get('model', load_data), strict=True)
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    Path(opts.save_dir).mkdir(parents=True)
    arguments = {key: value for key, value in vars(opts).items() if key != 'device'}
    with (Path(opts.save_dir) / 'args.json').open('w') as handle:
        json.dump(arguments, handle, indent=2)
    print('CSV metrics: {}'.format(opts.metrics_dir))
    with MetricsLogger(opts.metrics_dir, opts.run_name, opts.metrics_step) as metrics_logger:
        return _run(model, problem, config, load_data, opts, metrics_logger)


def _run(model, problem, config, load_data, opts, metrics_logger):
    if opts.val_dataset:
        val_dataset = problem.load_dataset(opts.val_dataset, config=config,
                                           data_format=opts.data_format, num_samples=opts.val_size)
    else:
        val_dataset = problem.make_training_dataset(config, opts.val_size, split_seed(opts.seed, 'validation'))
    if opts.eval_only:
        if not (opts.load_path or opts.resume):
            print('No checkpoint supplied: validating a randomly initialized model')
        return validate(model, val_dataset, opts, metrics_logger=metrics_logger,
                        epoch=load_data.get('epoch'))

    if opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    elif opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    else:
        baseline = NoBaseline()
    if opts.bl_warmup_epochs:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, opts.exp_beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr_model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    if opts.resume:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(opts.device)
        scheduler.load_state_dict(load_data['lr_scheduler'])
        baseline.load_state_dict(load_data['baseline'])
        random.setstate(load_data['python_rng_state'])
        np.random.set_state(load_data['numpy_rng_state'])
        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda and load_data['cuda_rng_state']:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        print('Resuming at epoch {}'.format(opts.epoch_start))
    last_epoch = opts.epoch_start + opts.n_epochs - 1
    for epoch in range(opts.epoch_start, last_epoch + 1):
        train_epoch(model, optimizer, baseline, scheduler, epoch, val_dataset, problem, None, opts,
                    metrics_logger=metrics_logger)
        baseline.epoch_callback(model, epoch)
        validate(model, val_dataset, opts, metrics_logger=metrics_logger, epoch=epoch,
                 step=(epoch + 1) * ((opts.epoch_size + opts.batch_size - 1) // opts.batch_size))
        scheduler.step()
        if opts.checkpoint_epochs and ((epoch + 1) % opts.checkpoint_epochs == 0 or epoch == last_epoch):
            checkpoint = {
                'schema_version': DATA_VERSION, 'epoch': epoch,
                'model': get_inner_model(model).state_dict(), 'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(), 'baseline': baseline.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if opts.use_cuda else [],
                'python_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state(),
            }
            path = Path(opts.save_dir) / 'epoch-{}.pt'.format(epoch)
            torch.save(checkpoint, str(path))
            print('Saved {}'.format(path))
    return model


if __name__ == '__main__':
    run(get_options())

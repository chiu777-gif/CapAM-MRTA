"""Training options. Environment argument definitions live in mrta.config."""
import argparse
from dataclasses import fields
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import torch
from problems.mrta.config import (DATA_VERSION, DEFAULT_SEED, MRTAConfig,
                                  add_environment_arguments, add_loading_arguments)
from utils.metrics_logger import add_metrics_arguments


def get_options(args=None):
    args = list(sys.argv[1:] if args is None else args)
    parser = argparse.ArgumentParser(description='Train CapAM on uniform MRTA scenarios')
    parser.add_argument('--problem', choices=['mrta'], default='mrta')
    add_environment_arguments(parser)
    add_loading_arguments(parser)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epoch_size', type=int, default=10,
                        help='New training scenarios per epoch; generated lazily')
    parser.add_argument('--val_size', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=5)
    parser.add_argument('--val_dataset', default=None,
                        help='Optional fixed validation file/directory; never used for training or rollout baseline')
    parser.add_argument('--model', choices=['attention'], default='attention')
    parser.add_argument('--encoder', type=str.lower, choices=['gcapcn', 'am', 'mamba'], default='gcapcn',
                        help='GCAPCN, AM self-attention, or Mamba encoder (case-insensitive)')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--n_encode_layers', type=int,
                        help='Encoder layers: GCAPCN requires 1; AM and Mamba default to 3')
    parser.add_argument('--tanh_clipping', type=float, default=10.)
    parser.add_argument('--normalization', choices=['batch', 'instance'], default='batch',
                        help='GCAPCN/AM normalization; Mamba always uses RMSNorm')
    parser.add_argument('--lr_model', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=1.)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=100, help='Epochs to run in this invocation')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max_grad_norm', type=float, default=1.)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--exp_beta', type=float, default=.8)
    parser.add_argument('--baseline', choices=['rollout', 'exponential', 'none'], default='rollout')
    parser.add_argument('--bl_alpha', type=float, default=.05)
    parser.add_argument('--bl_warmup_epochs', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Training data-loader workers; 0 works in restricted containers')
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--log_step', type=int, default=50)
    add_metrics_arguments(parser)
    parser.add_argument('--run_name', default='run')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save every N epochs AND the final epoch; 0 disables all checkpoints')
    loads = parser.add_mutually_exclusive_group()
    loads.add_argument('--load_path', help='Initialize model weights only; do not load optimizer/baseline')
    loads.add_argument('--resume', help='Resume a new-format checkpoint and its saved configuration')
    parser.add_argument('--no_progress_bar', action='store_true')

    # Saved values are defaults, explicit CLI options still win. Conflicting
    # physical/model settings are then rejected rather than silently overwritten.
    initial = parser.parse_known_args(args)[0]
    resume = initial.resume
    saved = None
    if resume:
        with (Path(resume).parent / 'args.json').open() as handle:
            saved = json.load(handle)
        if saved.get('schema_version') != DATA_VERSION:
            parser.error('Old checkpoints lack explicit physical units; use --load_path for weights instead')
        saved.setdefault('encoder', 'gcapcn')
        destinations = {action.dest for action in parser._actions}
        excluded = {'resume', 'load_path', 'eval_only', 'run_name', 'epoch_start', 'metrics_dir'}
        parser.set_defaults(**{key: value for key, value in saved.items()
                               if key in destinations and key not in excluded})
    elif initial.load_path:
        # Weight-only initialization still needs the checkpoint's architecture.
        args_path = Path(initial.load_path).parent / 'args.json'
        if args_path.is_file():
            with args_path.open() as handle:
                model_args = json.load(handle)
            model_args.setdefault('encoder', 'gcapcn')
            if model_args['encoder'].lower() == 'gcapcn':
                # Historical GCAPCN ignored the recorded layer count.
                model_args['n_encode_layers'] = 1
            architecture = ('encoder', 'embedding_dim', 'normalization', 'n_encode_layers', 'tanh_clipping')
            parser.set_defaults(**{key: model_args[key] for key in architecture if key in model_args})
    opts = parser.parse_args(args)
    if opts.n_encode_layers is None:
        opts.n_encode_layers = 1 if opts.encoder == 'gcapcn' else 3
    if opts.n_encode_layers < 1 or (opts.encoder == 'gcapcn' and opts.n_encode_layers != 1):
        parser.error('GCAPCN requires --n_encode_layers 1; AM/Mamba accept any positive layer count')
    try:
        config = MRTAConfig.from_options(opts)
    except ValueError as exc:
        parser.error(str(exc))
    opts.robot_counts = list(config.robot_counts)
    opts.n_agents = None
    if saved:
        protected = [field.name for field in fields(MRTAConfig)] + [
            'model', 'encoder', 'embedding_dim', 'normalization', 'n_encode_layers', 'tanh_clipping',
            'baseline', 'exp_beta', 'bl_warmup_epochs', 'seed', 'lr_model', 'lr_decay', 'max_grad_norm',
            'batch_size', 'epoch_size', 'val_size', 'eval_batch_size', 'val_dataset', 'data_format']
        differences = [name for name in protected if getattr(opts, name) != saved.get(name)]
        if differences:
            parser.error('Resume must preserve {}. Use --load_path for a new experiment.'.format(', '.join(differences)))
    for name in ('batch_size', 'epoch_size', 'val_size', 'eval_batch_size', 'n_epochs', 'log_step'):
        if getattr(opts, name) < 1:
            parser.error('{} must be positive'.format(name))
    if opts.embedding_dim < 8 or opts.embedding_dim % 8:
        parser.error('embedding_dim must be positive and divisible by 8 attention heads')
    if opts.checkpoint_epochs < 0 or opts.epoch_start < 0 or opts.num_workers < 0:
        parser.error('checkpoint_epochs, epoch_start and num_workers must be nonnegative')
    if opts.metrics_step < 0:
        parser.error('metrics_step must be nonnegative')
    if (not all(math.isfinite(value) for value in (opts.lr_model, opts.lr_decay, opts.max_grad_norm))
            or opts.lr_model <= 0 or not 0 < opts.lr_decay <= 1 or opts.max_grad_norm < 0):
        parser.error('lr_model > 0, 0 < lr_decay <= 1 and max_grad_norm >= 0 are required')
    if not 0 <= opts.exp_beta < 1 or not 0 < opts.bl_alpha < 1:
        parser.error('Require 0 <= exp_beta < 1 and 0 < bl_alpha < 1')
    if not math.isfinite(opts.tanh_clipping) or opts.tanh_clipping < 0:
        parser.error('tanh_clipping must be finite and nonnegative')
    if not 0 <= opts.seed < 2 ** 32:
        parser.error('seed must be in [0, 2**32)')
    if opts.bl_warmup_epochs < 0 or (opts.bl_warmup_epochs and opts.baseline != 'rollout'):
        parser.error('Baseline warmup requires rollout and a nonnegative epoch count')
    opts.hidden_dim = opts.embedding_dim  # Historical model-constructor metadata.
    opts.schema_version = DATA_VERSION
    opts.network_version = DATA_VERSION
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name += '_' + datetime.now().strftime('%Y%m%dT%H%M%S%f')
    opts.save_dir = str(Path(opts.output_dir) / '{}_{}'.format(opts.problem, opts.graph_size) / opts.run_name)
    opts.metrics_dir = str(Path(opts.metrics_dir) if opts.metrics_dir else Path(opts.save_dir) / 'metrics')
    return opts

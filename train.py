import time
# from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils.metrics_logger import CostAccumulator
from utils import move_to
from tqdm import tqdm
from problems.mrta.config import MRTAConfig, split_seed
from problems.mrta.data import collate_mrta


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, metrics_logger=None, epoch=None, step=None):
    # Validate
    print('Validating...')
    started = time.perf_counter()
    cost = rollout(model, dataset, opts, metrics_logger=metrics_logger, epoch=epoch, step=step)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost)) if len(cost) > 1 else 0.0))

    if metrics_logger is not None:
        stats = CostAccumulator()
        stats.update(cost)
        metrics_logger.write('validation_epoch', epoch=epoch, step=step,
                             duration_seconds=time.perf_counter() - started, **stats.summary())
    return avg_cost


def rollout(model, dataset, opts, metrics_logger=None, epoch=None, step=None):
    set_decode_type(model, "greedy")
    model.eval()
    costs = []
    loader = DataLoader(dataset, batch_size=opts.eval_batch_size, collate_fn=collate_mrta)
    for batch_id, batch in enumerate(loader):
        started = time.perf_counter()
        with torch.no_grad():
            cost, _ = model(move_to(batch, opts.device))
        cost = cost.detach().cpu()
        costs.append(cost)
        if metrics_logger is not None and metrics_logger.should_log_batch(batch_id + 1):
            stats = CostAccumulator()
            stats.update(cost)
            metrics_logger.write('validation_batch', epoch=epoch, batch=batch_id, step=step,
                                 duration_seconds=time.perf_counter() - started, **stats.summary())
    return torch.cat(costs, dim=0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts,
                metrics_logger=None):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * math.ceil(opts.epoch_size / opts.batch_size)
    start_time = time.time()
    epoch_costs = CostAccumulator()
    epoch_loss = epoch_actor_loss = 0.


    # All training/validation/baseline scenarios use the same physical config.
    training_dataset = baseline.wrap_dataset(problem.make_training_dataset(
        MRTAConfig.from_options(opts), opts.epoch_size, split_seed(opts.seed, 'train', epoch)))
    training_dataloader = DataLoader(
        training_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers,
        collate_fn=baseline.collate_fn)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        batch_started = time.perf_counter()
        values = train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        costs = values.pop('costs')
        epoch_costs.update(costs)
        epoch_loss += values['total_loss'] * costs.numel()
        epoch_actor_loss += values['actor_loss'] * costs.numel()
        if metrics_logger is not None and metrics_logger.should_log_batch(step + 1):
            batch_stats = CostAccumulator()
            batch_stats.update(costs)
            metrics_logger.write('train_batch', epoch=epoch, batch=batch_id, step=step + 1,
                                 duration_seconds=time.perf_counter() - batch_started,
                                 **batch_stats.summary(), **values)
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if metrics_logger is not None:
        metrics_logger.write('train_epoch', epoch=epoch, step=step, **epoch_costs.summary(),
                             actor_loss=epoch_actor_loss / epoch_costs.count,
                             total_loss=epoch_loss / epoch_costs.count,
                             learning_rate=optimizer.param_groups[0]['lr'], duration_seconds=epoch_duration)
    # Validation, baseline updates, scheduler and checkpoints are ordered in run.py.


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)


    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()

    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()

    loss.backward()

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)

    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)

    return {'costs': cost.detach(), 'actor_loss': reinforce_loss.item(), 'total_loss': loss.item(),
            'negative_log_likelihood': -log_likelihood.detach().mean().item(),
            'grad_norm': float(grad_norms[0][0]), 'clipped_grad_norm': float(grad_norms[1][0]),
            'learning_rate': optimizer.param_groups[0]['lr']}

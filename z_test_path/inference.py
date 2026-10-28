"""Profile obstacle-policy inference without charging planning or replay to it."""
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from problems.mrta.data import collate_mrta
from utils import move_to


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def evaluate_scenarios(model, dataset, opts, width, device):
    """Return scenario rows and one timing record per actual inference batch.

    The caller loads/warms the model. CUDA is synchronized around inference;
    all seconds here are wall-clock seconds, not asynchronous kernel launch time.
    """
    loader = DataLoader(dataset, batch_size=opts.eval_batch_size, collate_fn=collate_mrta)
    results, batches, index = [], [], 0
    with tqdm(loader, disable=opts.no_progress_bar) as progress:
        for batch in progress:
            batch = move_to(batch, device)
            synchronize(device)
            batch_size, tasks = batch['loc'].shape[:2]

            phase_started = time.perf_counter()
            best_sequences = best_costs = best_done = None
            remaining = 1 if opts.decode_strategy == 'greedy' else width
            limit = max(1, opts.max_calc_batch_size // batch_size)
            with torch.no_grad():
                while remaining:
                    samples = min(remaining, limit)
                    sequences, costs, done = model.sample_many(batch, batch_rep=samples)
                    done = torch.as_tensor(done, device=device).reshape(-1)
                    if best_costs is None:
                        best_sequences, best_costs, best_done = sequences, costs, done
                    else:
                        improved = costs < best_costs
                        best_sequences[improved] = sequences[improved]
                        best_done[improved] = done[improved]
                        best_costs = torch.minimum(best_costs, costs)
                    remaining -= samples
            synchronize(device)
            inference_seconds = time.perf_counter() - phase_started

            with torch.no_grad():
                state = model.problem.replay(batch, best_sequences)
                if not torch.allclose(best_costs, state.get_final_cost()[:, 0]):
                    raise RuntimeError('Decoded cost differs from path-table simulator replay')
                if not torch.equal(best_done, state.tasks_done_success[:, 0]):
                    raise RuntimeError('Decoded completion count differs from replay')
                trace, robot_sequences = model.problem.make_state(batch), []
                for step in range(best_sequences.size(1)):
                    robot_sequences.append(trace.robot_taking_decision[:, 0])
                    trace = trace.update(best_sequences[:, step])
                robot_sequences = torch.stack(robot_sequences, dim=1)

            cpu_sequences = best_sequences.detach().cpu().tolist()
            cpu_costs = best_costs.detach().cpu().tolist()
            cpu_done = best_done.detach().cpu().tolist()
            cpu_robots = robot_sequences.detach().cpu().tolist()
            counts = batch['n_agents'].detach().cpu().tolist()

            batch_id = len(batches)
            for row in range(batch_size):
                results.append({
                    'instance': dataset.names[index], 'cost': cpu_costs[row],
                    'tasks_done': cpu_done[row], 'task_count': tasks,
                    'completion_rate': cpu_done[row] / tasks, 'n_agents': counts[row],
                    'sequence': cpu_sequences[row], 'robot_sequence': cpu_robots[row],
                    'batch_id': batch_id, 'batch_scenarios': batch_size,
                    'inference_seconds': inference_seconds / batch_size,
                })
                index += 1
            batches.append(dict(batch_id=batch_id, scenarios=batch_size, total_tasks=tasks * batch_size,
                                inference_seconds=inference_seconds))
    return results, batches

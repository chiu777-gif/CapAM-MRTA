"""BiG-MRTA simulators with completion incentives or local cost matching.

upstream_taptc: the state transitions of main_bigmrta_TAPTC.py, including its
ceil(service time), time grid, simultaneous decisions and shared distanceMatrix.
capam_full_visit: the repository's continuous-time StateMRTA and all-task rule.
Completion retains the authors' policy and relaxes only on zero assignments.
Cost uses an adapted Hungarian minimum-immediate-penalty policy.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
import warnings

HERE = Path(__file__).resolve().parent
for path in (HERE / '.deps', HERE / 'upstream', HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache' / 'matplotlib'))

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch

import bigmrta_taptc as upstream
from problems.mrta.data import collate_mrta
from problems.mrta.state_mrta import StateMRTA

# NetworkX 2.8 warns on every matrix construction in this older source.
warnings.filterwarnings('ignore', category=FutureWarning,
                        module=r'bigmrta_taptc')


@dataclass
class Scenario:
    model_input: dict
    locations: np.ndarray
    starts: np.ndarray
    workload: np.ndarray
    capacities: np.ndarray
    deadlines: np.ndarray
    speed: float
    navigation: dict = None
    coordinate_scale: float = 100.

    @property
    def n_tasks(self):
        return len(self.locations)

    @property
    def n_robots(self):
        return len(self.starts)


def prepare(instance, coordinate_scale=100., workload_scale=100., taptc_source=None, navigation=None):
    """Retain replay tensors; use original physical values for TAPTC text.

    A float32 round trip can turn an exact service time of 20 into 20.000001,
    changing the native ceil to 21. Read physical TAPTC values directly instead.
    Generated files still use their saved normalized values without resampling.
    """
    n = int(instance['n_agents'])

    def array(key, scale=1.):
        return instance[key].detach().cpu().numpy().astype(np.float64) * scale

    scenario = Scenario(
        collate_mrta([instance]), array('loc', coordinate_scale),
        array('robots_start_location', coordinate_scale)[:n],
        array('workload', workload_scale),
        array('robots_work_capacity', workload_scale)[:n], array('deadline'),
        float(instance['max_speed']) * coordinate_scale,
    )
    scenario.navigation = navigation
    scenario.coordinate_scale = coordinate_scale
    if navigation is not None:
        tasks, slots = len(instance['loc']), len(instance['robots_start_location'])
        table = torch.zeros(1 + tasks + slots, 1 + tasks)
        table[1:tasks + 1, 1:] = torch.from_numpy(navigation['task_to_task'])
        table[tasks + 1:, 1:] = torch.from_numpy(navigation['robot_to_task'])
        scenario.model_input['path_distances'] = table[None]
    if taptc_source is not None:
        task_path = Path(taptc_source)
        robot_path = task_path.parent.parent / 'agent' / ('a' + task_path.stem.split('a', 1)[1] + '.txt')
        tasks = np.loadtxt(task_path, skiprows=1, ndmin=2)
        robots = np.loadtxt(robot_path, skiprows=1, ndmin=2)
        from problems.mrta.config import DEFAULT_CONFIG
        scenario.locations = tasks[:, 1:3].copy()
        scenario.workload = tasks[:, 3].copy()
        scenario.deadlines = tasks[:, 4].copy()
        scenario.starts = robots[:, 1:3].copy()
        scenario.capacities = robots[:, 3].copy()
        scenario.speed = DEFAULT_CONFIG.agent_max_speed
    return scenario


def robot_distances(scenario, robot):
    if scenario.navigation is None:
        locations = np.vstack((scenario.starts[robot], scenario.locations))
        return cdist(locations, locations)
    distance = np.zeros((scenario.n_tasks + 1, scenario.n_tasks + 1))
    distance[1:, 1:] = scenario.navigation['task_to_task']
    distance[0, 1:] = scenario.navigation['robot_to_task'][robot]
    distance[1:, 0] = distance[0, 1:]
    return distance * scenario.coordinate_scale


def matrices(scenario, ceil_service):
    times = {}
    work = np.r_[0., scenario.workload]
    for robot in range(scenario.n_robots):
        distance = robot_distances(scenario, robot)
        service = work / scenario.capacities[robot]
        if ceil_service:
            service = np.ceil(service)
        service = np.tile(service, (scenario.n_tasks + 1, 1))
        np.fill_diagonal(service, 0.)
        times['robot-{}'.format(robot + 1)] = distance / scenario.speed + service
    # Deliberately preserve upstream's use of the LAST robot's distance matrix
    # for every robot's incentive/range; time matrices remain robot-specific.
    return distance, times


def initial_arrays(scenario):
    state = np.zeros((scenario.n_robots, 9))
    state[:, 3] = 100.  # Q from the authors' TAPTC driver, not work capacity.
    state[:, 4] = 1e6
    return (state, ['r{}'.format(i + 1) for i in range(scenario.n_robots)],
            list(range(1, scenario.n_tasks + 1)), np.r_[0., scenario.deadlines])


def choose(t, robot, state, nodes, tasks, distance, times, deadlines, horizon):
    return upstream.getNextTask(
        t, robot, state, nodes, tasks, distance, times, deadlines,
        TotalTime=horizon, MaxCost=1000, incentiveModelType=6,
        isFullTaskState=True, isVisualize=False, matching_algorithm='hungarian',
    )[0]


OBJECTIVES = {
    'completion': 'Original deadline-feasible incentive matching; relax deadlines only on empty assignments in full-visit mode. '
                  'Heuristic for on-time completion, not a global maximum-completion solver.',
    'cost': 'Hungarian minimum predicted added deadline cost: 0 when finish <= deadline, otherwise finish/deadline. '
            'All remaining tasks are eligible, including late ones. Equal-cost assignments follow SciPy deterministic row/task order. '
            'If the current robot is unmatched when tasks are fewer than robots, select its minimum-cost task. '
            'Visits every task; local matching does not guarantee minimum final schedule cost.',
}


def choose_cost(robot, state, tasks, times, deadlines):
    """Minimize the sum of immediate penalties in the next matching.

    Every edge is allowed: late tasks must eventually be visited. Predictions
    include each robot's availability and its own path-based travel/service
    times. Busy robots compete using their future availability, as in upstream.
    """
    task_ids = np.asarray(tasks, dtype=int)
    finish = np.stack([
        state[r, 1] + times['robot-{}'.format(r + 1)][int(state[r, 0]), task_ids]
        for r in range(len(state))
    ])
    penalty = np.where(finish > deadlines[task_ids], finish / deadlines[task_ids], 0.)
    rows, columns = linear_sum_assignment(penalty)
    match = columns[rows == robot]
    column = int(match[0]) if len(match) else int(np.argmin(penalty[robot]))
    return int(task_ids[column])


def solve_upstream(scenario, objective='completion'):
    """Literal decision/state-update order of the original TAPTC driver.

    Logging, plotting, file I/O and history-array growth are omitted. None feeds
    back into the policy. The driver counts allocations as task completions;
    actual deadline completion is measured separately here from finish times.
    """
    if objective != 'completion':
        raise ValueError('upstream_taptc only supports objective=completion; cost requires capam_full_visit')
    distance, times = matrices(scenario, ceil_service=True)
    state, nodes, tasks, deadlines = initial_arrays(scenario)
    horizon = int(1.1 * max(scenario.deadlines))
    if horizon < 1:
        raise ValueError('The original simulator needs int(1.1 * max(deadline)) >= 1')
    finish = np.full(scenario.n_tasks, np.inf)
    events = []
    policy_seconds = 0.
    calls = 0
    proposals = np.zeros(scenario.n_robots, dtype=np.int64)
    # Exact np.linspace and ready comparison used upstream (not arange).
    for t in np.linspace(0, horizon, int(horizon + 1)):
        if not tasks:
            break
        for robot in range(scenario.n_robots):
            if state[robot, 1] - t <= 0:
                if state[robot, 0] == 0:
                    state[robot, 3] = 100.
                    state[robot, 4] = 1e6
                    if t > 0:
                        state[robot, 8] += 1
                else:
                    state[robot, 3] -= 1
                    state[robot, 4] -= state[robot, 2]
                    state[robot, 7] += 1
                state[robot, 5] += state[robot, 2]
        # All ready robots decide from the same snapshot, exactly as upstream.
        for robot in range(scenario.n_robots):
            if state[robot, 1] - t <= 0:
                started = time.perf_counter()
                proposals[robot] = choose(t, robot, state, nodes, tasks,
                                           distance, times, deadlines, horizon)
                duration = time.perf_counter() - started
                policy_seconds += duration
                state[robot, 6] += duration
                calls += 1
        for robot in range(scenario.n_robots):
            if state[robot, 1] - t <= 0:
                task = int(proposals[robot])
                if task:
                    tasks.remove(task)
                previous = int(state[robot, 0])
                state[robot, 0] = task
                state[robot, 1] += times['robot-{}'.format(robot + 1)][previous, task]
                state[robot, 2] = distance[previous, task]
                if task:
                    finish[task - 1] = state[robot, 1]
                    events.append({'decision_time': float(t), 'robot': robot,
                                   'task': task, 'finish_time': float(finish[task - 1]),
                                   'deadline_relaxed': False})
    return dict(finish_times=finish, events=events, policy_calls=calls,
                policy_seconds=policy_seconds, relaxed_decisions=0,
                isolated_relaxed_decisions=0, robot_state=state,
                upstream_travel_cost=float(state[:, 5].sum()))


def solve_full_visit(scenario, objective='completion'):
    """Configurable matching with StateMRTA's exact all-task dynamics."""
    if objective not in OBJECTIVES:
        raise ValueError('Unknown objective: ' + objective)
    distance, times = matrices(scenario, ceil_service=False)
    robots, nodes, tasks, deadlines = initial_arrays(scenario)
    horizon = max(1, int(1.1 * max(scenario.deadlines)))
    relaxed_deadlines = np.full_like(deadlines, np.inf)
    if scenario.navigation is not None:
        from z_test_path.path_adapter import PathState
        state = PathState.initialize(scenario.model_input)
    else:
        state = StateMRTA.initialize(scenario.model_input)
    events = []
    calls, relaxed_count, isolated_count = 0, 0, 0
    policy_seconds = 0.
    while tasks:
        robot = int(state.robot_taking_decision[0, 0])
        # Read availability from the common simulator, including float32 ties.
        destinations = state.robots_current_destination[0, :scenario.n_robots].numpy()
        # The policy uses local node 0 for each robot's start. PathState keeps
        # distinct start node IDs after the tasks in its rectangular table.
        robots[:, 0] = np.where(destinations > scenario.n_tasks, 0, destinations)
        robots[:, 1] = state.robots_next_decision_time[0, :scenario.n_robots].numpy()
        t = float(robots[robot, 1])
        started = time.perf_counter()
        task = (choose_cost(robot, robots, tasks, times, deadlines) if objective == 'cost' else
                choose(t, robot, robots, nodes, tasks, distance, times, deadlines, horizon))
        calls += 1
        relaxed = task == 0
        if relaxed:
            relaxed_count += 1
            task = choose(t, robot, robots, nodes, tasks, distance, times,
                          relaxed_deadlines, horizon)
            calls += 1
            if task == 0:
                # A rectangular matching can leave this robot unmatched when
                # fewer tasks than robots remain. Reuse the same policy with
                # only this robot competing, so it must get a positive edge.
                isolated_count += 1
                isolated = robots.copy()
                isolated[:, 3] = 0.
                isolated[robot, 3] = robots[robot, 3]
                task = choose(t, robot, isolated, nodes, tasks, distance, times,
                              relaxed_deadlines, horizon)
                calls += 1
        policy_seconds += time.perf_counter() - started
        if task not in tasks:
            raise RuntimeError('Upstream matching did not return an available task')
        previous = int(robots[robot, 0])
        state = state.update(torch.tensor([task], dtype=torch.long))
        finish = float(state.tasks_finish_time[0, task - 1])
        events.append({'decision_time': t, 'robot': robot, 'task': int(task),
                       'finish_time': finish, 'deadline_relaxed': relaxed})
        tasks.remove(task)
        robots[robot, 3] -= 1
        robots[robot, 4] -= distance[previous, task]
    return dict(finish_times=state.tasks_finish_time[0].numpy().copy(),
                events=events, policy_calls=calls, policy_seconds=policy_seconds,
                relaxed_decisions=relaxed_count,
                isolated_relaxed_decisions=isolated_count,
                robot_state=robots, upstream_travel_cost=None)


SOLVERS = {'upstream_taptc': solve_upstream, 'capam_full_visit': solve_full_visit}


def replay_full_visit(scenario, result):
    """Outside timed inference: verify every robot, finish time and final cost."""
    from problems.mrta.problem_mrta import MRTA
    if scenario.navigation is not None:
        from z_test_path.path_adapter import PathMRTA
        MRTA = PathMRTA
    sequence = torch.tensor([[event['task'] for event in result['events']]])
    state = MRTA.replay(scenario.model_input, sequence)
    np.testing.assert_array_equal(state.tasks_finish_time[0].numpy(), result['finish_times'])
    # A per-robot NumPy replay additionally checks the recorded robot IDs.
    available = np.zeros(scenario.n_robots)
    previous = np.zeros(scenario.n_robots, dtype=int)
    distances = [robot_distances(scenario, robot) for robot in range(scenario.n_robots)]
    for event in result['events']:
        robot, task = event['robot'], event['task'] - 1
        if not np.isclose(available[robot], event['decision_time'], atol=2e-3, rtol=1e-6):
            raise AssertionError('Robot availability differs during independent replay')
        finish = (available[robot] + distances[robot][previous[robot], task + 1]
                  / scenario.speed + scenario.workload[task] / scenario.capacities[robot])
        if not np.isclose(finish, event['finish_time'], atol=2e-3, rtol=1e-6):
            raise AssertionError('Finish time differs during independent replay')
        available[robot] = event['finish_time']
        previous[robot] = task + 1
    return float(state.get_final_cost()[0, 0]), int(state.tasks_done_success[0, 0])

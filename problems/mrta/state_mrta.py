"""One event-driven simulator used by training, inference and route replay."""
from typing import NamedTuple
import torch
from .metrics import completed_on_time, deadline_cost


class StateMRTA(NamedTuple):
    coords: torch.Tensor
    distance_matrix: torch.Tensor
    time_matrix: torch.Tensor
    deadline: torch.Tensor
    workload: torch.Tensor
    tasks_finish_time: torch.Tensor
    ids: torch.Tensor
    robots_initial_decision_sequence: torch.Tensor
    robots_next_decision_time: torch.Tensor
    robots_current_destination: torch.Tensor
    robots_start_point: torch.Tensor
    robots_work_capacity: torch.Tensor
    robots_start_location: torch.Tensor
    robots_current_destination_location: torch.Tensor
    current_time: torch.Tensor
    robot_taking_decision: torch.Tensor
    next_decision_time: torch.Tensor
    previous_decision_time: torch.Tensor
    tasks_done_success: torch.Tensor
    tasks_visited: torch.Tensor
    depot: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor
    n_agents: torch.Tensor
    max_speed: torch.Tensor
    n_nodes: int
    n_depot: int

    @property
    def visited(self):
        return self.visited_[:, :, self.n_depot:]

    @property
    def dist(self):
        return self.distance_matrix

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            values = {name: value[key] for name, value in self._asdict().items()
                      if torch.is_tensor(value) and name != 'i'}
            values['ids'] = torch.arange(values['coords'].size(0), device=self.coords.device)[:, None]
            return self._replace(**values)
        return tuple.__getitem__(self, key)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        if visited_dtype not in (torch.uint8, torch.bool):
            raise NotImplementedError('Compressed MRTA state masks are not supported')
        loc, depot = input['loc'], input['depot']
        batch, nodes, _ = loc.shape
        if depot.shape != (batch, 1, 2):
            raise ValueError('MRTA currently supports exactly one depot per scenario')
        slots = input['robots_start_location'].size(1)
        n_agents = input['n_agents'].reshape(batch, 1)
        speeds = input['max_speed'].reshape(batch, 1)
        if (speeds <= 0).any() or not torch.isfinite(speeds).all():
            raise ValueError('Robot speeds must be finite and positive')
        if ((n_agents < 1) | (n_agents > slots)).any():
            raise ValueError('Invalid active robot count')
        if (input['deadline'] <= 0).any() or (input['robots_work_capacity'] <= 0).any():
            raise ValueError('Deadlines and work capacities must be positive')
        coords = torch.cat((depot, loc), dim=1)
        distances = torch.cdist(coords, coords)
        robot_ids = torch.arange(slots, device=loc.device).expand(batch, slots)
        decision_times = loc.new_zeros(batch, slots).masked_fill(robot_ids >= n_agents, float('inf'))
        zero_time = loc.new_zeros(batch, 1)
        zero_index = torch.zeros(batch, 1, dtype=torch.long, device=loc.device)
        robot_destinations = torch.zeros(batch, slots, dtype=torch.long, device=loc.device)
        return StateMRTA(
            coords=coords, distance_matrix=distances,
            time_matrix=distances / speeds[:, :, None],
            deadline=input['deadline'], workload=input['workload'],
            tasks_finish_time=loc.new_zeros(batch, nodes),
            ids=torch.arange(batch, device=loc.device)[:, None],
            robots_initial_decision_sequence=robot_ids,
            robots_next_decision_time=decision_times,
            robots_current_destination=robot_destinations,
            robots_start_point=robot_destinations.clone(),
            robots_work_capacity=input['robots_work_capacity'],
            robots_start_location=input['robots_start_location'],
            robots_current_destination_location=input['robots_start_location'].clone(),
            current_time=zero_time, robot_taking_decision=zero_index,
            next_decision_time=zero_time.clone(), previous_decision_time=zero_time.clone(),
            tasks_done_success=zero_index.clone(), tasks_visited=zero_index.clone(),
            depot=zero_index.clone(), prev_a=zero_index.clone(),
            visited_=torch.zeros(batch, 1, nodes + 1, dtype=torch.bool, device=loc.device),
            lengths=zero_time.clone(), cur_coord=depot.clone(),
            i=torch.zeros(1, dtype=torch.long, device=loc.device),
            n_agents=n_agents, max_speed=speeds, n_nodes=nodes, n_depot=1,
        )

    def update(self, selected):
        selected = selected.reshape(-1).long()
        batch = self.coords.size(0)
        rows = torch.arange(batch, device=self.coords.device)
        if selected.numel() != batch or ((selected < 1) | (selected > self.n_nodes)).any():
            raise ValueError('Select one task numbered 1..N per scenario')
        if self.visited_[rows, 0, selected].any():
            raise ValueError('A task cannot be assigned twice')
        robot = self.robot_taking_decision[:, 0]
        destination = self.coords[rows, selected]
        distance = (self.robots_current_destination_location[rows, robot] - destination).norm(dim=-1)
        travel = distance / self.max_speed[:, 0]
        work = self.workload[rows, selected - 1] / self.robots_work_capacity[rows, robot]
        finish = self.robots_next_decision_time[rows, robot] + travel + work

        # Clone dynamic tensors: updating a state must not mutate input or sibling states.
        next_times = self.robots_next_decision_time.clone()
        next_times[rows, robot] = finish
        finish_times = self.tasks_finish_time.clone()
        finish_times[rows, selected - 1] = finish
        visited = self.visited_.clone()
        visited[rows, 0, selected] = True
        destinations = self.robots_current_destination.clone()
        start_points = self.robots_start_point.clone()
        start_points[rows, robot] = destinations[rows, robot]
        destinations[rows, robot] = selected
        robot_locations = self.robots_current_destination_location.clone()
        robot_locations[rows, robot] = destination
        next_time, next_robot = next_times.min(dim=-1, keepdim=True)
        return self._replace(
            prev_a=selected[:, None], previous_decision_time=self.current_time,
            current_time=self.next_decision_time, robot_taking_decision=next_robot,
            next_decision_time=next_time, robots_next_decision_time=next_times,
            robots_current_destination=destinations, robots_start_point=start_points,
            robots_current_destination_location=robot_locations,
            tasks_finish_time=finish_times, visited_=visited,
            tasks_done_success=completed_on_time(finish_times, self.deadline, visited[:, 0, 1:]),
            tasks_visited=self.tasks_visited + 1,
            lengths=self.lengths + distance[:, None], cur_coord=destination[:, None], i=self.i + 1,
        )

    def get_final_cost(self):
        if not self.all_finished():
            raise ValueError('Final cost requires all tasks to be assigned')
        return deadline_cost(self.tasks_finish_time, self.deadline)[:, None]

    def all_finished(self):
        return self.visited.all()

    def get_finished(self):
        return self.visited.all(dim=-1)

    def get_current_node(self):
        return self.robots_current_destination.gather(1, self.robot_taking_decision)

    def get_mask(self):
        # The depot is not an action; each task is allocated exactly once.
        mask = self.visited_.clone()
        mask[:, :, 0] = True
        return mask

    def construct_solutions(self, actions):
        return actions

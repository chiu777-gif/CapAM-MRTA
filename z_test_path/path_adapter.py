"""Local inference adapters: retain checkpoint tensors, query path distances."""
import torch

from nets.graph_encoder import GCAPCN
from problems.mrta.metrics import completed_on_time
from problems.mrta.problem_mrta import MRTA
from problems.mrta.state_mrta import StateMRTA


class PathState(StateMRTA):
    __slots__ = ()

    @staticmethod
    def initialize(data, visited_dtype=torch.bool):
        if visited_dtype not in (torch.bool, torch.uint8):
            raise ValueError('Compressed state masks are not supported')
        loc, depot = data['loc'], data['depot']
        batch, tasks, _ = loc.shape
        slots = data['robots_start_location'].size(1)
        if depot.shape != (batch, 1, 2):
            raise ValueError('Expected one depot placeholder per scenario')
        table = data['path_distances']
        if table.shape != (batch, 1 + tasks + slots, 1 + tasks):
            raise ValueError('path_distances has an invalid shape')
        if not torch.isfinite(table).all() or (table < 0).any():
            raise ValueError('Path distances must be finite and nonnegative')
        agents = data['n_agents'].reshape(batch, 1)
        speeds = data['max_speed'].reshape(batch, 1)
        if ((agents < 1) | (agents > slots)).any() or (speeds <= 0).any():
            raise ValueError('Invalid robot count or speed')
        robot_ids = torch.arange(slots, device=loc.device).expand(batch, slots)
        decision_times = loc.new_zeros(batch, slots).masked_fill(robot_ids >= agents, float('inf'))
        zero_time = loc.new_zeros(batch, 1)
        zero_index = torch.zeros(batch, 1, dtype=torch.long, device=loc.device)
        starts = tasks + 1 + robot_ids
        return PathState(
            coords=torch.cat((depot, loc), dim=1), distance_matrix=table,
            time_matrix=table / speeds[:, :, None], deadline=data['deadline'], workload=data['workload'],
            tasks_finish_time=loc.new_zeros(batch, tasks),
            ids=torch.arange(batch, device=loc.device)[:, None],
            robots_initial_decision_sequence=robot_ids, robots_next_decision_time=decision_times,
            robots_current_destination=starts, robots_start_point=starts.clone(),
            robots_work_capacity=data['robots_work_capacity'], robots_start_location=data['robots_start_location'],
            robots_current_destination_location=data['robots_start_location'].clone(),
            current_time=zero_time, robot_taking_decision=zero_index,
            next_decision_time=zero_time.clone(), previous_decision_time=zero_time.clone(),
            tasks_done_success=zero_index.clone(), tasks_visited=zero_index.clone(),
            depot=zero_index.clone(), prev_a=zero_index.clone(),
            visited_=torch.zeros(batch, 1, tasks + 1, dtype=torch.bool, device=loc.device),
            lengths=zero_time.clone(), cur_coord=depot.clone(),
            i=torch.zeros(1, dtype=torch.long, device=loc.device),
            n_agents=agents, max_speed=speeds, n_nodes=tasks, n_depot=1,
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
        start = self.robots_current_destination[rows, robot]
        distance = self.distance_matrix[rows, start, selected]
        destination = self.coords[rows, selected]
        finish = (self.robots_next_decision_time[rows, robot] + distance / self.max_speed[:, 0]
                  + self.workload[rows, selected - 1] / self.robots_work_capacity[rows, robot])
        next_times = self.robots_next_decision_time.clone()
        next_times[rows, robot] = finish
        finish_times = self.tasks_finish_time.clone()
        finish_times[rows, selected - 1] = finish
        visited = self.visited_.clone()
        visited[rows, 0, selected] = True
        destinations = self.robots_current_destination.clone()
        start_points = self.robots_start_point.clone()
        start_points[rows, robot] = start
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
            tasks_visited=self.tasks_visited + 1, lengths=self.lengths + distance[:, None],
            cur_coord=destination[:, None], i=self.i + 1,
        )


class PathMRTA(MRTA):
    @staticmethod
    def make_state(data):
        return PathState.initialize(data)

    @classmethod
    def replay(cls, data, sequences):
        if sequences.ndim != 2 or sequences.size(1) != data['loc'].size(1):
            raise ValueError('A solution must contain exactly one action per task')
        expected = torch.arange(1, sequences.size(1) + 1, device=sequences.device)
        if not (sequences.sort(-1).values == expected).all():
            raise ValueError('Each task must occur exactly once, numbered 1..N')
        state = cls.make_state(data)
        for step in range(sequences.size(1)):
            state = state.update(sequences[:, step])
        return state

    @classmethod
    def get_costs(cls, data, pi):
        return cls.replay(data, pi).get_final_cost().squeeze(-1), None


class PathGCAPCN(GCAPCN):
    def forward(self, data, mask=None):
        deadline = data['deadline']
        normalizer = deadline.max() if self.legacy_mode else deadline.amax(dim=1, keepdim=True)
        normalized_deadline = deadline / normalizer
        features = torch.cat((data['loc'], normalized_deadline[:, :, None]), dim=-1)
        batch, tasks, _ = features.shape
        spatial = data['path_distances'][:, 1:tasks + 1, 1:tasks + 1]
        deadline_difference = normalized_deadline[:, :, None] - normalized_deadline[:, None, :]
        # Preserve the checkpoint's deadline term; only spatial distance changes.
        distances = (spatial.square() + deadline_difference.square()).sqrt()
        identity = torch.eye(tasks, device=features.device).expand(batch, tasks, tasks)
        adjacency = (1 / distances) * (identity - 1).bool().float()
        adjacency = torch.where(torch.isfinite(adjacency), adjacency, torch.zeros_like(adjacency))
        normalizer = adjacency.max() if self.legacy_mode else adjacency.amax(dim=(1, 2), keepdim=True)
        adjacency = adjacency / normalizer.clamp_min(torch.finfo(adjacency.dtype).eps)
        degree = identity * (adjacency.sum(-1) - 1)[:, None].expand(batch, tasks, tasks)
        laplacian = degree - adjacency
        squared_laplacian = laplacian @ laplacian
        initial = self.init_embed(features)

        def capsule(values, projection):
            return projection(torch.cat((values, laplacian @ values, squared_laplacian @ values), dim=-1))

        transformed = torch.cat((capsule(initial, self.W_L_1_G1),
                                 capsule(initial.square(), self.W_L_1_G2),
                                 capsule(initial * initial.square(), self.W_L_1_G3)), dim=-1)
        transformed = self.normalization_1(self.activ(transformed) + initial)
        tasks_embedding = self.activ(self.W_F(transformed))
        embeddings = torch.cat((self.init_embed_depot(data['depot']), tasks_embedding), dim=1)
        return embeddings, embeddings.mean(dim=1)


def adapt_model(model, normalization='batch'):
    if model.encoder not in ('gcapcn', 'am', 'mamba'):
        raise ValueError('Unsupported encoder: {}'.format(model.encoder))
    if model.encoder == 'gcapcn':
        original = model.embedder
        adapter = PathGCAPCN(n_dim=original.n_dim, n_p=original.n_p, n_K=original.n_K,
                             node_dim=original.node_dim, normalization=normalization,
                             legacy_mode=original.legacy_mode)
        adapter.load_state_dict(original.state_dict(), strict=True)
        adapter.train(original.training)
        model.embedder = adapter
    # AM and Mamba do not construct a geometric distance graph. Preserve their
    # encoders and learned tensors; all three share the table-based simulator.
    model.problem = PathMRTA
    return model

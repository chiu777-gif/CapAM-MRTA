"""Single definition of physical scenario parameters and model-unit scales."""
from dataclasses import dataclass, fields
import hashlib
import math

DEFAULT_SEED = 1234
DATA_VERSION = 2
DATA_FORMATS = ('auto', 'taptc', 'legacy_normalized')


@dataclass(frozen=True)
class MRTAConfig:
    # Public parameters use physical map/work units and seconds.
    graph_size: int = 100
    robot_counts: tuple = (2, 3, 4, 5, 6, 7)
    map_size: float = 100.0
    agent_max_speed: float = 1.0
    deadline_min: float = 50.0
    deadline_max: float = 600.0
    workload_min: float = 10.0
    workload_max: float = 30.0
    work_capacity_min: float = 1.0
    work_capacity_max: float = 3.0
    coordinate_scale: float = 100.0
    workload_scale: float = 100.0

    def __post_init__(self):
        counts = tuple(self.robot_counts)
        object.__setattr__(self, 'robot_counts', counts)
        if not isinstance(self.graph_size, int) or self.graph_size < 1:
            raise ValueError('graph_size must be a positive integer')
        if (not counts or len(set(counts)) != len(counts)
                or any(not isinstance(n, int) or n < 1 for n in counts)):
            raise ValueError('robot_counts must contain distinct positive integers')
        for name in ('map_size', 'agent_max_speed', 'coordinate_scale', 'workload_scale'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('{} must be finite and positive'.format(name))
        for prefix in ('deadline', 'workload', 'work_capacity'):
            low, high = getattr(self, prefix + '_min'), getattr(self, prefix + '_max')
            if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
                raise ValueError('{} bounds must satisfy 0 < min <= max'.format(prefix))

    @classmethod
    def from_options(cls, opts):
        values = {field.name: getattr(opts, field.name)
                  for field in fields(cls) if hasattr(opts, field.name)}
        if getattr(opts, 'n_agents', None) is not None:
            values['robot_counts'] = (opts.n_agents,)
        return cls(**values)


DEFAULT_CONFIG = MRTAConfig()


def add_environment_arguments(parser, include_graph=True, loading_only=False):
    names = ('agent_max_speed', 'coordinate_scale', 'workload_scale') if loading_only else (
        field.name for field in fields(MRTAConfig))
    for name in names:
        if name == 'graph_size' and not include_graph:
            continue
        default = getattr(DEFAULT_CONFIG, name)
        if name == 'robot_counts':
            group = parser.add_mutually_exclusive_group()
            group.add_argument('--robot_counts', type=int, nargs='+', default=default,
                               help='Robot counts sampled uniformly during training')
            group.add_argument('--n_agents', type=int, default=None,
                               help='Use exactly this many robots during training')
        else:
            help_text = {
                'agent_max_speed': 'Physical map units/second (1 becomes 0.01 with default scales)',
                'coordinate_scale': 'Divide coordinates AND speed by this scale; keep seconds',
                'workload_scale': 'Divide workload AND work capacity by this scale',
            }.get(name, 'Physical scenario parameter: ' + name)
            parser.add_argument('--' + name, type=int if name == 'graph_size' else float,
                                default=default, help=help_text)


def add_loading_arguments(parser):
    parser.add_argument('--data_format', choices=DATA_FORMATS, default='auto',
                        help='auto: versioned data or TAPTC text; old dictionaries need explicit units')


def split_seed(seed, split, epoch=0):
    """Stable independent random streams for train/validation/baseline data."""
    value = '{}:{}:{}'.format(seed, split, epoch).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), 'little') % (2 ** 63)

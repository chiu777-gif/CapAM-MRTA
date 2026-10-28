# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions, with deadline relaxation only on empty assignments to visit all tasks. Its cost uses the same metric as CapAM.

## upstream_taptc

Planner: `visibility_dijkstra`; epsilon: `1e-06`. Shared preprocessing wall: 2.764983 s.

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | undefined (unassigned tasks) | 45.600% | 85.765 | 0.000 |
| 3 | 14 | undefined (unassigned tasks) | 59.571% | 168.669 | 0.000 |
| 4 | 24 | undefined (unassigned tasks) | 70.958% | 270.703 | 0.000 |
| 5 | 15 | undefined (unassigned tasks) | 75.200% | 405.028 | 0.000 |
| 6 | 16 | undefined (unassigned tasks) | 80.688% | 505.399 | 0.000 |
| 7 | 21 | undefined (unassigned tasks) | 84.190% | 573.551 | 0.000 |
| All | 100 | undefined (unassigned tasks) | 71.800% | 359.223 | 0.000 |

Unassigned tasks: 2702; relaxed decisions: 0.

Shared path preprocessing: planning 0.000000 s; cache read 2.490217 s; cache write 0.000000 s; hits 100, misses 0.

## capam_full_visit

Planner: `visibility_dijkstra`; epsilon: `1e-06`. Shared preprocessing wall: 2.764983 s.

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 265.868416 | 46.500% | 168.898 | 0.000 |
| 3 | 14 | 169.247045 | 60.429% | 190.281 | 0.000 |
| 4 | 24 | 111.557332 | 71.833% | 212.904 | 0.000 |
| 5 | 15 | 86.648102 | 75.867% | 239.667 | 0.000 |
| 6 | 16 | 65.876455 | 81.062% | 265.449 | 0.000 |
| 7 | 21 | 50.688962 | 84.238% | 289.806 | 0.000 |
| All | 100 | 111.237318 | 72.390% | 233.907 | 0.000 |

Unassigned tasks: 0; relaxed decisions: 2655.

Shared path preprocessing: planning 0.000000 s; cache read 2.490217 s; cache write 0.000000 s; hits 100, misses 0.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, path preprocessing/cache I/O, warmup, replay checks and output I/O. Path planning is shared across protocols, excludes cache I/O, and is zero on cache hits. Existing CapAM batch-32 amortized times are a different timing regime.

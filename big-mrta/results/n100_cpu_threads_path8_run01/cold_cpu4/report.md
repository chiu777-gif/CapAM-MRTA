# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions. Completion mode relaxes deadlines only on empty assignments; cost mode allows late-task edges and minimizes immediate penalty in each matching. Both visit every task and report the same cost metric as CapAM.

## capam_full_visit

Objective: `completion`. Original deadline-feasible incentive matching; relax deadlines only on empty assignments in full-visit mode. Heuristic for on-time completion, not a global maximum-completion solver.

Path worker processes: 8 (requested 8).

Planner: `visibility_dijkstra`; epsilon: `1e-06`. Shared preprocessing wall: 19.097695 s.

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 265.868416 | 46.500% | 162.604 | 1275.341 |
| 3 | 14 | 169.247045 | 60.429% | 174.306 | 1285.228 |
| 4 | 24 | 111.557332 | 71.833% | 194.990 | 1314.511 |
| 5 | 15 | 86.648102 | 75.867% | 214.388 | 1330.383 |
| 6 | 16 | 65.876455 | 81.062% | 240.014 | 1334.835 |
| 7 | 21 | 50.688962 | 84.238% | 265.991 | 1388.274 |
| All | 100 | 111.237318 | 72.390% | 213.880 | 1327.617 |

Unassigned tasks: 0; relaxed decisions: 2655.

Shared path preprocessing: planning 132.761727 s; cache read 0.000000 s; cache write 2.807826 s; hits 0, misses 100.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, path preprocessing/cache I/O, warmup, replay checks and output I/O. Path planning is shared across protocols, excludes cache I/O, and is zero on cache hits. Parallel planning totals sum scenario worker times; preprocessing wall includes worker startup/shutdown. Existing CapAM batch-32 amortized times are a different timing regime.

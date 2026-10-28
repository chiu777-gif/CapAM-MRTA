# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions. Completion mode relaxes deadlines only on empty assignments; cost mode allows late-task edges and minimizes immediate penalty in each matching. Both visit every task and report the same cost metric as CapAM.

## capam_full_visit

Objective: `completion`. Original deadline-feasible incentive matching; relax deadlines only on empty assignments in full-visit mode. Heuristic for on-time completion, not a global maximum-completion solver.

Path worker processes: 8 (requested 8).

Planner: `visibility_dijkstra`; epsilon: `1e-06`. Shared preprocessing wall: 1.946172 s.

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 265.868416 | 46.500% | 175.468 | 0.000 |
| 3 | 14 | 169.247045 | 60.429% | 195.100 | 0.000 |
| 4 | 24 | 111.557332 | 71.833% | 213.436 | 0.000 |
| 5 | 15 | 86.648102 | 75.867% | 243.349 | 0.000 |
| 6 | 16 | 65.876455 | 81.062% | 262.465 | 0.000 |
| 7 | 21 | 50.688962 | 84.238% | 299.552 | 0.000 |
| All | 100 | 111.237318 | 72.390% | 237.488 | 0.000 |

Unassigned tasks: 0; relaxed decisions: 2655.

Shared path preprocessing: planning 0.000000 s; cache read 2.554813 s; cache write 0.000000 s; hits 100, misses 0.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, path preprocessing/cache I/O, warmup, replay checks and output I/O. Path planning is shared across protocols, excludes cache I/O, and is zero on cache hits. Parallel planning totals sum scenario worker times; preprocessing wall includes worker startup/shutdown. Existing CapAM batch-32 amortized times are a different timing regime.

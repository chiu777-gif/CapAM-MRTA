# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions. Completion mode relaxes deadlines only on empty assignments; cost mode allows late-task edges and minimizes immediate penalty in each matching. Both visit every task and report the same cost metric as CapAM.

## capam_full_visit

Objective: `completion`. Original deadline-feasible incentive matching; relax deadlines only on empty assignments in full-visit mode. Heuristic for on-time completion, not a global maximum-completion solver.

Path worker processes: 8 (requested 8).

Planner: `visibility_dijkstra`; epsilon: `1e-06`. Shared preprocessing wall: 18.992108 s.

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 265.868416 | 46.500% | 161.130 | 1217.942 |
| 3 | 14 | 169.247045 | 60.429% | 172.697 | 1261.730 |
| 4 | 24 | 111.557332 | 71.833% | 194.182 | 1294.786 |
| 5 | 15 | 86.648102 | 75.867% | 216.348 | 1311.743 |
| 6 | 16 | 65.876455 | 81.062% | 237.959 | 1291.914 |
| 7 | 21 | 50.688962 | 84.238% | 265.573 | 1370.153 |
| All | 100 | 111.237318 | 72.390% | 213.190 | 1300.385 |

Unassigned tasks: 0; relaxed decisions: 2655.

Shared path preprocessing: planning 130.038487 s; cache read 0.000000 s; cache write 2.785144 s; hits 0, misses 100.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, path preprocessing/cache I/O, warmup, replay checks and output I/O. Path planning is shared across protocols, excludes cache I/O, and is zero on cache hits. Parallel planning totals sum scenario worker times; preprocessing wall includes worker startup/shutdown. Existing CapAM batch-32 amortized times are a different timing regime.

# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions. Completion mode relaxes deadlines only on empty assignments; cost mode allows late-task edges and minimizes immediate penalty in each matching. Both visit every task and report the same cost metric as CapAM.

## capam_full_visit

Objective: `completion`. Original deadline-feasible incentive matching; relax deadlines only on empty assignments in full-visit mode. Heuristic for on-time completion, not a global maximum-completion solver.

Path worker processes: 8 (requested 8).

Planner: `visibility_dijkstra`; epsilon: `0.01`. Shared preprocessing wall: 9.228189 s.

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |
|---:|---:|---:|---:|---:|---:|
| 2 | 51 | 246.800635 | 48.882% | 144.557 | 170.426 |
| 3 | 54 | 161.563356 | 62.333% | 161.013 | 174.269 |
| 4 | 48 | 109.969544 | 71.354% | 181.486 | 176.475 |
| 5 | 53 | 81.875874 | 77.283% | 198.028 | 179.289 |
| 6 | 49 | 64.243247 | 80.878% | 217.247 | 184.560 |
| 7 | 45 | 50.610229 | 84.044% | 236.789 | 188.834 |
| All | 300 | 121.181975 | 70.417% | 188.582 | 178.721 |

Unassigned tasks: 0; relaxed decisions: 8557.

Shared path preprocessing: planning 53.616384 s; cache read 0.000000 s; cache write 7.490606 s; hits 0, misses 300.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, path preprocessing/cache I/O, warmup, replay checks and output I/O. Path planning is shared across protocols, excludes cache I/O, and is zero on cache hits. Parallel planning totals sum scenario worker times; preprocessing wall includes worker startup/shutdown. Existing CapAM batch-32 amortized times are a different timing regime.

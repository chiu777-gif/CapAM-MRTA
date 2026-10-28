# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions. Completion mode relaxes deadlines only on empty assignments; cost mode allows late-task edges and minimizes immediate penalty in each matching. Both visit every task and report the same cost metric as CapAM.

## capam_full_visit

Objective: `completion`. Original deadline-feasible incentive matching; relax deadlines only on empty assignments in full-visit mode. Heuristic for on-time completion, not a global maximum-completion solver.

Path worker processes: 8 (requested 8).

Planner: `visibility_dijkstra`; epsilon: `0.01`. Shared preprocessing wall: 66.611314 s.

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |
|---:|---:|---:|---:|---:|---:|
| 2 | 51 | 248.246897 | 49.137% | 141.237 | 1636.645 |
| 3 | 54 | 163.586521 | 62.630% | 154.164 | 1685.051 |
| 4 | 48 | 113.879857 | 70.688% | 172.626 | 1665.904 |
| 5 | 53 | 81.854405 | 77.283% | 192.110 | 1712.777 |
| 6 | 49 | 63.976898 | 81.286% | 209.985 | 1714.705 |
| 7 | 45 | 50.468537 | 84.244% | 232.868 | 1735.517 |
| All | 300 | 122.349109 | 70.503% | 182.547 | 1691.070 |

Unassigned tasks: 0; relaxed decisions: 8539.

Shared path preprocessing: planning 507.320995 s; cache read 0.000000 s; cache write 7.508867 s; hits 0, misses 300.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, path preprocessing/cache I/O, warmup, replay checks and output I/O. Path planning is shared across protocols, excludes cache I/O, and is zero on cache hits. Parallel planning totals sum scenario worker times; preprocessing wall includes worker startup/shutdown. Existing CapAM batch-32 amortized times are a different timing regime.

# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions. Completion mode relaxes deadlines only on empty assignments; cost mode allows late-task edges and minimizes immediate penalty in each matching. Both visit every task and report the same cost metric as CapAM.

## capam_full_visit

Objective: `completion`. Original deadline-feasible incentive matching; relax deadlines only on empty assignments in full-visit mode. Heuristic for on-time completion, not a global maximum-completion solver.

Path worker processes: 8 (requested 8).

Planner: `visibility_dijkstra`; epsilon: `0.01`. Shared preprocessing wall: 48.415393 s.

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |
|---:|---:|---:|---:|---:|---:|
| 2 | 51 | 252.308362 | 48.490% | 140.445 | 1165.587 |
| 3 | 54 | 167.941723 | 61.852% | 154.850 | 1181.263 |
| 4 | 48 | 113.686140 | 70.708% | 171.965 | 1206.374 |
| 5 | 53 | 81.356969 | 77.736% | 193.176 | 1238.948 |
| 6 | 49 | 64.254200 | 81.000% | 210.971 | 1239.538 |
| 7 | 45 | 50.194980 | 84.178% | 231.874 | 1263.303 |
| All | 300 | 123.708878 | 70.280% | 182.631 | 1214.631 |

Unassigned tasks: 0; relaxed decisions: 8608.

Shared path preprocessing: planning 364.389343 s; cache read 0.000000 s; cache write 7.695045 s; hits 0, misses 300.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, path preprocessing/cache I/O, warmup, replay checks and output I/O. Path planning is shared across protocols, excludes cache I/O, and is zero on cache hits. Parallel planning totals sum scenario worker times; preprocessing wall includes worker startup/shutdown. Existing CapAM batch-32 amortized times are a different timing regime.

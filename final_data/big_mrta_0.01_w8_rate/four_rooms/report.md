# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions. Completion mode relaxes deadlines only on empty assignments; cost mode allows late-task edges and minimizes immediate penalty in each matching. Both visit every task and report the same cost metric as CapAM.

## capam_full_visit

Objective: `completion`. Original deadline-feasible incentive matching; relax deadlines only on empty assignments in full-visit mode. Heuristic for on-time completion, not a global maximum-completion solver.

Path worker processes: 8 (requested 8).

Planner: `visibility_dijkstra`; epsilon: `0.01`. Shared preprocessing wall: 73.267391 s.

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |
|---:|---:|---:|---:|---:|---:|
| 2 | 51 | 255.309479 | 47.980% | 157.604 | 1788.045 |
| 3 | 54 | 167.119719 | 61.796% | 172.382 | 1824.967 |
| 4 | 48 | 120.365991 | 69.354% | 194.433 | 1855.926 |
| 5 | 53 | 84.101209 | 76.887% | 211.866 | 1884.024 |
| 6 | 49 | 65.051984 | 80.776% | 233.178 | 1879.226 |
| 7 | 45 | 52.253179 | 83.911% | 257.729 | 1925.803 |
| All | 300 | 126.063734 | 69.740% | 203.105 | 1858.065 |

Unassigned tasks: 0; relaxed decisions: 8795.

Shared path preprocessing: planning 557.419436 s; cache read 0.000000 s; cache write 7.735019 s; hits 0, misses 300.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, path preprocessing/cache I/O, warmup, replay checks and output I/O. Path planning is shared across protocols, excludes cache I/O, and is zero on cache hits. Parallel planning totals sum scenario worker times; preprocessing wall includes worker startup/shutdown. Existing CapAM batch-32 amortized times are a different timing regime.

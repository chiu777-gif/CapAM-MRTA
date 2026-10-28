# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions. Completion mode relaxes deadlines only on empty assignments; cost mode allows late-task edges and minimizes immediate penalty in each matching. Both visit every task and report the same cost metric as CapAM.

## capam_full_visit

Objective: `cost`. Hungarian minimum predicted added deadline cost: 0 when finish <= deadline, otherwise finish/deadline. All remaining tasks are eligible, including late ones. Equal-cost assignments follow SciPy deterministic row/task order. If the current robot is unmatched when tasks are fewer than robots, select its minimum-cost task. Visits every task; local matching does not guarantee minimum final schedule cost.

Planner: `visibility_dijkstra`; epsilon: `1e-06`. Shared preprocessing wall: 1.883952 s.

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 777.524377 | 18.400% | 61.419 | 0.000 |
| 3 | 14 | 473.647727 | 27.500% | 62.052 | 0.000 |
| 4 | 24 | 342.311054 | 35.750% | 63.028 | 0.000 |
| 5 | 15 | 270.445510 | 43.467% | 63.763 | 0.000 |
| 6 | 16 | 225.464653 | 49.188% | 65.465 | 0.000 |
| 7 | 21 | 193.964813 | 53.048% | 65.005 | 0.000 |
| All | 100 | 343.591554 | 39.800% | 63.646 | 0.000 |

Unassigned tasks: 0; relaxed decisions: 0.

Shared path preprocessing: planning 0.000000 s; cache read 1.656078 s; cache write 0.000000 s; hits 100, misses 0.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, path preprocessing/cache I/O, warmup, replay checks and output I/O. Path planning is shared across protocols, excludes cache I/O, and is zero on cache hits. Existing CapAM batch-32 amortized times are a different timing regime.

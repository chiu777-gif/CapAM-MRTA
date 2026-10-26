# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions, with deadline relaxation only on empty assignments to visit all tasks. Its cost uses the same metric as CapAM.

## upstream_taptc

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario |
|---:|---:|---:|---:|---:|
| 2 | 147 | undefined (unassigned tasks) | 46.912% | 69.638 |
| 3 | 172 | undefined (unassigned tasks) | 60.721% | 135.894 |
| 4 | 183 | undefined (unassigned tasks) | 70.230% | 225.980 |
| 5 | 173 | undefined (unassigned tasks) | 76.393% | 315.997 |
| 6 | 149 | undefined (unassigned tasks) | 80.081% | 407.411 |
| 7 | 176 | undefined (unassigned tasks) | 83.886% | 480.576 |
| All | 1000 | undefined (unassigned tasks) | 70.104% | 274.918 |

Unassigned tasks: 28857; relaxed decisions: 0.

## capam_full_visit

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario |
|---:|---:|---:|---:|---:|
| 2 | 147 | 262.418053 | 47.803% | 148.208 |
| 3 | 172 | 166.905822 | 61.517% | 167.823 |
| 4 | 183 | 114.596447 | 71.115% | 185.688 |
| 5 | 173 | 82.245170 | 77.185% | 205.498 |
| 6 | 149 | 65.799310 | 80.685% | 223.395 |
| 7 | 176 | 49.415389 | 84.631% | 247.765 |
| All | 1000 | 120.984025 | 70.892% | 197.077 |

Unassigned tasks: 0; relaxed decisions: 28016.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, warmup, replay checks and output I/O. Existing CapAM batch-32 amortized times are a different timing regime.

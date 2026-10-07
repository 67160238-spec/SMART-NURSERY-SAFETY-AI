# hazard_object: model comparison on held-out images

- date: 2026-10-07 10:04
- dataset: `hazard_full` (built by tools/build_hazard_dataset.py; split by source, near-duplicates grouped)
- test images: 2513 (COCO hazard-free images capped at 1000)
- 95% bootstrap intervals in brackets (images resampled, 1000 reps)
- **These are web/COCO photos, not the nursery camera.** They rank models; they do not replace the QA clips (tools/eval_clips.py) or the Phase 2 Decision Gate.

| model | weights | sha256 | imgsz | person-guided zoom |
|---|---|---|---|---|
| base | `yolo11n.pt` | `0ebbc80d4a7680d1` | 960 | no |
| v1_full | `models/hazard/hazard_v1_full.pt` | `77897166ffa4760b` | 960 | no |
| v2 | `models/hazard/hazard_v2.pt` | `ef841aacd746ae3f` | 960 | no |
| v2_zoom | `models/hazard/hazard_v2.pt` | `ef841aacd746ae3f` | 960 | yes |

## 1. Box level, test split (AP50 %)

| model | source | person | knife | scissors |
|---|---|---|---|---|
| base | coco | 78.7 [76.5, 80.8] | 26.0 [18.9, 35.0] | 44.8 [23.0, 70.8] |
| base | hod | - | 31.9 [28.4, 36.0] | - |
| base | sohas | - | 5.7 [4.0, 8.5] | - |
| v1_full | coco | 73.0 [70.6, 75.3] | 15.7 [10.9, 22.8] | 11.4 [0.0, 33.1] |
| v1_full | hod | - | 72.5 [69.1, 76.3] | - |
| v1_full | sohas | - | 87.9 [84.7, 91.1] | - |
| v2 | coco | 74.0 [71.7, 76.3] | 21.9 [15.0, 30.3] | 42.7 [16.9, 71.5] |
| v2 | hod | - | 70.4 [66.8, 74.4] | - |
| v2 | sohas | - | 90.1 [87.1, 92.7] | - |
| v2_zoom | coco | 74.0 [71.7, 76.3] | 20.9 [14.2, 29.0] | 42.1 [16.0, 70.7] |
| v2_zoom | hod | - | 69.5 [65.9, 73.5] | - |
| v2_zoom | sohas | - | 89.1 [86.0, 92.0] | - |

## 2. Image level (what raises the alert), test split

Threshold `tuned` = min of the per-class F2-best thresholds on the validation split; `fixed` = 0.25 (today's config).

| model | threshold | source | hazard recall % | false-alarm rate % |
|---|---|---|---|---|
| base | fixed 0.25 | all | 32.9 [30.4, 35.6] | 0.8 [0.4, 1.4] |
| base | fixed 0.25 | coco | 51.9 [42.5, 60.9] | 1.0 [0.4, 1.7] |
| base | fixed 0.25 | hod | 43.3 [39.6, 47.4] | n/a |
| base | fixed 0.25 | sohas | 13.7 [10.6, 16.9] | 0.3 [0.0, 1.0] |
| base | tuned 0.05 | all | 55.2 [52.3, 58.0] | 3.1 [2.2, 4.0] |
| base | tuned 0.05 | coco | 79.8 [72.2, 86.8] | 3.6 [2.5, 4.8] |
| base | tuned 0.05 | hod | 68.8 [65.3, 72.2] | n/a |
| base | tuned 0.05 | sohas | 30.5 [26.5, 35.2] | 1.6 [0.3, 3.0] |
| v1_full | fixed 0.25 | all | 89.3 [87.6, 91.0] | 3.3 [2.3, 4.3] |
| v1_full | fixed 0.25 | coco | 48.1 [38.8, 58.1] | 2.7 [1.7, 3.8] |
| v1_full | fixed 0.25 | hod | 91.1 [88.9, 93.2] | n/a |
| v1_full | fixed 0.25 | sohas | 96.2 [94.4, 97.9] | 5.0 [2.8, 7.5] |
| v1_full | tuned 0.10 | all | 94.5 [93.2, 95.8] | 6.3 [5.0, 7.6] |
| v1_full | tuned 0.10 | coco | 66.3 [57.5, 75.5] | 5.5 [4.2, 6.8] |
| v1_full | tuned 0.10 | hod | 96.7 [95.3, 98.1] | n/a |
| v1_full | tuned 0.10 | sohas | 97.8 [96.4, 99.1] | 8.8 [5.7, 11.9] |
| v2 | fixed 0.25 | all | 89.8 [88.1, 91.4] | 3.9 [2.9, 5.0] |
| v2 | fixed 0.25 | coco | 52.9 [43.2, 61.5] | 3.4 [2.3, 4.6] |
| v2 | fixed 0.25 | hod | 90.6 [88.2, 92.8] | n/a |
| v2 | fixed 0.25 | sohas | 97.1 [95.6, 98.6] | 5.3 [3.0, 8.0] |
| v2 | tuned 0.20 | all | 91.4 [89.8, 92.9] | 5.1 [4.0, 6.4] |
| v2 | tuned 0.20 | coco | 58.7 [49.6, 67.6] | 4.4 [3.2, 5.7] |
| v2 | tuned 0.20 | hod | 92.2 [90.0, 94.2] | n/a |
| v2 | tuned 0.20 | sohas | 97.8 [96.4, 99.1] | 7.2 [4.6, 10.3] |
| v2_zoom | fixed 0.25 | all | 90.4 [88.7, 92.1] | 4.2 [3.2, 5.4] |
| v2_zoom | fixed 0.25 | coco | 55.8 [46.6, 64.6] | 3.8 [2.7, 5.0] |
| v2_zoom | fixed 0.25 | hod | 91.2 [88.9, 93.4] | n/a |
| v2_zoom | fixed 0.25 | sohas | 97.3 [95.8, 98.7] | 5.6 [3.3, 8.5] |
| v2_zoom | tuned 0.20 | all | 92.1 [90.6, 93.6] | 5.6 [4.4, 6.9] |
| v2_zoom | tuned 0.20 | coco | 62.5 [54.2, 71.7] | 5.0 [3.7, 6.3] |
| v2_zoom | tuned 0.20 | hod | 92.6 [90.4, 94.7] | n/a |
| v2_zoom | tuned 0.20 | sohas | 98.2 [97.0, 99.3] | 7.5 [4.9, 10.6] |

## 2b. Recall by object size (far objects are small), test split, threshold 0.25

Box recall of knife + scissors (either label counts). Size = sqrt(box area / image area).

| model | small (<3%) | medium (3-8%) | large (>8%) |
|---|---|---|---|
| base | 12.8 [0.0, 23.8] | 7.7 [3.5, 12.9] | 26.0 [23.5, 28.7] |
| v1_full | 0.0 [0.0, 0.0] | 44.2 [35.2, 55.2] | 79.4 [76.9, 82.0] |
| v2 | 23.1 [0.0, 42.9] | 46.8 [37.7, 57.7] | 78.3 [75.6, 80.9] |
| v2_zoom | 23.1 [0.0, 42.9] | 48.7 [39.8, 58.6] | 78.8 [76.1, 81.4] |
| (objects) | 39 | 156 | 1205 |

## 2c. Simulated distance (test hazard images shrunk x2, x3, x4), threshold 0.25

Same scene, object made N times smaller = about N times farther from the camera.

| model | x1: image recall / box recall | x2: image recall / box recall | x3: image recall / box recall | x4: image recall / box recall |
|---|---|---|---|---|
| base | 32.9 [30.3, 35.5] / 23.6 [21.2, 26.3] | 26.8 [24.2, 29.5] / 18.9 [16.8, 21.2] | 16.1 [14.2, 18.2] / 10.5 [8.9, 12.2] | 8.7 [7.2, 10.3] / 5.6 [4.4, 6.9] |
| v1_full | 89.3 [87.5, 91.1] / 73.3 [70.0, 76.4] | 83.2 [81.2, 85.4] / 66.4 [63.2, 69.8] | 68.7 [66.2, 71.4] / 50.9 [47.8, 54.3] | 52.4 [49.7, 55.2] / 36.6 [33.7, 39.4] |
| v2 | 89.8 [88.0, 91.5] / 73.3 [70.1, 76.3] | 82.0 [79.9, 84.2] / 65.1 [62.0, 68.5] | 64.7 [62.2, 67.4] / 49.1 [46.2, 52.4] | 47.6 [44.8, 50.5] / 34.2 [31.4, 37.2] |
| v2_zoom | 90.4 [88.9, 92.0] / 73.9 [70.8, 77.0] | 86.7 [84.7, 88.7] / 69.6 [66.5, 72.9] | 77.4 [75.1, 79.5] / 59.9 [56.8, 63.0] | 64.2 [61.5, 66.9] / 48.1 [45.0, 51.4] |

## 3. Per-class thresholds chosen on validation (F2)

| model | knife | scissors |
|---|---|---|
| base | 0.05 | 0.40 |
| v1_full | 0.20 | 0.10 |
| v2 | 0.25 | 0.20 |
| v2_zoom | 0.25 | 0.20 |

## 4. Paired differences vs `base` (test split, points; + = better)

| model | knife AP50 (all) | scissors AP50 (coco) | person AP50 (coco) | hazard recall @ 0.25 | small-object recall @ 0.25 | false-alarm rate @ 0.25 (− = better) |
|---|---|---|---|---|---|---|
| v1_full | 49.2 [45.2, 52.8] | -33.4 [-60.8, -10.8] | -5.7 [-6.7, -4.5] | 56.4 [53.4, 59.3] | -12.8 [-23.8, 0.0] | 2.4 [1.5, 3.4] |
| v2 | 50.0 [46.2, 53.4] | -2.1 [-28.6, 18.1] | -4.7 [-5.5, -3.7] | 56.9 [54.0, 59.8] | 10.3 [0.0, 20.0] | 3.0 [2.0, 4.2] |
| v2_zoom | 49.0 [45.1, 52.5] | -2.6 [-29.3, 17.7] | -4.7 [-5.5, -3.7] | 57.6 [54.7, 60.3] | 10.3 [0.0, 20.0] | 3.4 [2.5, 4.5] |

## Notes

- scissors test boxes: 14 (COCO only). Small n: read the interval, not the point.
- Web knife labels are incomplete (checked by eye); test labels were NOT completed by a model, so a correct box on an unlabelled knife counts as a false positive for every model alike.
- Sohas hard negatives (smartphone, purse, bill, card held in the hand) are in the false-alarm rate of the `sohas` rows.

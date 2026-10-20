# observer-blur-coverage

How often does a fixed-size face blur circle fully cover a face, in head-worn and in room-view video?

This repository measures that question for the face blur in [video-deid](https://github.com/kbjohnson-penn/video-deid), the video stage of the MedVidDeID pipeline (Mopidevi et al., SSRN 2025, doi 10.2139/ssrn.5194782) that de-identifies the Observer clinical video dataset. It uses public ground truth only. No Observer data was used.

## What is measured

In video-deid (commit `eb5ff161`, the commit MedVidDeID pins as its `video_deid` submodule), each tracked person's facial keypoints are averaged into one point. A box is built around that single point with a constant margin, and the blur is a circle drawn from that box:

- [`blur/processing.py` lines 45 to 46](https://github.com/kbjohnson-penn/video-deid/blob/eb5ff161d182d2985fc456438ea27422926bb85e/video_deid/blur/processing.py#L45-L46): `calculate_bounding_box` on one point with `margin=FACE_MARGIN`
- [`config.py` line 16](https://github.com/kbjohnson-penn/video-deid/blob/eb5ff161d182d2985fc456438ea27422926bb85e/video_deid/config.py#L16): `FACE_MARGIN = 50`
- [`blur/techniques.py` line 27](https://github.com/kbjohnson-penn/video-deid/blob/eb5ff161d182d2985fc456438ea27422926bb85e/video_deid/blur/techniques.py#L27): radius = 2 x half the box side, so 100 px

The radius does not change with face size or video resolution, and the command line has no option for it.

`gt_geometry.py` assumes perfect detection: it centres a 100 px circle exactly on the mean of the ground-truth facial keypoints and checks how much of an estimated face box falls inside it.

- Face: ground-truth COCO facial keypoints, with the nose and both eyes inside the frame.
- Face box: ear-to-ear span wide, from 0.6 span above to 0.8 span below the keypoint mean, sampled on a 9 x 9 grid.
- "Not fully covered": at least one grid point of the face box lies outside the circle.
- Also reported: a facial keypoint outside the circle, less than 90% of the box covered, face span in pixels.
- Paired comparison: only (sequence, frame) pairs present in both the head-worn and the room annotations.

## Data source (not included)

[EgoHumans](https://github.com/rawalkhirodkar/egohumans) (Khirodkar et al., ICCV 2023, [arXiv 2305.16487](https://arxiv.org/abs/2305.16487)): people wearing Aria glasses (head-worn RGB, 1408 x 1408) filmed at the same moments by static GoPro cameras (3840 x 2160).

- Licence: the EgoHumans code repository is Apache-2.0. The project page states no dataset licence. The COCO-style annotation files carry a CC BY-NC-SA 2.0 licence entry. Follow the dataset authors' terms.
- Only the two COCO keypoint annotation files were used, from the benchmark folder linked in the EgoHumans [download instructions](https://github.com/rawalkhirodkar/egohumans/blob/main/assets/DOWNLOAD.md): `person_keypoints_ego_rgb.json` and `person_keypoints_exo.json`. No images were downloaded or used.
- This repository contains no frames, images or annotation files, only code and aggregate numbers.

## How to run

Python 3.9 or newer, standard library only.

```
mkdir data
# put person_keypoints_ego_rgb.json and person_keypoints_exo.json in data/
head -c 201326592 data/person_keypoints_exo.json > data/person_keypoints_exo.part.json
python3 gt_geometry.py data/person_keypoints_ego_rgb.json data/person_keypoints_exo.part.json results/gt_geometry.json
python3 make_tables.py results/gt_geometry.json results/
```

The published numbers come from the first 201,326,592 bytes of the room-view file, because the download stopped there (Google Drive quota). That prefix covers the tagging sequences 001 to 007. The `head -c` line reproduces exactly that input, and the script parses up to the last complete annotation. With the full file, more sequences are paired and the numbers will change. `results/input_hashes.txt` lists the sha256 of both inputs.

## Results

Paired tagging sequences 001 to 007, 100 px circle, perfect detection assumed. Wilson 95% intervals.

| View | Face instances | Face box not fully covered | Facial keypoint outside circle | Face span p10 / p50 / p90 (px) |
|---|---|---|---|---|
| Head-worn (Aria glasses) | 21,877 | 2.5% [2.3, 2.7] | 0.6% | 19 / 32 / 64 |
| Room (4K GoPro) | 78,748 | 13.9% [13.7, 14.2] | 5.6% | 39 / 60 / 122 |

- Head-worn minus room, "not fully covered", bootstrapped over cameras: -14.0 to -8.7 percentage points (95% interval).
- Per room camera: 4.7% to 21.8% not fully covered. Per glasses camera: 2.1% to 2.9%.
- Seated lego assembly sequences 001 to 006 (head-worn only, the room file was cut before them): 7.2% to 15.4% not fully covered.
- Face size in pixels mattered more than camera type. In this data, faces were larger in the 4K room views than in the glasses.

Files in `results/`: `summary.csv` (every set and view), `per_camera.csv`, `per_person.csv`, `bootstrap_diff.txt`, `gt_geometry.json` (raw output), `input_hashes.txt`.

## Caveats

- Only blur geometry is measured. Detection misses, Kalman filter lag and the full-frame fallbacks of video-deid were not measured, and the detector itself was not run.
- All ground-truth keypoints carry visibility 2, so a turned-away head with nose and eyes projected into the frame still counts as a face. The face box is an anthropometric assumption.
- Clipping of the blur box at frame edges is not modelled.
- Frames are correlated, so the camera-level bootstrap is the honest interval. The frame-level Wilson intervals are optimistic.
- Tagging is an outdoor running game, not a clinic room. Observer's room view uses a 360-degree camera, and its resolution is not stated in the paper, so these figures may not transfer. Faces in panoramic footage likely occupy fewer pixels.

## Licence

Code and result tables: MIT, see `LICENSE`. EgoHumans data and video-deid code are not part of this repository and keep their own licences.

# AIrPPG

Lightweight multi-ROI rPPG experiments for heart-rate estimation from webcam/RGB face videos.

The current implementation focus is **Task 1: video preprocessing with MediaPipe Face Mesh landmark tracking and ROI extraction**. The generated Task 1 artifacts are the input for Task 2 signal extraction.

## 1. Install `uv`

This project uses Python 3.11 and [`uv`](https://docs.astral.sh/uv/) for environment management.

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal after installation, then verify:

```powershell
uv --version
```

## 2. Set up the project environment

From the repository root:

```powershell
uv sync
```

Start Jupyter Lab:

```powershell
uv run jupyter lab
```

Open:

```text
notebooks/task1_video_preprocessing.ipynb
```

## 3. Dataset layout

Dataset files are not committed to git. Put the local UBFC dataset under:

```text
datasets/UBFC_DATASET/
  DATASET_1/
    10-gt/
      vid.avi
      gtdump.xmp
    ...
  DATASET_2/
    subject1/
      vid.avi
      ground_truth.txt
    ...
```

The notebook discovers videos recursively under `datasets/UBFC_DATASET` by default.

You can override the dataset path without editing code:

```powershell
$env:DATASET_ROOT="D:/path/to/UBFC_DATASET"
```

## 4. Run Task 1 for one video

Use the first discovered dataset video:

```powershell
uv run jupyter lab
```

Or choose a specific video:

```powershell
$env:VIDEO_PATH="D:/path/to/UBFC_DATASET/DATASET_2/subject1/vid.avi"
uv run jupyter lab
```

Task 1 runs full video by default. For a quick debug run only:

```powershell
$env:MAX_FRAMES="300"
$env:RESIZE_WIDTH="320"
$env:OVERLAY_SAMPLE_COUNT="4"
uv run jupyter lab
```

To return to full-video processing:

```powershell
$env:MAX_FRAMES="None"
```

## 5. Run Task 1 for the full dataset

Batch mode processes every discovered video and mirrors the dataset split structure in `outputs/`:

```powershell
$env:PROCESS_ALL_VIDEOS="1"
$env:MAX_FRAMES="None"
uv run jupyter lab
```

For a small batch smoke test:

```powershell
$env:PROCESS_ALL_VIDEOS="1"
$env:BATCH_MAX_VIDEOS="2"
$env:MAX_FRAMES="30"
uv run jupyter lab
```

Useful environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATASET_ROOT` | `datasets/UBFC_DATASET` | Root folder scanned for videos. |
| `VIDEO_PATH` | first discovered video | Single video to process. |
| `PROCESS_ALL_VIDEOS` | `0` | Set to `1` to process all videos. |
| `BATCH_MAX_VIDEOS` | `None` | Limit number of videos in batch mode for debugging. |
| `MAX_FRAMES` | `None` | Limit frames per video for debugging. `None` means full video. |
| `RESIZE_WIDTH` | `640` | Resize width while preserving aspect ratio. |
| `TARGET_FPS` | `None` | Optional FPS downsampling. |
| `OVERLAY_SAMPLE_COUNT` | `8` | Number of PNG ROI overlay samples to save. |
| `SAVE_OVERLAY_VIDEO` | `0` | Set to `1` to save overlay MP4. |
| `SKIP_EXISTING` | `1` | Batch mode skips videos whose `.npz` and `.json` already exist. |

## 6. Task 1 outputs

Single-video and batch outputs are written under:

```text
outputs/task1_preprocessing/
```

Batch mode mirrors the dataset split:

```text
outputs/task1_preprocessing/
  DATASET_1/
    10-gt/
      vid_roi_data.npz
      vid_metadata.json
      vid_overlay_samples/
        overlay_000.png
        ...
  DATASET_2/
    subject1/
      vid_roi_data.npz
      vid_metadata.json
      vid_overlay_samples/
        overlay_000.png
        ...
  task1_dataset_manifest.json
```

### `.npz` arrays

`*_roi_data.npz` contains:

| Array | Shape | Meaning for Task 2 |
| --- | --- | --- |
| `frame_indices` | `(N,)` | Original video frame indices that were processed. Use this to align with ground truth. |
| `landmarks_px` | `(N, 478, 2)` | Face Mesh landmark coordinates in pixels. Useful for motion/jitter quality features. |
| `roi_polygons_px` | `(N, 3, P, 2)` | Forehead, left cheek, right cheek ROI polygons in pixels. |
| `roi_masks` | `(N, 3, H, W)` | Binary ROI masks. Use these to compute mean RGB per ROI per frame. |
| `valid` | `(N,)` | Whether the frame has valid face landmarks and ROI masks. |
| `error_reasons` | `(N,)` | Error reason for invalid frames, e.g. `no_face`. |
| `roi_names` | `(3,)` | ROI axis labels: `forehead`, `left_cheek`, `right_cheek`. |

### `.json` metadata

`*_metadata.json` contains:

- source video path;
- original FPS, resolution, frame count, duration;
- preprocessing config;
- ROI landmark indices;
- valid/invalid frame counts;
- failure statistics;
- overlay sample paths.

## 7. Quick `.npz` inspection

Use the helper script:

```powershell
uv run python inspect_task1_npz.py
```

Or pass a specific artifact:

```powershell
uv run python inspect_task1_npz.py outputs/task1_preprocessing/DATASET_1/10-gt/vid_roi_data.npz
```

## 8. What Task 2 should consume

Task 2 should read the Task 1 `.npz` and source video, then compute per-ROI RGB time series:

```text
forehead_R(t), forehead_G(t), forehead_B(t)
left_cheek_R(t), left_cheek_G(t), left_cheek_B(t)
right_cheek_R(t), right_cheek_G(t), right_cheek_B(t)
```

Use `roi_masks` to select pixels inside each ROI and `valid` / `error_reasons` to avoid silently using failed detections. Use `frame_indices` and video FPS from metadata to align the signal with ground truth.

Generated data should stay under ignored local folders such as `outputs/`, `artifacts/`, or `datasets/`.

"""Mean-RGB extraction from preprocessing ROI masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from airppg.signal_extraction.loaders import get_roi_names


def extract_mean_rgb(
    video_path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    max_frames: int | None = None,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    frame_indices = arrays["frame_indices"].astype(np.int64)
    roi_masks = arrays["roi_masks"].astype(bool)
    valid = arrays["valid"].astype(bool)
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
        roi_masks = roi_masks[:max_frames]
        valid = valid[:max_frames]

    fps = float(metadata["video"]["fps"])
    timestamps = frame_indices.astype(np.float64) / fps
    mean_rgb = np.full((len(frame_indices), roi_masks.shape[1], 3), np.nan, dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source video: {video_path}")

    iterator = range(len(frame_indices))
    if progress:
        iterator = tqdm(iterator, desc=f"RGB {Path(video_path).stem}", unit="frame", leave=False)

    last_frame = -1
    try:
        for out_idx in iterator:
            if not valid[out_idx]:
                continue
            frame_idx = int(frame_indices[out_idx])
            if frame_idx != last_frame + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                last_frame = frame_idx - 1
            ok, frame_bgr = cap.read()
            last_frame += 1
            if not ok:
                continue
            mask_h, mask_w = roi_masks.shape[2:]
            frame_h, frame_w = frame_bgr.shape[:2]
            if frame_h != mask_h or frame_w != mask_w:
                frame_bgr = cv2.resize(frame_bgr, (mask_w, mask_h), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            for roi_idx in range(roi_masks.shape[1]):
                mask = roi_masks[out_idx, roi_idx]
                if mask.any():
                    mean_rgb[out_idx, roi_idx] = frame_rgb[mask].mean(axis=0)
    finally:
        cap.release()

    return {
        "frame_indices": frame_indices,
        "timestamps": timestamps,
        "valid": valid,
        "mean_rgb": mean_rgb,
        "roi_names": get_roi_names(arrays, metadata),
    }

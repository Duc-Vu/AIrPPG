"""Signal extraction from video frames using ROI masks from Task 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm.auto import tqdm


def load_task1_artifact(npz_path: Path) -> dict[str, np.ndarray]:
    """Load Task 1 ROI data artifact.
    
    Args:
        npz_path: Path to the .npz file from Task 1.
        
    Returns:
        Dictionary containing arrays: frame_indices, landmarks_px, roi_polygons_px,
        roi_masks, valid, error_reasons, roi_names.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_task1_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load Task 1 metadata JSON.
    
    Args:
        metadata_path: Path to the metadata JSON file from Task 1.
        
    Returns:
        Dictionary containing video metadata, config, ROI info, etc.
    """
    import json
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_rgb_signals_from_video(
    video_path: Path,
    roi_masks: np.ndarray,
    frame_indices: np.ndarray,
    valid: np.ndarray,
    resize_width: int | None = None,
    progress_bar: bool = True,
) -> dict[str, np.ndarray]:
    """Extract RGB signals from video frames using ROI masks.
    
    Args:
        video_path: Path to the source video file.
        roi_masks: Array of shape (N, 3, H, W) with binary ROI masks for each frame.
        frame_indices: Array of frame indices that were processed in Task 1.
        valid: Boolean array indicating which frames have valid ROI masks.
        resize_width: Optional width to resize frames (should match Task 1 config).
        progress_bar: Whether to show progress bar.
        
    Returns:
        Dictionary with keys:
        - 'rgb_signals': Array of shape (N, 3, 3) with mean RGB values for each ROI.
        - 'frame_indices': The frame indices (aligned with rgb_signals).
        - 'valid': The valid flags (aligned with rgb_signals).
        - 'roi_names': Array of ROI names ['forehead', 'left_cheek', 'right_cheek'].
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    try:
        n_frames = len(frame_indices)
        n_rois = roi_masks.shape[1] if len(roi_masks.shape) > 1 else 3
        
        rgb_signals = np.full((n_frames, n_rois, 3), np.nan, dtype=np.float32)
        
        iterator = range(n_frames)
        if progress_bar:
            iterator = tqdm(iterator, desc="Extracting RGB signals", unit="frame")
        
        for i in iterator:
            frame_idx = frame_indices[i]
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            
            if not ret:
                continue
            
            # Resize if needed
            if resize_width is not None:
                h, w = frame_bgr.shape[:2]
                if w != resize_width:
                    scale = resize_width / float(w)
                    new_size = (resize_width, max(1, int(round(h * scale))))
                    frame_bgr = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Extract mean RGB for each ROI
            for roi_idx in range(n_rois):
                mask = roi_masks[i, roi_idx]
                if mask.sum() > 0:  # Valid ROI
                    # Extract RGB values where mask is 1
                    for channel in range(3):
                        rgb_signals[i, roi_idx, channel] = frame_rgb[:, :, channel][mask == 1].mean()
        
        roi_names = np.array(['forehead', 'left_cheek', 'right_cheek'], dtype=object)
        
        return {
            'rgb_signals': rgb_signals,
            'frame_indices': frame_indices,
            'valid': valid,
            'roi_names': roi_names,
        }
    finally:
        cap.release()


def extract_rgb_signals_from_task1(
    task1_npz_path: Path,
    task1_metadata_path: Path,
    video_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Extract RGB signals using Task 1 artifacts.
    
    This is a convenience function that loads Task 1 artifacts and extracts
    RGB signals from the video.
    
    Args:
        task1_npz_path: Path to Task 1 .npz file.
        task1_metadata_path: Path to Task 1 metadata JSON.
        video_path: Optional path to video. If None, uses path from metadata.
        
    Returns:
        Dictionary with rgb_signals, frame_indices, valid, roi_names.
    """
    # Load Task 1 artifacts
    task1_data = load_task1_artifact(task1_npz_path)
    metadata = load_task1_metadata(task1_metadata_path)
    
    # Get video path
    if video_path is None:
        video_path = Path(metadata['video']['path'])
    
    # Get resize width from metadata
    resize_width = metadata['config'].get('resize_width')
    
    # Extract signals
    return extract_rgb_signals_from_video(
        video_path=video_path,
        roi_masks=task1_data['roi_masks'],
        frame_indices=task1_data['frame_indices'],
        valid=task1_data['valid'],
        resize_width=resize_width,
    )

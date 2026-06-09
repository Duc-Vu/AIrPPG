"""Face Mesh landmark ROI polygon and mask helpers."""

from __future__ import annotations

import cv2
import numpy as np

from airppg.schemas import ROI_NAMES

EXPECTED_LANDMARKS = 478
FOREHEAD_IDX = np.array([109, 10, 338, 337, 336, 296, 334, 293, 300, 70, 63, 105, 66, 107], dtype=np.int32)
LEFT_CHEEK_IDX = np.array([50, 101, 118, 119, 100, 47, 126, 209, 49, 203, 205, 187], dtype=np.int32)
RIGHT_CHEEK_IDX = np.array([280, 330, 347, 348, 329, 277, 355, 429, 279, 423, 425, 411], dtype=np.int32)
ROI_LANDMARKS = {
    "forehead": FOREHEAD_IDX,
    "left_cheek": LEFT_CHEEK_IDX,
    "right_cheek": RIGHT_CHEEK_IDX,
}


def landmarks_to_pixels(face_landmarks: object, width: int, height: int) -> np.ndarray:
    points = np.zeros((len(face_landmarks.landmark), 2), dtype=np.float32)
    for idx, landmark in enumerate(face_landmarks.landmark):
        points[idx] = (landmark.x * width, landmark.y * height)
    return points


def polygon_area(polygon: np.ndarray) -> float:
    if polygon.size == 0:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_polygon(polygon: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = np.asarray(polygon, dtype=np.float32).copy()
    if clipped.size == 0:
        return clipped.reshape(0, 2)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(0, width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(0, height - 1))
    return clipped


def make_mask(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon.size == 0:
        return mask.astype(bool)
    pts = np.round(polygon).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def pack_roi_polygons(polygons: dict[str, np.ndarray]) -> np.ndarray:
    max_points = max((polygon.shape[0] for polygon in polygons.values()), default=0)
    packed = np.full((len(ROI_NAMES), max_points, 2), np.nan, dtype=np.float32)
    for idx, name in enumerate(ROI_NAMES):
        polygon = polygons[name].astype(np.float32, copy=False)
        packed[idx, : polygon.shape[0], :] = polygon
    return packed


def extract_roi_polygons(
    landmarks_px: np.ndarray,
    width: int,
    height: int,
    min_area_px: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, bool, str | None]:
    polygons: dict[str, np.ndarray] = {}
    masks = np.zeros((len(ROI_NAMES), height, width), dtype=bool)
    for roi_idx, name in enumerate(ROI_NAMES):
        polygon = clip_polygon(landmarks_px[ROI_LANDMARKS[name]], width, height)
        polygons[name] = polygon
        if polygon_area(polygon) < min_area_px:
            return polygons, masks, False, f"roi_area_too_small:{name}"
        masks[roi_idx] = make_mask(polygon, (height, width))
        if not masks[roi_idx].any():
            return polygons, masks, False, f"roi_mask_empty:{name}"
    return polygons, masks, True, None


def draw_rois(frame_rgb: np.ndarray, polygons: dict[str, np.ndarray], valid: bool = True) -> np.ndarray:
    canvas = frame_rgb.copy()
    colors = {"forehead": (255, 180, 0), "left_cheek": (0, 220, 100), "right_cheek": (0, 160, 255)}
    for name, polygon in polygons.items():
        if polygon.size == 0:
            continue
        pts = np.round(polygon).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=colors.get(name, (255, 255, 255)), thickness=2)
        cv2.putText(canvas, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.get(name, (255, 255, 255)), 1, cv2.LINE_AA)
    if not valid:
        cv2.putText(canvas, "INVALID", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    return canvas

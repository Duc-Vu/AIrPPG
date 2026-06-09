from __future__ import annotations

from pathlib import Path

import numpy as np

from airppg.fusion.metrics import FUSION_STRATEGY_NAMES, aggregate_metric_rows, build_metric_rows
from airppg.fusion.quality import FEATURE_NAMES, compute_quality_features, compute_roi_weights
from airppg.fusion.strategies import (
    average_fuse_signals,
    estimate_fused_hr,
    estimate_single_roi_hr,
    make_windows,
    quality_weighted_fuse_signals,
)
from airppg.metrics import align_ground_truth
from airppg.preprocessing.dataset import assign_dataset_splits, sample_id_for_video, source_relpath
from airppg.preprocessing.ground_truth import load_and_normalize_ground_truth
from airppg.preprocessing.roi import clip_polygon, extract_roi_polygons, make_mask
from airppg.signal import estimate_hr_trace, preprocess_trace
from airppg.signal_extraction.baselines import run_baselines


def test_ground_truth_parsers_dataset_1_and_2(tmp_path: Path) -> None:
    d1 = tmp_path / "gtdump.xmp"
    d1.write_text("0,70,98,0.1\n1000,72,99,0.2\n", encoding="utf-8")
    parsed1 = load_and_normalize_ground_truth(d1, "DATASET_1")
    assert parsed1["metadata"]["source_format"] == "ubfc_dataset_1_gtdump_xmp"
    np.testing.assert_allclose(parsed1["arrays"]["timestamps"], [0.0, 1.0])
    np.testing.assert_allclose(parsed1["arrays"]["spo2_percent"], [98.0, 99.0])

    d2 = tmp_path / "ground_truth.txt"
    d2.write_text("0.1 0.2 0.3\n71 72 73\n0 1 2\n", encoding="utf-8")
    parsed2 = load_and_normalize_ground_truth(d2, "DATASET_2")
    assert parsed2["metadata"]["missing_fields"] == ["spo2_percent"]
    np.testing.assert_allclose(parsed2["arrays"]["heart_rate_bpm"], [71.0, 72.0, 73.0])
    assert np.isnan(parsed2["arrays"]["spo2_percent"]).all()


def test_dataset_paths_and_splits_are_portable_and_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "datasets" / "UBFC_DATASET"
    videos = [root / "DATASET_2" / f"subject{i}" / "vid.avi" for i in range(1, 7)]
    for video in videos:
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"")
    ids = [sample_id_for_video(video, root) for video in videos]
    assert ids == [sample_id_for_video(video, root) for video in videos]
    assert len(set(ids)) == len(ids)
    assert source_relpath(videos[0], root, "datasets/UBFC_DATASET").startswith(
        "datasets/UBFC_DATASET/DATASET_2/subject1/"
    )
    split_a = assign_dataset_splits(videos, root, {"train": 0.8, "val": 0.1, "test": 0.1}, 42)
    split_b = assign_dataset_splits(videos, root, {"train": 0.8, "val": 0.1, "test": 0.1}, 42)
    assert split_a == split_b
    assert set(split_a.values()) == {"train", "val", "test"}


def test_roi_clipping_masks_and_small_area_invalidation() -> None:
    polygon = np.array([[-5, -5], [5, 0], [5, 5], [0, 5]], dtype=np.float32)
    clipped = clip_polygon(polygon, width=4, height=4)
    assert clipped.min() >= 0
    assert clipped.max() <= 3
    mask = make_mask(clipped, (4, 4))
    assert mask.dtype == bool
    assert mask.any()

    landmarks = np.zeros((478, 2), dtype=np.float32)
    _, masks, valid, reason = extract_roi_polygons(landmarks, width=8, height=8, min_area_px=1.0)
    assert not valid
    assert reason == "roi_area_too_small:forehead"
    assert masks.shape == (3, 8, 8)


def test_signal_preprocessing_and_hr_estimation_on_sinusoid() -> None:
    fps = 30.0
    duration = 20.0
    timestamps = np.arange(int(fps * duration), dtype=np.float64) / fps
    signal = np.sin(2.0 * np.pi * 1.5 * timestamps)
    signal[10:15] = np.nan
    processed = preprocess_trace(signal, fps, 42.0, 210.0)
    assert np.isfinite(processed).all()
    hr, centers = estimate_hr_trace(processed, timestamps, fps, 10.0, 1.0, 42.0, 210.0)
    assert centers.size == hr.size
    assert np.nanmedian(hr) == 90.0


def test_alignment_and_baseline_shapes_on_synthetic_rgb() -> None:
    gt = {"timestamps": np.array([0.0, 1.0]), "heart_rate_bpm": np.array([60.0, 90.0])}
    np.testing.assert_allclose(align_ground_truth(gt, np.array([0.5])), [75.0])

    fps = 30.0
    timestamps = np.arange(300, dtype=np.float64) / fps
    pulse = np.sin(2.0 * np.pi * 1.5 * timestamps)
    mean_rgb = np.zeros((timestamps.size, 3, 3), dtype=np.float32)
    mean_rgb[..., 0] = 120.0
    mean_rgb[..., 1] = (100.0 + pulse[:, None]).astype(np.float32)
    mean_rgb[..., 2] = 80.0
    mean_rgb[5:8, 1, :] = np.nan

    baselines = run_baselines(mean_rgb, fps, 42.0, 210.0)
    assert set(baselines) == {"green", "chrom", "pos"}
    for result in baselines.values():
        assert result["signal"].shape == (timestamps.size,)
        assert result["per_roi"].shape == (timestamps.size, 3)
        assert np.isfinite(result["signal"]).all()

def test_fusion_window_geometry_matches_signal_extraction() -> None:
    fps = 29.97
    n_frames = 1000
    windows = make_windows(n_frames, fps, 30.0, 1.0)
    window_size = min(max(3, int(round(30.0 * fps))), n_frames)
    step_size = max(1, int(round(1.0 * fps)))
    assert windows == [
        (start, start + window_size) for start in range(0, n_frames - window_size + 1, step_size)
    ]


def test_fusion_single_roi_hr_on_synthetic_sinusoid() -> None:
    fps = 30.0
    timestamps = np.arange(int(fps * 20.0), dtype=np.float64) / fps
    pulse = np.sin(2.0 * np.pi * 1.5 * timestamps)
    rppg_roi_signals = np.stack(
        [np.stack([pulse, pulse * 0.8, pulse * 1.2], axis=1) for _ in range(3)],
        axis=0,
    ).astype(np.float32)
    single_roi_hr, centers = estimate_single_roi_hr(
        rppg_roi_signals, timestamps, fps, 10.0, 1.0, 42.0, 210.0
    )
    assert single_roi_hr.shape == (centers.size, 3, 3)
    np.testing.assert_allclose(np.nanmedian(single_roi_hr, axis=0), np.full((3, 3), 90.0))


def test_fusion_quality_weights_are_normalized_with_fallback() -> None:
    features = np.full((2, 3, 3, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    features[0, :, :, 0] = 1.0
    features[0, :, :, 1] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    features[0, :, :, 2] = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    features[0, :, :, 3] = np.array([0.0, 3.0, 6.0], dtype=np.float32)
    features[0, :, :, 4] = np.array([5.0, 2.0, 1.0], dtype=np.float32)
    features[1, :, :, 0] = np.array([0.2, 0.9, 0.1], dtype=np.float32)
    weights = compute_roi_weights(features)
    assert weights.shape == (2, 3, 3)
    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=2), 1.0)
    np.testing.assert_allclose(weights[1, :, :], np.tile([0.0, 1.0, 0.0], (3, 1)))


def test_fusion_artifact_schema_and_metric_rows_on_synthetic_arrays(tmp_path: Path) -> None:
    fps = 30.0
    timestamps = np.arange(int(fps * 12.0), dtype=np.float64) / fps
    pulse = np.sin(2.0 * np.pi * 1.5 * timestamps)
    rppg_roi_signals = np.stack(
        [np.stack([pulse, pulse * 0.9, pulse * 1.1], axis=1) for _ in range(3)],
        axis=0,
    ).astype(np.float32)
    valid = np.ones(timestamps.size, dtype=bool)
    windows = make_windows(timestamps.size, fps, 10.0, 1.0)
    single_roi_hr, centers = estimate_single_roi_hr(
        rppg_roi_signals, timestamps, fps, 10.0, 1.0, 42.0, 210.0
    )
    features = compute_quality_features(
        rppg_roi_signals, valid, windows, fps, single_roi_hr, 42.0, 210.0
    )
    weights = compute_roi_weights(features)
    average_signals = average_fuse_signals(rppg_roi_signals)
    average_hr, _ = estimate_fused_hr(
        average_signals, timestamps, fps, 10.0, 1.0, 42.0, 210.0
    )
    weighted_signals = quality_weighted_fuse_signals(
        rppg_roi_signals, weights, windows, average_signals
    )
    weighted_hr, _ = estimate_fused_hr(
        weighted_signals, timestamps, fps, 10.0, 1.0, 42.0, 210.0
    )
    baseline_hr = average_hr.copy()
    baseline_signals = average_signals.copy()
    gt_aligned = np.full(centers.shape, 90.0, dtype=np.float32)
    method_names = np.array(["green", "chrom", "pos"])
    roi_names = np.array(["forehead", "left_cheek", "right_cheek"])
    rows = build_metric_rows(
        "sample",
        "test",
        method_names,
        roi_names,
        centers,
        gt_aligned,
        single_roi_hr,
        average_hr,
        weighted_hr,
        baseline_hr,
        rppg_roi_signals,
        average_signals,
        weighted_signals,
        baseline_signals,
        fps,
        42.0,
        210.0,
    )
    assert len(rows) == 3 * len(FUSION_STRATEGY_NAMES)
    assert {row["strategy"] for row in rows} == set(FUSION_STRATEGY_NAMES)
    assert aggregate_metric_rows(rows)

    artifact_path = tmp_path / "fusion_signals.npz"
    np.savez_compressed(
        artifact_path,
        schema_version=np.array("fusion_v1"),
        sample_id=np.array("sample"),
        split=np.array("test"),
        timestamps=timestamps.astype(np.float32),
        valid=valid,
        roi_names=roi_names,
        method_names=method_names,
        fusion_strategy_names=np.array(FUSION_STRATEGY_NAMES),
        feature_names=np.array(FEATURE_NAMES),
        hr_timestamps=centers.astype(np.float32),
        gt_aligned_to_hr=gt_aligned,
        baseline_hr=baseline_hr,
        single_roi_hr=single_roi_hr,
        average_fusion_hr=average_hr,
        weighted_fusion_hr=weighted_hr,
        roi_quality_features=features,
        roi_weights=weights,
        average_fused_rppg_signals=average_signals,
        weighted_fused_rppg_signals=weighted_signals,
    )
    with np.load(artifact_path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "sample_id",
            "split",
            "timestamps",
            "valid",
            "roi_names",
            "method_names",
            "fusion_strategy_names",
            "feature_names",
            "hr_timestamps",
            "gt_aligned_to_hr",
            "baseline_hr",
            "single_roi_hr",
            "average_fusion_hr",
            "weighted_fusion_hr",
            "roi_quality_features",
            "roi_weights",
            "average_fused_rppg_signals",
            "weighted_fused_rppg_signals",
        }
        assert required <= set(data.files)
        np.testing.assert_allclose(data["roi_weights"].sum(axis=2), 1.0)
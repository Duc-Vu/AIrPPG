"""Dataset discovery, portable IDs, and split helpers for preprocessing."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from airppg.paths import project_relpath, relpath_from_root
from airppg.schemas import VIDEO_EXTENSIONS


def discover_dataset_videos(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        return []
    return sorted(
        path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def dataset_relative_parts(video_path: Path, dataset_root: Path) -> tuple[str, ...]:
    try:
        return video_path.resolve().relative_to(dataset_root.resolve()).parts
    except ValueError:
        return video_path.parts[-3:]


def source_dataset_name(video_path: Path, dataset_root: Path) -> str:
    parts = dataset_relative_parts(video_path, dataset_root)
    return parts[0] if parts and parts[0].startswith("DATASET_") else "UNSORTED"


def video_subject_name(video_path: Path, dataset_root: Path) -> str:
    parts = dataset_relative_parts(video_path, dataset_root)
    if len(parts) >= 2 and parts[0].startswith("DATASET_"):
        return parts[1]
    return video_path.parent.name


def slug(text: str) -> str:
    chars: list[str] = []
    previous_sep = False
    for char in text.lower():
        if char.isalnum():
            chars.append(char)
            previous_sep = False
        elif not previous_sep:
            chars.append("-")
            previous_sep = True
    return "".join(chars).strip("-") or "sample"


def sample_id_for_video(video_path: Path, dataset_root: Path) -> str:
    source = source_dataset_name(video_path, dataset_root)
    source_suffix = {"DATASET_1": "d1", "DATASET_2": "d2"}.get(source, slug(source))
    subject = slug(video_subject_name(video_path, dataset_root))
    stem = slug(video_path.stem)
    rel = "/".join(dataset_relative_parts(video_path, dataset_root))
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    return f"ubfc_{source_suffix}_{subject}_{stem}_{digest}"


def stable_split_key(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def split_counts_for_n(n: int, ratios: dict[str, float]) -> dict[str, int]:
    if n <= 0:
        return {"train": 0, "val": 0, "test": 0}
    test_count = int(round(n * float(ratios.get("test", 0.1))))
    val_count = int(round(n * float(ratios.get("val", 0.1))))
    if n >= 3:
        test_count = max(1, test_count)
        val_count = max(1, val_count)
    while val_count + test_count > max(0, n - 1):
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break
    return {"train": n - val_count - test_count, "val": val_count, "test": test_count}


def _assign_splits_for_group(videos: list[Path], dataset_root: Path, ratios: dict[str, float], seed: int) -> dict[Path, str]:
    ordered = sorted(videos, key=lambda path: stable_split_key(sample_id_for_video(path, dataset_root), seed))
    counts = split_counts_for_n(len(ordered), ratios)
    assignments: dict[Path, str] = {}
    idx = 0
    for split in ("test", "val", "train"):
        for path in ordered[idx : idx + counts[split]]:
            assignments[path] = split
        idx += counts[split]
    return assignments


def assign_dataset_splits(videos: list[Path], dataset_root: Path, ratios: dict[str, float], seed: int) -> dict[Path, str]:
    if not videos:
        return {}
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in videos:
        groups[source_dataset_name(path, dataset_root)].append(path)
    if len(groups) > 1 and all(len(paths) >= 3 for paths in groups.values()):
        assignments: dict[Path, str] = {}
        for paths in groups.values():
            assignments.update(_assign_splits_for_group(paths, dataset_root, ratios, seed))
        return assignments
    return _assign_splits_for_group(videos, dataset_root, ratios, seed)


def source_relpath(path: Path, dataset_root: Path, dataset_root_hint: str) -> str:
    try:
        relative = path.resolve().relative_to(dataset_root.resolve()).as_posix()
        return f"{dataset_root_hint.rstrip('/')}/{relative}" if dataset_root_hint else relative
    except ValueError:
        return project_relpath(path)


def video_output_dir(output_root: Path, split: str, sample_id: str) -> Path:
    return output_root / split / sample_id


def manifest_item_for_paths(
    video_path: Path,
    dataset_root: Path,
    dataset_root_hint: str,
    output_root: Path,
    split: str,
    sample_id: str,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    out_dir = video_output_dir(output_root, split, sample_id)
    item: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "status": status,
        "source_dataset": source_dataset_name(video_path, dataset_root),
        "subject": video_subject_name(video_path, dataset_root),
        "video_relpath": source_relpath(video_path, dataset_root, dataset_root_hint),
        "output_dir": relpath_from_root(out_dir, output_root),
        "roi_npz": relpath_from_root(out_dir / "roi_data.npz", output_root),
        "metadata_json": relpath_from_root(out_dir / "metadata.json", output_root),
        "ground_truth_npz": relpath_from_root(out_dir / "ground_truth.npz", output_root),
    }
    if error:
        item["error"] = error
    return item

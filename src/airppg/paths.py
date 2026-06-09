"""Path and JSON helpers used by notebooks and pipelines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def resolve_project_root(cwd: Path | None = None) -> Path:
    """Return the repository root whether the process starts in repo root or notebooks/."""
    root = (cwd or Path.cwd()).resolve()
    return root.parent if root.name == "notebooks" else root


def resolve_input_path(path: str | Path, project_root: Path | None = None) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (resolve_project_root(project_root) / p).resolve()


def project_relpath(path: str | Path, project_root: Path | None = None) -> str:
    root = resolve_project_root(project_root)
    p = Path(path).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def relpath_from_root(path: str | Path, root: str | Path) -> str:
    p = Path(path).resolve()
    base = Path(root).resolve()
    try:
        return p.relative_to(base).as_posix()
    except ValueError:
        return p.as_posix()


def portable_config_path(path: str | Path, project_root: Path | None = None) -> str:
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    root = resolve_project_root(project_root)
    try:
        return p.resolve().relative_to(root).as_posix()
    except ValueError:
        return p.name


def resolve_relative(path_text: str | Path, base: str | Path) -> Path:
    p = Path(path_text)
    return p.resolve() if p.is_absolute() else (Path(base) / p).resolve()


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return to_builtin(value.item())
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, ensure_ascii=False, indent=2, allow_nan=False)

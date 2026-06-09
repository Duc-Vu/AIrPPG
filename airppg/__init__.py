"""Local import shim for the src-layout package in Unicode Windows workspaces."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "src" / "airppg")]

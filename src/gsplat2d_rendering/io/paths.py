"""Resolves a user-supplied model path to an actual .ply file.

Accepts either a direct path to a .ply, or a training output directory
(`<model_dir>/point_cloud/iteration_<N>/point_cloud.ply`, the standard
Gaussian-Splatting-family training layout) -- so a caller can point at
whichever one is on hand without manually navigating into the iteration
subfolder.
"""
from __future__ import annotations

from pathlib import Path


def resolve_ply_path(path: str | Path, iterations: int = 30000) -> Path:
    path = Path(path)
    if path.suffix.lower() == ".ply":
        if not path.is_file():
            raise FileNotFoundError(f"PLY not found: {path}")
        return path

    candidate = path / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"PLY not found at {candidate} -- pass a direct .ply path, or a training "
            f"model directory together with the right 'iterations' value (got {iterations})"
        )
    return candidate

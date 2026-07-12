"""Loads a trained 2D Gaussian Splatting PLY into a GaussianModel.

PLY layout is the 2D-GS training format's own contract (not specific to any
downstream project): `f_rest_` properties are laid out `[N, 3*K]` with
axis-1 ordered `[R_0..R_{K-1}, G_0..G_{K-1}, B_0..B_{K-1}]`, so the reshape
must be `reshape(-1, 3, K).transpose(0, 2, 1)` -- reshaping directly to
`(-1, K, 3)` silently scrambles the RGB channels into the wrong colors.
Quaternions are stored `(w, x, y, z)`. Note `scale_` may have 2 or 3
properties -- 2D-GS surfels only need 2 (in-plane) scale axes, unlike
3D-GS's 3.

A source .ply may already have been through offline compression (e.g. a
lower SH degree already baked in, or -- from some other tool -- int8-packed
rotation/normals). `compression_level`/`target_sh_degree` here operate on
whatever the file *currently* contains: re-"compressing" an already-reduced
SH degree just clamps to `min(current, target)` (a no-op if target is
already looser), it never restores detail that's already gone. int8-packed
fields are explicitly rejected below rather than silently reinterpreted as
raw floats, which would corrupt every quaternion/normal without raising an
error.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from gsplat2d_rendering._log import info, warning
from gsplat2d_rendering.compression import apply_compression, prune_low_opacity
from gsplat2d_rendering.model import GaussianModel


def detect_sh_degree(path: str | Path) -> int:
    el = PlyData.read(str(path)).elements[0]
    n_rest = sum(1 for p in el.properties if p.name.startswith("f_rest_"))
    if n_rest == 0:
        return 0
    for deg in range(1, 4):
        if 3 * ((deg + 1) ** 2 - 1) == n_rest:
            return deg
    raise ValueError(f"Cannot determine SH degree: {n_rest} f_rest_ properties in {path}")


def _sorted_props(el, prefix: str) -> list[str]:
    names = [p.name for p in el.properties if p.name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def _check_float_dtype(el, names: list[str]) -> None:
    for name in names:
        kind = el[name].dtype.kind
        if kind not in ("f",):
            raise ValueError(
                f"Property '{name}' is {el[name].dtype} (not floating point). This loader "
                "doesn't support int8/quantized PLY properties (e.g. from an offline "
                "int8-rotation compression pass) -- reinterpreting quantization codes as "
                "raw floats would silently corrupt the model instead of failing loudly."
            )


def _stack(el, names: list[str]) -> np.ndarray:
    _check_float_dtype(el, names)
    return np.stack([np.asarray(el[name]) for name in names], axis=1).astype(np.float32)


def _sanitize(arr: np.ndarray, label: str) -> np.ndarray:
    n_bad = int(np.sum(~np.isfinite(arr)))
    if n_bad > 0:
        warning(__name__, f"{label}: zeroing {n_bad:,} NaN/Inf values")
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def load_gaussian_model(
    path: str | Path,
    sh_degree: int = -1,
    device: str = "cuda",
    compression_level: int = 0,
    target_sh_degree: int = 1,
    opacity_threshold: float = 0.0,
) -> GaussianModel:
    """compression_level: 0 (none) - 3 (aggressive), see compression.py.
    target_sh_degree only applies at compression_level 2. opacity_threshold:
    0.0 (default, off) permanently drops splats at/under this activated
    opacity when loading -- see compression.py's prune_low_opacity for why
    this can be a large fraction of a real model."""
    ply_path = str(path)
    el = PlyData.read(ply_path).elements[0]

    xyz = _sanitize(_stack(el, ["x", "y", "z"]), "xyz")
    _check_float_dtype(el, ["opacity"])
    opacity = _sanitize(np.asarray(el["opacity"], dtype=np.float32)[..., None], "opacity")
    scaling = _sanitize(_stack(el, _sorted_props(el, "scale_")), "scale")
    rotation = _sanitize(_stack(el, _sorted_props(el, "rot_")), "rotation")
    features_dc = _sanitize(_stack(el, _sorted_props(el, "f_dc_")), "f_dc")[:, np.newaxis, :]

    degree = detect_sh_degree(ply_path) if sh_degree < 0 else sh_degree
    f_rest_names = _sorted_props(el, "f_rest_")
    if f_rest_names:
        f_rest_flat = _sanitize(_stack(el, f_rest_names), "f_rest")
        k = (degree + 1) ** 2 - 1
        if f_rest_flat.shape[1] != 3 * k:
            raise ValueError(
                f"sh_degree={degree} implies {3 * k} f_rest_ properties, but {ply_path} has "
                f"{f_rest_flat.shape[1]}. Pass sh_degree=-1 to auto-detect, or check that this "
                "PLY hasn't already been degree-reduced by another tool."
            )
        features_rest = f_rest_flat.reshape((-1, 3, k)).transpose(0, 2, 1)
    else:
        features_rest = np.zeros((xyz.shape[0], 0, 3), dtype=np.float32)

    arrays = {
        "xyz": xyz, "opacity": opacity, "scaling": scaling,
        "rotation": rotation, "features_dc": features_dc, "features_rest": features_rest,
    }
    # Before compression, not after: pruning shrinks N, so compression
    # (fp16 conversion, SH truncation) then has less to do.
    arrays, n_pruned = prune_low_opacity(arrays, opacity_threshold)
    arrays, degree = apply_compression(arrays, degree, compression_level, target_sh_degree)

    def to_device(arr: np.ndarray) -> torch.Tensor:
        # Preserves arr's own dtype (float32 normally, float16 when
        # compression_level >= 1 -- see compression.py's to_fp16_safe)
        # rather than forcing float32 here, which would silently undo the
        # memory/bandwidth savings apply_compression just produced.
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device)

    model = GaussianModel(
        xyz=to_device(arrays["xyz"]),
        raw_opacity=to_device(arrays["opacity"]),
        raw_scaling=to_device(arrays["scaling"]),
        raw_rotation=to_device(arrays["rotation"]),
        features_dc=to_device(arrays["features_dc"]),
        features_rest=to_device(arrays["features_rest"]),
        active_sh_degree=degree,
    )
    prune_note = f", pruned {n_pruned:,} at opacity<={opacity_threshold}" if n_pruned else ""
    info(__name__, f"Loaded {model.num_points:,} splats (SH degree {degree}, "
         f"compression level {compression_level}{prune_note}) from {ply_path}")
    return model

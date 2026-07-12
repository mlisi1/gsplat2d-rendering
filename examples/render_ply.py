#!/usr/bin/env python3
"""Minimal end-to-end example: load a trained 2D-GS PLY, build/load an
octree index for culling, render one frame from an orbit camera, save it.

Usage:
    python examples/render_ply.py /path/to/point_cloud.ply out.png [--build-index]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import gsplat2d_rendering as gs2d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--build-index", action="store_true", help="build/cache an octree for culling")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = gs2d.load_gaussian_model(args.ply_path, device=args.device)

    xyz_np = model.xyz.float().cpu().numpy()
    octree = gs2d.load_or_build_octree(args.ply_path, xyz_np, build_index=args.build_index)
    if octree is not None:
        model.reorder_(torch.from_numpy(octree.flat_indices).to(args.device))

    renderer = gs2d.Renderer(model, device=args.device, octree=octree, culling_enabled=octree is not None)

    # Orbit camera: look at the scene's centroid from a distance proportional
    # to its extent, so this works on any scene without hand-tuned numbers.
    center = np.median(xyz_np, axis=0)
    # 90th percentile, not max: a handful of far-flung outlier splats (common
    # in real trained scenes) would otherwise zoom the default view out
    # absurdly far to fit them.
    radius = float(np.percentile(np.linalg.norm(xyz_np - center, axis=1), 90))
    eye = center + np.array([0.0, -2.0 * radius, 0.6 * radius])
    intrinsics = gs2d.Intrinsics.from_fov(args.width, args.height, fov_x=np.radians(60))
    camera = gs2d.Camera.look_at(eye, center, intrinsics, device=args.device)

    result = renderer.render(camera)
    print(f"Rendered {result.num_rendered:,} splats -> {args.out_path}")
    Image.fromarray(result.rgb).save(args.out_path)


if __name__ == "__main__":
    main()

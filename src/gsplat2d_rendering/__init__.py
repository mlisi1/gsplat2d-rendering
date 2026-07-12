"""General-purpose rendering for trained 2D Gaussian Splatting (surfel)
models: load a PLY, optionally build an octree for culling/LOD, render from
any camera pose via diff-surfel-rasterization. See README.md.
"""
from gsplat2d_rendering._log import (
    NORMAL,
    SILENT,
    VERBOSE,
    get_verbosity,
    set_verbosity,
)
from gsplat2d_rendering.camera import Camera, Intrinsics
from gsplat2d_rendering.culling import (
    Octree,
    build_octree,
    load_octree,
    load_or_build_octree,
    save_octree,
)
from gsplat2d_rendering.io import detect_sh_degree, load_gaussian_model, resolve_ply_path
from gsplat2d_rendering.model import GaussianModel
from gsplat2d_rendering.render import (
    Bounds,
    Renderer,
    RenderMode,
    RenderResult,
    depth_to_normal,
    depth_to_points,
)

__all__ = [
    "GaussianModel",
    "load_gaussian_model",
    "detect_sh_degree",
    "resolve_ply_path",
    "Camera",
    "Intrinsics",
    "Octree",
    "build_octree",
    "save_octree",
    "load_octree",
    "load_or_build_octree",
    "Renderer",
    "RenderResult",
    "depth_to_normal",
    "depth_to_points",
    "RenderMode",
    "Bounds",
    "set_verbosity",
    "get_verbosity",
    "SILENT",
    "NORMAL",
    "VERBOSE",
]

__version__ = "0.1.0"

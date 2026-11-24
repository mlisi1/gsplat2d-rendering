# SPDX-License-Identifier: Apache-2.0
"""Builds the surfel rasterizer extension (gsplat2d_rendering._raster._C).
Needs the installed torch and its CUDA toolkit, so install with
`pip install --no-build-isolation -e .`. Set TORCH_CUDA_ARCH_LIST to the
target GPUs' compute capabilities (e.g. "8.6;8.7;12.0")."""
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

csrc = Path("src/gsplat2d_rendering/_raster/csrc")
sources = [str(csrc / "torch_binding.cpp")] + sorted(str(p) for p in (csrc / "core").glob("*.cu"))

setup(
    ext_modules=[
        CUDAExtension(
            name="gsplat2d_rendering._raster._C",
            sources=sources,
            include_dirs=[str((csrc).resolve())],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

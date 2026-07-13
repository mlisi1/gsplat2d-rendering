"""Partial (row-range) reads of a 2D-GS PLY, for callers that keep only a
spatial subset of a model resident at a time (e.g. a disk-backed chunk
streaming system). Split out of io/ply.py rather than growing that file,
following this repo's own "one file, one concern" convention.

Requires the source PLY to already be laid out so that a chunk's rows are
*contiguous* -- this module does not do fancy/scattered indexing, since a
scattered read defeats the whole point of a bounded, sequential disk read
per chunk. Producing such a file (physically reordering a model into
chunk-contiguous order) is the caller's job -- see write_gaussian_model in
io/ply.py and build_octree in culling/octree.py, whose flat_indices is
exactly the permutation needed to build one.
"""
from __future__ import annotations

from pathlib import Path

from plyfile import PlyData

from gsplat2d_rendering.io.ply import _gaussian_model_from_element
from gsplat2d_rendering.model import GaussianModel


class _RowRangeElement:
    """Minimal element-like view over a row-sliced structured array.
    `PlyElement.data[a:b]` is a plain numpy structured array -- it has no
    `.properties` (property *names* are metadata belonging to the source
    element as a whole, not to any particular row range), so this borrows
    that list from the full element rather than re-deriving it, while
    `__getitem__` indexes into the row-sliced data. `_gaussian_model_from_element`
    only ever does `el[name]` / `el.properties`, so this is a complete stand-in."""
    __slots__ = ("data", "properties")

    def __init__(self, data, properties):
        self.data = data
        self.properties = properties

    def __getitem__(self, name):
        return self.data[name]


class ChunkedPlyReader:
    """Opens and memory-maps a chunk-contiguous PLY once, then serves many
    cheap row-range reads against it. A caller doing a single one-off range
    read can use `load_gaussian_model_range` below, but a caller issuing many
    reads against the same file over its lifetime (e.g. a disk-backed chunk
    streaming system walking a camera around a scene) should hold one of
    these instead: `PlyData.read` re-parses the header and re-establishes
    the mmap on every call, which is wasted, repeated work once a file is
    read from dozens or hundreds of times per session.

    Relies on PlyData.read's default mmap='c' (copy-on-write memory map):
    the 2D-GS vertex schema has no list-typed properties, so plyfile can
    memory-map the whole element as one fixed-stride structured array
    instead of falling back to its row-by-row Python reader, making
    `el.data[row_start:row_start + row_count]` a slice of that array rather
    than a full-file materialization. Safe to call `read_range` concurrently
    from multiple threads on one instance -- it's read-only access to the
    shared mmap, no mutable state.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._element = PlyData.read(self.path).elements[0]

    def read_range(
        self,
        row_start: int,
        row_count: int,
        sh_degree: int = -1,
        device: str = "cuda",
        compression_level: int = 0,
        target_sh_degree: int = 1,
    ) -> GaussianModel:
        """Loads only rows [row_start, row_start + row_count) of the vertex
        element. opacity_threshold is deliberately not exposed here --
        unlike compression, pruning changes which rows exist at all, so it
        has to happen once when a chunked file is built (io/ply.py's
        write_gaussian_model), not as a variable per-range-read parameter
        that would silently desync a chunk's row layout from whatever
        manifest offsets a caller computed against the unpruned file."""
        row_slice = _RowRangeElement(
            self._element.data[row_start:row_start + row_count], self._element.properties
        )
        return _gaussian_model_from_element(
            row_slice, sh_degree, device, compression_level, target_sh_degree,
            opacity_threshold=0.0,
            source=f"{self.path}[{row_start}:{row_start + row_count}]",
            # A range read is -- by construction -- one piece of a caller's
            # larger partitioning scheme (a chunk streaming session can issue
            # hundreds of these), not a standalone notable event the way a
            # whole-file load is -- see _gaussian_model_from_element's own
            # docstring for why that's NORMAL-level but this is VERBOSE-level.
            verbose_log=True,
        )


def load_gaussian_model_range(
    path: str | Path,
    row_start: int,
    row_count: int,
    sh_degree: int = -1,
    device: str = "cuda",
    compression_level: int = 0,
    target_sh_degree: int = 1,
) -> GaussianModel:
    """One-off convenience wrapper around ChunkedPlyReader -- opens the file,
    reads a single range, and lets the PlyData/mmap get garbage-collected.
    Callers issuing many range reads against the same file should hold a
    ChunkedPlyReader themselves instead of calling this repeatedly, to avoid
    re-opening and re-parsing the file on every call."""
    return ChunkedPlyReader(path).read_range(
        row_start, row_count, sh_degree, device, compression_level, target_sh_degree,
    )

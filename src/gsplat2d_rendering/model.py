"""In-memory 2D Gaussian Splat model: raw optimized tensors + the activation
functions that turn them into renderable quantities.

Deliberately holds only what a *trained* model needs at render time -- no
optimizer state, no densification, no training-only bookkeeping. Activation
functions (exp / sigmoid / normalize) match the 2D-GS paper's own convention
(scale is stored log-space, opacity as an inverse-sigmoid logit, rotation as
a raw un-normalized quaternion) -- this is the PLY file format's contract,
not a choice specific to any particular caller.

Raw tensors may be float16 (see compression.py) to halve resting VRAM and
the memory traffic masking/gathering has to move -- but every accessor below
always hands back float32: the rasterizer CUDA kernel is hardcoded float32
(third_party/diff-surfel-rasterization has no dtype dispatch at all), so
fp16 storage is an implementation detail of this class, never something a
caller needs to know about. `.float()` is a no-op (returns self, no copy)
when a tensor is already float32, so this costs nothing when uncompressed.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from gsplat2d_rendering._log import info, verbose


@dataclass
class GaussianModel:
    xyz: torch.Tensor              # [N, 3]
    raw_opacity: torch.Tensor      # [N, 1], pre-sigmoid
    raw_scaling: torch.Tensor      # [N, 2 or 3], pre-exp (log-space)
    raw_rotation: torch.Tensor     # [N, 4], un-normalized quaternion (w, x, y, z)
    features_dc: torch.Tensor      # [N, 1, 3]
    features_rest: torch.Tensor    # [N, K, 3]
    active_sh_degree: int

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.xyz.float()

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_opacity.float())

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self.raw_scaling.float())

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.raw_rotation.float())

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self.features_dc.float(), self.features_rest.float()), dim=1)

    @property
    def num_points(self) -> int:
        return self.xyz.shape[0]

    @staticmethod
    def _activate(xyz, raw_opacity, raw_scaling, raw_rotation, features_dc, features_rest):
        """Shared by render_fields (boolean-mask path) and the renderer's
        contiguous-slice-gather path (see reorder_) -- same activation
        formulas, different way of arriving at the (already-reduced-to-
        candidates) raw tensors fed in here."""
        return (
            xyz.float(),
            torch.sigmoid(raw_opacity.float()),
            torch.exp(raw_scaling.float()),
            torch.nn.functional.normalize(raw_rotation.float()),
            torch.cat((features_dc.float(), features_rest.float()), dim=1),
        )

    def render_fields(self, mask: torch.Tensor | None = None):
        """(xyz, opacity, scaling, rotation, features), activated, restricted
        to `mask` if given. Indexes the *raw* tensors before activating them
        rather than activating-then-indexing (what naive `get_opacity[mask]`
        etc. does) -- sigmoid/exp/normalize/cat are nontrivial per-point
        ops, so doing them over the full model and discarding most of the
        result makes their cost independent of how much culling actually
        removes. This is why it belongs on the model, not the caller: only
        this class knows which raw field maps to which activation.

        Boolean-mask indexing here is O(N) regardless of mask sparsity (it
        has to scan/compact the full N-length mask) -- fine for the
        brute-force (mask=None) and no-octree cases, but
        render.rasterizer.SplatRenderer's main per-frame path reaches
        `_activate` directly with contiguous-slice-gathered (not
        boolean-masked) raw tensors instead, for exactly this reason."""
        if mask is None:
            return self._activate(self.xyz, self.raw_opacity, self.raw_scaling,
                                   self.raw_rotation, self.features_dc, self.features_rest)
        return self._activate(
            self.xyz[mask], self.raw_opacity[mask], self.raw_scaling[mask],
            self.raw_rotation[mask], self.features_dc[mask], self.features_rest[mask],
        )

    def reorder_(self, perm: torch.Tensor, verbose_log: bool = False) -> None:
        """In-place permutation of every raw per-splat tensor -- called
        once at load time (not per-frame) to put the model into an
        octree's leaf-contiguous order (perm = that octree's
        flat_indices), so a leaf's points become directly sliceable as
        model.xyz[node_offsets[j]:node_offsets[j+1]] with no further
        indirection. This is what makes the renderer's contiguous-slice
        gather possible instead of boolean-mask indexing -- see
        culling/octree.py's module docstring for why that distinction
        matters: boolean masking is O(N) to compact regardless of how few
        points survive, contiguous slicing of a pre-sorted array is not.

        verbose_log routes the summary line through verbose() instead of
        info(): a one-off whole-model reorder at load time is worth NORMAL
        visibility, but a caller reordering the same model repeatedly (e.g.
        chunk streaming re-reordering its composited model on every rebuild)
        would otherwise flood NORMAL-level output with one line per rebuild."""
        self.xyz = self.xyz[perm].contiguous()
        self.raw_opacity = self.raw_opacity[perm].contiguous()
        self.raw_scaling = self.raw_scaling[perm].contiguous()
        self.raw_rotation = self.raw_rotation[perm].contiguous()
        self.features_dc = self.features_dc[perm].contiguous()
        self.features_rest = self.features_rest[perm].contiguous()
        log_fn = verbose if verbose_log else info
        log_fn(__name__, f"Reordered {self.num_points:,} splats to octree leaf-contiguous order")


def concat_gaussian_models(models: list[GaussianModel]) -> GaussianModel:
    """Concatenates several models' per-splat tensors into one, in list
    order -- e.g. for a caller that keeps a model resident as separate
    spatial pieces (chunks) and needs one combined model to hand to a
    renderer. Raises rather than silently coercing on a mismatched device
    or SH degree across inputs: either signals a caller bug (chunks loaded
    onto different devices, or from files at different compression levels),
    not something with one obviously-correct fallback."""
    if not models:
        raise ValueError("concat_gaussian_models: empty model list")
    device = models[0].xyz.device
    degree = models[0].active_sh_degree
    k = models[0].features_rest.shape[1]
    for m in models[1:]:
        if m.xyz.device != device:
            raise ValueError(
                f"concat_gaussian_models: device mismatch ({m.xyz.device} vs {device})"
            )
        if m.features_rest.shape[1] != k:
            raise ValueError(
                f"concat_gaussian_models: SH degree mismatch ({m.active_sh_degree} vs {degree})"
            )
    return GaussianModel(
        xyz=torch.cat([m.xyz for m in models], dim=0),
        raw_opacity=torch.cat([m.raw_opacity for m in models], dim=0),
        raw_scaling=torch.cat([m.raw_scaling for m in models], dim=0),
        raw_rotation=torch.cat([m.raw_rotation for m in models], dim=0),
        features_dc=torch.cat([m.features_dc for m in models], dim=0),
        features_rest=torch.cat([m.features_rest for m in models], dim=0),
        active_sh_degree=degree,
    )

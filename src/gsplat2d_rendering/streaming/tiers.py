"""Pure tier-residency math for chunk streaming: given which chunks are
frustum-visible right now, decide which chunks should be VRAM-resident
(strict + margin), and which should be RAM-resident, entirely in terms of
adjacency-graph hop counts -- no I/O, no GPU, no state. Split out of
`ChunkManager.update()` specifically so this decision can be tested/reasoned
about on its own, independent of the transition pool / disk I/O it drives.

Three residency questions, all hop-count-based (see `adjacency.bfs_expand`),
each answering something different:

- `vram_margin_hops` expands *outward from the strict frustum-visible set*:
  chunks this many hops from a visible chunk are ALSO promoted to actual
  VRAM residency (composited into the rendered model) even though they're
  outside the frustum right now -- so a small camera rotation across this
  margin needs no rebuild latency, since a caller's own cheap per-frame GPU
  cull is what then hides/shows them.
- `ram_margin_hops` expands *outward from the VRAM tier* (strict + margin)
  into CPU-only prefetch -- same shape, one tier further out.
- `max_load_hops` bounds *overall reach*: no chunk more than this many hops
  from the strict set is ever considered for either margin, regardless of
  how large `vram_margin_hops`/`ram_margin_hops` are. Needed whenever the
  underlying visibility test has no far-plane clip (a chunk aligned with
  the camera's view direction could otherwise pass the test regardless of
  true distance, ballooning residency toward the whole scene) -- `None`
  disables the bound. Seeded from the strict set itself (never a single
  "nearest chunk" reference point) so it can only ever ADD reach for the
  margins, never exclude a chunk the caller's own visibility test already
  found.

Both margins default to 0 (off) -- no separate enable/disable flag needed,
0 hops of expansion is already a well-defined "don't expand" state.

The propagation invariant this is meant to guarantee (given a reasonably
connected adjacency graph -- see adjacency.py's own module docstring for why
that graph has to actually be complete for this to hold): every strict-VRAM
chunk's neighbors are themselves VRAM (strict or margin) whenever
`vram_margin_hops >= 1`; every VRAM-margin chunk's neighbors are VRAM or RAM
whenever `ram_margin_hops >= 1`. Nothing here enforces margin_hops >= 1 --
0 is a legitimate, deliberate "no buffer tier" choice -- but a caller that
wants zero-latency margin buffering around visible geometry needs at least
1 hop of each."""
from __future__ import annotations

from dataclasses import dataclass

from gsplat2d_rendering.streaming.adjacency import bfs_expand


@dataclass(frozen=True)
class TierSets:
    strict: frozenset[int]  # frustum-visible right now
    margin: frozenset[int]  # adjacency-promoted VRAM residents, not frustum-visible
    ram: frozenset[int]     # CPU-only prefetch, beyond the VRAM tier

    @property
    def vram(self) -> frozenset[int]:
        return self.strict | self.margin


def compute_desired_tiers(
    adjacency: dict[int, list[int]],
    frustum_visible: set[int],
    vram_margin_hops: int,
    ram_margin_hops: int,
    max_load_hops: int | None = None,
) -> TierSets:
    """`frustum_visible` must be non-empty -- callers whose own visibility
    test finds nothing (e.g. a degenerate starting camera pose) should
    substitute a fallback seed (see `adjacency.nearest_chunk_id`) before
    calling this, rather than this function guessing one on their behalf."""
    if max_load_hops is not None:
        reachable = bfs_expand(adjacency, frustum_visible, max_load_hops)
    else:
        reachable = None

    vram_expanded = bfs_expand(adjacency, frustum_visible, vram_margin_hops)
    if reachable is not None:
        vram_expanded &= reachable
    margin = vram_expanded - frustum_visible
    vram = frustum_visible | margin

    ram_expanded = bfs_expand(adjacency, vram, ram_margin_hops)
    if reachable is not None:
        ram_expanded &= reachable
    ram = ram_expanded - vram

    return TierSets(strict=frozenset(frustum_visible), margin=frozenset(margin), ram=frozenset(ram))

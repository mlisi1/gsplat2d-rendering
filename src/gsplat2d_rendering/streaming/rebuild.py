"""Throttled, backgrounded rebuild-and-swap scheduler: runs a caller-supplied
`stitch_fn` (assemble one composited result from whatever's currently
resident) on a timer, off the calling thread, and hands the result back
through `drain()` once it lands.

Purely time-gated, not gated on anything else being idle -- a caller
driving this from continuous per-frame updates (see manager.py's own
`update()`) may have other background work (e.g. disk reads) running
indefinitely, and gating a rebuild on "everything else settled" would mean
newly-resident data finishes loading in the background but never actually
gets folded into what's rendered until activity stops.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Callable, Generic, TypeVar

import torch

R = TypeVar("R")

DEFAULT_THROTTLE_S = 0.4
DEFAULT_OOM_COOLDOWN_S = 2.0


class RebuildScheduler(Generic[R]):
    """`stitch_fn(snapshot) -> R` does the actual (potentially expensive,
    GPU-uploading) assembly work; this class only owns the throttle/
    backoff/threading around calling it. `on_oom`, if given, fires exactly
    once per `torch.cuda.OutOfMemoryError` -- for a caller that wants an
    edge-triggered notification distinct from any per-item OOM signal it
    might also be getting from a `TransitionPool` elsewhere (a rebuild OOM
    means the *whole* composited result didn't fit, not any one input item,
    so there's no single item to blame or cool down)."""

    def __init__(self, stitch_fn: Callable[[dict], R],
                 throttle_s: float = DEFAULT_THROTTLE_S,
                 oom_cooldown_s: float = DEFAULT_OOM_COOLDOWN_S,
                 on_oom: Callable[[], None] | None = None):
        self._stitch_fn = stitch_fn
        self.throttle_s = throttle_s
        self.oom_cooldown_s = oom_cooldown_s
        self._on_oom = on_oom

        self._busy = False
        self._pending: tuple[R, frozenset[int]] | None = None
        self._last_kick = 0.0
        self._oom_cooldown_until = 0.0
        self.last_rebuilt_ids: frozenset[int] = frozenset()

    def sync(self, snapshot: dict[int, object]) -> R:
        """Blocking, immediate rebuild -- for a caller's own startup path,
        where there's nothing yet to throttle against and the result is
        needed right away, not on the next drain()."""
        result = self._stitch_fn(snapshot)
        self.last_rebuilt_ids = frozenset(snapshot.keys())
        return result

    def maybe_kick(self, snapshot: dict[int, object]) -> None:
        """No-op if already busy, throttled, cooling down from a prior OOM,
        or `snapshot` is empty. Otherwise spawns a background rebuild over
        a shallow copy of `snapshot` (so a caller mutating its own dict
        afterward can't race the background thread's read of it)."""
        if self._busy or not snapshot:
            return
        now = time.monotonic()
        if now - self._last_kick < self.throttle_s or now < self._oom_cooldown_until:
            return
        self._busy = True
        self._last_kick = now
        threading.Thread(target=self._worker, args=(dict(snapshot),), daemon=True).start()

    def _worker(self, snapshot: dict[int, object]) -> None:
        try:
            result = self._stitch_fn(snapshot)
            self._pending = (result, frozenset(snapshot.keys()))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._oom_cooldown_until = time.monotonic() + self.oom_cooldown_s
            if self._on_oom is not None:
                self._on_oom()
            traceback.print_exc()
        except Exception:
            traceback.print_exc()
        finally:
            self._busy = False

    def drain(self) -> R | None:
        """Call once per frame/tick. Returns the most recently landed
        result, or None if nothing has landed since the last call."""
        pending = self._pending
        if pending is None:
            return None
        self._pending = None
        result, ids = pending
        self.last_rebuilt_ids = ids
        return result

    def reset(self) -> None:
        """Forgets the last-rebuilt-ids marker -- for a caller that just
        invalidated its own resident set wholesale (e.g. a compression-level
        change) and wants the next `maybe_kick` to be treated as a genuine
        change rather than a no-op because the id set happens to match."""
        self.last_rebuilt_ids = frozenset()

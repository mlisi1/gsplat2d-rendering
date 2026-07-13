"""Bounded background worker pool for chunk streaming's disk reads. Generic
over what "read a chunk" means (`load_fn`) and what a caller wants to tag a
transition with (`tag`, e.g. a compression level) -- this module only owns
concurrency bounding, OOM cooldown, and thread-safe result handoff, not
chunk-streaming policy (which tier gets priority, how many hops out a chunk
should be prefetched, etc. -- see tiers.py/manager.py for that).

A single "keep last-known-good state" idiom for the two things that fail
transiently:
- A read that raises `MemoryError` (host-RAM exhaustion, not CUDA -- this
  pool never touches the GPU itself, see manager.py for where the one GPU
  upload chunk streaming does happen) puts that chunk id on a cooldown
  instead of retrying it every frame, and records it as `last_oom_chunk_id`
  for a caller to surface a warning from.
- Any other exception during a read is logged and the transition is
  reported as failed (`model=None`), leaving whatever tier state the caller
  already had for that chunk untouched -- a caller should never let a
  failed read silently evict a chunk that was already working.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

DEFAULT_MAX_CONCURRENT = 3
DEFAULT_OOM_COOLDOWN_S = 2.0


@dataclass
class TransitionResult(Generic[T]):
    chunk_id: int
    kind: str            # caller-defined, e.g. "to_vram_from_disk" / "to_ram_from_disk"
    payload: T | None     # None means the transition failed (OOM or otherwise)
    tag: object = None    # caller-defined, e.g. the setting a stale result should be checked against


class TransitionPool(Generic[T]):
    """Runs up to `max_concurrent` `load_fn(chunk_id) -> T` calls at once,
    each on its own daemon thread, and hands landed results back through
    `drain()` -- a caller's own per-frame `update()`-style loop calls
    `drain()` once, then `spawn()` for whatever new work that frame's state
    calls for, checking `slot_available()`/`cooled_down()` itself to decide
    what and how much to spawn (this class enforces the concurrency/cooldown
    bounds, but has no opinion on *which* chunk id or `kind` a caller should
    prioritize -- that's the caller's own tier policy)."""

    def __init__(self, load_fn: Callable[[int], T],
                 max_concurrent: int = DEFAULT_MAX_CONCURRENT,
                 oom_cooldown_s: float = DEFAULT_OOM_COOLDOWN_S,
                 on_oom: Callable[[int], None] | None = None):
        """`on_oom`, if given, fires exactly once per OOM event (chunk_id) --
        for a caller that needs an edge-triggered notification rather than
        polling `last_oom_chunk_id` (which just holds the *latest* chunk id
        and isn't cleared by this class, so it can't by itself distinguish
        "still the same old event" from "the same chunk id OOM'd again")."""
        self._load_fn = load_fn
        self.max_concurrent = max_concurrent
        self.oom_cooldown_s = oom_cooldown_s
        self._on_oom = on_oom

        # Guards inflight/_pending/_oom_cooldown_until only -- never held
        # across load_fn itself, which can be a slow disk read.
        self._lock = threading.Lock()
        self.inflight: set[int] = set()
        self._pending: list[TransitionResult[T]] = []
        self._oom_cooldown_until: dict[int, float] = {}
        self.last_oom_chunk_id: int | None = None

    def cooled_down(self, chunk_id: int) -> bool:
        return time.monotonic() >= self._oom_cooldown_until.get(chunk_id, 0.0)

    def slot_available(self) -> bool:
        return len(self.inflight) < self.max_concurrent

    def spawn(self, chunk_id: int, kind: str, tag: object = None) -> None:
        """Caller's responsibility to have already checked `slot_available()`
        and `cooled_down(chunk_id)` -- this method doesn't re-check either,
        so a caller spawning without checking can exceed `max_concurrent`."""
        self.inflight.add(chunk_id)
        threading.Thread(target=self._worker, args=(chunk_id, kind, tag), daemon=True).start()

    def _worker(self, chunk_id: int, kind: str, tag: object) -> None:
        try:
            payload = self._load_fn(chunk_id)
            result = TransitionResult(chunk_id, kind, payload, tag)
        except MemoryError:
            self._oom_cooldown_until[chunk_id] = time.monotonic() + self.oom_cooldown_s
            self.last_oom_chunk_id = chunk_id
            if self._on_oom is not None:
                self._on_oom(chunk_id)
            result = TransitionResult(chunk_id, kind, None, tag)
        except Exception:
            traceback.print_exc()
            result = TransitionResult(chunk_id, kind, None, tag)
        with self._lock:
            self._pending.append(result)
            self.inflight.discard(chunk_id)

    def drain(self) -> list[TransitionResult[T]]:
        """Returns (and clears) every transition that has landed since the
        last call -- call once per frame/tick, before deciding what new
        work to spawn."""
        with self._lock:
            pending, self._pending = self._pending, []
        return pending


def spawn_tier_transitions(
    pool: TransitionPool, primary: set[int], secondary: set[int],
    primary_kind: str, secondary_kind: str, tag: object = None,
) -> None:
    """Default two-tier scheduling policy for a pool shared between a
    priority tier (`primary`, e.g. promotions into a rendered composition)
    and a prefetch tier (`secondary`, e.g. margin buffering) -- reserves at
    least one worker slot for `secondary` whenever it's non-empty, so a
    steady stream of `primary` candidates under continuous churn can't
    starve `secondary` out entirely (which would silently turn prefetch
    into a no-op under exactly the sustained-motion case it exists for).

    Both id sets should already be reduced to genuine candidates by the
    caller (not already resident, not in flight, past any cooldown) -- this
    function only allocates pool slots between the two, it doesn't filter."""
    primary, secondary = list(primary), list(secondary)
    primary_slot_cap = (pool.max_concurrent - 1) if secondary else pool.max_concurrent
    for i, chunk_id in enumerate(primary):
        if i >= primary_slot_cap or not pool.slot_available():
            break
        pool.spawn(chunk_id, primary_kind, tag=tag)
    for chunk_id in secondary:
        if not pool.slot_available():
            return
        pool.spawn(chunk_id, secondary_kind, tag=tag)

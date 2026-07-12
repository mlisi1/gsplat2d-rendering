"""Opt-in per-stage timing collection for SplatRenderer/Renderer.

Disabled by default. Each GPU stage boundary costs a `torch.cuda.synchronize()`
to time honestly, which is real overhead the hot path shouldn't pay just to
have profiling *available* -- so `lap()` is a no-op unless `enable()` was
called, and render() itself takes no profiling argument. Call
`enable_profiling()` once (e.g. from a debug flag), then pull results anytime
with `get_last_timings()` / `get_profiling_stats()`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class StageStats:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.min_ms = ms if ms < self.min_ms else self.min_ms
        self.max_ms = ms if ms > self.max_ms else self.max_ms

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms if self.count else 0.0,
            "max_ms": self.max_ms,
            "total_ms": self.total_ms,
        }


class Profiler:
    """Stage-timing collector shared by SplatRenderer and Renderer (the
    latter records its own CPU-copy stage into the same instance, so a
    caller sees one combined breakdown regardless of which layer it used).

    `sync_fn` is called before each lap so GPU stages are timed honestly;
    passing `lambda: None` on CPU-only paths keeps this class backend-agnostic.
    """

    def __init__(self, sync_fn: Callable[[], None] = lambda: None):
        self._sync_fn = sync_fn
        self.enabled = False
        self._last: dict[str, float] | None = None
        self._stats: dict[str, StageStats] = {}
        self._t = 0.0

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def reset(self) -> None:
        self._last = None
        self._stats.clear()

    def start(self) -> None:
        if not self.enabled:
            return
        self._last = {}
        self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        if not self.enabled or self._last is None:
            return
        self._sync_fn()
        now = time.perf_counter()
        elapsed_ms = (now - self._t) * 1000.0
        self._t = now
        self._last[name] = elapsed_ms
        self._stats.setdefault(name, StageStats()).add(elapsed_ms)

    def last(self) -> dict[str, float] | None:
        """Per-stage timings (ms) for the most recently profiled render call,
        or None if profiling was off or nothing has rendered yet."""
        return dict(self._last) if self._last is not None else None

    def stats(self) -> dict[str, dict[str, float]]:
        """Per-stage {count, mean_ms, min_ms, max_ms, total_ms}, accumulated
        since the profiler was enabled or last reset()."""
        return {name: s.as_dict() for name, s in self._stats.items()}

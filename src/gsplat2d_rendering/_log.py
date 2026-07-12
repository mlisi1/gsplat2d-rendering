"""Lightweight, print-based logging with 3 verbosity levels and per-module
tags. Not stdlib `logging`: handlers/formatters/root-logger configuration
are real ceremony for what every caller here actually wants -- "which of my
own messages show up" -- and a library pulling in a root-logger dependency
is more likely to surprise a host application than help it.

Levels (set_verbosity(N), default NORMAL):
  SILENT  (0) -- errors only
  NORMAL  (1) -- + warnings + essential info   (default)
  VERBOSE (2) -- + detailed/diagnostic info

`error()` always prints regardless of level -- a caller that set SILENT
still needs to know when something is actually broken, not just quiet.
"""
from __future__ import annotations

import sys

SILENT = 0
NORMAL = 1
VERBOSE = 2

_LEVELS = (SILENT, NORMAL, VERBOSE)
_level = NORMAL
_PACKAGE_PREFIX = "gsplat2d_rendering."


def set_verbosity(level: int) -> None:
    global _level
    if level not in _LEVELS:
        raise ValueError(f"verbosity must be SILENT(0)/NORMAL(1)/VERBOSE(2), got {level!r}")
    _level = level


def get_verbosity() -> int:
    return _level


def _tag(module: str) -> str:
    # `module` is meant to be called with __name__, so callers never
    # hand-maintain a short tag string that can drift out of sync with the
    # file it's in -- strip the redundant package prefix for a shorter,
    # still-unambiguous console tag (e.g. "culling.cache", not
    # "gsplat2d_rendering.culling.cache").
    name = module[len(_PACKAGE_PREFIX):] if module.startswith(_PACKAGE_PREFIX) else module
    return f"[gsplat2d_rendering:{name}]"


def error(module: str, msg: str) -> None:
    """Always printed (to stderr), regardless of verbosity -- see module docstring."""
    print(f"{_tag(module)} ERROR: {msg}", file=sys.stderr)


def warning(module: str, msg: str) -> None:
    if _level >= NORMAL:
        print(f"{_tag(module)} WARNING: {msg}")


def info(module: str, msg: str) -> None:
    if _level >= NORMAL:
        print(f"{_tag(module)} {msg}")


def verbose(module: str, msg: str) -> None:
    if _level >= VERBOSE:
        print(f"{_tag(module)} {msg}")

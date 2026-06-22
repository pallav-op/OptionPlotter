"""Snapshot sources and clocks for the live scheduler.

These are the seam between the computation engine and the outside world. The real
market-data feed is deferred (it needs the generated protobuf classes and a network
connection), but the scheduler only depends on the small :class:`SnapshotSource`
protocol below — so it is fully testable today and the real feed slots in later as
just another ``SnapshotSource`` implementation.

  SnapshotSource         — async ``get_next()`` → the next ChainSnapshot (or None to stop)
    IterableSnapshotSource — replay/test: yields from a list/iterator
    LatestSnapshotSource   — live: returns whatever the feed last pushed via set_latest()

  Clock                  — wall-clock + sleep, injectable so tests run deterministically
    RealClock              — time.time + asyncio.sleep
    ManualClock            — virtual time advanced by sleep(), no real waiting
"""
from __future__ import annotations

import asyncio
import time
from typing import Iterable, Iterator, Protocol, runtime_checkable

from .models import ChainSnapshot


# ---------------------------------------------------------------------------
# Snapshot sources
# ---------------------------------------------------------------------------

@runtime_checkable
class SnapshotSource(Protocol):
    async def get_next(self) -> ChainSnapshot | None:
        """Return the next snapshot to compute, or None when the stream is exhausted."""
        ...


class IterableSnapshotSource:
    """Feed snapshots from an in-memory iterable (replay / testing).

    Each ``get_next`` pops the next item; returns None when exhausted.
    """

    def __init__(self, snapshots: Iterable[ChainSnapshot]) -> None:
        self._it: Iterator[ChainSnapshot] = iter(snapshots)

    async def get_next(self) -> ChainSnapshot | None:
        return next(self._it, None)


class LatestSnapshotSource:
    """Hold the most recent snapshot pushed by a live feed.

    The feed (or its adapter) calls :meth:`set_latest` whenever a new book snapshot
    arrives; the scheduler calls :meth:`get_next` once per second and computes against
    whatever is current. Returns None until the first snapshot has been set.
    """

    def __init__(self) -> None:
        self._latest: ChainSnapshot | None = None

    def set_latest(self, snapshot: ChainSnapshot) -> None:
        self._latest = snapshot

    async def get_next(self) -> ChainSnapshot | None:
        return self._latest


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------

@runtime_checkable
class Clock(Protocol):
    def time(self) -> float:
        """Current wall-clock time in seconds."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Asynchronously sleep for ``seconds`` (no-op if <= 0)."""
        ...


class RealClock:
    """Production clock: real wall time and real asyncio sleeps."""

    def time(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)


class ManualClock:
    """Deterministic virtual clock for tests — ``sleep`` advances time instantly."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def time(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._now += seconds

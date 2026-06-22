"""Live 1-second scheduler.

Drives ``compute_one_tick`` for every active plot once per second, batches the
results, and hands the batch to a ``publish`` sink. The sink is where a WebSocket
layer plugs in later (deferred); today it defaults to collecting batches in memory.

Design notes
------------
* One plot's failure never stops the others — each compute is wrapped in try/except;
  a failing plot is set to ``status="error"`` (which removes it from ``active_plots``,
  quarantining a repeatedly-failing plot after a single bad tick).
* Slow plots are logged (over ``slow_warn_ms``). They are also rejected up front by
  validation Stage 6 (performance), so this is a runtime backstop.
* Optional hard per-tick timeout: when ``per_tick_timeout`` is set, each compute runs
  in a thread executor under ``asyncio.wait_for``. A hung/slow plot is abandoned after
  the timeout (its thread lingers under the GIL but no longer blocks the loop) and the
  plot is quarantined. With ``per_tick_timeout=None`` (default) computes run inline.
* The loop tick is aligned to the wall-clock second via an injectable :class:`Clock`,
  so tests can run deterministically with a :class:`ManualClock`.

The batch payloads use the exact message shapes from the spec (``append_points`` /
``plot_error``) so the deferred WebSocket layer only has to serialise and send them.
"""
from __future__ import annotations

import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .models import ChainSnapshot, PlotResult, PlotRuntime
from .registry import PlotRegistry
from .runtime import compute_one_tick
from .sources import Clock, RealClock, SnapshotSource

_logger = logging.getLogger("plot_compute.scheduler")


# ---------------------------------------------------------------------------
# Message builders (spec §19) — plain dicts; WebSocket layer serialises later
# ---------------------------------------------------------------------------

def make_append_points(plot: PlotRuntime, result: PlotResult, timestamp_ms: int) -> dict:
    return {
        "type": "append_points",
        "plot_id": plot.plot_id,
        "version": plot.version,
        "points": {name: [[timestamp_ms, value]] for name, value in result.series.items()},
    }


def make_plot_error(plot: PlotRuntime) -> dict:
    return {
        "type": "plot_error",
        "plot_id": plot.plot_id,
        "version": plot.version,
        "error": plot.error,
    }


def make_batch(updates: list[dict]) -> dict:
    return {"type": "batch_append_points", "updates": updates}


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class LiveScheduler:
    def __init__(
        self,
        registry: PlotRegistry,
        source: SnapshotSource,
        *,
        publish: Callable[[list[dict]], None] | None = None,
        per_tick_timeout: float | None = None,
        slow_warn_ms: float = 20.0,
        clock: Clock | None = None,
        executor: ThreadPoolExecutor | None = None,
        max_workers: int = 4,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry
        self._source = source
        self._publish = publish
        self._per_tick_timeout = per_tick_timeout
        self._slow_warn_ms = slow_warn_ms
        self._clock = clock or RealClock()
        self._logger = logger or _logger
        self._running = False

        self._owns_executor = executor is None and per_tick_timeout is not None
        self._executor = executor
        if self._owns_executor:
            self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                                thread_name_prefix="plot-compute")

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False

    async def run(self, *, max_ticks: int | None = None) -> int:
        """Run the live loop until the source is exhausted, ``stop`` is called, or
        ``max_ticks`` ticks have completed. Returns the number of ticks processed."""
        self._running = True
        ticks = 0
        try:
            while self._running:
                snapshot = await self._source.get_next()
                if snapshot is None:
                    break

                batch = await self.tick(snapshot)
                if self._publish is not None:
                    self._publish(batch)

                ticks += 1
                if max_ticks is not None and ticks >= max_ticks:
                    break

                await self._sleep_until_next_second()
        finally:
            self._running = False
            if self._owns_executor and self._executor is not None:
                self._executor.shutdown(wait=False)
        return ticks

    async def tick(self, snapshot: ChainSnapshot) -> list[dict]:
        """Process one snapshot across all active plots; return the batch of updates."""
        updates: list[dict] = []
        for plot in self._registry.active_plots():
            update = await self._compute_plot(plot, snapshot)
            if update is not None:
                updates.append(update)
        return updates

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _compute_plot(self, plot: PlotRuntime, snapshot: ChainSnapshot) -> dict | None:
        t0 = self._clock.time()
        try:
            if self._per_tick_timeout is None:
                result = compute_one_tick(plot, snapshot)
            else:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, compute_one_tick, plot, snapshot),
                    timeout=self._per_tick_timeout,
                )
        except asyncio.TimeoutError:
            plot.status = "error"
            plot.error = f"compute exceeded per-tick timeout of {self._per_tick_timeout:.3f}s"
            self._logger.error("plot %s timed out: %s", plot.plot_id, plot.error)
            return make_plot_error(plot)
        except Exception as exc:  # noqa: BLE001 - isolate one plot's failure from the rest
            plot.status = "error"
            plot.error = str(getattr(exc, "message", exc))
            self._logger.error("plot %s failed: %s", plot.plot_id, plot.error)
            return make_plot_error(plot)

        elapsed_ms = (self._clock.time() - t0) * 1000.0
        if elapsed_ms > self._slow_warn_ms:
            self._logger.warning(
                "slow plot %s: %.1f ms (warn > %.1f ms)",
                plot.plot_id, elapsed_ms, self._slow_warn_ms,
            )
        return make_append_points(plot, result, snapshot.timestamp_ms)

    async def _sleep_until_next_second(self) -> None:
        now = self._clock.time()
        next_tick = math.floor(now) + 1
        await self._clock.sleep(next_tick - now)

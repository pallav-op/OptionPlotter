# `plot_compute` — Full Pipeline Reference

A guide to the whole library: the end-to-end data flow, then every module and every
function in it. Written to be **repo-agnostic** — names and responsibilities matter
more than the exact file layout, so you can map these onto a differently-structured
codebase.

The library lets a user write a tiny Python function:

```python
def compute(ctx):
    return {"value": ...}        # or {"series": {"name": value, ...}}
```

…and the library does everything else: parse the market-data proto, present a clean
columnar view, run the function safely, validate it adversarially before it goes live,
replay it from the start of the day, then tick it once per second.

---

## 1. The data flow (one diagram)

```
 market-data proto (BookPublisherOptionChainDataMessage)
        │   proto_adapter.build_chain_snapshot()
        ▼
 ChainSnapshot              ── columnar numpy arrays, one row per option
        │   build_context() wraps it as ctx.chain (ChainView) + ctx.state + ctx.window
        ▼
 user compute(ctx)  ──────► {"value": x}  or  {"series": {...}}
        │   result_format.validate_result()  → normalised {name: float}
        ▼
 runtime.compute_one_tick()  ── appends (ts, value) to plot.output_series,
        │                        pushes into the rolling-window store
        ├── historical: runtime.replay_plot()   (day start → "live")
        └── live:       scheduler.LiveScheduler  (once per second → batch → publish sink)

 Before any of the live/replay path runs, a plot must pass:
   validation.validate_plot_code()  ── 6 stages, incl. the adversarial robustness battery
   PlotRegistry.create_plot()       ── owns lifecycle, attaches the robustness report
```

Two design rules hold the whole thing together:

1. **One compute path.** `compute_one_tick` is used identically for replay and live, so
   historical and real-time values are guaranteed consistent.
2. **Validate before you trust.** No plot reaches replay/live until `validate_plot_code`
   has run it against an adversarial battery (syntax, safety, NaN/zero/extreme data,
   and *input mutation*). The runtime then trusts validated code.

---

## 2. Module-by-module, function-by-function

### `errors.py` — typed exceptions
- **`ValidationError(stage, message)`** — raised by any validation stage; `stage` is one
  of `syntax/safety/contract/robustness/result_format/performance`. The string carries a
  human-readable reason (for `robustness`, the full formatted report).
- **`RuntimePlotError(plot_id, version, message)`** — raised when *validated* code still
  throws at runtime (e.g. live tick). Lets the scheduler attribute a failure to a plot.

### `models.py` — core data structures
- **`ChainSnapshot`** *(frozen dataclass)* — one market snapshot in **columnar** form:
  scalars (`seq_no`, `key`, `timestamp_ns`, `timestamp_ms`) plus one numpy array per
  field. 1-D arrays (length = #options): `option_type`, `strike_px`, `delta`, `iv`,
  `moneyness`, `has_quote`, `volume_since_day_start`, … 2-D arrays (shape
  `[#options, #levels]`): `bid_price/bid_quantity/bid_order_count` and the `ask_*` trio.
  Columnar = one numpy op filters/aggregates all rows at once.
- **`PlotRuntime`** *(mutable dataclass)* — everything about one plot: `plot_id`,
  `version`, `code`, compiled `compute_fn`, `status`, `state` (per-plot dict persisted
  across ticks), `output_series` (`{name: [(ts, value), …]}`), `last_computed_ts`,
  `error`, and `robustness_report` (attached at registration).
- **`PlotResult`** — a validated single tick’s output, always normalised to
  `series: {name: float}` (a single-value plot uses the key `"value"`).

### `helpers.py` — safe math injected into user code
- **`safe_div(a, b, default=0.0)`** — division that returns `default` on zero/NaN/inf
  denominator or non-finite result. The single most important guard for indicators.
- **`clip(x, lo, hi)`**, **`is_nan(x)`**, **`nz(x, default=0.0)`** (NaN→default).
  All four are placed in the user namespace so `compute` can call them without imports.

### `proto_adapter.py` — proto → ChainSnapshot
- **`build_chain_snapshot(proto_msg)`** — the only converter. **Duck-typed**: reads via
  `getattr`/`dict.get`, so it works with real protobuf objects, plain dicts, or the test
  generator’s dataclasses. It (1) concatenates `call_options` + `put_options` into rows
  and records `option_type`, (2) builds each 1-D column with `np.fromiter`, (3) fills the
  ragged 2-D book arrays to a common `max_levels` (missing → `price=NaN, qty=0, oc=0`),
  (4) converts `timestamp_ns → timestamp_ms`.
- `_get`, `_get_list` — internal duck-typed attribute/list accessors.

### `chain_view.py` — the `ctx.chain` query API
- **`ChainView(snapshot)`** — wraps a snapshot; the user-facing aggregation surface.
  - **`sum / mean / min / max(field, **filters)`**, **`count(**filters)`**,
    **`weighted_mean(field, weight_field, **filters)`** — reduce a filtered column to a
    float. `mean/min/max` return NaN on empty/all-NaN selections (no numpy warnings);
    `sum` returns 0.0; `count` returns an int.
  - `_resolve_field(field)` — maps a field name to a column. Plain names → 1-D array;
    `bid_price_l1 / ask_quantity_l2 / …` → the right column of the 2-D book array
    (`_l1`→index 0). Out-of-range levels return NaN (prices) or 0 (quantities).
  - `_build_mask(**filters)` — turns filter kwargs into a boolean row mask:
    `option_type`, `expiry_timestamp`, `min/max_strike`, `min/max_delta`,
    `min/max_moneyness`, `has_quote`. Unknown filter → `ValueError`.

### `rolling.py` — `ctx.window` storage + math
- **`RollingWindowStore(max_age_ms=DEFAULT_MAX_AGE_MS)`** — per-series deque of
  `(ts, value)` over a plot’s **own output**.
  - **`push(name, ts, value)`** — append; evict entries older than
    `latest_ts − max_age_ms` (default 24h). **This is the memory-leak fix**: bounded
    retention that never truncates a realistic intraday window, regardless of query
    order. The most recent point is always kept (so `last` works).
  - **`mean / sum / min / max(name, window)`**, **`last(name)`** — v1 aggregations over
    a window like `"5m"`, `"30s"`, `"1h"`.
  - **`std(name, window, ddof=1)`** — sample std; NaN if <2 points.
  - **`zscore(name, window)`** — `(latest − mean) / std`; NaN if std is 0 or <2 points.
  - **`ema(name, window, span=None)`** — exponentially-weighted mean over the windowed
    values (`alpha = 2/(span+1)`, `span` defaults to the window’s point count).
  - **`correlation(a, b, window)`** — Pearson correlation between two own-output series,
    aligned by timestamp; NaN if <2 aligned pairs or zero variance.
  - `clear()` — drop all series. `_parse_window_ms`, `_pairs_in_window`,
    `_values_in_window` — internals (window string → ms; windowed slice).

### `context.py` — what `compute(ctx)` receives
- **`IndicatorContext`** — the `ctx`: `ts`, `seq_no`, `key`, `chain` (ChainView),
  `state` (the plot’s persistent dict), `window` (WindowAPI).
- **`WindowAPI(store)`** — thin pass-through to the rolling store exposing
  `mean/sum/min/max/last/std/zscore/ema/correlation`.
- **`build_context(plot, snapshot, rolling_store)`** — assembles a fresh `ctx` for one
  tick from the plot’s state and its rolling store.

### `executor.py` — compile + sandbox user code
- **`SAFE_BUILTINS`** — the only names user code can see: the four helpers plus a
  whitelist of safe builtins (`abs/min/max/sum/len/range/...`). No `open`, `__import__`,
  `eval`, etc.
- **`compile_user_code(code)`** — `compile(code, "<user_plot>", "exec")`.
- **`build_user_namespace(code_obj)`** — exec the code with `__builtins__` replaced by
  `SAFE_BUILTINS`; returns the namespace (where `compute` lives).
- **`execute_compute(fn, ctx)`** — call the function (kept as a seam for future
  instrumentation).

### `result_format.py` — return-value contract (Stage 5)
- **`validate_result(result)`** — accept `{"value": finite}` or
  `{"series": {name: finite, …}}`; reject `None`, non-dicts, wrong keys, bool, non-numbers,
  and NaN/inf. Returns the normalised `{name: float}`. Used both at validation time and on
  every runtime tick. *(Lives in its own module so `validation` and `robustness` can both
  import it without a cycle.)*

### `edge_cases.py` — the adversarial market-data battery
- **`EdgeCase`** *(frozen dataclass)* — `name`, `category`, `description`, `snapshot`
  (+ `n_options`/`n_levels`).
- **`INVARIANTS`** — the four per-tick rules a case must uphold (see robustness): no
  exception, **no input mutation**, valid format, finite values.
- **`build_edge_cases()`** — returns ~33 hostile `ChainSnapshot`s across categories:
  `structural` (empty/calls-only/no-levels/dense), `nan`, `zero`, `flat` (zero variance),
  `extreme` (overflow/underflow/negative/±inf), `quote` (no-quote/mixed), `normal`.
- **`save_edge_cases(dir)`** / **`load_edge_cases(dir)`** — persist/restore the battery
  (`edge_cases.pkl` + `manifest.json` with invariants & cases + a generated `README.md`).
- `_snapshot(...)`, `_render_readme(...)` — internal builder & doc generator.

### `robustness.py` — Stage 4 engine
- **Failure kinds** `EXCEPTION`, `MUTATED_INPUT`, `BAD_FORMAT`, `NON_FINITE` and the
  `FAILURE_KINDS` tuple.
- **`run_robustness_suite(compute_fn, cases=None, repeat=3)`** — runs the function
  against every case for `repeat` consecutive ticks and returns a `RobustnessReport`.
  Never raises.
- **`_run_case(...)`** — per case: a fresh state + rolling store; each tick runs on a
  **deep copy** of the snapshot so mutation can’t leak into the battery. Check order per
  tick: **exception → input-mutation → bad-format/non-finite**. (`_clone_snapshot`,
  `_fingerprint`, `_detect_mutation` implement the mutation guard by comparing array
  contents before/after.)
- **`CaseResult`** — per-case `passed`, `failure_kind`, `detail`, last `output`.
- **`RobustnessReport`** — `ok()`, `n_passed/n_failed`, `failures`, `kind_counts()`
  (always reports all four kinds), `category_scores()`, and **`format(show_passes=)`**
  for the printable pass/fail summary.

### `validation.py` — the 6-stage gate
- **`validate_plot_code(code, strict_robustness=True, robustness_repeat=3)`** — the entry
  point. Runs **syntax → safety → contract → robustness → performance** and returns
  `(compute_fn, RobustnessReport)`. (Result-format is enforced inside robustness and at
  runtime.) Raises `ValidationError` on the first failing stage.
- **`validate_syntax`** — `ast.parse`.
- **`validate_ast_safety`** — `_UnsafeCodeVisitor` blocks imports and unsafe calls
  (`open/eval/exec/compile/__import__/globals/locals/vars/dir/getattr/setattr/delattr`).
- **`validate_compute_signature`** — compile + check `compute` exists, is callable, takes
  exactly one positional arg; returns the compiled fn.
- **`run_robustness_validation(fn, strict, repeat)`** — runs the battery; if `strict` and
  any case fails, raises `ValidationError("robustness", report.format())`.
- **`run_performance_test`** — 100 runs on a sample context; warn >5 ms, reject >50 ms.
- **`build_sample_snapshot` / `build_sample_context`** — a healthy sample used by the
  performance stage.

### `runtime.py` — the single compute path
- **`compute_one_tick(plot, snapshot)`** — build ctx → run `compute_fn` (wrap user errors
  as `RuntimePlotError`) → `validate_result` → append to `output_series` + push to the
  rolling store. Returns a `PlotResult`. **Used by both replay and live.**
- **`replay_plot(plot, snapshots)`** — set `backfilling`, clear state/series/rolling,
  run every snapshot through `compute_one_tick`, end at `live`. Bails if `deleted`.
- `_get_rolling_store(plot_id)` / `_clear_rolling_store` — per-plot rolling stores.

### `registry.py` — plot lifecycle owner
- **`PlotRegistry`** holds all `PlotRuntime`s.
  - **`create_plot(plot_id, code, strict_robustness=True)`** — validate, build the runtime
    (`status="created"`), attach the robustness report. Caller then drives replay.
  - **`update_plot(plot_id, code, strict_robustness=True)`** — validate *new* code first;
    on failure the old version keeps running; on success bump `version`, clear
    state/series, re-attach report.
  - **`delete_plot / pause_plot / resume_plot / get_plot / active_plots / all_plots`** —
    `active_plots` returns only `status=="live"` (so an errored plot is auto-quarantined).

### `sources.py` — the feed/clock seam (stand-in for the deferred connection layer)
- **`SnapshotSource`** *(Protocol)* — `async get_next() -> ChainSnapshot | None`.
  - **`IterableSnapshotSource(iterable)`** — replay/testing; pops the next snapshot.
  - **`LatestSnapshotSource`** — live; `set_latest(snap)` is called by the feed, and each
    `get_next()` returns the current latest. (A real market-feed adapter is just another
    `SnapshotSource`.)
- **`Clock`** *(Protocol)* — `time()` + `async sleep(seconds)`.
  - **`RealClock`** — wall time + `asyncio.sleep`. **`ManualClock`** — virtual time for
    deterministic tests (sleep advances instantly).

### `scheduler.py` — the live 1-second loop
- **`LiveScheduler(registry, source, publish=None, per_tick_timeout=None,
  slow_warn_ms=20, clock=RealClock(), …)`**
  - **`run(max_ticks=None)`** — loop: pull a snapshot, `tick()`, hand the batch to
    `publish`, sleep until the next wall-second. Stops when the source is exhausted,
    `stop()` is called, or `max_ticks` is hit.
  - **`tick(snapshot)`** — process every `active_plots()` plot; return the batch.
  - **`_compute_plot(plot, snapshot)`** — run one plot with **error isolation**: on any
    exception/`RuntimePlotError` set `status="error"` and emit a `plot_error` (a repeatedly
    failing plot is thereby quarantined out of `active_plots`). With `per_tick_timeout` set,
    the compute runs in a thread executor under `asyncio.wait_for`, so a hung/slow plot is
    abandoned instead of blocking the others. Slow plots (over `slow_warn_ms`) are logged.
  - **`make_append_points / make_plot_error / make_batch`** — build the spec’s wire
    payloads (`append_points`, `plot_error`, `batch_append_points`). The WebSocket layer
    (deferred) only has to serialise and send these dicts.

### `__init__.py` — public API
Re-exports the names above (`PlotRegistry`, `build_chain_snapshot`, `compute_one_tick`,
`replay_plot`, `LiveScheduler`, sources/clocks, the rolling store, validation +
robustness symbols, helpers, and errors).

---

## 3. End-to-end usage

```python
from plot_compute import (
    PlotRegistry, build_chain_snapshot, replay_plot,
    LiveScheduler, IterableSnapshotSource,
)

registry = PlotRegistry()
code = '''
def compute(ctx):
    c = ctx.chain.sum(option_type="CALL", field="bid_quantity_l1")
    p = ctx.chain.sum(option_type="PUT",  field="bid_quantity_l1")
    imb = safe_div(c - p, c + p)
    return {"series": {"imbalance": imb, "imb_z": nz(ctx.window.zscore("imbalance", "5m"))}}
'''

# 1) Register — runs the full 6-stage gate incl. the adversarial battery.
plot = registry.create_plot("imbalance", code)     # raises ValidationError if unsafe
print(plot.robustness_report.format(show_passes=False))

# 2) Backfill from day start (same compute path as live).
snapshots = [build_chain_snapshot(m) for m in todays_proto_messages]
replay_plot(plot, snapshots)                        # status → "live"

# 3) Go live: one tick per second; batches go to your publish sink (→ WebSocket later).
import asyncio
sched = LiveScheduler(registry, IterableSnapshotSource(more_snapshots),
                      publish=lambda batch: send_to_clients(batch))
asyncio.run(sched.run())
```

## 4. What’s intentionally not here yet

- **WebSocket transport.** The scheduler already emits the exact `append_points` /
  `plot_error` / `batch_append_points` payloads; a transport just serialises and sends
  them, and a real market feed plugs in as a `SnapshotSource`.
- **Generated protobuf classes.** `build_chain_snapshot` is duck-typed and ready; only the
  `.proto`-generated module and feed connection are outstanding.

## 5. Run it

```bash
python -m pytest tests/ -q          # 94 tests
python run_robustness_demo.py       # full lifecycle incl. mutation rejection
```

---

## 6. Plot statuses — what they mean and what comes next

`PlotRuntime.status` is a plain string field. The table below shows every value that
is **actually set by the code** (not just mentioned in comments) and what can follow it.

| status | set by | what it means | possible next statuses |
|---|---|---|---|
| `"created"` | `registry.create_plot()` after validation passes; also `registry.update_plot()` after a successful code update | Plot exists and is validated but no historical replay has been run yet. No data in `output_series`. | `"backfilling"` (caller starts replay) |
| `"backfilling"` | `runtime.replay_plot()` at the start of the loop | Replay is in progress. `output_series` and `state` were just cleared; `compute_one_tick` is filling them with historical data. | `"live"` (replay finished); `"deleted"` (caller deleted mid-replay, checked on each tick) |
| `"live"` | `runtime.replay_plot()` at the end; also `registry.resume_plot()` | Replay done. Plot is included in `registry.active_plots()` so the scheduler processes it every second. | `"paused"` (pause_plot); `"error"` (scheduler catches a runtime exception); `"created"` (update_plot success — version bumped, state cleared, needs re-replay) |
| `"paused"` | `registry.pause_plot()` — only transitions from `"live"` | Excluded from `active_plots()`; scheduler skips it. State and output_series are preserved. | `"live"` (resume_plot) |
| `"error"` | `scheduler._compute_plot()` on any unhandled exception or timeout | Plot's `compute_fn` threw at runtime. Plot is quarantined — excluded from `active_plots()` and will not be re-tried. `plot.error` holds the message. | No automatic recovery. Caller must call `update_plot()` with fixed code (→ `"created"`) or `delete_plot()`. |
| `"deleted"` | `registry.delete_plot()` | Plot removed from the registry dict. Status is set on the object before deletion for any in-flight reference. `replay_plot` checks this flag and bails early. | Terminal — the object is gone from the registry. |

**Statuses listed in the spec but not set on real plots in the current code:**

- `"validating"` — appears in the spec and is set on *internal dummy objects* used by the
  robustness harness and sample-context builder, but never on a real `PlotRuntime` returned
  to a caller. Validation happens before the runtime object is created.
- `"recomputing"` — described in the spec as the state during a code-update replay. Not set
  anywhere in the current code; `update_plot` reuses `"created"` for that transition.

---

## 7. Mid-market creation: exact state flow and edge cases

### The happy path (plot created at 10:30, market open 09:30–16:00)

```
wall time    status        what happens in code
─────────────────────────────────────────────────────────────────────────────
10:30:00     —             registry.create_plot("p", code)
                           └─ validate_plot_code() runs internally with dummy
                              PlotRuntime(status="validating") for harness
                           └─ validation passes → real PlotRuntime created
             "created"     plot.output_series = {}, plot.state = {}

10:30:00     "created"     caller: replay_plot(plot, historical_snaps_09:30_to_10:30)
             "backfilling" └─ output_series and state cleared (were already empty)
                           └─ _rolling_stores["p"] cleared
                           └─ compute_one_tick() called for each of ~3600 snaps
                              each tick: appends to output_series, updates state,
                              pushes to rolling store
                           ⚠ THIS TAKES ~4-6 seconds of real wall time.
                             During this 4-6s the live feed is running and
                             generating new snaps (10:30:00–10:30:05-ish).
                             THOSE SNAPS ARE NOT CAPTURED. The library has no
                             buffer for them. This is a known gap.

~10:30:05    "live"        replay_plot() loop exits normally
                           plot.output_series has 3600 (ts, value) tuples
                           plot.state has accumulated as if running since 09:30
                           rolling store has last 24h of output values (= all 3600)

10:30:05     "live"        caller: asyncio.run(sched.run())
                           └─ scheduler calls active_plots() → ["p"]
                           └─ source.get_next() → current live snapshot (~10:30:05)
                           └─ compute_one_tick() appends tick 3601 to output_series
                           └─ gap: output_series jumps from ts=10:30:00 to ts=10:30:05
                              (4-5 missing seconds, never computed)

10:31:00–    "live"        one tick per second; output_series grows continuously
16:00:00
```

### Market close — what the code actually does

```
wall time    status        what happens in code
─────────────────────────────────────────────────────────────────────────────
16:00:00     "live"        live feed stops sending new snapshots.
                           LatestSnapshotSource._latest stays fixed at the
                           last snapshot it received (16:00:00 data).

16:00:01     "live"        scheduler wakes up, calls source.get_next()
                           → returns the same 16:00:00 snapshot (not None)
                           → compute_one_tick() runs again on stale data
                           → appends another (16:00:00_ms, value) to output_series
                           ⚠ DUPLICATE TIMESTAMPS. The same snapshot timestamp
                             is emitted every second after close.
                           ⚠ The scheduler loop NEVER terminates on its own —
                             LatestSnapshotSource never returns None after first set.
                             run() keeps going until stop() is called or process exits.

after 16:00  "live"        Nothing changes status. No automatic "market_closed" state.
                           The process must be externally stopped (sched.stop() or
                           process shutdown).
```

### Process restart next day — what the code actually does

```
wall time    status        what happens in code
─────────────────────────────────────────────────────────────────────────────
next morning  —            Process restarts. Python imports plot_compute fresh.

                           PlotRegistry.__init__() → self._plots = {}  (empty dict)
                           _rolling_stores = {}  (module-level, empty)

                           ALL of the following are gone:
                             • Every PlotRuntime object (plot_id, version, code)
                             • Every output_series (yesterday's computed history)
                             • Every plot.state (running sums, accumulators)
                             • Every rolling store (window history)

                           The library remembers NOTHING. There is no persistence.
                           The only durable data is whatever the caller stored
                           externally (e.g. the code string in a database).

09:30:00      —            Caller must:
                             1. Re-call create_plot(plot_id, code) for every plot
                                → re-runs the full 6-stage validation (again)
                                → status = "created"
                             2. Re-call replay_plot(plot, today's_snaps_from_09:30)
                                → status = "backfilling" → "live"
                             3. Restart scheduler.run()
```

### What is and isn't preserved across the replay→live boundary

| Thing | Preserved? | Where |
|---|---|---|
| `output_series` (all computed points) | ✅ Yes — live ticks append to the same list | `PlotRuntime.output_series` |
| `plot.state` (ctx.state dict) | ✅ Yes — same dict object, never cleared between replay and live | `PlotRuntime.state` |
| Rolling window data | ✅ Yes — `_rolling_stores["p"]` is warm from replay; first live tick has full window history | module-level `_rolling_stores` dict in runtime.py |
| `plot.version` | ✅ Yes — unchanged | `PlotRuntime.version` |
| Data gap during replay (~4-6s) | ❌ No — snapshots arriving during replay are not captured | Not implemented |
| Data after market close | ❌ No — scheduler keeps ticking stale data indefinitely | Not implemented |
| Anything after process restart | ❌ No — no persistence layer | Not implemented |

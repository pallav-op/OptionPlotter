import json

import pytest

from plot_compute import (
    PlotRegistry,
    RobustnessReport,
    ValidationError,
    build_edge_cases,
    load_edge_cases,
    run_robustness_suite,
    save_edge_cases,
)
from plot_compute.robustness import BAD_FORMAT, EXCEPTION, MUTATED_INPUT, NON_FINITE
from plot_compute.validation import validate_compute_signature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROBUST_CODE = """
def compute(ctx):
    c = ctx.chain.sum(option_type="CALL", field="bid_quantity_l1")
    p = ctx.chain.sum(option_type="PUT", field="bid_quantity_l1")
    return {"value": safe_div(c - p, c + p)}
"""

# Naive mean(iv) — returns NaN on all-NaN / no-quote / empty selections
NAN_RETURN_CODE = """
def compute(ctx):
    return {"value": ctx.chain.mean(option_type="CALL", field="iv", has_quote=True)}
"""

# Raw division by a count that can be zero — raises ZeroDivisionError
DIVZERO_CODE = """
def compute(ctx):
    iv = ctx.chain.mean(field="iv")
    n = ctx.chain.count(option_type="CALL")
    return {"value": iv / n}
"""


def _fn(code):
    return validate_compute_signature(code)


# ---------------------------------------------------------------------------
# Suite behaviour
# ---------------------------------------------------------------------------

def test_robust_fn_passes_all_cases():
    report = run_robustness_suite(_fn(ROBUST_CODE))
    assert report.ok()
    assert report.n_passed == report.total
    assert report.n_failed == 0


def test_nan_return_is_flagged():
    report = run_robustness_suite(_fn(NAN_RETURN_CODE))
    assert not report.ok()
    # the all_nan_iv and all_no_quote cases must be among the failures
    failed_names = {r.name for r in report.failures}
    assert "all_nan_iv" in failed_names
    assert "all_no_quote" in failed_names
    # those failures are classified as non-finite returns
    for r in report.failures:
        if r.name in ("all_nan_iv", "all_no_quote"):
            assert r.failure_kind == NON_FINITE


def test_exception_is_caught_not_propagated():
    report = run_robustness_suite(_fn(DIVZERO_CODE))
    assert not report.ok()
    # puts_only / calls_only drive count(CALL)==0 → ZeroDivisionError
    kinds = {r.failure_kind for r in report.failures}
    assert EXCEPTION in kinds
    div_fail = next(r for r in report.failures if r.failure_kind == EXCEPTION)
    assert "ZeroDivisionError" in div_fail.detail


def test_bad_format_is_flagged():
    report = run_robustness_suite(_fn("def compute(ctx): return [1, 2, 3]"))
    assert not report.ok()
    assert all(r.failure_kind == BAD_FORMAT for r in report.failures)


# ---------------------------------------------------------------------------
# Input-mutation detection
# ---------------------------------------------------------------------------

# Writes into the shared snapshot in place. `[:] = 0` is a no-op on the empty
# chain (so that case still passes), but mutates every non-empty case.
MUTATING_CODE = """
def compute(ctx):
    ctx.chain._snap.bid_quantity[:] = 0
    return {"value": 1.0}
"""


def test_mutation_is_flagged():
    report = run_robustness_suite(_fn(MUTATING_CODE))
    assert not report.ok()
    muts = [r for r in report.failures if r.failure_kind == MUTATED_INPUT]
    # every non-empty case mutated and was caught
    assert len(muts) >= 25
    # the detail names the offending field
    assert "bid_quantity" in muts[0].detail


def test_mutation_does_not_corrupt_the_battery():
    # Run a mutating function, then a clean one — the clean one must still pass,
    # proving the harness isolated each run on a deep copy of the battery.
    run_robustness_suite(_fn(MUTATING_CODE))
    clean = run_robustness_suite(_fn(ROBUST_CODE))
    assert clean.ok()


def test_mutation_rejected_by_create_plot():
    r = PlotRegistry()
    with pytest.raises(ValidationError) as exc_info:
        r.create_plot("mutator", MUTATING_CODE)
    assert exc_info.value.stage == "robustness"
    assert "mutated_input" in exc_info.value.message
    with pytest.raises(KeyError):
        r.get_plot("mutator")


def test_mutation_allowed_under_nonstrict():
    r = PlotRegistry()
    plot = r.create_plot("mutator", MUTATING_CODE, strict_robustness=False)
    assert plot.robustness_report.kind_counts()[MUTATED_INPUT] > 0


def test_kind_counts_includes_all_kinds():
    report = run_robustness_suite(_fn(ROBUST_CODE))
    counts = report.kind_counts()
    # every known kind is present as a key, even with zero failures
    assert set(counts) == {EXCEPTION, MUTATED_INPUT, BAD_FORMAT, NON_FINITE}


# ---------------------------------------------------------------------------
# Report shape & formatting
# ---------------------------------------------------------------------------

def test_report_counts_consistent():
    report = run_robustness_suite(_fn(NAN_RETURN_CODE))
    assert report.n_passed + report.n_failed == report.total
    assert report.total == len(build_edge_cases())


def test_report_category_scores():
    report = run_robustness_suite(_fn(ROBUST_CODE))
    scores = report.category_scores()
    # every category fully passes for the robust fn
    for cat, (passed, total) in scores.items():
        assert passed == total, cat


def test_report_format_is_informative():
    report = run_robustness_suite(_fn(NAN_RETURN_CODE))
    text = report.format()
    assert "ROBUSTNESS REPORT" in text
    assert "FAIL" in text
    assert "by category" in text
    assert "by failure" in text
    # a failing case shows its description and reason
    assert "all_nan_iv" in text


def test_report_is_dataclass_with_results():
    report = run_robustness_suite(_fn(ROBUST_CODE))
    assert isinstance(report, RobustnessReport)
    assert all(hasattr(r, "name") and hasattr(r, "passed") for r in report.results)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_create_plot_strict_rejects_buggy_code():
    r = PlotRegistry()
    with pytest.raises(ValidationError) as exc_info:
        r.create_plot("buggy", NAN_RETURN_CODE)
    assert exc_info.value.stage == "robustness"
    # plot must NOT have been registered
    with pytest.raises(KeyError):
        r.get_plot("buggy")


def test_create_plot_nonstrict_allows_buggy_but_reports():
    r = PlotRegistry()
    plot = r.create_plot("buggy", NAN_RETURN_CODE, strict_robustness=False)
    assert plot.robustness_report is not None
    assert not plot.robustness_report.ok()
    assert plot.robustness_report.n_failed > 0


def test_create_plot_attaches_report_on_success():
    r = PlotRegistry()
    plot = r.create_plot("good", ROBUST_CODE)
    assert plot.robustness_report is not None
    assert plot.robustness_report.ok()


def test_invalid_update_keeps_old_robust_version():
    r = PlotRegistry()
    r.create_plot("p", ROBUST_CODE)
    with pytest.raises(ValidationError):
        r.update_plot("p", NAN_RETURN_CODE)  # strict → rejected
    plot = r.get_plot("p")
    assert plot.version == 1
    assert plot.robustness_report.ok()


# ---------------------------------------------------------------------------
# Edge-case battery + persistence
# ---------------------------------------------------------------------------

def test_every_case_is_documented():
    for c in build_edge_cases():
        assert c.name and c.category and c.description
        assert len(c.description) > 10  # not a placeholder


def test_save_load_roundtrip(tmp_path):
    d = tmp_path / "vd"
    save_edge_cases(str(d))
    assert (d / "edge_cases.pkl").exists()
    assert (d / "manifest.json").exists()
    assert (d / "README.md").exists()

    manifest = json.loads((d / "manifest.json").read_text())
    fresh = build_edge_cases()
    assert manifest["n_cases"] == len(fresh)
    assert len(manifest["cases"]) == len(fresh)
    # the mutation invariant is documented in the manifest
    kinds = {inv["kind"] for inv in manifest["invariants"]}
    assert MUTATED_INPUT in kinds

    loaded = load_edge_cases(str(d))
    assert [c.name for c in loaded] == [c.name for c in fresh]


def test_load_falls_back_to_fresh_when_absent(tmp_path):
    loaded = load_edge_cases(str(tmp_path / "does_not_exist"))
    assert len(loaded) == len(build_edge_cases())

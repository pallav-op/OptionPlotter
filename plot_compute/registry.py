from __future__ import annotations

from .errors import ValidationError
from .models import PlotRuntime
from .validation import validate_plot_code


class PlotRegistry:
    """Manages the lifecycle of all PlotRuntime instances."""

    def __init__(self) -> None:
        self._plots: dict[str, PlotRuntime] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_plot(self, plot_id: str, code: str) -> PlotRuntime:
        """Validate code and create a new PlotRuntime.

        The caller is responsible for driving historical replay after creation.
        Raises ValidationError if any validation stage fails.
        """
        if plot_id in self._plots:
            raise ValueError(f"Plot {plot_id!r} already exists. Use update_plot to modify it.")

        compute_fn = validate_plot_code(code)
        plot = PlotRuntime(
            plot_id=plot_id,
            version=1,
            code=code,
            compute_fn=compute_fn,
            status="created",
        )
        self._plots[plot_id] = plot
        return plot

    def update_plot(self, plot_id: str, code: str) -> PlotRuntime:
        """Update the compute code of an existing plot.

        If validation fails the old version continues running and ValidationError is raised.
        On success: version is incremented and state/series are cleared (caller replays).
        """
        plot = self._get_existing(plot_id)

        compute_fn = validate_plot_code(code)  # raises ValidationError on failure

        # Validation passed — update in place
        plot.version += 1
        plot.code = code
        plot.compute_fn = compute_fn
        plot.state.clear()
        plot.output_series.clear()
        plot.last_computed_ts = None
        plot.error = None
        plot.status = "created"
        return plot

    def delete_plot(self, plot_id: str) -> None:
        plot = self._get_existing(plot_id)
        plot.status = "deleted"
        del self._plots[plot_id]

    def pause_plot(self, plot_id: str) -> None:
        plot = self._get_existing(plot_id)
        if plot.status == "live":
            plot.status = "paused"

    def resume_plot(self, plot_id: str) -> None:
        plot = self._get_existing(plot_id)
        if plot.status == "paused":
            plot.status = "live"

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_plot(self, plot_id: str) -> PlotRuntime:
        return self._get_existing(plot_id)

    def active_plots(self) -> list[PlotRuntime]:
        return [p for p in self._plots.values() if p.status == "live"]

    def all_plots(self) -> list[PlotRuntime]:
        return list(self._plots.values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_existing(self, plot_id: str) -> PlotRuntime:
        plot = self._plots.get(plot_id)
        if plot is None:
            raise KeyError(f"Plot {plot_id!r} not found.")
        return plot

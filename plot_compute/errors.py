class ValidationError(Exception):
    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


class RuntimePlotError(Exception):
    def __init__(self, plot_id: str, version: int, message: str) -> None:
        self.plot_id = plot_id
        self.version = version
        self.message = message
        super().__init__(f"plot {plot_id} v{version}: {message}")

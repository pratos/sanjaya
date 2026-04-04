"""REPL execution logging helpers."""

from rich.console import Console

_console = Console()


class REPLEnvLogger:
    """Logger for code execution inside the REPL environment."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def _print(self, message: str) -> None:
        if self.enabled:
            _console.print(message)

    def log_execution_start(self, index: int, total: int, code: str) -> None:
        self._print(
            f"[cyan]Running code block {index}/{total} ({len(code)} chars)[/]\n[dim]{code[:400]}[/]"
        )

    def log_execution_end(self, index: int, total: int, execution_time: float) -> None:
        self._print(f"[cyan]Finished code block {index}/{total} in {execution_time:.2f}s[/]")

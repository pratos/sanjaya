"""High-level colorful logger for RLM orchestration."""

from rich.console import Console

_console = Console()


class ColorfulLogger:
    """Simple console logger with optional output suppression."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def _print(self, message: str) -> None:
        if self.enabled:
            _console.print(message)

    def log_query_start(self, query: str) -> None:
        self._print(f"[bold cyan]Query:[/] {query}")

    def log_initial_messages(self, messages: list[dict[str, str]]) -> None:
        self._print(f"[dim]Initialized message history ({len(messages)} message(s))[/]")

    def log_model_response(self, response: str, has_tool_calls: bool) -> None:
        tool_hint = "with code blocks" if has_tool_calls else "without code blocks"
        self._print(f"[blue]Model response ({tool_hint}):[/] {response[:300]}")

    def log_execution_result(self, index: int, total: int, stdout: str, stderr: str, result: object) -> None:
        self._print(f"[green]Executed code block {index}/{total}[/]")
        if stdout:
            self._print(f"[dim]stdout:[/] {stdout[:300]}")
        if stderr:
            self._print(f"[red]stderr:[/] {stderr[:300]}")
        if result is not None:
            self._print(f"[dim]result:[/] {result!r}")

    def log_final_response(self, final_answer: object) -> None:
        self._print(f"[bold green]Final answer:[/] {final_answer}")

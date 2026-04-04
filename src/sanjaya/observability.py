"""Observability configuration and helpers for Pydantic Logfire."""

from contextlib import nullcontext
from types import ModuleType
from typing import Any

_logfire_configured = False
_logfire_module: ModuleType | None = None


def _import_logfire() -> ModuleType | None:
    """Import and cache logfire module."""
    global _logfire_module

    if _logfire_module is not None:
        return _logfire_module

    try:
        import logfire as logfire_module
    except ImportError:
        return None

    _logfire_module = logfire_module
    return _logfire_module


def configure_logfire(enabled: bool = True) -> bool:
    """Configure Pydantic Logfire for observability.

    Reads LOGFIRE_TOKEN and LOGFIRE_SERVICE_NAME from settings.

    Args:
        enabled: Whether to enable Logfire instrumentation

    Returns:
        True if Logfire was configured, False if disabled or unavailable
    """
    global _logfire_configured

    if not enabled:
        return False

    if _logfire_configured:
        return True

    logfire_module = _import_logfire()
    if logfire_module is None:
        return False

    from .settings import get_settings

    settings = get_settings()

    if not settings.logfire_token:
        # No token configured, skip silently
        return False

    logfire_module.configure(
        token=settings.logfire_token,
        service_name=settings.logfire_service_name,
        send_to_logfire=True,
    )

    # Instrument Pydantic AI (captures all agent runs automatically)
    # Prefer semantic convention v2 with explicit content capture; fall back safely for
    # older compatible versions of pydantic-ai/logfire.
    try:
        logfire_module.instrument_pydantic_ai(
            include_content=True,
            include_binary_content=True,
            version=2,
        )
    except TypeError:
        try:
            logfire_module.instrument_pydantic_ai(
                include_content=True,
                include_binary_content=True,
            )
        except TypeError:
            logfire_module.instrument_pydantic_ai()

    _logfire_configured = True
    return True


def is_logfire_configured() -> bool:
    """Return whether Logfire is currently configured."""
    return _logfire_configured


def get_logfire(enabled: bool = True) -> ModuleType | None:
    """Get configured logfire module or None.

    Args:
        enabled: Whether observability should be enabled

    Returns:
        Configured logfire module or None if unavailable/unconfigured
    """
    if not enabled:
        return None

    if not configure_logfire(enabled=True):
        return None

    return _import_logfire()


def span(msg_template: str, **attributes: Any):
    """Create a Logfire span context manager.

    Falls back to a no-op if Logfire is not configured.

    Args:
        msg_template: Message template with {placeholders} for attributes
        **attributes: Span attributes (referenced in template with {name=} syntax)

    Returns:
        Context manager for the span
    """
    logfire_module = get_logfire(enabled=True)
    if logfire_module is None:
        return nullcontext()

    return logfire_module.span(msg_template, **attributes)


def log_info(msg_template: str, **kwargs: Any) -> None:
    """Log an info message to Logfire.

    Args:
        msg_template: Message template with {placeholders}
        **kwargs: Values for placeholders

    No-op if Logfire is not configured.
    """
    logfire_module = get_logfire(enabled=True)
    if logfire_module is None:
        return

    logfire_module.info(msg_template, **kwargs)


def log_error(msg_template: str, **kwargs: Any) -> None:
    """Log an error message to Logfire.

    Args:
        msg_template: Message template with {placeholders}
        **kwargs: Values for placeholders

    No-op if Logfire is not configured.
    """
    logfire_module = get_logfire(enabled=True)
    if logfire_module is None:
        return

    logfire_module.error(msg_template, **kwargs)


def shutdown_logfire(timeout_ms: int = 2000) -> None:
    """Flush and shutdown Logfire to avoid hanging at exit.

    Args:
        timeout_ms: Maximum time to wait for flush in milliseconds (default: 2s)
    """
    if not _logfire_configured:
        return

    logfire_module = _import_logfire()
    if logfire_module is None:
        return

    try:
        # Force flush with timeout
        if hasattr(logfire_module, "force_flush"):
            logfire_module.force_flush(timeout_ms)
        # Shutdown the provider
        if hasattr(logfire_module, "shutdown"):
            logfire_module.shutdown()
    except Exception:
        pass  # Best effort - don't fail on shutdown

"""Logfire configuration and setup."""

from __future__ import annotations

from typing import Any


def get_logfire(enabled: bool = True) -> Any | None:
    """Get configured Logfire module, or None if unavailable/disabled."""
    if not enabled:
        return None
    try:
        import logfire
        return logfire
    except ImportError:
        return None


def configure_logfire(service_name: str = "sanjaya") -> None:
    """Configure logfire with default settings."""
    logfire = get_logfire()
    if logfire is None:
        return

    try:
        from ..settings import get_settings
        settings = get_settings()
        logfire.configure(
            service_name=settings.logfire_service_name or service_name,
            token=settings.logfire_token,
        )
    except Exception:
        pass

    try:
        logfire.instrument_pydantic_ai()
    except Exception:
        pass

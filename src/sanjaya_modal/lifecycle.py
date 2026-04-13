"""Deploy / stop / health-check Moondream Photon on Modal."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

APP_NAME = "sanjaya-moondream"
_APP_FILE = Path(__file__).parent / "app.py"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModalEndpoint:
    """A deployed Modal endpoint with auth credentials."""

    base_url: str
    auth_token: str

    def __str__(self) -> str:
        return self.base_url


# ── Public helpers ───────────────────────────────────────────


def deploy(gpu: str = "L4") -> ModalEndpoint:
    """Deploy Moondream Photon to Modal and return the endpoint info.

    Returns a ``ModalEndpoint`` with ``.base_url`` (the ``/v1`` URL) and
    ``.auth_token`` (Bearer token for authenticated requests).

    Example base_url: ``https://user--sanjaya-moondream-server.modal.run/v1``
    """
    # Ensure .env is loaded so MOONDREAM_API_KEY propagates to Modal
    _load_dotenv()

    env = os.environ.copy()
    env["SANJAYA_MODAL_GPU"] = gpu

    result = subprocess.run(
        ["modal", "deploy", str(_APP_FILE)],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"modal deploy failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )

    # Parse endpoint URL from deploy output
    url_match = re.search(r"https://\S+\.modal\.run", result.stdout + result.stderr)
    if not url_match:
        raise RuntimeError(
            f"Could not find endpoint URL in modal output:\n{result.stdout}"
        )

    base = url_match.group(0).rstrip("/")

    # Read the auth token that app.py resolved/generated at import time.
    # We import it lazily because app.py has a modal dependency and runs
    # .env parsing + token generation at module scope.
    from sanjaya_modal.app import AUTH_TOKEN

    return ModalEndpoint(base_url=f"{base}/v1", auth_token=AUTH_TOKEN)


def stop() -> None:
    """Stop the Modal app (scales containers to zero immediately)."""
    subprocess.run(
        ["modal", "app", "stop", APP_NAME],
        capture_output=True,
        text=True,
    )
    logger.info("Stopped Modal app %s", APP_NAME)


def warm_up(base_url: str, timeout: int = 300, poll_interval: int = 5) -> None:
    """Block until Moondream Photon responds (triggers cold start).

    First request triggers container creation → Photon init → model load.
    Subsequent requests are fast.
    """
    deadline = time.time() + timeout
    last_err: Exception | None = None

    while time.time() < deadline:
        try:
            # Hit the health endpoint; any HTTP response proves the
            # container is alive and Photon has finished loading.
            health_url = base_url.rstrip("/") + "/health"
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "sanjaya-modal/0.1"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                logger.info("Photon ready (HTTP %s)", resp.status)
                return
        except urllib.error.HTTPError:
            # 404/405 etc. — server is up, just no route for bare /v1
            logger.info("Photon ready (got HTTP error, server is up)")
            return
        except Exception as exc:
            last_err = exc
            logger.debug("Waiting for Photon: %s", exc)
            time.sleep(poll_interval)

    raise TimeoutError(
        f"Moondream Photon not ready after {timeout}s. Last error: {last_err}"
    )


# ── Context manager ──────────────────────────────────────────


class ModalMoondream:
    """Context manager that deploys and tears down Moondream on Modal.

    Usage::

        with ModalMoondream(gpu="L4") as endpoint:
            os.environ["MOONDREAM_BASE_URL"] = endpoint.base_url
            os.environ["MOONDREAM_AUTH_TOKEN"] = endpoint.auth_token
            # ... run benchmarks ...
        # container stopped automatically
    """

    def __init__(self, gpu: str = "L4", stop_on_exit: bool = True):
        self.gpu = gpu
        self.stop_on_exit = stop_on_exit
        self.endpoint: ModalEndpoint | None = None

    def __enter__(self) -> ModalEndpoint:
        self.endpoint = deploy(gpu=self.gpu)
        warm_up(self.endpoint.base_url)
        return self.endpoint

    def __exit__(self, *_: Any) -> None:
        if self.stop_on_exit:
            stop()


# ── Internal ─────────────────────────────────────────────────


def _load_dotenv() -> None:
    """Load project .env into os.environ (no external deps)."""
    dotenv = _PROJECT_ROOT / ".env"
    if not dotenv.exists():
        return
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)

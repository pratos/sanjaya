"""Modal app — Moondream Photon on GPU.

Runs the ``moondream`` Python package with ``local=True`` (Photon engine)
behind a thin FastAPI server that exposes the same ``/v1/caption`` and
``/v1/query`` endpoints that sanjaya's MoondreamVisionClient expects.

Auth: every request must include ``Authorization: Bearer <token>`` where
the token matches the ``SANJAYA_MODAL_AUTH_TOKEN`` secret.  The health
endpoint is exempt so the warm-up poller can reach it unauthenticated.

Deploy manually:
    SANJAYA_MODAL_GPU=L4 modal deploy src/sanjaya_modal/app.py

Or use the lifecycle helpers / runner which handle this automatically.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import modal

_GPU = os.environ.get("SANJAYA_MODAL_GPU", "L4")

app = modal.App("sanjaya-moondream")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "moondream>=1.1.0",
        "starlette>=0.40",
        "pillow>=10.0",
    )
)

# ── Read secrets from local .env at deploy time ─────────────
_DOTENV = Path(__file__).resolve().parent.parent.parent / ".env"
_MOONDREAM_KEY = ""
_AUTH_TOKEN = ""
if _DOTENV.exists():
    for line in _DOTENV.read_text().splitlines():
        if line.startswith("MOONDREAM_API_KEY="):
            _MOONDREAM_KEY = line.split("=", 1)[1].strip().strip("'\"")
        elif line.startswith("SANJAYA_MODAL_AUTH_TOKEN="):
            _AUTH_TOKEN = line.split("=", 1)[1].strip().strip("'\"")
_MOONDREAM_KEY = os.environ.get("MOONDREAM_API_KEY", _MOONDREAM_KEY)
_AUTH_TOKEN = os.environ.get("SANJAYA_MODAL_AUTH_TOKEN", _AUTH_TOKEN)

# Auto-generate a token if none is configured
if not _AUTH_TOKEN:
    _AUTH_TOKEN = secrets.token_urlsafe(32)
    print(f"[sanjaya_modal] Generated auth token (no SANJAYA_MODAL_AUTH_TOKEN set)")

# Export so lifecycle.py can read it back
AUTH_TOKEN = _AUTH_TOKEN

_secrets_dict: dict[str, str] = {"SANJAYA_MODAL_AUTH_TOKEN": _AUTH_TOKEN}
if _MOONDREAM_KEY:
    _secrets_dict["MOONDREAM_API_KEY"] = _MOONDREAM_KEY


@app.function(
    image=image,
    gpu=_GPU,
    memory=4096,
    timeout=86400,
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    secrets=[modal.Secret.from_dict(_secrets_dict)],
)
@modal.asgi_app()
def server():
    """Initialise Photon and return a FastAPI app."""
    import base64
    import io
    import os
    import time

    import moondream as md
    from PIL import Image
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    # ── Auth ─────────────────────────────────────────────────
    _auth_token = os.environ.get("SANJAYA_MODAL_AUTH_TOKEN", "")

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        """Reject requests without a valid Bearer token.

        The ``/v1/health`` endpoint is exempt so warm-up polling works
        without credentials.
        """

        async def dispatch(self, request: Request, call_next):
            if request.url.path.rstrip("/").endswith("/health"):
                return await call_next(request)

            if not _auth_token:
                # No token configured → open access (shouldn't happen)
                return await call_next(request)

            auth = request.headers.get("authorization", "")
            if auth == f"Bearer {_auth_token}":
                return await call_next(request)

            return JSONResponse(
                {"error": "unauthorized", "detail": "Missing or invalid Bearer token"},
                status_code=401,
            )

    # ── Boot Photon ──────────────────────────────────────────
    api_key = os.environ.get("MOONDREAM_API_KEY", "")
    print(f"Initializing Moondream Photon (local=True, gpu={_GPU}) ...")
    t0 = time.time()
    model = md.vl(api_key=api_key, local=True)
    print(f"Photon ready in {time.time() - t0:.1f}s")

    # ── Helpers ──────────────────────────────────────────────
    TOKENS_PER_IMAGE = 729

    def _decode(image_url: str) -> Image.Image:
        payload = image_url.split(",", 1)[1] if "," in image_url else image_url
        return Image.open(io.BytesIO(base64.b64decode(payload)))

    # ── Routes ───────────────────────────────────────────────

    async def health(request):
        return JSONResponse({"status": "ok", "engine": "photon", "gpu": _GPU})

    async def caption(request):
        body = await request.json()
        img = _decode(body["image_url"])
        length = body.get("length", "normal")
        result = model.caption(img, length=length)
        text = result["caption"]
        out_tokens = max(1, int(len(text.split()) * 1.3))
        return JSONResponse({
            "caption": text,
            "metrics": {"input_tokens": TOKENS_PER_IMAGE, "output_tokens": out_tokens},
        })

    async def query(request):
        body = await request.json()
        img = _decode(body["image_url"])
        result = model.query(img, body["question"])
        text = result["answer"]
        out_tokens = max(1, int(len(text.split()) * 1.3))
        return JSONResponse({
            "answer": text,
            "metrics": {"input_tokens": TOKENS_PER_IMAGE, "output_tokens": out_tokens},
        })

    return Starlette(
        routes=[
            Route("/v1/health", health, methods=["GET"]),
            Route("/v1/caption", caption, methods=["POST"]),
            Route("/v1/query", query, methods=["POST"]),
        ],
        middleware=[Middleware(BearerAuthMiddleware)],
    )

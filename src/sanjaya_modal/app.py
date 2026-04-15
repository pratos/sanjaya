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
    scaledown_window=3600,
    secrets=[modal.Secret.from_dict({
        **_secrets_dict,
        # Photon auto-batches concurrent inference calls up to this limit.
        # Default is 4; raise to match typical frame counts per clip.
        "KESTREL_MAX_BATCH_SIZE": "8",
    })],
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
        """Decode a base64 data-URI into a fully-loaded RGB PIL Image.

        Forces an eager ``.load()`` so broken/truncated JPEGs fail here
        with a clear error instead of crashing deep inside Photon's
        ``_image_to_bytes`` re-encode step.
        """
        payload = image_url.split(",", 1)[1] if "," in image_url else image_url
        img = Image.open(io.BytesIO(base64.b64decode(payload)))
        img.load()  # force full decode — catches truncated streams
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    # ── Inference helpers (run in threads so Photon can auto-batch) ──

    import asyncio

    def _caption_sync(img: Image.Image, length: str) -> dict:
        return model.caption(img, length=length)

    def _query_sync(img: Image.Image, question: str) -> dict:
        return model.query(img, question)

    # ── Routes ───────────────────────────────────────────────

    async def health(request):
        return JSONResponse({"status": "ok", "engine": "photon", "gpu": _GPU})

    async def caption(request):
        body = await request.json()
        try:
            img = _decode(body["image_url"])
        except Exception as e:
            return JSONResponse(
                {"error": "bad_image", "detail": str(e)}, status_code=400,
            )
        length = body.get("length", "normal")
        result = await asyncio.to_thread(_caption_sync, img, length)
        text = result["caption"]
        out_tokens = max(1, int(len(text.split()) * 1.3))
        return JSONResponse({
            "caption": text,
            "metrics": {"input_tokens": TOKENS_PER_IMAGE, "output_tokens": out_tokens},
        })

    async def batch_caption(request):
        body = await request.json()
        images = body.get("images", [])
        length = body.get("length", "normal")

        # Decode all images first (CPU-bound, fast)
        decoded: list[tuple[int, Image.Image | str]] = []
        for i, item in enumerate(images):
            try:
                decoded.append((i, _decode(item["image_url"])))
            except Exception as e:
                decoded.append((i, f"[caption error: {e}]"))

        # Run inference concurrently — Photon auto-batches at GPU level
        async def _cap(idx: int, img_or_err):
            if isinstance(img_or_err, str):
                return idx, img_or_err, 0
            result = await asyncio.to_thread(_caption_sync, img_or_err, length)
            text = result["caption"]
            out_tokens = max(1, int(len(text.split()) * 1.3))
            return idx, text, out_tokens

        results = await asyncio.gather(*[_cap(i, v) for i, v in decoded])
        results_sorted = sorted(results, key=lambda r: r[0])

        captions = [text for _, text, _ in results_sorted]
        total_in = sum(TOKENS_PER_IMAGE for _, v in decoded if not isinstance(v, str))
        total_out = sum(ot for _, _, ot in results_sorted)
        return JSONResponse({
            "captions": captions,
            "metrics": {"input_tokens": total_in, "output_tokens": total_out},
        })

    async def query(request):
        body = await request.json()
        try:
            img = _decode(body["image_url"])
        except Exception as e:
            return JSONResponse(
                {"error": "bad_image", "detail": str(e)}, status_code=400,
            )
        result = await asyncio.to_thread(_query_sync, img, body["question"])
        text = result["answer"]
        out_tokens = max(1, int(len(text.split()) * 1.3))
        return JSONResponse({
            "answer": text,
            "metrics": {"input_tokens": TOKENS_PER_IMAGE, "output_tokens": out_tokens},
        })

    async def batch_query(request):
        body = await request.json()
        queries = body.get("queries", [])

        # Decode all images first
        decoded: list[tuple[int, Image.Image | str, str]] = []
        for i, item in enumerate(queries):
            try:
                decoded.append((i, _decode(item["image_url"]), item["question"]))
            except Exception as e:
                decoded.append((i, f"[query error: {e}]", ""))

        # Run inference concurrently — Photon auto-batches at GPU level
        async def _qry(idx: int, img_or_err, question: str):
            if isinstance(img_or_err, str):
                return idx, img_or_err, 0
            result = await asyncio.to_thread(_query_sync, img_or_err, question)
            text = result["answer"]
            out_tokens = max(1, int(len(text.split()) * 1.3))
            return idx, text, out_tokens

        results = await asyncio.gather(*[_qry(i, v, q) for i, v, q in decoded])
        results_sorted = sorted(results, key=lambda r: r[0])

        answers = [text for _, text, _ in results_sorted]
        total_in = sum(TOKENS_PER_IMAGE for _, v, _ in decoded if not isinstance(v, str))
        total_out = sum(ot for _, _, ot in results_sorted)
        return JSONResponse({
            "answers": answers,
            "metrics": {"input_tokens": total_in, "output_tokens": total_out},
        })

    return Starlette(
        routes=[
            Route("/v1/health", health, methods=["GET"]),
            Route("/v1/caption", caption, methods=["POST"]),
            Route("/v1/batch/caption", batch_caption, methods=["POST"]),
            Route("/v1/query", query, methods=["POST"]),
            Route("/v1/batch/query", batch_query, methods=["POST"]),
        ],
        middleware=[Middleware(BearerAuthMiddleware)],
    )

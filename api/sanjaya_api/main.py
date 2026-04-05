"""FastAPI application for Sanjaya HUD backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sanjaya_api.routes.health import router as health_router
from sanjaya_api.routes.runs import router as runs_router

app = FastAPI(
    title="Sanjaya API",
    description="FastAPI bridge for VideoRLM orchestration monitoring",
    version="0.1.0",
)

# CORS — allow the Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5100",
        "http://127.0.0.1:5100",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(runs_router)

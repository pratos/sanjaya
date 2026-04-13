#!/usr/bin/env python3
"""Run sanjaya benchmarks with Moondream Photon on a Modal GPU.

Handles the full lifecycle: deploy → warm-up → benchmark → teardown.
Photon runs locally on the Modal GPU; sanjaya talks to it via the same
REST API as Moondream Cloud (MOONDREAM_BASE_URL env var override).

Usage:
    # Run all 12 video prompts on an L4
    uv run python -m sanjaya_modal.runner

    # Use a faster GPU, pass extra args to the benchmark script
    uv run python -m sanjaya_modal.runner --gpu A10 -- --prompt 1 2 3

    # Run document benchmarks instead
    uv run python -m sanjaya_modal.runner --script scripts/run_demo_documents.py

    # Keep the Modal app running after the benchmark (for follow-up runs)
    uv run python -m sanjaya_modal.runner --keep-alive
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure src/ is importable when run as a script
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sanjaya_modal.lifecycle import ModalMoondream, stop  # noqa: E402

_PROJECT_ROOT = _SRC.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sanjaya benchmarks with Moondream on Modal GPU.",
    )
    parser.add_argument(
        "--gpu",
        default="L4",
        help="Modal GPU type (default: L4). Options: T4, L4, A10, L40S, A100, H100",
    )
    parser.add_argument(
        "--script",
        default="scripts/run_demo_prompts.py",
        help="Benchmark script to run (default: scripts/run_demo_prompts.py)",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Don't stop the Modal app after the run (useful for follow-up runs)",
    )

    args, extra = parser.parse_known_args()

    print(f"{'=' * 60}")
    print(f"  Sanjaya Benchmark — Moondream Photon on Modal ({args.gpu})")
    print(f"  Script: {args.script}")
    if extra:
        print(f"  Extra args: {extra}")
    print(f"{'=' * 60}\n")

    with ModalMoondream(gpu=args.gpu, stop_on_exit=not args.keep_alive) as endpoint:
        print(f"✅ Moondream Photon ready at {endpoint.base_url}")
        print(f"   Auth token: {endpoint.auth_token[:8]}...{endpoint.auth_token[-4:]}\n")

        # Pass the Modal endpoint to the benchmark via env vars.
        # MoondreamVisionClient picks up MOONDREAM_BASE_URL and
        # MOONDREAM_AUTH_TOKEN automatically.
        env = os.environ.copy()
        env["MOONDREAM_BASE_URL"] = endpoint.base_url
        env["MOONDREAM_AUTH_TOKEN"] = endpoint.auth_token

        script_path = _PROJECT_ROOT / args.script
        cmd = ["uv", "run", "python", str(script_path)] + extra

        print(f"Running: {' '.join(cmd)}\n")
        t0 = time.time()

        result = subprocess.run(cmd, env=env, cwd=str(_PROJECT_ROOT))

        elapsed = time.time() - t0
        minutes = elapsed / 60

        print(f"\n{'=' * 60}")
        print(f"  Benchmark finished in {minutes:.1f} min")
        print(f"  Modal GPU cost estimate: ~${elapsed / 3600 * _gpu_rate(args.gpu):.2f}")
        if not args.keep_alive:
            print("  Modal app stopped.")
        else:
            print("  Modal app kept alive (use `modal app stop sanjaya-moondream` to stop).")
        print(f"{'=' * 60}")

        sys.exit(result.returncode)


def _gpu_rate(gpu: str) -> float:
    """Approximate $/hr for Modal GPUs (as of 2025)."""
    rates = {
        "T4": 0.59,
        "L4": 0.80,
        "A10": 1.10,
        "L40S": 1.95,
        "A100": 2.10,
        "A100-80GB": 2.50,
        "H100": 3.95,
    }
    return rates.get(gpu, 1.00)


if __name__ == "__main__":
    main()

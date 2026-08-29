"""Render one portrait + one background with every Krea 2 checkpoint in a
directory, through the real ``EmbeddedImageClient``.

Not a unit test — this loads 12 GB of weights per checkpoint and needs a
GPU, so it lives here rather than in ``tests/``. It's the end-to-end
check that the ComfyUI→diffusers conversion in
:mod:`lucidium.providers.krea_checkpoint` actually produces a working
pipeline, for both the fp8-scaled and int8-convrot quantizations.

Usage::

    python scripts/krea_live_render.py <models_dir> [<out_dir>]

Writes ``<out_dir>/<checkpoint stem>-<workflow>.png`` and prints the
per-render wall clock. A non-zero exit means at least one render failed.

Set ``KREA_LIVE_QUICK=1`` to render only the first case per checkpoint —
enough to prove the conversion loads and samples, at roughly half the
GPU time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucidium.providers.embedded_image_client import (
    EmbeddedImageClient,
    _is_krea_model_path,
)

# Two renders per checkpoint: the portrait bucket (832x1216) exercises
# the character path, the background bucket (1536x1024) exercises the
# landscape one and is the render that historically OOMs first.
_CASES: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "character.workflow.json",
        {
            "positive_prompt": (
                "full body portrait of a young woman with short auburn hair, "
                "wearing a green wool coat, standing, plain grey backdrop, "
                "soft studio lighting, head fully visible"
            ),
            "face_prompt": "calm confident expression, looking at viewer",
            "negative_extras": "",
        },
    ),
    (
        "background.workflow.json",
        {
            "positive_prompt": (
                "a quiet cobblestone street in a European old town at dusk, "
                "warm lamplight, light rain, no people"
            ),
            "negative_extras": "",
        },
    ),
)


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    models_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("krea_renders")
    out_dir.mkdir(parents=True, exist_ok=True)

    # One-shot CLI script: blocking directory scan is harmless here.
    checkpoints = sorted(p for p in models_dir.iterdir() if p.is_file() and _is_krea_model_path(p))  # noqa: ASYNC240
    if not checkpoints:
        print(f"no Krea 2 checkpoints found in {models_dir}")
        return 1
    print(f"found {len(checkpoints)} Krea 2 checkpoint(s):")
    for path in checkpoints:
        print(f"  - {path.name}")

    failures = 0
    for path in checkpoints:
        # A fresh client per checkpoint so each one pays a cold load and
        # nothing leaks across (12 GB resident apiece — they can't be
        # co-resident on a 24 GiB card).
        client = EmbeddedImageClient(
            models_dir=str(models_dir),
            model_name=path.name,
            # Skip rembg: this script is checking the diffusion pipeline,
            # and the background cut is family-independent.
            bg_remover=lambda data: data,
        )
        cases = _CASES[:1] if os.environ.get("KREA_LIVE_QUICK") else _CASES
        for workflow, params in cases:
            label = f"{path.stem} / {workflow}"
            started = time.monotonic()
            try:
                png = await client.generate(workflow, params, seed=12345)
            except Exception as exc:
                failures += 1
                print(f"FAIL {label}: {type(exc).__name__}: {exc}")
                continue
            elapsed = time.monotonic() - started
            target = out_dir / f"{path.stem}-{workflow.split('.')[0]}.png"
            target.write_bytes(png)
            print(f"OK   {label}: {len(png)} bytes in {elapsed:.1f}s -> {target}")
        await client.aclose()

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

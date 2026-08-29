"""One-off REAL Qwen-Image validation on this machine.

Drives the production download + EmbeddedImageClient path end to end so
the ACTUAL first-run flow runs, not a hand-placed file:

  * download_model(MODEL_CATALOG["qwen-image"]) -> fetches (idempotent if
    already present) the fp8 distill checkpoint under the catalog's
    local_filename, the same name the first-run wizard saves,
  * _load_qwen_pipeline (transformer from single-file + Qwen2.5-VL text
    encoder / VAE / scheduler from Qwen/Qwen-Image),
  * _apply_qwen_offload (block-level streaming / fp8 on this 24 GiB card),
  * generate() -> Qwen text2img recipe,
  * regenerate_from_image() -> Qwen img2img from the generated portrait.

Not part of the test suite (needs the ~20 GiB checkpoint + a GPU). Run:
    .venv/Scripts/python.exe scripts/_qwen_real_run.py
"""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

from PIL import Image

from lucidium.providers.embedded_image_client import EmbeddedImageClient
from lucidium.providers.embedded_models import (
    MODEL_CATALOG,
    download_model,
    list_models,
    pick_default_model,
)

MODELS_DIR = Path("C:/Users/conra/.cache/lucidium-qwen-test/models")
OUT_DIR = Path("C:/Users/conra/.cache/lucidium-qwen-test/out")


def _vram_peak_gib() -> float:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


async def main() -> None:
    # One-shot benchmark script: blocking path I/O at startup is harmless.
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- integrated download flow (the first-run wizard's one-click path) ---
    spec = MODEL_CATALOG["qwen-image"]
    print(f">>> download_model({spec.key}) -> {spec.local_filename} (idempotent)")
    t0 = time.monotonic()
    last_pct = -10

    def _on_progress(done: int, total: int | None) -> None:
        nonlocal last_pct
        if total:
            pct = int(done * 100 / total)
            if pct >= last_pct + 10:
                last_pct = pct
                print(f"    download {pct:3d}%  ({done / 1024**3:.1f}/{total / 1024**3:.1f} GiB)")

    path = download_model(spec, MODELS_DIR, on_progress=_on_progress)
    print(
        f"    checkpoint at {path} ({path.stat().st_size / 1024**3:.1f} GiB) "
        f"in {time.monotonic() - t0:.1f}s"
    )

    # The loader picks the checkpoint exactly as the running app does — by
    # listing the dir and resolving the configured/first model. Verify the
    # download landed under a name the production picker actually selects.
    available = list_models(MODELS_DIR)
    print(f"    list_models -> {available}")
    picked = pick_default_model(MODELS_DIR, spec.local_filename)
    print(f"    pick_default_model -> {picked.name if picked else None}")
    assert picked == path, f"picker chose {picked}, expected {path}"

    # identity bg remover so we don't pull rembg/onnxruntime for this probe.
    client = EmbeddedImageClient(
        models_dir=str(MODELS_DIR),
        bg_remover=lambda b: b,
    )

    print(">>> text2img (generate) — loads pipeline + block offload")
    t0 = time.monotonic()
    png = await client.generate(
        "character.json",
        {
            "positive_prompt": (
                "full body portrait of a knight in ornate silver armor, "
                "standing, neutral background, detailed"
            ),
            "subject_kind": "human",
        },
        seed=12345,
    )
    dt = time.monotonic() - t0
    img = Image.open(io.BytesIO(png))
    (OUT_DIR / "text2img.png").write_bytes(png)
    print(f"    text2img OK: {img.size} {img.mode} {len(png)} bytes in {dt:.1f}s")
    print(f"    peak VRAM: {_vram_peak_gib():.1f} GiB")

    print(">>> img2img (regenerate_from_image) — character change from the portrait")
    t0 = time.monotonic()
    png2 = await client.regenerate_from_image(
        "character.json",
        png,
        {
            "positive_prompt": (
                "full body portrait of a knight, now wearing a red cloak "
                "over the silver armor, standing, neutral background"
            ),
            "subject_kind": "human",
        },
        seed=12345,
    )
    dt = time.monotonic() - t0
    if png2 is None:
        print("    img2img DECLINED (returned None) — UNEXPECTED for a Qwen checkpoint")
    else:
        img2 = Image.open(io.BytesIO(png2))
        (OUT_DIR / "img2img.png").write_bytes(png2)
        print(f"    img2img OK: {img2.size} {img2.mode} {len(png2)} bytes in {dt:.1f}s")
    print(f"    peak VRAM: {_vram_peak_gib():.1f} GiB")
    await client.aclose()
    print(">>> DONE — outputs in", OUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())

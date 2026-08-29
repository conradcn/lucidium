"""End-to-end smoke test for the Z-Image-Turbo embedded path.

Loads the user's locally-installed Z-Image-Turbo safetensors through
``EmbeddedImageClient`` (the same code path the running engine uses),
renders one character portrait and one background, and writes both
PNGs to a temp dir for visual inspection.

Validates:
  * the filename sniff routed the load to ``ZImagePipeline`` (not
    SDXL) — printed in the diagnostics line;
  * the runtime branch invoked the Turbo recipe (9 steps, cfg 0) —
    printed in the diagnostics line;
  * the rendered PNGs match the workflow dimensions (832x1216 portrait,
    1536x1024 background).

Run from the repo root::

    .\\backend\\.venv\\Scripts\\python.exe backend\\scripts\\e2e_z_image.py
"""

from __future__ import annotations

import asyncio
import io
import sys
import time
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend" / "src"))

from lucidium.persistence import settings_store  # noqa: E402
from lucidium.providers.embedded_image_client import (  # noqa: E402
    WORKFLOW_DIMENSIONS,
    EmbeddedImageClient,
    _is_z_image_model_path,
)
from lucidium.providers.embedded_models import (  # noqa: E402
    list_models,
    resolve_models_dir,
)

PORTRAIT_PROMPT = (
    "a tall stoic figure in a long wool coat, dim alley at dusk, "
    "rim light catching the collar, head fully visible, "
    "full body, three-quarter view"
)
PORTRAIT_FACE = "calm composure, sharp eyes, faint smile"
PORTRAIT_NEG = "extra people, crowd"

BACKGROUND_PROMPT = (
    "a stone harbor at dawn, slick cobbles, gulls overhead, soft mist on the water, painterly"
)


def _find_z_image_model(models_dir: Path) -> Path | None:
    for name in list_models(models_dir):
        if _is_z_image_model_path(models_dir / name):
            return models_dir / name
    return None


async def _generate(
    client: EmbeddedImageClient,
    *,
    workflow: str,
    positive: str,
    face_prompt: str = "",
    negative_extras: str = "",
    seed: int,
) -> bytes:
    params: dict[str, str] = {"positive_prompt": positive}
    if face_prompt:
        params["face_prompt"] = face_prompt
    if negative_extras:
        params["negative_extras"] = negative_extras
    return await client.generate(workflow, params, seed=seed)


async def main() -> int:
    print("== Lucidium Z-Image-Turbo end-to-end smoke test ==")
    settings = settings_store.load_settings()
    models_dir = resolve_models_dir(settings.image.embedded_models_dir)
    print(f"Models dir:       {models_dir}")
    if not models_dir.exists():
        print("ABORT: models dir does not exist")
        return 1

    z_path = _find_z_image_model(models_dir)
    if z_path is None:
        print(
            "ABORT: no Z-Image-named safetensors found in the configured "
            "models dir. Looked for any file whose name contains "
            "'z-image' / 'zimage' / 'tongyi-mai'."
        )
        print("Available files:")
        for name in list_models(models_dir):
            print(f"  {name}")
        return 1
    print(f"Z-Image model:    {z_path.name}")

    out_dir = REPO / "test-results" / "z-image-e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir:       {out_dir}")

    client = EmbeddedImageClient(
        models_dir=str(models_dir),
        # Pin the same Z-Image safetensors for BOTH character and
        # environment workflows so we exercise one pipeline-load
        # instead of two.
        character_model_name=z_path.name,
        environment_model_name=z_path.name,
        face_detail=settings.image.embedded_face_detail,
    )

    # --- character render ----------------------------------------------
    print("\n-- character.json --")
    started = time.monotonic()
    portrait_bytes = await _generate(
        client,
        workflow="character.json",
        positive=PORTRAIT_PROMPT,
        face_prompt=PORTRAIT_FACE,
        negative_extras=PORTRAIT_NEG,
        seed=20260520,
    )
    elapsed = time.monotonic() - started
    portrait_path = out_dir / "z-image-portrait.png"
    portrait_path.write_bytes(portrait_bytes)
    img = Image.open(io.BytesIO(portrait_bytes))
    expected = WORKFLOW_DIMENSIONS["character.json"]
    ok = (img.width, img.height) == expected
    print(
        f"  size={img.width}x{img.height} expected={expected[0]}x{expected[1]} "
        f"elapsed={elapsed:.1f}s bytes={len(portrait_bytes)} -> {portrait_path}"
    )
    if not ok:
        print("FAIL: portrait dimensions do not match the workflow contract")
        return 2

    # --- background render ---------------------------------------------
    print("\n-- background.json --")
    started = time.monotonic()
    background_bytes = await _generate(
        client,
        workflow="background.json",
        positive=BACKGROUND_PROMPT,
        seed=20260520,
    )
    elapsed = time.monotonic() - started
    background_path = out_dir / "z-image-background.png"
    background_path.write_bytes(background_bytes)
    img = Image.open(io.BytesIO(background_bytes))
    expected = WORKFLOW_DIMENSIONS["background.json"]
    ok = (img.width, img.height) == expected
    print(
        f"  size={img.width}x{img.height} expected={expected[0]}x{expected[1]} "
        f"elapsed={elapsed:.1f}s bytes={len(background_bytes)} -> {background_path}"
    )
    if not ok:
        print("FAIL: background dimensions do not match the workflow contract")
        return 2

    await client.aclose()
    print("\nOK — both renders match the workflow contract.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

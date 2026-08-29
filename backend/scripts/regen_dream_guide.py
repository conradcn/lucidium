"""Regenerate the bundled dream-guide PNG via ComfyUI + Dreamshaper XL.

Submits the project's existing character workflow (RMBG transparent
background, FaceDetailer, ControlNet pose) but swaps the checkpoint
to Dreamshaper XL and prompts for a nondescript ethereal spirit
rather than a specific character. Writes over
``backend/workflows/placeholders/dream_guide.png``.

Run with the project's backend venv after confirming ComfyUI is up at
http://127.0.0.1:8000 and that ``dreamshaperXL10_alpha2Xl10.safetensors``
is in the checkpoints directory.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[2]
WORKFLOW_TEMPLATE = REPO / "backend" / "workflows" / "character.json"
OUTPUT_PATH = REPO / "backend" / "workflows" / "placeholders" / "dream_guide.png"

COMFY_URL = "http://127.0.0.1:8000"
CHECKPOINT = "dreamshaperXL10_alpha2Xl10.safetensors"

POSITIVE_PROMPT = (
    "ethereal spirit guide, nondescript humanoid silhouette, soft "
    "luminescent figure, translucent flowing robes, no facial features, "
    "no hair detail, gentle inner glow, dreamlike, mystical, calm pose, "
    "full body, standing, centered, masterpiece, highly detailed, soft "
    "rim light, painterly, ethereal palette of pale gold and cool blue"
)
NEGATIVE_PROMPT = (
    "specific character, recognizable face, distinct hair, jewelry, "
    "weapons, armor, anime, cartoon, deformed, asymmetrical, "
    "extra limbs, watermark, text, logo, cropped, close-up, head shot, "
    "bust shot, portrait crop, cut off legs, cut off feet, out of frame, "
    "back view, side view, profile, facing away"
)
FACE_PROMPT = "soft glowing veil where a face would be, ethereal, no defined features"


def build_workflow() -> dict[str, Any]:
    template = json.loads(WORKFLOW_TEMPLATE.read_text(encoding="utf-8"))
    # Swap checkpoint to Dreamshaper XL.
    template["1"]["inputs"]["ckpt_name"] = CHECKPOINT
    # Apply our prompts directly (no PLACEHOLDER substitution since we
    # write the workflow ourselves).
    template["2"]["inputs"]["text"] = POSITIVE_PROMPT
    template["3"]["inputs"]["text"] = NEGATIVE_PROMPT
    template["12"]["inputs"]["wildcard"] = FACE_PROMPT
    # A reproducible seed for the bundled placeholder.
    template["20"]["inputs"]["noise_seed"] = 7341981
    template["12"]["inputs"]["seed"] = 7341981
    return template


async def submit_and_wait(workflow: dict[str, Any], timeout_s: float = 600.0) -> bytes:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        resp = await client.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
        print(f"[comfy] submitted prompt_id={prompt_id}", flush=True)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(2.0)
            history = await client.get(f"{COMFY_URL}/history/{prompt_id}")
            data = history.json()
            if prompt_id not in data:
                continue
            entry = data[prompt_id]
            outputs = entry.get("outputs", {})
            for _node_id, out in outputs.items():
                images = out.get("images") or []
                for image in images:
                    # PreviewImage emits ``type="temp"``; SaveImage
                    # emits ``type="output"``. Either is the rendered
                    # PNG we want — just refetch it via /view.
                    image_type = image.get("type", "output")
                    filename = image["filename"]
                    subfolder = image.get("subfolder", "")
                    print(
                        f"[comfy] fetching {filename} "
                        f"(type={image_type!r} subfolder={subfolder!r})",
                        flush=True,
                    )
                    img_resp = await client.get(
                        f"{COMFY_URL}/view",
                        params={
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": image_type,
                        },
                    )
                    img_resp.raise_for_status()
                    return img_resp.content
        raise TimeoutError(f"comfy did not return output within {timeout_s}s")


async def main() -> int:
    print(f"[regen-guide] writing to {OUTPUT_PATH}", flush=True)
    workflow = build_workflow()
    image_bytes = await submit_and_wait(workflow)
    if not image_bytes.startswith(b"\x89PNG"):
        print(
            f"[regen-guide] FATAL: response is not a PNG (got {image_bytes[:8]!r})",
            file=sys.stderr,
        )
        return 1
    OUTPUT_PATH.write_bytes(image_bytes)
    print(f"[regen-guide] wrote {len(image_bytes)} bytes to {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Render the main-menu carousel: background + matching dream guide
for several distinct (genre, visual_style) combinations.

Outputs go to ``frontend/public/main-menu/``: each pair becomes
``<slug>-bg.png`` (background, 1536x1024) and ``<slug>-guide.png``
(transparent dream guide, ~832x1216). A manifest at
``frontend/public/main-menu/manifest.json`` lists the slugs in display
order so the renderer can import them at build time.

Run with the project's backend venv after confirming ComfyUI is up at
http://127.0.0.1:8000 and that ``dreamshaperXL10_alpha2Xl10.safetensors``
is in the checkpoints directory. ~6 backgrounds (KSampler 25 steps)
plus ~6 character runs (FaceDetailer + RMBG) — budget ~10-15 min on
a 4090.

The MENU_COMBO data table + prompt builders live in
``lucidium.orchestration.menu_combos`` so the runtime preview pipeline
can re-render the same dream-guide character in the player-chosen
visual style during the New Game interview without duplicating the
prompt strings.
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
# Make ``lucidium.*`` importable when the script is run as a stand-alone.
sys.path.insert(0, str(REPO / "backend" / "src"))

from lucidium.orchestration.menu_combos import (  # noqa: E402  (path-prepend)
    COMBOS,
    MenuCombo,
    background_positive,
    guide_face,
    guide_negative_extras,
    guide_positive,
)

WORKFLOW_BG = REPO / "backend" / "workflows" / "background.json"
WORKFLOW_CHAR = REPO / "backend" / "workflows" / "character.json"
OUTPUT_DIR = REPO / "frontend" / "public" / "main-menu"
TS_MANIFEST_OUT = REPO / "frontend" / "src" / "app" / "menuManifest.generated.ts"

COMFY_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# ComfyUI submission
# ---------------------------------------------------------------------------


def build_background_workflow(combo: MenuCombo, seed: int) -> dict[str, Any]:
    template = json.loads(WORKFLOW_BG.read_text(encoding="utf-8"))
    template["2"]["inputs"]["text"] = background_positive(combo)
    # Strong NO-FIGURE / NO-FOREGROUND-CLUTTER negatives. The dream
    # guide is composited in front of these backgrounds at runtime;
    # any humans, animals, vehicles, or large foreground objects
    # the SD model adds will visibly clash with the figure overlay.
    template["3"]["inputs"]["text"] = (
        "low quality, blurry, watermark, text, "
        "people, characters, person, human, figure, silhouette, "
        "animals, camel, dog, cat, horse, "
        "vehicle, car, automobile, spacecraft in foreground, "
        "foreground subject, large foreground object, centered subject, "
        "object blocking center, cropped, out of frame"
    )
    template["5"]["inputs"]["seed"] = seed
    return template


def build_guide_workflow(combo: MenuCombo, seed: int) -> dict[str, Any]:
    template = json.loads(WORKFLOW_CHAR.read_text(encoding="utf-8"))
    # Use Dreamshaper XL like the original dream guide — works with
    # the pony workflow's nodes (CheckpointLoaderSimple is generic).
    template["1"]["inputs"]["ckpt_name"] = "dreamshaperXL10_alpha2Xl10.safetensors"
    template["2"]["inputs"]["text"] = guide_positive(combo)
    template["3"]["inputs"]["text"] = template["3"]["inputs"]["text"].replace(
        "PLACEHOLDER_NEGATIVE_EXTRAS", guide_negative_extras()
    )
    template["12"]["inputs"]["wildcard"] = guide_face(combo)
    template["20"]["inputs"]["noise_seed"] = seed
    template["12"]["inputs"]["seed"] = seed
    return template


async def submit_and_fetch(
    client: httpx.AsyncClient,
    workflow: dict[str, Any],
    *,
    timeout_s: float = 600.0,
) -> bytes:
    resp = await client.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
    resp.raise_for_status()
    prompt_id = resp.json()["prompt_id"]

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await asyncio.sleep(2.0)
        history = await client.get(f"{COMFY_URL}/history/{prompt_id}")
        data = history.json()
        if prompt_id not in data:
            continue
        outputs = data[prompt_id].get("outputs", {})
        for _node, out in outputs.items():
            for image in out.get("images", []) or []:
                image_type = image.get("type", "output")
                view = await client.get(
                    f"{COMFY_URL}/view",
                    params={
                        "filename": image["filename"],
                        "subfolder": image.get("subfolder", ""),
                        "type": image_type,
                    },
                )
                view.raise_for_status()
                return view.content
    raise TimeoutError(f"comfy did not produce output within {timeout_s}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        for index, combo in enumerate(COMBOS):
            seed = 1_000_000 + index * 17  # deterministic but distinct
            bg_path = OUTPUT_DIR / f"{combo.slug}-bg.png"
            guide_path = OUTPUT_DIR / f"{combo.slug}-guide.png"

            # Skip whichever PNGs already exist — the script is
            # idempotent and tolerant of partial runs. Re-running adds
            # only the new combos.
            if not bg_path.exists():
                print(f"[main-menu] [{combo.slug}] background…", flush=True)
                bg = build_background_workflow(combo, seed)
                bg_bytes = await submit_and_fetch(client, bg)
                bg_path.write_bytes(bg_bytes)
                print(
                    f"[main-menu] [{combo.slug}] background written ({len(bg_bytes)} bytes)",
                    flush=True,
                )
            else:
                print(f"[main-menu] [{combo.slug}] background cached", flush=True)

            if not guide_path.exists():
                print(f"[main-menu] [{combo.slug}] dream guide…", flush=True)
                guide = build_guide_workflow(combo, seed + 1)
                guide_bytes = await submit_and_fetch(client, guide)
                guide_path.write_bytes(guide_bytes)
                print(
                    f"[main-menu] [{combo.slug}] dream guide written ({len(guide_bytes)} bytes)",
                    flush=True,
                )
            else:
                print(f"[main-menu] [{combo.slug}] dream guide cached", flush=True)

            manifest.append(
                {
                    "slug": combo.slug,
                    "label": combo.location_label,
                    "genre": combo.genre,
                    "visual_style": combo.visual_style,
                    "background": f"{combo.slug}-bg.png",
                    "guide": f"{combo.slug}-guide.png",
                    "title_color": combo.title_color,
                    "title_font": combo.title_font,
                    "title_letter_spacing": combo.title_letter_spacing,
                    "title_weight": combo.title_weight,
                    "title_shade_color": combo.title_shade_color,
                }
            )

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps({"pairs": manifest}, indent=2), encoding="utf-8")
    # Also emit a typed TS module so the renderer can import the
    # manifest at build time without a runtime fetch.
    TS_MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    ts_lines = [
        "// AUTO-GENERATED by backend/scripts/regen_main_menu.py — do not edit.",
        "",
        "export interface MenuPair {",
        "  slug: string;",
        "  label: string;",
        "  genre: string;",
        "  visual_style: string;",
        "  background: string;",
        "  guide: string;",
        "  title_color: string;",
        "  title_font: string;",
        "  title_letter_spacing: string;",
        "  title_weight: string;",
        "  title_shade_color: string;",
        "}",
        "",
        # Relative path (no leading slash). The packaged Electron
        # app loads index.html via file:// — a leading-slash URL
        # would resolve to the FILESYSTEM root, not the app root,
        # and the menu carousel's PNGs would silently 404. Vite's
        # ``base: "./"`` config emits relative URLs in index.html
        # for the same reason; this string follows that convention.
        'export const MENU_BASE = "main-menu/";',
        "",
        "export const MENU_PAIRS: MenuPair[] = " + json.dumps(manifest, indent=2) + ";",
        "",
    ]
    TS_MANIFEST_OUT.write_text("\n".join(ts_lines), encoding="utf-8")
    print(
        f"[main-menu] wrote {len(manifest)} pairs + {manifest_path} + {TS_MANIFEST_OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

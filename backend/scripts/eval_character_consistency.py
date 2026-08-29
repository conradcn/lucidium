"""Render the same character under N varied beat states, composite the
results into a single image, and save it for human grading.

Used by the /autoresearch loop targeting character render quality:
the loop iterates on prompt builders / workflow params, runs this
script after each change, and the agent reads the composite to
score consistency on four axes (face identity, framing, pose
sensibility, outfit/state coherence).

Usage:
    python eval_character_consistency.py [--label <iter-name>]

Outputs:
    backend/eval/character/<timestamp>-<label>.png   (2x2 composite)
    backend/eval/character/<timestamp>-<label>/      (per-variant PNGs)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend" / "src"))

from lucidium.domain.character import Character  # noqa: E402
from lucidium.domain.world import WorldState  # noqa: E402
from lucidium.orchestration.prompts import image_prompts  # noqa: E402

WORKFLOW_PATH = REPO / "backend" / "workflows" / "character.json"
OUTPUT_DIR = REPO / "backend" / "eval" / "character"
COMFY_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Fixed world. The visual style is deliberately a tight noir prompt so
# anatomy errors are easy to spot (deep shadows hide some flaws but
# silhouette + framing remain readable).
WORLD = WorldState(
    game_name="The Salt Lantern",
    setting="A stone harbor at dawn",
    genre="Mystery",
    visual_style=(
        "film noir black and white, deep shadows, cigarette smoke, 35mm grain, high contrast"
    ),
    overall_plot_direction="Find the missing keeper.",
)

# Same identity-anchoring fields across every variant — only outfit /
# pose / expression change. Seed is fixed so face-detail noise is
# deterministic; if the loop's prompt change improves identity
# preservation, the four faces should look more similar to each other.
BASE_CHARACTER_FIELDS: dict[str, Any] = {
    "is_player": True,
    "name": "Iris Vale",
    "description": "a wry archivist nursing an old grudge",
    "gender": "female",
    "age": 28,
    "ethnicity": "Welsh",
    "skin": "pale",
    "hair_color": "auburn",
    "hairstyle": "single loose braid",
    "eye_color": "grey-green",
    "build": "slight",
    "bust": "moderate",
    "seed": 1_234_567_891,
}


@dataclass(frozen=True)
class Variant:
    label: str
    outfit: str
    pose: str
    expression: str
    seed_offset: int = 0


# Four state-varying variants for pose/framing/outfit testing,
# plus two extra renders of the BASELINE outfit/pose/expression at
# different stochastic seeds — those last two test render-to-render
# face-identity consistency (same prompt, different noise → should
# still look like the same person).
VARIANTS: list[Variant] = [
    # Variants follow the RENDER-FRIENDLY VALUES rule from
    # text_gen.py: terse comma-list outfit (no motion verbs —
    # those caused garment-drop in iter4), verbose semantic-anchor
    # pose (short pose tags collapse to standing-portrait, narrow
    # tags lose grip too).
    Variant(
        label="alert-coat",
        outfit="charcoal wool coat, grey sweater, black trousers, boots",
        pose="standing alert, weight on one foot, hands relaxed at sides",
        expression="watchful, lips slightly parted",
    ),
    Variant(
        label="kneeling-glove",
        outfit="charcoal coat, leather gloves, dark scarf, boots",
        pose="kneeling on one knee, peering down at the cobblestones, both hands braced on the ground",
        expression="focused, brow drawn together",
    ),
    Variant(
        label="seated-reading",
        outfit="charcoal coat, oxblood cardigan, dark trousers, boots",
        pose="seated on a low stone wall, leaning forward, holding a small leather notebook open",
        expression="engrossed, faint smile",
    ),
    Variant(
        label="running-alarmed",
        outfit="charcoal coat, grey sweater, black trousers, boots",
        pose="running full-tilt toward the camera, one arm pumping, the other reaching forward",
        expression="alarmed, mouth set, eyes wide",
    ),
    # Render-to-render consistency probes: identical prompt as
    # alert-coat, only the stochastic seed shifts. The face should
    # look like the same person — if it drifts, single-seed identity
    # preservation is fragile.
    Variant(
        label="alert-coat-noise-2",
        outfit="charcoal wool coat, grey sweater, black trousers, boots",
        pose="standing, hands relaxed at sides",
        expression="watchful",
        seed_offset=1,
    ),
    Variant(
        label="alert-coat-noise-3",
        outfit="charcoal wool coat, grey sweater, black trousers, boots",
        pose="standing, hands relaxed at sides",
        expression="watchful",
        seed_offset=2,
    ),
]


# ---------------------------------------------------------------------------
# ComfyUI submission
# ---------------------------------------------------------------------------


def build_workflow(character: Character) -> dict[str, Any]:
    """Build a ComfyUI workflow with the production prompt builders.

    Mirrors what ``ComfyUiImageClient._instantiate_workflow`` does at
    runtime: PLACEHOLDER_<KEY> string substitution before parse, then
    seed injection.
    """
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    positive = image_prompts.portrait_prompt(world=WORLD, character=character)
    face = image_prompts.portrait_face_prompt(character=character)
    neg_extras = image_prompts.portrait_negative_extras(character=character)
    raw = raw.replace("PLACEHOLDER_POSITIVE_PROMPT", json.dumps(positive)[1:-1])
    raw = raw.replace("PLACEHOLDER_FACE_PROMPT", json.dumps(face)[1:-1])
    raw = raw.replace("PLACEHOLDER_NEGATIVE_EXTRAS", json.dumps(neg_extras)[1:-1])
    graph = json.loads(raw)
    for node in graph.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        if "seed" in inputs:
            inputs["seed"] = character.seed
        if "noise_seed" in inputs:
            inputs["noise_seed"] = character.seed
    return graph


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
        history = (await client.get(f"{COMFY_URL}/history/{prompt_id}")).json()
        if prompt_id not in history:
            continue
        outputs = history[prompt_id].get("outputs", {})
        # Prefer the latest node's output (FaceDetailer + RMBG result).
        ranked = sorted(
            ((k, v) for k, v in outputs.items()),
            key=lambda kv: int(kv[0]) if kv[0].isdigit() else 10**9,
            reverse=True,
        )
        for _node_id, out in ranked:
            for image in out.get("images", []) or []:
                view = await client.get(
                    f"{COMFY_URL}/view",
                    params={
                        "filename": image["filename"],
                        "subfolder": image.get("subfolder", ""),
                        "type": image.get("type", "output"),
                    },
                )
                view.raise_for_status()
                return view.content
    raise TimeoutError(f"comfy timed out for prompt {prompt_id}")


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def composite_grid(per_variant: list[tuple[Variant, bytes]], out_path: Path) -> None:
    """Stitch the variant PNGs into a labeled 2x3 grid (6 tiles).
    Top row: state-varying variants (pose/outfit/expression test).
    Bottom row: render-to-render consistency probes (same prompt,
    different stochastic seed — face identity check)."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(io.BytesIO(b)).convert("RGBA") for _v, b in per_variant]
    cols = 3
    rows = (len(per_variant) + cols - 1) // cols
    tile_w, tile_h = 320, 468  # 832x1216 scaled down
    label_h = 28
    cell_h = tile_h + label_h
    canvas = Image.new("RGBA", (tile_w * cols, cell_h * rows), (24, 24, 28, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for idx, (variant, _bytes) in enumerate(per_variant):
        col, row = idx % cols, idx // cols
        x, y = col * tile_w, row * cell_h
        draw.rectangle([x, y, x + tile_w, y + label_h], fill=(40, 42, 50, 255))
        draw.text((x + 8, y + 5), variant.label, fill=(220, 220, 220, 255), font=font)
        thumb = images[idx].resize((tile_w, tile_h), Image.LANCZOS)
        canvas.paste(thumb, (x, y + label_h), thumb)
    canvas.convert("RGB").save(out_path, "PNG", optimize=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    label = "iter"
    if "--label" in sys.argv:
        idx = sys.argv.index("--label")
        if idx + 1 < len(sys.argv):
            label = sys.argv[idx + 1]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    per_iter_dir = OUTPUT_DIR / f"{stamp}-{label}"
    per_iter_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] rendering {len(VARIANTS)} variants -> {per_iter_dir}", flush=True)
    started = time.monotonic()

    per_variant: list[tuple[Variant, bytes]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        for v in VARIANTS:
            fields = dict(BASE_CHARACTER_FIELDS)
            fields["seed"] = fields["seed"] + v.seed_offset
            character = Character(
                outfit=v.outfit,
                pose=v.pose,
                expression=v.expression,
                **fields,
            )
            t0 = time.monotonic()
            print(f"[eval] [{v.label}] submitting…", flush=True)
            workflow = build_workflow(character)
            png = await submit_and_fetch(client, workflow)
            (per_iter_dir / f"{v.label}.png").write_bytes(png)
            per_variant.append((v, png))
            print(
                f"[eval] [{v.label}] {len(png)} bytes in {time.monotonic() - t0:.1f}s",
                flush=True,
            )

    composite_path = OUTPUT_DIR / f"{stamp}-{label}.png"
    composite_grid(per_variant, composite_path)
    elapsed = time.monotonic() - started
    print(f"[eval] composite written: {composite_path} (total {elapsed:.1f}s)", flush=True)
    print(str(composite_path))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

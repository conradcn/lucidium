"""ComfyUI FaceDetailer live test.

Renders the SAME body+seed twice with the SAME face_prompt: once
through the production ``character.json`` workflow (FaceDetailer
node enabled, wildcard fed by ``face_prompt``) and once through a
runtime-modified copy of the workflow with FaceDetailer bypassed
(image flows around node 12 — VAEDecode → RMBG → Preview directly).

If the FaceDetailer is doing real work (running its bbox detector,
inpainting the face region with the wildcard prompt active), the
two outputs must diverge in the face region. If the outputs are
identical or near-identical, FaceDetailer is being skipped — the
"face detailer doesn't seem to be triggering" report.

Excluded from the default run via the ``live`` marker. Requires a
running ComfyUI server reachable at the user's configured
``ImageSettings.base_url`` AND that server's Impact-Pack node set
loaded (FaceDetailer is custom, not core).

Run with::

    pytest tests/integration/test_live_face_detailer.py -m live -v
"""

from __future__ import annotations

import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from lucidium.persistence import settings_store
from lucidium.providers.image_client import ComfyUiImageClient

# Where the workflow JSONs live. Mirrors Session._default_workflow_root.
_WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflows"


# Note: ``live`` mark is applied per-test (not file-level) so the
# pure-logic ``test_workflow_bypass_helper_rewires_chain`` below
# can run in the default suite without requiring ComfyUI.


def _comfy_reachable(url: str) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            return client.get(f"{url}/system_stats").status_code == 200
    except Exception:
        return False


def _settings_or_skip():
    settings = settings_store.load_settings()
    base = (settings.image.base_url or "").strip()
    if not base:
        pytest.skip("settings.image.base_url is unset")
    if not _comfy_reachable(base):
        pytest.skip(f"ComfyUI not reachable at {base}")
    return settings


def _bypass_face_detailer(graph: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of the workflow with the FaceDetailer node
    (id ``12``) cut out of the chain. Downstream nodes that read from
    ``["12", 0]`` (image) get rewired to read from the FaceDetailer's
    upstream image source instead — node 6's VAEDecode output. The
    FaceDetailer node itself stays in the graph but unreferenced
    (ComfyUI executes only nodes that downstream sinks pull from)."""
    out = deepcopy(graph)
    face_node = out.get("12")
    if face_node is None or face_node.get("class_type") != "FaceDetailer":
        raise RuntimeError(
            "workflow doesn't have a FaceDetailer at node id 12 — "
            "this test pins the bypass on that exact node id; "
            "update both if you renumber the workflow"
        )
    upstream_image = face_node["inputs"]["image"]  # e.g. ["6", 0]
    # Walk every node and rewire any input that referenced ["12", 0]
    # to the FaceDetailer's upstream source. Skip the FaceDetailer
    # node itself.
    for node_id, node in out.items():
        if node_id == "12":
            continue
        inputs = node.get("inputs", {})
        for slot_name, slot_value in inputs.items():
            if (
                isinstance(slot_value, list)
                and len(slot_value) == 2
                and slot_value[0] == "12"
                and slot_value[1] == 0
            ):
                inputs[slot_name] = list(upstream_image)
    return out


def _face_region(image: Image.Image) -> bytes:
    width, height = image.size
    box = (0, 0, width, height // 3)
    cropped = image.crop(box).convert("RGB")
    return cropped.tobytes()


def _diff_ratio(a: bytes, b: bytes) -> float:
    if len(a) != len(b):
        return 1.0
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b, strict=True) if x != y) / len(a)


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_face_detailer_modifies_output(tmp_path: Path) -> None:
    """End-to-end ComfyUI test: render with FaceDetailer enabled vs
    bypassed; the face region must differ.

    The bypassed workflow is written to a temp directory so we can
    point ``ComfyUiImageClient`` at it via ``workflow_root``. The
    base ``character.json`` is read directly from the production
    workflow dir for the enabled-side render. Same seed, same body
    prompt, same face_prompt across both — the only difference is
    whether node 12 is in the execution chain.
    """
    settings = _settings_or_skip()

    base_workflow_path = _WORKFLOW_DIR / "character.json"
    if not base_workflow_path.exists():
        pytest.skip(f"character.json not present at {base_workflow_path}")
    base_graph = json.loads(base_workflow_path.read_text(encoding="utf-8"))

    # Verify the FaceDetailer node is present and the test's bypass
    # transform applies cleanly. Skip cleanly if the workflow has
    # been restructured — better than a misleading failure.
    if "12" not in base_graph or base_graph["12"].get("class_type") != "FaceDetailer":
        pytest.skip(
            "character.json does not have FaceDetailer at node 12; "
            "test needs updating to find the new node id"
        )

    bypassed = _bypass_face_detailer(base_graph)

    # Materialise both workflow trees on disk. ComfyUiImageClient
    # reads from ``workflow_root / <workflow filename>``.
    enabled_root = tmp_path / "enabled"
    bypass_root = tmp_path / "bypass"
    enabled_root.mkdir()
    bypass_root.mkdir()
    (enabled_root / "character.json").write_text(json.dumps(base_graph), encoding="utf-8")
    # The other workflow files used by the rest of the engine
    # (background.json etc.) aren't needed here — the test only
    # invokes character.json — but copy them through so the client
    # could fall back without surprises if ComfyUI complains.
    for sibling in _WORKFLOW_DIR.glob("*.json"):
        if sibling.name != "character.json":
            (enabled_root / sibling.name).write_text(
                sibling.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (bypass_root / sibling.name).write_text(
                sibling.read_text(encoding="utf-8"), encoding="utf-8"
            )
    (bypass_root / "character.json").write_text(json.dumps(bypassed), encoding="utf-8")

    enabled_client = ComfyUiImageClient(settings.image, workflow_root=enabled_root)
    bypass_client = ComfyUiImageClient(settings.image, workflow_root=bypass_root)

    body_prompt = (
        "a young scholar in a long wool coat, full body, standing "
        "centered, head and feet visible, neutral pose, soft window light"
    )
    face_prompt = "fierce scowl, brows knit tight, lips pressed thin, jaw set"
    seed = 31337

    enabled_png = await enabled_client.generate(
        "character.json",
        {
            "positive_prompt": body_prompt,
            "face_prompt": face_prompt,
            "negative_extras": "",
        },
        seed=seed,
    )
    bypass_png = await bypass_client.generate(
        "character.json",
        {
            "positive_prompt": body_prompt,
            "face_prompt": face_prompt,
            "negative_extras": "",
        },
        seed=seed,
    )

    assert enabled_png and bypass_png, (
        "both renders must produce non-empty bytes; one or both "
        "ComfyUI workflows failed to return an image"
    )

    enabled_image = Image.open(io.BytesIO(enabled_png))
    bypass_image = Image.open(io.BytesIO(bypass_png))
    assert enabled_image.size == bypass_image.size, (
        f"enabled vs bypass renders disagreed on size — "
        f"{enabled_image.size} vs {bypass_image.size} — workflow "
        f"bypass altered the latent dimensions, which it shouldn't"
    )

    enabled_face = _face_region(enabled_image)
    bypass_face = _face_region(bypass_image)
    diff = _diff_ratio(enabled_face, bypass_face)

    # 3% is the floor — empirically a working FaceDetailer pass
    # rewrites enough of the face region that 10-30% of pixels
    # differ from the bypassed version. Setting the threshold this
    # low gives generous headroom; if FaceDetailer is genuinely a
    # no-op (skipping inpaint because the bbox detector found
    # nothing, or the wildcard isn't reaching CLIPTextEncode), the
    # diff sits at the same noise floor as two identical renders
    # (~0%). 3% catches the no-op case while surviving small
    # ComfyUI-side stochasticity.
    assert diff > 0.03, (
        f"face region pixel diff between FaceDetailer-enabled and "
        f"FaceDetailer-bypassed renders is only {diff:.1%}. "
        f"FaceDetailer is not modifying the face — likely the bbox "
        f"detector isn't finding the face (raise threshold to check), "
        f"the wildcard isn't being passed through, or the node is "
        f"erroring silently. Check ComfyUI server logs for the run."
    )


def test_workflow_bypass_helper_rewires_chain() -> None:
    """Pure unit-style sanity check on the bypass helper: pre-fix it
    skipped the rewiring step and just removed node 12, leaving
    downstream sinks pointed at a missing node. This guards the
    helper itself so the live test above can't false-pass on a
    bypass that broke ComfyUI submission entirely (which would
    return a clearly-different image — wrong reason)."""
    fake = {
        "6": {"class_type": "VAEDecode", "inputs": {}},
        "12": {
            "class_type": "FaceDetailer",
            "inputs": {"image": ["6", 0], "wildcard": "expr"},
        },
        "14": {
            "class_type": "RMBG",
            "inputs": {"image": ["12", 0]},
        },
        "7": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["14", 0]},
        },
    }
    out = _bypass_face_detailer(fake)
    assert out["14"]["inputs"]["image"] == ["6", 0], (
        "downstream RMBG must read directly from VAEDecode after "
        "FaceDetailer is bypassed; helper failed to rewire"
    )
    # FaceDetailer node stays in the dict but is now orphaned —
    # ComfyUI only executes nodes pulled by sinks.
    assert "12" in out

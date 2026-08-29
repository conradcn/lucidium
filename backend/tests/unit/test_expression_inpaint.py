"""Expression-only portrait refresh: face-targeted img2img inpaint.

When a beat changes ONLY a character's expression, the engine refreshes
just the face on the existing portrait (img2img at 0.5 denoise, OpenCV
face mask grown 30 px with a soft fade) instead of re-rendering the whole
figure. Covers:

  * the soft-faded dilated mask geometry,
  * ``_run_expression_inpaint``'s safe fallbacks (no face / no pipeline),
  * ``EmbeddedImageClient.regenerate_expression`` routing + declines,
  * the orchestration detection (``_portrait_identity_hash`` /
    ``_find_expression_base`` / ``_build_portrait_render``).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lucidium.providers.embedded_image_client import (
    EmbeddedImageClient,
    _expression_inpaint_mask,
    _run_expression_inpaint,
)

# --------------------------------------------------------------------------
# Mask geometry: dilated rectangle + Gaussian soft fade
# --------------------------------------------------------------------------


def test_expression_mask_solid_core_soft_edge_black_outside():
    mask = _expression_inpaint_mask(
        crop_size_h=200,
        crop_size_w=200,
        face_top=60,
        face_left=60,
        face_height=80,
        face_width=80,
        boundary=30,
    )
    px = mask.load()
    # Core over the face is fully opaque.
    assert px[100, 100] >= 250
    # Far corner, well outside the grown box, is black.
    assert px[2, 2] <= 8
    # The grown rect edge (face_top - boundary == 30) sits inside a soft
    # fade — a partial value, not a hard 0/255 step.
    edge = px[100, 30]
    assert 10 < edge < 245, f"expected a soft-faded edge, got {edge}"
    # There IS a gradient (more than just black + white present).
    distinct = {mask.getpixel((100, y)) for y in range(0, 200)}
    assert len(distinct) > 3


def test_expression_mask_wider_boundary_covers_more():
    """A larger boundary radius grows the opaque region outward."""
    narrow = _expression_inpaint_mask(
        crop_size_h=200,
        crop_size_w=200,
        face_top=80,
        face_left=80,
        face_height=40,
        face_width=40,
        boundary=10,
    )
    wide = _expression_inpaint_mask(
        crop_size_h=200,
        crop_size_w=200,
        face_top=80,
        face_left=80,
        face_height=40,
        face_width=40,
        boundary=40,
    )
    # Total opacity, summed over every pixel. ``tobytes()`` rather than
    # ``getdata()``: the mask is a single-channel ``L`` image so its raw
    # buffer is one byte per pixel, and ``getdata()`` is deprecated from
    # Pillow 12 (removed in 14) in favour of a ``get_flattened_data``
    # that does not exist on Pillow 11. ``tobytes()`` works on both.
    assert sum(wide.tobytes()) > sum(narrow.tobytes())


# --------------------------------------------------------------------------
# _run_expression_inpaint — safe fallbacks
# --------------------------------------------------------------------------


def _png_bytes(w: int = 64, h: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), (10, 20, 30, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_run_expression_inpaint_no_face_returns_none(monkeypatch):
    """No detectable face -> ``None`` so the caller re-renders fully
    rather than inpainting a guessed region of a good portrait."""
    monkeypatch.setattr(
        "lucidium.providers.embedded_image_client._detect_face_bboxes",
        lambda *a, **k: [],
    )
    out = _run_expression_inpaint(
        object(),
        _png_bytes(),
        expression_prompt="smiling warmly",
        negative="",
        seed=1,
    )
    assert out is None


def test_run_expression_inpaint_no_inpaint_pipeline_returns_none(monkeypatch):
    monkeypatch.setattr(
        "lucidium.providers.embedded_image_client._detect_face_bboxes",
        lambda *a, **k: [(20, 16, 44, 44)],
    )
    monkeypatch.setattr(
        "lucidium.providers.embedded_image_client._get_or_build_inpaint_pipeline",
        lambda _p: None,
    )
    out = _run_expression_inpaint(
        object(),
        _png_bytes(),
        expression_prompt="smiling warmly",
        negative="",
        seed=1,
    )
    assert out is None


def test_run_expression_inpaint_targets_largest_face(monkeypatch):
    """The single portrait subject = the largest detected face, and it's
    inpainted with the 30 px dilated mask (boundary_px forwarded)."""
    monkeypatch.setattr(
        "lucidium.providers.embedded_image_client._detect_face_bboxes",
        lambda *a, **k: [(0, 0, 10, 10), (20, 16, 60, 60)],
    )
    monkeypatch.setattr(
        "lucidium.providers.embedded_image_client._get_or_build_inpaint_pipeline",
        lambda _p: object(),
    )
    seen: dict[str, Any] = {}

    def fake_one_face(*, face_bbox, boundary_px, strength, base_image, **_kw):
        seen["face_bbox"] = face_bbox
        seen["boundary_px"] = boundary_px
        seen["strength"] = strength
        return base_image

    monkeypatch.setattr("lucidium.providers.embedded_image_client._inpaint_one_face", fake_one_face)
    out = _run_expression_inpaint(
        object(),
        _png_bytes(),
        expression_prompt="grinning",
        negative="",
        seed=7,
    )
    assert out is not None
    assert seen["face_bbox"] == (20, 16, 60, 60)  # the larger box
    assert seen["boundary_px"] == 30
    assert seen["strength"] == 0.5


# --------------------------------------------------------------------------
# EmbeddedImageClient.regenerate_expression — routing + declines
# --------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self, name: str = "test-stub") -> None:
        self.config = type("Cfg", (), {"_name_or_path": name})()


@pytest.fixture
def stocked_models_dir(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    (models / "test-checkpoint.safetensors").write_bytes(b"fake")
    return models


@pytest.mark.asyncio
async def test_regenerate_expression_human_calls_runner(stocked_models_dir):
    calls: list[dict[str, Any]] = []

    def fake_runner(pipeline, base_png, *, expression_prompt, negative, seed):
        calls.append({"expression_prompt": expression_prompt, "negative": negative, "seed": seed})
        return b"NEW-EXPRESSION-PNG"

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        pipeline_factory=lambda _path, _device: _FakePipeline(),
        bg_remover=lambda b: b,
        expression_inpaint_runner=fake_runner,
    )
    out = await client.regenerate_expression(
        "character.json",
        b"BASE-PNG",
        {"face_prompt": "soft smile", "negative_extras": "", "subject_kind": "human"},
        seed=42,
    )
    assert out == b"NEW-EXPRESSION-PNG"
    assert len(calls) == 1
    assert calls[0]["expression_prompt"] == "soft smile"
    # Seed is derived (Knuth-multiplied) so it diverges from the body seed.
    assert calls[0]["seed"] != 42


@pytest.mark.asyncio
async def test_regenerate_expression_nonhuman_declines(stocked_models_dir):
    """Non-human subjects have no Haar-detectable face -> decline (None)
    so the caller does a full render."""
    called = False

    def fake_runner(*a, **k):
        nonlocal called
        called = True
        return b"x"

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        pipeline_factory=lambda _path, _device: _FakePipeline(),
        bg_remover=lambda b: b,
        expression_inpaint_runner=fake_runner,
    )
    out = await client.regenerate_expression(
        "character.json",
        b"BASE-PNG",
        {"face_prompt": "menacing", "subject_kind": "nonhuman"},
        seed=1,
    )
    assert out is None
    assert called is False


@pytest.mark.asyncio
async def test_regenerate_expression_empty_face_prompt_declines(stocked_models_dir):
    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        pipeline_factory=lambda _path, _device: _FakePipeline(),
        bg_remover=lambda b: b,
        expression_inpaint_runner=lambda *a, **k: b"x",
    )
    out = await client.regenerate_expression(
        "character.json",
        b"BASE-PNG",
        {"face_prompt": "", "subject_kind": "human"},
        seed=1,
    )
    assert out is None


@pytest.mark.asyncio
async def test_regenerate_expression_z_image_declines(stocked_models_dir):
    """Z-Image has no SDXL inpaint path -> decline (None)."""
    called = False

    def fake_runner(*a, **k):
        nonlocal called
        called = True
        return b"x"

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        pipeline_factory=lambda _path, _device: _FakePipeline(name="Z-Image-Turbo"),
        bg_remover=lambda b: b,
        expression_inpaint_runner=fake_runner,
    )
    out = await client.regenerate_expression(
        "character.json",
        b"BASE-PNG",
        {"face_prompt": "smiling", "subject_kind": "human"},
        seed=1,
    )
    assert out is None
    assert called is False


# --------------------------------------------------------------------------
# Orchestration: identity hash + base lookup + render routing
# --------------------------------------------------------------------------


from lucidium.domain.character import Character, CharacterImage, CharacterKind  # noqa: E402
from lucidium.domain.world import WorldState  # noqa: E402
from lucidium.orchestration import assets  # noqa: E402


def _make_character(**overrides: object) -> Character:
    base: dict[str, object] = {
        "id": "char-1",
        "is_player": False,
        "name": "Test",
        "description": "A test character.",
        "gender": "female",
        "age": 30,
        "ethnicity": "white",
        "skin": "pale",
        "hair_color": "brown",
        "hairstyle": "long",
        "eye_color": "green",
        "build": "lean",
        "bust": "B",
        "outfit": "charcoal trench coat",
        "pose": "standing tall",
        "expression": "neutral",
        "effects": "",
        "seed": 12345,
        "kind": CharacterKind.human,
    }
    base.update(overrides)
    return Character(**base)


def _make_world() -> WorldState:
    return WorldState(
        game_name="Test",
        setting="A test setting",
        genre="Mystery",
        visual_style="ink-noir",
    )


def test_identity_hash_ignores_expression_only():
    world = _make_world()
    a = _make_character(expression="neutral")
    b = _make_character(expression="beaming with joy")
    # Identity (expression-excluded) is identical...
    assert assets._portrait_identity_hash(world, a) == assets._portrait_identity_hash(world, b)
    # ...but the full prompt hash differs (expression DID change).
    assert assets._portrait_prompt_hash(world, a) != assets._portrait_prompt_hash(world, b)


def test_identity_hash_changes_with_non_expression_attrs():
    world = _make_world()
    a = _make_character(outfit="charcoal trench coat")
    b = _make_character(outfit="red ballgown")
    assert assets._portrait_identity_hash(world, a) != assets._portrait_identity_hash(world, b)


def test_find_expression_base_matches_identity_on_disk(tmp_path):
    img_file = tmp_path / "portrait.png"
    img_file.write_bytes(_png_bytes())
    character = _make_character(
        images=[
            CharacterImage(
                path=str(img_file),
                prompt_hash="old-full-hash",
                identity_hash="IDENT-1",
                attributes_snapshot={"expression": "neutral"},
            )
        ]
    )
    assert assets._find_expression_base(character, "IDENT-1") is not None
    # Different identity -> no base (a non-expression change happened).
    assert assets._find_expression_base(character, "IDENT-2") is None
    # Empty identity -> never matches.
    assert assets._find_expression_base(character, "") is None


def test_find_expression_base_skips_missing_file(tmp_path):
    character = _make_character(
        images=[
            CharacterImage(
                path=str(tmp_path / "gone.png"),
                prompt_hash="h",
                identity_hash="IDENT-1",
                attributes_snapshot={},
            )
        ]
    )
    assert assets._find_expression_base(character, "IDENT-1") is None


class _FakeImageClientNoInpaint:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, workflow, params, *, seed):
        self.generate_calls += 1
        return b"FULL-RENDER"


class _FakeImageClientWithInpaint(_FakeImageClientNoInpaint):
    def __init__(self, result: bytes | None) -> None:
        super().__init__()
        self.regen_calls = 0
        self._result = result

    async def regenerate_expression(self, workflow, base_png, params, *, seed):
        self.regen_calls += 1
        return self._result


class _Session:
    def __init__(self) -> None:
        self.settings = type(
            "S", (), {"image": type("I", (), {"portrait_workflow": "character.json"})()}
        )()


@pytest.mark.asyncio
async def test_build_render_uses_inpaint_when_base_exists(tmp_path):
    img_file = tmp_path / "portrait.png"
    img_file.write_bytes(_png_bytes())
    character = _make_character(
        expression="beaming",
        images=[
            CharacterImage(
                path=str(img_file),
                prompt_hash="old",
                identity_hash="IDENT-1",
                attributes_snapshot={"expression": "neutral"},
            )
        ],
    )
    client = _FakeImageClientWithInpaint(result=b"INPAINTED")
    render = assets._build_portrait_render(
        session=_Session(),
        image_client=client,
        character=character,
        positive="p",
        face="beaming face",
        negative_extras="",
        identity_hash="IDENT-1",
    )
    out = await render()
    assert out == b"INPAINTED"
    assert client.regen_calls == 1
    assert client.generate_calls == 0


@pytest.mark.asyncio
async def test_build_render_falls_back_to_full_when_inpaint_declines(tmp_path):
    img_file = tmp_path / "portrait.png"
    img_file.write_bytes(_png_bytes())
    character = _make_character(
        images=[
            CharacterImage(
                path=str(img_file),
                prompt_hash="old",
                identity_hash="IDENT-1",
                attributes_snapshot={},
            )
        ],
    )
    client = _FakeImageClientWithInpaint(result=None)  # backend declines
    render = assets._build_portrait_render(
        session=_Session(),
        image_client=client,
        character=character,
        positive="p",
        face="f",
        negative_extras="",
        identity_hash="IDENT-1",
    )
    out = await render()
    assert out == b"FULL-RENDER"
    assert client.regen_calls == 1
    assert client.generate_calls == 1


@pytest.mark.asyncio
async def test_build_render_full_when_backend_lacks_inpaint(tmp_path):
    """A backend without ``regenerate_expression`` (ComfyUI / fixtures)
    always full-renders, even if an identity match exists."""
    img_file = tmp_path / "portrait.png"
    img_file.write_bytes(_png_bytes())
    character = _make_character(
        images=[
            CharacterImage(
                path=str(img_file),
                prompt_hash="old",
                identity_hash="IDENT-1",
                attributes_snapshot={},
            )
        ],
    )
    client = _FakeImageClientNoInpaint()
    render = assets._build_portrait_render(
        session=_Session(),
        image_client=client,
        character=character,
        positive="p",
        face="f",
        negative_extras="",
        identity_hash="IDENT-1",
    )
    out = await render()
    assert out == b"FULL-RENDER"
    assert client.generate_calls == 1

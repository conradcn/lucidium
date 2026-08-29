"""Embedded image client must always render character / portrait
workflows at the canonical SDXL+Pony portrait bucket (832x1216, 2:3),
NOT a square fallback.

Earlier shape: ``WORKFLOW_DIMENSIONS`` was keyed by ``character.json``
but the runtime default in ``ImageSettings.portrait_workflow`` is
``portrait.workflow.json`` — the dict miss fell back to (1024, 1024)
square. Pony Diffusion is only trained on its bucket aspect ratios;
straying off-bucket yields cropped figures, broken anatomy, and the
"why does my portrait look like a head with no body" problem the
user reported. Pin both names map to portrait dims, plus the
backgrounds map to the landscape bucket."""

from __future__ import annotations

import pytest

from lucidium.config import (
    DEFAULT_IMAGE_BACKGROUND_WORKFLOW,
    DEFAULT_IMAGE_PORTRAIT_WORKFLOW,
)
from lucidium.providers.embedded_image_client import (
    _is_character_workflow,
    _resolve_dimensions,
)


def test_canonical_character_filename_maps_to_sdxl_portrait_bucket() -> None:
    assert _resolve_dimensions("character.json") == (832, 1216)


def test_runtime_default_portrait_workflow_maps_to_sdxl_portrait_bucket() -> None:
    # The actual default the engine ships — anything else is a bug.
    assert _resolve_dimensions(DEFAULT_IMAGE_PORTRAIT_WORKFLOW) == (832, 1216)


def test_canonical_background_filename_maps_to_landscape_bucket() -> None:
    assert _resolve_dimensions("background.json") == (1536, 1024)


def test_runtime_default_background_workflow_maps_to_landscape_bucket() -> None:
    assert _resolve_dimensions(DEFAULT_IMAGE_BACKGROUND_WORKFLOW) == (1536, 1024)


def test_unknown_workflow_falls_back_to_portrait_not_square() -> None:
    """Failure-mode of the old code was a square (1024,1024) fallback
    which silently wrecked Pony output. The new fallback is the
    portrait bucket — most callers want characters, and a square
    misrender is worse than a portrait misrender."""
    w, h = _resolve_dimensions("custom-character-v3.workflow.json")
    assert (w, h) == (832, 1216)
    # Sanity: the bucket is portrait (taller than wide), 2:3.
    assert h > w
    assert pytest.approx(w / h, rel=0.01) == 832 / 1216


def test_unknown_background_flavoured_workflow_routes_to_landscape() -> None:
    w, h = _resolve_dimensions("scene-night.workflow.json")
    assert (w, h) == (1536, 1024)


def test_runtime_default_portrait_routes_through_character_pipeline() -> None:
    """If ``portrait.workflow.json`` is treated as a background, the
    FaceDetailer is skipped and the whole render lands at landscape
    dims — explicitly pin that the runtime default is recognised as
    a character workflow."""
    assert _is_character_workflow(DEFAULT_IMAGE_PORTRAIT_WORKFLOW) is True
    assert _is_character_workflow("character.json") is True
    assert _is_character_workflow("portrait.workflow.json") is True


def test_background_workflows_routed_through_environment_pipeline() -> None:
    assert _is_character_workflow(DEFAULT_IMAGE_BACKGROUND_WORKFLOW) is False
    assert _is_character_workflow("background.json") is False
    assert _is_character_workflow("scene-loop.workflow.json") is False
    assert _is_character_workflow("environment-day.json") is False

"""Environment entity: a generated background tied to a Location."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from .ids import new_id


def _random_seed() -> int:
    return secrets.randbits(64)


class Environment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    location_label: str
    prompt: str
    image_path: str | None = None
    prompt_hash: str
    # A few words describing the lighting in this scene — e.g.,
    # "harsh midday sunlight", "candlelit gloom", "blue dusk through
    # leaded glass". Drives both the background prompt AND the
    # portrait prompts for any character on this stage so faces
    # match the room. Updated when the LLM regenerates the
    # environment or when the player edits the field directly.
    lighting: str = ""
    # Deterministic render seed. Same prompt + same seed = same
    # background image, so the player's "Rerender" button mutates
    # this field to force a different draw without changing the
    # prompt itself. Mirrors ``Character.seed``. Default is a fresh
    # random 64-bit value per environment; existing saves that
    # predate this field receive a fresh seed at load time (Pydantic
    # uses the default_factory when the JSON is missing the key).
    seed: int = Field(default_factory=_random_seed)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

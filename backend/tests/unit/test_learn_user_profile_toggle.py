"""``Settings.learn_user_profile`` is the master toggle for the
summarizer's profile-inference step. These tests pin the schema
contract — the field exists, defaults to True, accepts overrides.
The actual gating (discarding ``application.user_profile`` when
the toggle is off) lives in
``world_refresh_handler``; runtime tests of that path require a
full session+LLM harness which is out of scope for the unit
suite. Pinning the schema here is enough to catch a refactor
that drops the field by mistake.
"""

from __future__ import annotations

from lucidium.domain.settings import Settings


def test_learn_user_profile_defaults_true() -> None:
    """Fresh installs learn from play. The opt-out is explicit."""
    s = Settings()
    assert s.learn_user_profile is True


def test_learn_user_profile_persists_when_set_false() -> None:
    """The settings_update handler does a shallow merge per-key,
    so a False value coming through the patch must round-trip
    cleanly through model_validate (no implicit fallback to the
    True default)."""
    raw = Settings().model_dump()
    raw["learn_user_profile"] = False
    s = Settings.model_validate(raw)
    assert s.learn_user_profile is False


def test_learn_user_profile_round_trips_through_json() -> None:
    """Save/load: persisting Settings to disk and reading back
    preserves the toggle. Pin this so a partial migration that
    forgets to write the field doesn't silently flip the default
    on top of an opt-out."""
    s = Settings(learn_user_profile=False)
    payload = s.model_dump_json()
    rehydrated = Settings.model_validate_json(payload)
    assert rehydrated.learn_user_profile is False

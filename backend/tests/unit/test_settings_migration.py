"""Tolerance for unknown keys in the on-disk settings file.

When an experimental field is added and later reverted (or a
field gets renamed), the player's ``settings.json`` may still
carry the old key. ``Settings`` declares ``extra="forbid"``, so
naive ``model_validate_json`` blows up — and the failure is
upstream of the Settings UI, so the player can't recover by
clicking anywhere.

These tests pin the lenient-load contract:

  * Unknown keys at any nesting depth get stripped silently.
  * The cleaned file is rewritten so the next boot is clean.
  * A KNOWN field whose value the schema rejects is dropped back
    to its default rather than raising — the load happens inside
    ``Session.__init__``, upstream of every WebSocket connection,
    so raising locks the player out of the Settings UI entirely.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from lucidium.domain.settings import Settings
from lucidium.persistence import settings_store
from lucidium.persistence.settings_store import load_settings


def test_unknown_top_level_keys_get_stripped(tmp_path: Path) -> None:
    """A top-level field that the schema doesn't declare anymore
    (e.g. a reverted experimental flag) shouldn't crash the load."""
    target = tmp_path / "settings.json"
    target.write_text(
        json.dumps(
            {
                "mature_content": True,
                "obsolete_experiment_flag": "yes",
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(target)
    assert settings.mature_content is True
    # File has been rewritten without the obsolete key.
    rewritten = json.loads(target.read_text(encoding="utf-8"))
    assert "obsolete_experiment_flag" not in rewritten
    assert rewritten["mature_content"] is True


def test_unknown_nested_keys_get_stripped(tmp_path: Path) -> None:
    """The ``image`` sub-model also forbids extras. A reverted
    audio feature leaves ``image.embedded_ambient_audio`` etc.
    in the file — those must strip too."""
    target = tmp_path / "settings.json"
    target.write_text(
        json.dumps(
            {
                "image": {
                    "backend": "embedded",
                    "embedded_ambient_audio": True,
                    "embedded_audio_model_name": "",
                    "embedded_ambient_clip_seconds": 30.0,
                },
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(target)
    assert settings.image.backend.value == "embedded"
    rewritten = json.loads(target.read_text(encoding="utf-8"))
    for key in (
        "embedded_ambient_audio",
        "embedded_audio_model_name",
        "embedded_ambient_clip_seconds",
    ):
        assert key not in rewritten["image"], f"{key} should have been stripped"


def test_known_field_with_invalid_value_is_dropped_to_default(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An out-of-range value on a KNOWN key must not raise.

    ``llm_max_in_flight`` is capped at 64; 9999 fails validation and
    stripping can't help (the key IS declared). Raising here would take
    down every WebSocket connection, so the field is dropped back to its
    default, logged, and the file rewritten clean. Sibling keys in the
    same sub-model survive.
    """
    target = tmp_path / "settings.json"
    default_llm = Settings().concurrency.llm_max_in_flight
    target.write_text(
        json.dumps(
            {
                "mature_content": True,
                "concurrency": {"llm_max_in_flight": 9999, "image_max_in_flight": 3},
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger=settings_store.__name__):
        settings = load_settings(target)

    assert settings.concurrency.llm_max_in_flight == default_llm
    # Only the offending field reverted; the rest of the file took effect.
    assert settings.concurrency.image_max_in_flight == 3
    assert settings.mature_content is True
    assert "concurrency.llm_max_in_flight" in caplog.text

    rewritten = json.loads(target.read_text(encoding="utf-8"))
    assert rewritten["concurrency"]["llm_max_in_flight"] == default_llm
    assert rewritten["concurrency"]["image_max_in_flight"] == 3


def test_unsalvageable_settings_fall_back_to_defaults(tmp_path: Path) -> None:
    """When the bad value can't be narrowed to a droppable field
    (here a whole sub-model given the wrong JSON type), the loader
    still hands back a usable ``Settings`` instead of raising."""
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"concurrency": 7}), encoding="utf-8")
    settings = load_settings(target)
    assert settings.concurrency.llm_max_in_flight >= 1


def test_clean_settings_file_loads_normally(tmp_path: Path) -> None:
    """Path doesn't perturb a clean file: a settings.json with
    only declared keys reads back identically (mtime + content),
    no spurious rewrite."""
    target = tmp_path / "settings.json"
    payload = (
        json.dumps({"mature_content": False, "first_time_setup_complete": True}, indent=2) + "\n"
    )
    target.write_text(payload, encoding="utf-8")
    before_mtime = target.stat().st_mtime_ns
    settings = load_settings(target)
    assert settings.mature_content is False
    assert settings.first_time_setup_complete is True
    after_mtime = target.stat().st_mtime_ns
    # No rewrite when the file was already clean.
    assert before_mtime == after_mtime


def test_corrupt_json_falls_back_to_defaults(tmp_path: Path) -> None:
    """If the file isn't valid JSON at all (truncated mid-write,
    manual corruption), fall back to defaults rather than refuse
    to start the engine."""
    target = tmp_path / "settings.json"
    target.write_text('{"image": {"backend":', encoding="utf-8")  # truncated
    settings = load_settings(target)
    # Defaults loaded; engine is operable.
    assert settings.mature_content is False


def test_missing_file_returns_defaults(tmp_path: Path) -> None:
    target = tmp_path / "does_not_exist.json"
    settings = load_settings(target)
    assert settings.mature_content is False

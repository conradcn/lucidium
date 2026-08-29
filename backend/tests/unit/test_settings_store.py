from __future__ import annotations

from pathlib import Path

from lucidium.config import settings_path
from lucidium.domain.settings import Settings
from lucidium.persistence.settings_store import load_settings, save_settings


def test_load_settings_returns_defaults_when_missing(tmp_app_data: Path) -> None:
    assert not settings_path().exists()
    loaded = load_settings()
    assert isinstance(loaded, Settings)
    assert loaded.image.base_url.startswith("http://127.0.0.1")


def test_settings_round_trip(tmp_app_data: Path) -> None:
    settings = Settings()
    settings = settings.model_copy(update={"typewriter_speed_chars_per_sec": 30})
    save_settings(settings)
    loaded = load_settings()
    assert loaded.typewriter_speed_chars_per_sec == 30

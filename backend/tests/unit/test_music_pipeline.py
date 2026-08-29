"""ACE-Step background-music pipeline wiring.

These tests stub the music client so they don't reach a real
ACE-Step server. They pin:

  * The world's ``music_path`` / ``music_prompt`` /
    ``music_prompt_hash`` get updated when ``ensure_music_for_world``
    runs against a non-empty prompt.
  * The same call is idempotent — re-running with the same prompt
    against an already-rendered file does no work.
  * ``Settings.music.enabled = False`` defaults; the field's
    fingerprint round-trips through JSON.
  * The text_gen prompt builder mentions the music_change rule
    when ``music_enabled=True`` and explicitly forbids it when
    ``False``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lucidium.domain.character import Character
from lucidium.domain.dialog import DialogTree
from lucidium.domain.game import Game
from lucidium.domain.settings import MusicSettings, Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.assets import (
    ensure_music_for_world,
    music_prompt_hash,
)
from lucidium.orchestration.prompts import text_gen as text_gen_prompts


class _StubMusicClient:
    """Returns deterministic bytes for any prompt — never reaches
    a real ACE-Step server."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def generate_music(
        self,
        prompt: str,
        *,
        seed: int,
        **_kwargs,
    ) -> bytes:
        self.calls.append((prompt, seed))
        # Minimal MP3 frame header (just enough that a player /
        # browser doesn't immediately reject the file).
        return b"\xff\xfb\x90\x00" + b"\x00" * 1024


class _SessionStub:
    def __init__(self, *, game: Game, settings: Settings) -> None:
        self.game = game
        self.settings = settings

    @property
    def saves_root(self) -> Path | None:
        return None

    def install_game(self, g: Game) -> None:
        self.game = g


def _player() -> Character:
    return Character(
        name="Iris",
        description="archivist",
        gender="female",
        age=28,
        ethnicity="local",
        skin="pale",
        hair_color="auburn",
        hairstyle="braid",
        eye_color="grey",
        build="slight",
        bust="moderate",
        outfit="wool coat",
        pose="standing",
        expression="alert",
        seed=1,
        is_player=True,
    )


def _make_session(*, settings: Settings | None = None) -> _SessionStub:
    settings = settings or Settings(music=MusicSettings(enabled=True))
    world = WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink wash",
    )
    player = _player()
    game = Game(
        world=world,
        characters={player.id: player},
        dialog_tree=DialogTree(),
    )
    return _SessionStub(game=game, settings=settings)


@pytest.mark.asyncio
async def test_ensure_music_renders_and_updates_world(tmp_path: Path) -> None:
    """First-call path: music_path is None, helper invokes the
    client, writes the file, sets all three world fields."""
    session = _make_session()
    client = _StubMusicClient()
    asset = await ensure_music_for_world(
        session=session,
        music_client=client,
        prompt="cinematic orchestral, low strings, slow, melancholic",
        saves_root=tmp_path,
    )
    assert asset is not None
    assert asset.kind == "music"
    assert asset.image_path.exists()
    # World fields point at the rendered file.
    world = session.game.world
    assert world.music_path == str(asset.image_path)
    assert world.music_prompt.startswith("cinematic orchestral")
    expected_hash = music_prompt_hash(
        "cinematic orchestral, low strings, slow, melancholic",
        session.settings.music.model_name,
    )
    assert world.music_prompt_hash == expected_hash
    # One ACE-Step call fired.
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_ensure_music_idempotent_on_same_prompt(tmp_path: Path) -> None:
    """Second call with the same prompt + same model = no work
    done. Prompt hash matches; the existing file passes the
    freshness check; the audio client is never invoked."""
    session = _make_session()
    client = _StubMusicClient()
    prompt = "synth-noir, soft pads, distant brass, mid-tempo, moody"
    first = await ensure_music_for_world(
        session=session,
        music_client=client,
        prompt=prompt,
        saves_root=tmp_path,
    )
    assert first is not None
    second = await ensure_music_for_world(
        session=session,
        music_client=client,
        prompt=prompt,
        saves_root=tmp_path,
    )
    assert second is None
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_ensure_music_skips_when_client_none(tmp_path: Path) -> None:
    """When music gen is disabled, ``Session.music_client()``
    returns None and the helper short-circuits cleanly without
    raising."""
    session = _make_session(settings=Settings(music=MusicSettings(enabled=False)))
    asset = await ensure_music_for_world(
        session=session,
        music_client=None,
        prompt="anything",
        saves_root=tmp_path,
    )
    assert asset is None


@pytest.mark.asyncio
async def test_ensure_music_skips_on_empty_prompt(tmp_path: Path) -> None:
    """Empty / whitespace prompts fall through. The world_init
    LLM emits empty for music_change when nothing should swap;
    the helper must not call the audio backend with empty text."""
    session = _make_session()
    client = _StubMusicClient()
    asset = await ensure_music_for_world(
        session=session,
        music_client=client,
        prompt="   ",
        saves_root=tmp_path,
    )
    assert asset is None
    assert client.calls == []


def test_music_settings_defaults_disabled() -> None:
    """Default install: music gen is OFF. Players opt in via the
    Settings UI checkbox."""
    s = Settings()
    assert s.music.enabled is False
    # Default model name matches the ACE-Step API server's stock
    # DiT slot — letting an out-of-the-box server hit /v1/init
    # without the player picking a model first.
    assert s.music.model_name.startswith("acestep-")


def test_music_settings_round_trips_through_json() -> None:
    """The Settings serializer + lenient migration both have to
    handle the music sub-model."""
    s = Settings(
        music=MusicSettings(
            enabled=True,
            base_url="http://localhost:9000",
            model_name="custom-fork",
        )
    )
    rehydrated = Settings.model_validate_json(s.model_dump_json())
    assert rehydrated.music.enabled is True
    assert rehydrated.music.base_url == "http://localhost:9000"
    assert rehydrated.music.model_name == "custom-fork"


def test_text_gen_prompt_includes_music_rule_when_enabled() -> None:
    """When the player has music gen on, the storyteller prompt
    must explain how to emit music_change beats. Pin the rule
    string so a refactor doesn't silently drop it."""
    from lucidium.domain.character import Character

    char = Character(
        name="Mira",
        description="scrivener",
        gender="female",
        age=34,
        ethnicity="local",
        skin="fair",
        hair_color="dark",
        hairstyle="braid",
        eye_color="hazel",
        build="slim",
        bust="small",
        outfit="coat",
        pose="standing",
        expression="alert",
        seed=2,
    )
    world = WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink",
    )
    msgs = text_gen_prompts.build(
        world=world,
        history=[],
        on_stage={char.id: char},
        off_stage={},
        chosen_option_text=None,
        music_enabled=True,
        current_music_prompt="cinematic orchestral",
    )
    user_text = msgs[-1]["content"]
    assert "MUSIC RULE" in user_text
    assert "music_change" in user_text
    assert "cinematic orchestral" in user_text  # current prompt surfaced
    # And the disabled-language doesn't appear.
    assert "Background music generation is DISABLED" not in user_text


def test_text_gen_prompt_forbids_music_change_when_disabled() -> None:
    """When music gen is off, the prompt must explicitly tell the
    LLM to set music_change to null, so token spend on irrelevant
    music prompts stays zero."""
    char = Character(
        name="Mira",
        description="scrivener",
        gender="female",
        age=34,
        ethnicity="local",
        skin="fair",
        hair_color="dark",
        hairstyle="braid",
        eye_color="hazel",
        build="slim",
        bust="small",
        outfit="coat",
        pose="standing",
        expression="alert",
        seed=2,
    )
    world = WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink",
    )
    msgs = text_gen_prompts.build(
        world=world,
        history=[],
        on_stage={char.id: char},
        off_stage={},
        chosen_option_text=None,
        music_enabled=False,
    )
    user_text = msgs[-1]["content"]
    assert "Background music generation is DISABLED" in user_text


def test_flatten_music_inventory_picks_dit_only() -> None:
    """The handler-side flattener has to keep ONLY DiT model names
    out of the inventory payload; the LM list is irrelevant to
    the engine (we drive prompts via OpenRouter and ask the
    server for ``init_llm=False``). Default model floats to the
    head of the list so the Settings dropdown's first option is
    the server's stock choice."""
    from lucidium.api.handlers import _flatten_music_inventory

    body = {
        "data": {
            "models": [
                {"name": "acestep-v15-turbo", "is_default": True},
                {"name": "acestep-v15-xl-base", "is_default": False},
                {"name": "acestep-v15-xl-turbo", "is_default": False},
            ],
            "default_model": "acestep-v15-turbo",
            "lm_models": [{"name": "acestep-5Hz-lm-1.7B"}],
            "loaded_lm_model": "acestep-5Hz-lm-1.7B",
            "llm_initialized": True,
        },
        "code": 200,
    }
    out = _flatten_music_inventory(body)
    assert out == [
        "acestep-v15-turbo",
        "acestep-v15-xl-base",
        "acestep-v15-xl-turbo",
    ]


def test_flatten_music_inventory_fallback_walks_unknown_shape() -> None:
    """When the server returns an unfamiliar shape (e.g. a flat
    list of dicts at the root), the flattener still pulls model
    names out so the dropdown isn't empty."""
    from lucidium.api.handlers import _flatten_music_inventory

    body = [{"name": "fork-a"}, {"name": "fork-b"}]
    assert _flatten_music_inventory(body) == ["fork-a", "fork-b"]

"""Pin a batch of session-side bug fixes from the user's list:

* Adventure genre is in the hardcoded list.
* Speaker auto-onstage: when a beat's speaker_id isn't in
  entering_character_ids, _apply_node still places them on
  stage so the renderer shows their portrait.
* Image-client dispose drops pipelines synchronously so a
  settings change frees VRAM before the next render's load.
"""

from __future__ import annotations

from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
)
from lucidium.domain.game import Game
from lucidium.domain.world import WorldState
from lucidium.orchestration.prompts.interview import HARDCODED_GENRE_OPTIONS


def test_adventure_is_a_hardcoded_genre_option() -> None:
    """User asked for Adventure as a genre. Pin it so a future
    list reshuffle can't silently drop it again."""
    assert "Adventure" in HARDCODED_GENRE_OPTIONS


# ---------- Speaker auto-onstage --------------------------------------------


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink wash",
    )


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
        kind=CharacterKind.human,
    )


def _npc(npc_id: str, name: str) -> Character:
    return Character(
        id=npc_id,
        name=name,
        description="an npc",
        gender="male",
        age=40,
        ethnicity="local",
        skin="tan",
        hair_color="grey",
        hairstyle="short",
        eye_color="brown",
        build="stocky",
        bust="n/a",
        outfit="oilskin",
        pose="standing",
        expression="watchful",
        seed=2,
        kind=CharacterKind.human,
    )


def test_speaker_implicitly_added_to_on_stage_when_entering_omits_them() -> None:
    """The reported symptom: the LLM emits a beat where Hale
    speaks but doesn't list ``hale`` in entering_character_ids.
    Without the implicit-onstage rule, the renderer shows no
    portrait for Hale even though he's clearly in the scene.
    """
    from lucidium.api.handlers import _apply_node

    hale = _npc("hale", "Hale")
    player = _player()
    game = Game(
        world=_world(),
        characters={player.id: player, hale.id: hale},
        environments={},
        dialog_tree=DialogTree(),
        on_stage=[],
    )
    node = DialogNode(
        id="n1",
        text="The lamp hisses softly.",
        speaker_id="hale",
        # entering_character_ids deliberately empty — the bug.
        entering_character_ids=[],
        leaving_character_ids=[],
        new_characters=[],
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )

    next_game = _apply_node(game, node)

    assert "hale" in next_game.on_stage, "speaker should be implicitly placed on stage"


def test_speaker_implicit_rule_skips_player_character() -> None:
    """Player is the lens, never on stage. If the player is
    listed as the speaker, they MUST NOT end up in on_stage."""
    from lucidium.api.handlers import _apply_node

    player = _player()
    game = Game(
        world=_world(),
        characters={player.id: player},
        environments={},
        dialog_tree=DialogTree(),
        on_stage=[],
    )
    node = DialogNode(
        id="n1",
        text="You step inside.",
        speaker_id=player.id,
        entering_character_ids=[],
        leaving_character_ids=[],
        new_characters=[],
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )

    next_game = _apply_node(game, node)

    assert player.id not in next_game.on_stage


def test_speaker_implicit_rule_respects_removed_flag() -> None:
    """A character marked ``removed=True`` (player-dismissed)
    must NOT be re-staged just because they appear as a
    speaker — the dismiss flag is sticky."""
    from lucidium.api.handlers import _apply_node

    hale = _npc("hale", "Hale").model_copy(update={"removed": True})
    player = _player()
    game = Game(
        world=_world(),
        characters={player.id: player, hale.id: hale},
        environments={},
        dialog_tree=DialogTree(),
        on_stage=[],
    )
    node = DialogNode(
        id="n1",
        text="Hale would speak, if he were here.",
        speaker_id="hale",
        entering_character_ids=[],
        leaving_character_ids=[],
        new_characters=[],
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )

    next_game = _apply_node(game, node)

    assert "hale" not in next_game.on_stage


def test_speaker_already_on_stage_is_idempotent() -> None:
    """Speaker already on stage stays exactly once — the
    implicit rule must not duplicate the entry."""
    from lucidium.api.handlers import _apply_node

    hale = _npc("hale", "Hale")
    player = _player()
    game = Game(
        world=_world(),
        characters={player.id: player, hale.id: hale},
        environments={},
        dialog_tree=DialogTree(),
        on_stage=["hale"],
    )
    node = DialogNode(
        id="n1",
        text="The lamp hisses.",
        speaker_id="hale",
        entering_character_ids=[],
        leaving_character_ids=[],
        new_characters=[],
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )

    next_game = _apply_node(game, node)

    assert next_game.on_stage.count("hale") == 1


# ---------- Image-client dispose drops pipelines synchronously --------------


def test_dispose_image_client_drops_pipelines_inline() -> None:
    """Pin: when the user changes model in Settings, the OLD
    client's pipeline cache must be cleared SYNCHRONOUSLY so
    the new client's load doesn't compete with stale pipelines
    in VRAM. Earlier this was fire-and-forget, which left the
    next render either OOMing or silently using the OLD
    checkpoint until the engine restarted."""
    from lucidium.domain.settings import Settings
    from lucidium.orchestration.session import Session

    class _StubClient:
        # Intentionally NO ``aclose`` attribute — the dispose
        # helper checks for it and skips the asyncio.run path
        # when missing. The synchronous pipeline drop (what this
        # test pins) doesn't depend on aclose, so we only test
        # that surface here. A separate test would exercise the
        # ComfyUI-style http-client teardown, but it's flaky on
        # Windows because asyncio.run + ProactorEventLoop leaks
        # ResourceWarnings under pytest's strict filter.
        def __init__(self) -> None:
            self._pipelines: dict[str, object] = {
                "old.safetensors": object(),
                "second.safetensors": object(),
            }
            self._inference_locks: dict[str, object] = {
                "old.safetensors": object(),
            }
            self._evicted: set[str] = {"old.safetensors"}

    session = Session.__new__(Session)  # bypass full constructor
    session.settings = Settings()
    # Use a stub logger to avoid touching the real logging stack.
    client = _StubClient()
    session._dispose_image_client(client)

    # Pipelines AND bookkeeping cleared inline. Without this, a
    # subsequent _image_factory() call would build a new client
    # while OLD pipelines still pinned VRAM.
    assert client._pipelines == {}
    assert client._inference_locks == {}
    assert client._evicted == set()

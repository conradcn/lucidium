"""The player character (PC) MUST NEVER get a portrait generated.

The PC is the camera lens — the story is rendered from their
perspective, never as an on-stage figure with a portrait. Three
defensive guards enforce this invariant:

  1. ``_apply_node`` refuses to add the PC's id to ``on_stage``
     even if the LLM lists it in ``entering_character_ids``.
  2. ``ensure_assets`` skips the PC if they somehow slip into
     ``on_stage`` (defense-in-depth for future refactors that
     might bypass guard #1).
  3. ``ensure_portraits_for`` refuses to render any id in its
     ``character_ids`` argument that resolves to ``is_player``.

These tests cover all three so a future change can't silently
introduce a PC portrait on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lucidium.api.handlers import _apply_node
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
)
from lucidium.domain.game import Game
from lucidium.domain.settings import ImageSettings, Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.assets import (
    ensure_assets,
    ensure_portraits_for,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _npc(name: str) -> Character:
    return Character(
        name=name,
        description=f"the {name}",
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
        pose="leaning",
        expression="watchful",
        seed=7,
        kind=CharacterKind.human,
    )


def _seed_game(player: Character, npc: Character, *, on_stage: list[str]) -> Game:
    parent = DialogNode(
        parent_id=None,
        speaker_id=None,
        text="x",
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )
    return Game(
        world=_world(),
        characters={player.id: player, npc.id: npc},
        dialog_tree=DialogTree(
            nodes={parent.id: parent},
            root_id=parent.id,
            committed_path=[parent.id],
        ),
        current_node_id=parent.id,
        on_stage=list(on_stage),
    )


class _CountingImageClient:
    """Tracks every ``generate()`` call. The asset pipeline calls
    ``generate`` once per render; a PC portrait sneaking through
    would surface as a call whose params reference the PC's id /
    name."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, workflow, params, *, seed):
        self.calls.append({"workflow": workflow, "params": params, "seed": seed})
        # Return a minimal PNG. Image bytes are written to disk but
        # content_filter is the NoOp default for unit tests, so the
        # bytes don't need to be a real PNG.
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class _StubSession:
    def __init__(self, *, game: Game) -> None:
        self.game = game
        self.settings = Settings(image=ImageSettings())
        self.emit = None  # nothing to receive notices in this test
        self.commits = 0

    def install_game(self, g: Game) -> None:
        self.game = g

    async def commit(self) -> None:
        self.commit_blocking()

    def commit_blocking(self) -> None:
        self.commits += 1


# ---------------------------------------------------------------------------
# Guard 1: _apply_node
# ---------------------------------------------------------------------------


def test_apply_node_drops_pc_from_explicit_entering_character_ids() -> None:
    """If an LLM beat wrongly lists the PC's id under
    ``entering_character_ids``, ``_apply_node`` must filter it out.
    The implicit-speaker-entering branch already did this; we're
    closing the same hole for the explicit list.
    """
    player = _player()
    npc = _npc("Hale")
    game = _seed_game(player, npc, on_stage=[])

    node = DialogNode(
        parent_id=game.current_node_id,
        speaker_id=None,
        text="a beat",
        # The bug surface: the LLM lists BOTH the PC and an NPC as
        # entering. The engine should accept the NPC and drop the PC.
        entering_character_ids=[player.id, npc.id],
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )
    updated = _apply_node(game, node)

    assert player.id not in updated.on_stage, (
        f"PC id={player.id} must not appear in on_stage; saw {updated.on_stage}"
    )
    assert npc.id in updated.on_stage


def test_apply_node_keeps_npc_when_pc_id_alone_in_entering() -> None:
    """Even if the LLM lists the PC alone (silly but possible),
    on_stage stays empty afterward — no portrait fodder accumulates.
    """
    player = _player()
    npc = _npc("Hale")
    game = _seed_game(player, npc, on_stage=[])

    node = DialogNode(
        parent_id=game.current_node_id,
        speaker_id=None,
        text="solo beat",
        entering_character_ids=[player.id],
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )
    updated = _apply_node(game, node)

    assert updated.on_stage == []


# ---------------------------------------------------------------------------
# Guard 2: ensure_assets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_assets_skips_pc_even_when_on_stage(
    tmp_path: Path,
) -> None:
    """If a future refactor breaks the on_stage gate, the asset
    pipeline itself MUST refuse to render the PC. The test
    deliberately puts the PC into ``on_stage`` (an invalid state)
    and asserts no portrait render call lands.
    """
    player = _player()
    npc = _npc("Hale")
    # The PC is on stage here — a state the rest of the engine would
    # never produce, but this test pins the assets-pipeline guard.
    game = _seed_game(player, npc, on_stage=[player.id, npc.id])
    session = _StubSession(game=game)
    client = _CountingImageClient()

    generated = await ensure_assets(
        session=session,
        image_client=client,
        saves_root=tmp_path,
    )

    # The NPC's portrait may render (or skip, depending on existing
    # files in tmp_path — there are none, so it renders); the PC's
    # absolutely must not.
    pc_renders = [a for a in generated if a.kind == "portrait" and a.target_id == player.id]
    assert pc_renders == [], f"PC portrait must not be generated; saw {pc_renders}"
    # Sanity: the NPC did render so the test isn't a false negative
    # because the pipeline silently no-op'd.
    npc_renders = [a for a in generated if a.kind == "portrait" and a.target_id == npc.id]
    assert len(npc_renders) == 1
    # And the image client got called exactly once (NPC), not twice.
    portrait_calls = [
        c for c in client.calls if c["workflow"] == session.settings.image.portrait_workflow
    ]
    assert len(portrait_calls) == 1


# ---------------------------------------------------------------------------
# Guard 3: ensure_portraits_for
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_portraits_for_skips_pc_in_caller_id_list(
    tmp_path: Path,
) -> None:
    """Speculative pre-renders accept a caller-supplied list of ids.
    The asset pipeline guards against a caller that includes the PC
    by accident — refusing rather than producing a silent FPV
    portrait on disk.
    """
    player = _player()
    npc = _npc("Hale")
    game = _seed_game(player, npc, on_stage=[])
    session = _StubSession(game=game)
    client = _CountingImageClient()

    generated = await ensure_portraits_for(
        session=session,
        image_client=client,
        character_ids=[player.id, npc.id],
        saves_root=tmp_path,
    )

    assert all(a.target_id != player.id for a in generated)
    assert any(a.target_id == npc.id for a in generated)
    # One generate() call (the NPC), not two.
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_ensure_portraits_for_renders_pc_when_allow_player_true(
    tmp_path: Path,
) -> None:
    """The one legitimate manual path — the player clicked Rerender
    on themselves in the Cast tab — passes ``allow_player=True`` and
    must succeed. Pins the escape hatch so a future tightening
    doesn't accidentally lock the user out of refreshing their own
    portrait after editing an attribute.
    """
    player = _player()
    npc = _npc("Hale")
    game = _seed_game(player, npc, on_stage=[])
    session = _StubSession(game=game)
    client = _CountingImageClient()

    generated = await ensure_portraits_for(
        session=session,
        image_client=client,
        character_ids=[player.id],
        saves_root=tmp_path,
        allow_player=True,
    )

    assert any(a.target_id == player.id for a in generated)
    assert len(client.calls) == 1

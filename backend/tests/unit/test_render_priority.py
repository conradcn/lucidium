"""Character renders take priority over the environment.

``ensure_assets`` (the synchronous opening/new-game asset pass)
dispatches every on-stage portrait BEFORE it renders the active-scene
background. On a single GPU the image client serialises the two; a
player registers a missing or stale face far more than a missing
backdrop, so the character work must hit the renderer first. The proof
is the order of the image client's ``generate`` calls — portrait
workflow before background workflow.

(The ongoing, turn-by-turn priority — and the rule that a background is
only ever *presented* for the current scene — lives in the per-entity
``RenderScheduler``; see ``test_render_scheduler.py``.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import DialogNode, DialogNodeState, DialogTree
from lucidium.domain.environment import Environment
from lucidium.domain.game import Game
from lucidium.domain.settings import ImageSettings, Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.assets import ensure_assets


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink wash",
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


class _CountingImageClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, workflow, params, *, seed):
        self.calls.append({"workflow": workflow, "params": params, "seed": seed})
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class _StubSession:
    def __init__(self, *, game: Game) -> None:
        self.game = game
        self.settings = Settings(image=ImageSettings())
        self.emit = None
        self.commits = 0

    def install_game(self, g: Game) -> None:
        self.game = g

    async def commit(self) -> None:
        self.commit_blocking()

    def commit_blocking(self) -> None:
        self.commits += 1


def _seed_game_with_env(npc: Character, env: Environment) -> Game:
    node = DialogNode(
        parent_id=None,
        speaker_id=None,
        text="x",
        location_id=env.id,
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )
    return Game(
        world=_world(),
        characters={npc.id: npc},
        environments={env.id: env},
        dialog_tree=DialogTree(
            nodes={node.id: node},
            root_id=node.id,
            committed_path=[node.id],
        ),
        current_node_id=node.id,
        on_stage=[npc.id],
    )


@pytest.mark.asyncio
async def test_ensure_assets_renders_portraits_before_background(
    tmp_path: Path,
) -> None:
    """The on-stage portrait must be dispatched to the image client
    BEFORE the active-scene background — character priority on the
    shared GPU."""
    npc = _npc("Hale")
    env = Environment(
        location_label="harbor",
        prompt="stone harbor at dawn",
        prompt_hash="seed-hash",  # recomputed inside ensure_assets
    )
    game = _seed_game_with_env(npc, env)
    session = _StubSession(game=game)
    client = _CountingImageClient()

    generated = await ensure_assets(
        session=session,
        image_client=client,
        saves_root=tmp_path,
    )

    kinds = [a.kind for a in generated]
    assert "portrait" in kinds and "background" in kinds, kinds
    assert kinds.index("portrait") < kinds.index("background"), kinds

    portrait_wf = session.settings.image.portrait_workflow
    background_wf = session.settings.image.background_workflow
    order = [c["workflow"] for c in client.calls]
    assert portrait_wf in order and background_wf in order, order
    assert order.index(portrait_wf) < order.index(background_wf), (
        f"portrait must be dispatched before background; got {order}"
    )

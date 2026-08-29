"""Shared scaffolding for the dispatch-level handler tests.

These tests drive handlers the way the WebSocket server does — through
:meth:`HandlerRegistry.dispatch` with a raw payload dict — rather than
calling the handler function with an already-constructed payload model.
That distinction matters: ``dispatch`` is where ``C2S_PAYLOAD_BY_TYPE``
validation runs, and validation is the ONLY thing standing between a
path-shaped ``save_id`` on the wire and ``shutil.rmtree``. A test that
hands the handler a pre-built ``C2SSavesDelete`` has already skipped the
check it means to be exercising.

Not a ``test_*`` module, so pytest doesn't collect it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lucidium.api.handlers import (
    HandlerContext,
    HandlerRegistry,
    build_default_registry,
)
from lucidium.api.messages import Envelope, MessageType
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import DialogNode, DialogNodeState, DialogTree
from lucidium.domain.game import Game
from lucidium.domain.settings import Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.session import Session

OutboundList = list[tuple[MessageType, Any]]


class NullImage:
    """Image client that never renders. The handlers under test don't
    generate imagery; this exists so ``Session`` has a client object
    rather than falling through to a real backend."""

    async def generate(self, *_a: Any, **_kw: Any) -> Any:  # pragma: no cover
        raise AssertionError("no image generation expected in these tests")

    async def aclose(self) -> None:
        return None


class ScriptedLlm:
    """LLM client that replays a fixed list of responses in order.

    Only ``new_game/surprise_me`` needs one; every other handler here is
    network- and model-free.
    """

    def __init__(self, responses: list[str], *, fallback: str | None = None) -> None:
        self._responses = list(responses)
        self._fallback = fallback
        self.prompts: list[Any] = []

    async def complete(self, prompt: Any = None, **_kw: Any) -> Any:
        self.prompts.append(prompt)
        if self._responses:
            value = self._responses.pop(0)
        elif self._fallback is not None:
            value = self._fallback
        else:
            raise AssertionError("ScriptedLlm ran out of responses")

        async def gen():
            yield value

        return gen()

    async def aclose(self) -> None:
        return None


def make_session(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    llm_client: Any | None = None,
) -> Session:
    """A real ``Session`` rooted at a temp saves directory."""
    saves_root = tmp_path / "saves"
    saves_root.mkdir(parents=True, exist_ok=True)
    return Session(
        settings=settings or Settings(),
        llm_client=llm_client,
        image_client=NullImage(),
        saves_root=saves_root,
    )


def make_registry() -> HandlerRegistry:
    return build_default_registry()


async def dispatch(
    registry: HandlerRegistry,
    ctx: HandlerContext,
    message_type: MessageType,
    payload: dict[str, Any] | None = None,
) -> OutboundList:
    """Send one frame through ``dispatch`` and collect every emission."""
    envelope = Envelope(type=message_type, payload=payload or {})
    return [msg async for msg in registry.dispatch(envelope, ctx)]


def types_of(messages: OutboundList) -> list[MessageType]:
    return [m[0] for m in messages]


# ---------------------------------------------------------------------------
# Game fixtures
# ---------------------------------------------------------------------------


def make_world(name: str = "Embers") -> WorldState:
    return WorldState(
        game_name=name,
        setting="stone harbor",
        genre="mystery",
        visual_style="ink wash",
    )


def make_character(
    *,
    name: str,
    is_player: bool = False,
    char_id: str | None = None,
) -> Character:
    kwargs: dict[str, Any] = {}
    if char_id is not None:
        kwargs["id"] = char_id
    return Character(
        name=name,
        description="a figure on the quay",
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
        is_player=is_player,
        kind=CharacterKind.human,
        **kwargs,
    )


def make_game(*, node_count: int = 2) -> tuple[Game, list[str], Character, Character]:
    """A committed linear chain plus a player and one NPC.

    Returns ``(game, node_ids, player, npc)``.
    """
    ids = [f"n{i}" for i in range(node_count)]
    nodes = {
        nid: DialogNode(
            id=nid,
            parent_id=ids[i - 1] if i > 0 else None,
            text=f"beat {i}",
            state=DialogNodeState.committed,
            premise_hash="h" * 64,
        )
        for i, nid in enumerate(ids)
    }
    player = make_character(name="Iris", is_player=True, char_id="iris")
    npc = make_character(name="Mira", char_id="mira")
    game = Game(
        world=make_world(),
        characters={player.id: player, npc.id: npc},
        on_stage=[npc.id],
        dialog_tree=DialogTree(nodes=nodes, root_id=ids[0], committed_path=list(ids)),
        current_node_id=ids[-1],
    )
    return game, ids, player, npc

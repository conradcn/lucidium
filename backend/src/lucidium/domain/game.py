"""Game aggregate root — the unit of one playthrough."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import GAME_SCHEMA_VERSION
from .character import Character
from .dialog import DialogTree
from .environment import Environment
from .ids import new_id
from .world import WorldState

_log = logging.getLogger(__name__)


class CostTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens_in: int = 0
    tokens_out: int = 0
    image_calls: int = 0
    llm_calls: int = 0
    latency_ms_total: int = 0
    dollar_estimate: float = 0.0


class Game(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    schema_version: int = GAME_SCHEMA_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    world: WorldState
    characters: dict[str, Character] = Field(default_factory=dict)
    dialog_tree: DialogTree = Field(default_factory=DialogTree)
    environments: dict[str, Environment] = Field(default_factory=dict)
    current_node_id: str | None = None
    on_stage: list[str] = Field(default_factory=list)
    cost_telemetry: CostTelemetry = Field(default_factory=CostTelemetry)

    @model_validator(mode="after")
    def _check_invariants(self) -> Game:
        # Exactly zero or one player character; not less, not more.
        players = [c for c in self.characters.values() if c.is_player]
        if len(players) > 1:
            msg = f"expected at most one player character, got {len(players)}"
            raise ValueError(msg)

        # current_node_id must point into the dialog tree when set.
        if self.current_node_id is not None and self.current_node_id not in self.dialog_tree.nodes:
            msg = f"current_node_id {self.current_node_id!r} is not in dialog_tree.nodes"
            raise ValueError(msg)

        # on_stage must reference real characters.
        for cid in self.on_stage:
            if cid not in self.characters:
                msg = f"on_stage references unknown character {cid!r}"
                raise ValueError(msg)

        # committed_path's tail, when non-empty, must equal current_node_id.
        committed = self.dialog_tree.committed_path
        if committed and self.current_node_id is not None and committed[-1] != self.current_node_id:
            msg = (
                "committed_path tail does not match current_node_id "
                f"({committed[-1]!r} vs {self.current_node_id!r})"
            )
            raise ValueError(msg)
        return self


def with_speaker_on_stage(game: Game) -> Game:
    """Audibility implies visibility: normalize ``game`` so the
    character speaking the CURRENT beat is on stage.

    A voice with no body on screen is the one stage bug the player
    always notices — a name tag over dialog while the stage shows
    nobody, or shows everyone except the one talking. Rather than
    chase every writer of ``on_stage`` (per-beat entering/leaving,
    the summarizer's idle-cleanup pass, retcon rewriting a beat's
    ``speaker_id``, undo, save migration), the invariant is repaired
    at the single point every mutation funnels through —
    ``Session.install_game`` — so no future caller can install a
    state that violates it.

    Two repairs, matching the two ways the pair can disagree:

    * Speaker is a real, present character but missing from
      ``on_stage`` → put them on stage. Speaking is unambiguous
      evidence of presence.
    * Speaker was ``removed`` (narrated dead, or dismissed from the
      cast panel) → drop the attribution instead of resurrecting
      them. ``removed`` is an explicit fiat that outranks a stray
      beat, and undo restores ``removed=False`` from the snapshot if
      the player wants them back. The line still ships; it just
      reads as narration rather than as a name tag pointing at an
      empty stage.

    A ``speaker_id`` that names no known character (a hallucinated
    id, or the literal ``"narrator"``) is left alone — the renderer
    already draws those as unattributed narration.

    Returns ``game`` unchanged when the invariant already holds, so
    the common path costs one dict lookup and no copies.
    """
    if game.current_node_id is None:
        return game
    node = game.dialog_tree.nodes.get(game.current_node_id)
    if node is None or not node.speaker_id:
        return game
    speaker = game.characters.get(node.speaker_id)
    # Unknown id -> narration; the player character is the camera
    # lens and never has a portrait on stage (FR-034a).
    if speaker is None or speaker.is_player:
        return game

    if speaker.removed:
        _log.warning(
            "beat %s is spoken by removed character %s (%s); dropping the "
            "speaker attribution so no off-stage name tag is rendered",
            node.id,
            speaker.id,
            speaker.removed_reason or "removed",
        )
        stripped = node.model_copy(update={"speaker_id": None})
        nodes = {**game.dialog_tree.nodes, node.id: stripped}
        return game.model_copy(
            update={"dialog_tree": game.dialog_tree.model_copy(update={"nodes": nodes})}
        )

    if node.speaker_id in game.on_stage:
        return game
    return game.model_copy(update={"on_stage": [*game.on_stage, node.speaker_id]})

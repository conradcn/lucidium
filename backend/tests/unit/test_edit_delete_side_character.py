"""Edit / delete handlers for the New Game side-character list.

The Add button on the Review step appends a stub Character to
``interview.side_characters``. The two handlers exercised here let
the player rename or remove those stubs before pressing Begin —
without going through the LLM (expansion still happens at confirm
time, regardless of which name the stub ends up with).
"""

from __future__ import annotations

import pytest

from lucidium.api.errors import NotFoundError, SchemaError
from lucidium.api.handlers import (
    HandlerContext,
    new_game_add_side_character_handler,
    new_game_delete_side_character_handler,
    new_game_edit_side_character_handler,
)
from lucidium.api.messages import (
    C2SNewGameAddSideCharacter,
    C2SNewGameDeleteSideCharacter,
    C2SNewGameEditSideCharacter,
    MessageType,
)
from lucidium.orchestration.session import InterviewState


class _StubSession:
    def __init__(self) -> None:
        self.interview = InterviewState(setting="harbor town")
        self.llm_calls = 0

    async def llm_text(self, *_a, **_kw):
        self.llm_calls += 1
        return "{}", []


async def _drain(gen):
    return [m async for m in gen]


async def _add(session, description: str) -> str:
    ctx = HandlerContext(session=session)
    msgs = await _drain(
        await new_game_add_side_character_handler(
            C2SNewGameAddSideCharacter(description=description),
            ctx,
        )
    )
    # The handler emits one state_patch with a single add op whose path
    # carries the new stub's id as its last segment.
    op = msgs[0][1].ops[0]
    return op.path.rsplit("/", 1)[-1]


@pytest.mark.asyncio
async def test_edit_renames_stub_and_keeps_descriptions_in_sync() -> None:
    session = _StubSession()
    stub_id = await _add(session, "grizzled lighthouse keeper")

    ctx = HandlerContext(session=session)
    msgs = await _drain(
        await new_game_edit_side_character_handler(
            C2SNewGameEditSideCharacter(
                character_id=stub_id,
                description="weathered keeper",
            ),
            ctx,
        )
    )

    sides = session.interview.side_characters
    descs = session.interview.side_character_descriptions
    assert len(sides) == 1
    assert sides[0].id == stub_id
    assert sides[0].name == "weathered keeper"
    assert sides[0].description == "weathered keeper"
    # Parallel list must stay aligned: confirm-time expansion reads
    # ``side_character_descriptions[i]`` against
    # ``side_characters[i]``, so a drift here would re-expand the
    # OLD description and the edit would be silently lost.
    assert descs == ["weathered keeper"]

    # Echo: one replace op carrying the new name.
    assert len(msgs) == 1
    msg_type, body = msgs[0]
    assert msg_type == MessageType.s2c_state_patch
    op = body.ops[0]
    assert op.op == "replace"
    assert op.path == f"/interview/side_characters/{stub_id}"
    assert op.value == {"id": stub_id, "name": "weathered keeper"}

    # No LLM round-trip; expansion is still deferred to confirm.
    assert session.llm_calls == 0


@pytest.mark.asyncio
async def test_edit_rejects_empty_description() -> None:
    session = _StubSession()
    stub_id = await _add(session, "the keeper")

    ctx = HandlerContext(session=session)
    with pytest.raises(SchemaError):
        await new_game_edit_side_character_handler(
            C2SNewGameEditSideCharacter(
                character_id=stub_id,
                description="   ",
            ),
            ctx,
        )

    # State unchanged.
    assert session.interview.side_characters[0].name == "the keeper"


@pytest.mark.asyncio
async def test_edit_unknown_id_raises_not_found() -> None:
    session = _StubSession()
    await _add(session, "the keeper")

    ctx = HandlerContext(session=session)
    with pytest.raises(NotFoundError):
        await new_game_edit_side_character_handler(
            C2SNewGameEditSideCharacter(
                character_id="nonexistent",
                description="x",
            ),
            ctx,
        )


@pytest.mark.asyncio
async def test_delete_drops_stub_from_both_lists() -> None:
    session = _StubSession()
    a = await _add(session, "first NPC")
    b = await _add(session, "second NPC")
    assert len(session.interview.side_characters) == 2

    ctx = HandlerContext(session=session)
    msgs = await _drain(
        await new_game_delete_side_character_handler(
            C2SNewGameDeleteSideCharacter(character_id=a),
            ctx,
        )
    )

    sides = session.interview.side_characters
    descs = session.interview.side_character_descriptions
    assert len(sides) == 1
    assert sides[0].id == b
    # The parallel descriptions list must shrink in lock-step.
    assert descs == ["second NPC"]

    assert len(msgs) == 1
    msg_type, body = msgs[0]
    assert msg_type == MessageType.s2c_state_patch
    op = body.ops[0]
    assert op.op == "remove"
    assert op.path == f"/interview/side_characters/{a}"


@pytest.mark.asyncio
async def test_delete_unknown_id_raises_not_found() -> None:
    session = _StubSession()
    await _add(session, "the keeper")

    ctx = HandlerContext(session=session)
    with pytest.raises(NotFoundError):
        await new_game_delete_side_character_handler(
            C2SNewGameDeleteSideCharacter(character_id="nonexistent"),
            ctx,
        )

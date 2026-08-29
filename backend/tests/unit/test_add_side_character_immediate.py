"""``c2s/new_game/add_side_character`` is non-blocking.

The handler used to round-trip through the LLM to expand the
one-line description into a fully populated NPC. That made the
Add button laggy — every NPC cost a full LLM call before the UI
updated.

The new shape: the handler appends a STUB Character (description
as name, empty body fields) to ``interview.side_characters`` and
returns immediately. The actual LLM expansion runs at confirm
time, in parallel with the other side characters, before the
opening world is materialised.

Stubs are detected by an empty ``outfit`` field — real expansions
always fill outfit/pose/expression. The stub's id is preserved
so the patch keys the renderer already received still resolve.
"""

from __future__ import annotations

import pytest

from lucidium.api.handlers import (
    HandlerContext,
    new_game_add_side_character_handler,
)
from lucidium.api.messages import (
    C2SNewGameAddSideCharacter,
    MessageType,
)
from lucidium.orchestration.session import InterviewState


class _StubSession:
    """Minimal session — only the bits the handler reads."""

    def __init__(self) -> None:
        self.interview = InterviewState(setting="harbor town")
        self.llm_calls = 0

    async def llm_text(self, *_, **__):
        self.llm_calls += 1
        return "{}", []


async def _drain(gen):
    return [m async for m in gen]


@pytest.mark.asyncio
async def test_add_side_character_does_not_call_llm() -> None:
    session = _StubSession()
    ctx = HandlerContext(session=session)

    result = await new_game_add_side_character_handler(
        C2SNewGameAddSideCharacter(description="grizzled lighthouse keeper"),
        ctx,
    )
    messages = await _drain(result)

    assert session.llm_calls == 0, "add_side_character must be non-blocking — no LLM round-trip"
    assert len(session.interview.side_characters) == 1
    stub = session.interview.side_characters[0]
    # Stub markers: empty outfit / pose / expression so confirm can
    # detect-and-expand. Description carries through as the name.
    assert stub.outfit == ""
    assert stub.pose == ""
    assert stub.expression == ""
    assert stub.name == "grizzled lighthouse keeper"
    # Description is also appended to the parallel descriptions list.
    assert session.interview.side_character_descriptions == ["grizzled lighthouse keeper"]
    # State patch fired with the stub's id + name so the UI updates
    # immediately.
    assert len(messages) == 1
    msg_type, payload = messages[0]
    assert msg_type == MessageType.s2c_state_patch
    op = payload.ops[0]
    assert op.path == f"/interview/side_characters/{stub.id}"
    assert op.value == {"id": stub.id, "name": stub.name}


@pytest.mark.asyncio
async def test_add_side_character_rejects_empty_description() -> None:
    session = _StubSession()
    ctx = HandlerContext(session=session)
    from lucidium.api.errors import SchemaError

    with pytest.raises(SchemaError):
        await new_game_add_side_character_handler(
            C2SNewGameAddSideCharacter(description="   "),
            ctx,
        )

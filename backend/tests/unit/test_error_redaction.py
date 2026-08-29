"""Client-facing error text must not echo internals.

Two leak sites are covered: the generic ``from_exception`` fallback
(which used to ship ``str(exc)``, i.e. absolute filesystem paths) and
the payload-validation path in the handler registry (which used to ship
the full pydantic repr, including ``input_value``).
"""

from __future__ import annotations

import re

import pytest

from lucidium.api.errors import ErrorCode, SchemaError, from_exception
from lucidium.api.handlers import HandlerContext, HandlerRegistry
from lucidium.api.messages import Envelope, MessageType

_REF = re.compile(r"\(ref: [0-9a-f]{8}\)")


def test_unmapped_exception_does_not_leak_message_text() -> None:
    payload = from_exception(OSError("C:/Users/alice/secret"))
    assert payload.code == ErrorCode.internal
    assert "alice" not in payload.message
    assert "secret" not in payload.message
    assert _REF.search(payload.message), payload.message


def test_correlation_ids_are_per_error() -> None:
    first = from_exception(OSError("boom"))
    second = from_exception(OSError("boom"))
    assert first.message != second.message


@pytest.mark.asyncio
async def test_validation_error_message_omits_input_value() -> None:
    registry = HandlerRegistry()
    envelope = Envelope(
        type=MessageType.c2s_saves_load,
        payload={"save_id": "C:/Users/alice/secret"},
    )

    with pytest.raises(SchemaError) as excinfo:
        async for _ in registry.dispatch(envelope, HandlerContext(session=None)):  # type: ignore[arg-type]
            pass

    message = excinfo.value.message
    assert "input_value" not in message
    assert "alice" not in message
    assert "secret" not in message
    assert _REF.search(message), message

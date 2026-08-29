"""``ws_server._encode`` serialises the payload exactly once.

The fast path hand-writes the envelope literal instead of round-tripping
the payload through ``Envelope``, so the thing worth pinning is that the
output is still byte-identical to what the model would have produced —
otherwise a renderer parsing the wire format sees a silent shape change.
"""

from __future__ import annotations

import json

from lucidium.api.messages import (
    Envelope,
    MessageType,
    PatchOp,
    S2CError,
    S2CStatePatch,
)
from lucidium.api.ws_server import _encode


def _via_envelope(message_type: MessageType, payload) -> str:
    """What ``_encode`` used to do."""
    return Envelope(type=message_type, payload=payload.model_dump(mode="json")).model_dump_json()


def test_encode_matches_the_envelope_round_trip_byte_for_byte() -> None:
    payload = S2CStatePatch(
        ops=[
            PatchOp(op="replace", path="/on_stage", value=["a", "b"]),
            PatchOp(op="add", path="/world/plot", value="the tide turns"),
        ]
    )
    assert _encode(MessageType.s2c_state_patch, payload) == _via_envelope(
        MessageType.s2c_state_patch, payload
    )


def test_encode_round_trips_back_through_envelope() -> None:
    payload = S2CError(code="schema_error", message="nope — em dash 0x2014", recoverable=True)
    encoded = _encode(MessageType.s2c_error, payload)

    parsed = Envelope.model_validate_json(encoded)
    assert parsed.type is MessageType.s2c_error
    assert parsed.payload["message"] == "nope — em dash 0x2014"

    # And it is valid, self-describing JSON with no extra keys.
    assert set(json.loads(encoded)) == {"type", "payload", "protocol_version"}


def test_encode_handles_an_empty_payload() -> None:
    payload = S2CStatePatch(ops=[])
    assert _encode(MessageType.s2c_state_patch, payload) == _via_envelope(
        MessageType.s2c_state_patch, payload
    )

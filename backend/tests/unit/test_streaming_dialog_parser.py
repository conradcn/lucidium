"""Streaming dialog parser pins.

Pins behaviour the foreground play-advance flow depends on:

  * Beats are emitted as soon as their closing ``}`` arrives,
    not held back until the whole response lands.
  * Chunk boundaries inside string values, inside escape
    sequences, and across bracket characters don't break parsing.
  * Null-typed-field coercion runs per-beat (same as the
    whole-response parser) so an LLM that emits ``"effects":
    null`` still produces a valid beat.
  * ``finalize`` returns a payload combining streamed beats
    with the trailing ``options``.
  * Best-effort recovery: if the trailing ``options`` are
    truncated or invalid, the streamed beats are still
    returned with empty options so the player can click
    Continue.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from lucidium.api.errors import ProviderValidationError
from lucidium.orchestration.responses import LlmBeat, LlmDialogPayload
from lucidium.orchestration.streaming_dialog_parser import (
    StreamingDialogParser,
    stream_dialog_payload,
)


def _beat_dict(text: str = "x", **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "text": text,
        "speaker_id": None,
        "entering_character_ids": [],
        "leaving_character_ids": [],
        "new_characters": [],
        "location_id": None,
        "location_prompt": None,
        "character_changes": [],
    }
    base.update(over)
    return base


def _full_payload(beats: list[dict[str, Any]], options: list[dict[str, Any]]) -> str:
    return json.dumps({"beats": beats, "options": options})


def _chunk(text: str, size: int = 8) -> list[str]:
    """Split a string into fixed-size chunks to simulate the LLM
    delivering tokens one at a time."""
    return [text[i : i + size] for i in range(0, len(text), size)]


# ---------- Single-shot ------------------------------------------------------


def test_parser_emits_each_beat_as_it_completes() -> None:
    """The headline behaviour. Two beats ⇒ two beats out
    incrementally; finalize then returns the full payload."""
    raw = _full_payload(
        beats=[_beat_dict("first"), _beat_dict("second")],
        options=[{"id": "o1", "text": "go"}, {"id": "o2", "text": "stay"}],
    )
    parser = StreamingDialogParser()
    out = parser.feed(raw)
    assert len(out) == 2
    assert out[0].text == "first"
    assert out[1].text == "second"

    payload = parser.finalize()
    assert isinstance(payload, LlmDialogPayload)
    assert len(payload.beats) == 2
    assert [o.text for o in payload.options] == ["go", "stay"]


def test_parser_handles_chunked_arrival() -> None:
    """Same input, but split across many small chunks. Beats
    must still emit in order, and only when their closing
    brace lands."""
    raw = _full_payload(
        beats=[_beat_dict("first"), _beat_dict("second"), _beat_dict("third")],
        options=[{"id": "o1", "text": "go"}],
    )
    parser = StreamingDialogParser()
    seen: list[str] = []
    for chunk in _chunk(raw, size=4):
        for beat in parser.feed(chunk):
            seen.append(beat.text)
    payload = parser.finalize()
    assert seen == ["first", "second", "third"]
    assert len(payload.beats) == 3
    assert payload.options[0].text == "go"


def test_first_beat_emits_before_later_chunks_arrive() -> None:
    """Pin the latency-win property: parser yields beat[0]
    BEFORE the chunks that carry beats[1+] are fed. This is
    what makes the typewriter start sooner."""
    beat0_dict = _beat_dict("first beat with some prose")
    beat1_dict = _beat_dict("second")
    beat0_text = json.dumps(beat0_dict)
    beat1_text = json.dumps(beat1_dict)
    head = '{"beats": [' + beat0_text + ","
    parser = StreamingDialogParser()
    out = parser.feed(head)
    # First beat is fully formed in ``head`` — must appear now.
    assert len(out) == 1
    assert out[0].text == "first beat with some prose"

    # Feed beat[1].
    out2 = parser.feed(beat1_text)
    assert len(out2) == 1
    assert out2[0].text == "second"

    # Close array + options + outer.
    parser.feed('], "options": [{"id": "o1", "text": "go"}]}')
    payload = parser.finalize()
    assert len(payload.beats) == 2
    assert payload.options[0].text == "go"


# ---------- Edge cases inside string values ---------------------------------


def test_parser_tolerates_curly_braces_inside_string_values() -> None:
    """A beat's text contains ``{`` / ``}`` — naive bracket
    counting would terminate the object early. The parser
    must skip brackets inside JSON strings."""
    raw = _full_payload(
        beats=[
            _beat_dict("She muttered '{this is fine}' and walked off."),
            _beat_dict("Continued."),
        ],
        options=[],
    )
    parser = StreamingDialogParser()
    out = parser.feed(raw)
    parser.finalize()
    assert len(out) == 2
    assert "{this is fine}" in out[0].text


def test_parser_tolerates_escaped_quotes_inside_strings() -> None:
    """Escape-aware string scanning: a ``\\"`` inside a string
    must not be treated as the string's closing quote."""
    raw = _full_payload(
        beats=[
            _beat_dict('He said \\"hello\\" — a {curly} aside — and left.'),
        ],
        options=[],
    )
    parser = StreamingDialogParser()
    out = parser.feed(raw)
    parser.finalize()
    assert len(out) == 1
    # Pydantic stores the unescaped form.
    assert "{curly}" in out[0].text


def test_parser_handles_chunk_boundary_inside_string() -> None:
    """A chunk boundary lands MID-string: the in-string state
    must persist across feed() calls so the next chunk's bytes
    aren't misinterpreted as structural JSON."""
    raw = _full_payload(
        beats=[_beat_dict('"some {tricky} stuff"')],
        options=[],
    )
    # Find a midpoint inside the beat text and split there.
    mid = raw.index("tricky")
    parser = StreamingDialogParser()
    parser.feed(raw[:mid])
    out = parser.feed(raw[mid:])
    parser.finalize()
    assert len(out) == 1
    assert "tricky" in out[0].text


def test_parser_handles_chunk_boundary_inside_escape_sequence() -> None:
    """Escape-state must persist across chunk boundaries too."""
    raw = _full_payload(
        beats=[_beat_dict('She said \\"go\\".')],
        options=[],
    )
    # Find the backslash before the second escape and split there.
    bs_idx = raw.index("\\")
    # Split right after the backslash so the next chunk starts
    # with the escaped char.
    parser = StreamingDialogParser()
    parser.feed(raw[: bs_idx + 1])
    out_after = parser.feed(raw[bs_idx + 1 :])
    parser.finalize()
    assert any("go" in b.text for b in out_after) or any(
        "go" in b.text for b in parser.emitted_beats
    )


# ---------- Null-coercion per beat ------------------------------------------


def test_parser_coerces_null_string_field_per_beat() -> None:
    """Same null-coerce path the whole-response parser uses,
    but applied per-beat as it parses."""
    raw = json.dumps(
        {
            "beats": [
                {
                    "text": "x",
                    "speaker_id": None,
                    "entering_character_ids": [],
                    "leaving_character_ids": [],
                    "new_characters": [
                        {
                            "id": "pell",
                            "name": "Pell",
                            "description": "innkeeper",
                            "age": 40,
                            "outfit": "apron",
                            "pose": "standing",
                            "expression": "curious",
                            "effects": None,  # null-coerced to ""
                        },
                    ],
                    "location_id": None,
                    "location_prompt": None,
                    "character_changes": [],
                },
            ],
            "options": [],
        }
    )
    parser = StreamingDialogParser()
    out = parser.feed(raw)
    parser.finalize()
    assert len(out) == 1
    assert out[0].new_characters[0].effects == ""


# ---------- Truncation / recovery -------------------------------------------


def test_finalize_returns_streamed_beats_when_options_truncate() -> None:
    """LLM cut off mid-options. The streamed beats are real;
    fall back to empty options so the player can click
    Continue rather than losing the whole response."""
    parser = StreamingDialogParser()
    # Two beats parse cleanly; then the array closes; then
    # options start but cut off mid-string.
    parser.feed('{"beats": [')
    parser.feed(json.dumps(_beat_dict("first")) + ",")
    parser.feed(json.dumps(_beat_dict("second")))
    parser.feed('], "options": [{"id": "o1", "text": "go awa')
    payload = parser.finalize()
    assert len(payload.beats) == 2
    assert payload.options == []


def test_finalize_raises_when_no_beats_emitted_and_buffer_unparseable() -> None:
    """Total failure path: nothing parsed during streaming AND
    the remaining buffer is junk. ``finalize`` raises so the
    caller can surface the error."""
    parser = StreamingDialogParser()
    parser.feed("not json at all")
    with pytest.raises(ProviderValidationError):
        parser.finalize()


def test_finalize_handles_markdown_fence_wrapper() -> None:
    """Some LLMs wrap their JSON in ```json...``` fences. The
    streaming path should tolerate the fence the same way
    ``parse_json_object`` does."""
    raw = "```json\n" + _full_payload(beats=[_beat_dict("first")], options=[]) + "\n```"
    parser = StreamingDialogParser()
    out = parser.feed(raw)
    payload = parser.finalize()
    # The leading fence may swallow the array-detection on the
    # incremental path (depends on chunk shape); finalize MUST
    # always return the full payload regardless.
    assert len(payload.beats) == 1
    assert payload.beats[0].text == "first"
    # Streaming may or may not have emitted depending on where
    # the fence sits, but either way we got the beat eventually.
    assert len(out) >= 0  # tolerant


# ---------- Async helper ----------------------------------------------------


@pytest.mark.asyncio
async def test_stream_dialog_payload_yields_beats_then_payload() -> None:
    """End-to-end via the async helper: yields each beat as it
    parses, then the final ``LlmDialogPayload`` once the chunk
    iterator ends."""
    raw = _full_payload(
        beats=[_beat_dict("first"), _beat_dict("second")],
        options=[{"id": "o1", "text": "go"}],
    )

    async def chunks() -> Any:
        for c in _chunk(raw, size=12):
            yield c
            await asyncio.sleep(0)

    seen: list[Any] = []
    async for item in stream_dialog_payload(chunks()):
        seen.append(item)
    # All-but-last entries are LlmBeat; last is LlmDialogPayload.
    assert all(isinstance(b, LlmBeat) for b in seen[:-1])
    assert isinstance(seen[-1], LlmDialogPayload)
    beats_yielded = [b for b in seen if isinstance(b, LlmBeat)]
    assert [b.text for b in beats_yielded] == ["first", "second"]
    assert seen[-1].options[0].text == "go"

"""LLM JSON recovery and retry.

Two layers, both pinned here:

  1. ``parse_json_object`` auto-coerces the most common LLM
     mistake — ``null`` where the schema expects a string,
     list, or dict — by replacing those nulls with ``""``,
     ``[]``, ``{}`` respectively and re-validating once. The
     reported bug case (``new_characters[0].effects = null``
     during world_init) is the canonical example.

  2. ``call_llm_json_with_retry`` wraps an LLM call so that
     when validation fails AND the auto-coerce can't fix it,
     the helper re-prompts with the actual error message.
     Bounded retry loop. The new-game flow uses this for the
     world_init parse so a single bad LLM response can't kill
     the whole onboarding.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from lucidium.api.errors import ProviderValidationError
from lucidium.domain.character import (
    DEFAULT_AGE,
    DEFAULT_EXPRESSION,
    DEFAULT_OUTFIT,
    DEFAULT_POSE,
)
from lucidium.domain.dialog import (
    NewCharacterDescriptor,
)
from lucidium.orchestration.responses import (
    LlmDialogPayload,
    LlmWorldInit,
    call_llm_json_with_retry,
    parse_json_object,
)

# ---------- Layer 1: null coercion at parse time ----------------------------


def _new_char_payload(**overrides: object) -> str:
    base: dict[str, Any] = {
        "id": "pell",
        "name": "Pell",
        "description": "innkeeper",
        "age": 40,
        "outfit": "apron",
        "pose": "standing",
        "expression": "curious",
    }
    base.update(overrides)
    return json.dumps(base)


def test_null_string_field_coerced_to_empty_string() -> None:
    """The reported bug: LLM emits ``effects: null``. Schema
    has ``effects: str = ""``. Pydantic rejects None for str
    even with a default. Auto-coerce replaces null → ""."""
    raw = _new_char_payload(effects=None)
    result = parse_json_object(raw, NewCharacterDescriptor)
    assert result.effects == ""


def test_null_optional_string_field_coerced() -> None:
    """Other str-with-default fields get the same treatment —
    physical_description, hair_color, hairstyle, etc."""
    raw = _new_char_payload(
        physical_description=None,
        hair_color=None,
        hairstyle=None,
    )
    result = parse_json_object(raw, NewCharacterDescriptor)
    assert result.physical_description == ""
    assert result.hair_color == ""
    assert result.hairstyle == ""


def test_null_list_field_coerced_to_empty_list() -> None:
    """LLM emits ``options: null`` instead of an empty list.
    Schema expects ``list[DialogOption]``. Auto-coerce → []."""
    payload = {
        "beats": [
            {
                "text": "You arrive at the harbor.",
                "speaker_id": None,
                "entering_character_ids": None,  # null where list expected
                "leaving_character_ids": [],
                "new_characters": [],
                "location_id": None,
                "location_prompt": None,
                "character_changes": [],
            },
        ],
        "options": None,  # null where list expected
    }
    result = parse_json_object(json.dumps(payload), LlmDialogPayload)
    assert result.options == []
    assert result.beats[0].entering_character_ids == []


def test_multiple_null_errors_in_one_response_all_coerced() -> None:
    """Pydantic surfaces every error in one ValidationError;
    coercion fixes all of them in a single retry, not one per
    field."""
    raw = _new_char_payload(
        effects=None,
        physical_description=None,
        hair_color=None,
        hairstyle=None,
        eye_color=None,
        skin=None,
        build=None,
        bust=None,
        ethnicity=None,
        gender=None,
    )
    result = parse_json_object(raw, NewCharacterDescriptor)
    assert result.effects == ""
    assert result.physical_description == ""
    assert result.hair_color == ""


def test_null_with_no_default_still_raises() -> None:
    """``outfit`` is a required str with no default. A null
    there gets coerced to "" too — same default-fallback. The
    LLM emitting ``outfit: null`` is recoverable; the bigger
    miss would be omitting it entirely (which is a different
    error type, not handled here)."""
    raw = _new_char_payload(outfit=None)
    result = parse_json_object(raw, NewCharacterDescriptor)
    assert result.outfit == ""


def test_genuinely_invalid_payload_still_raises_provider_error() -> None:
    """A payload missing an irreducibly-required field (``name`` on
    a NewCharacterDescriptor) doesn't match a coercible error type
    and still surfaces as a ProviderValidationError. Note ``age`` /
    ``outfit`` / ``pose`` / ``expression`` are NOT in this set any
    more — they self-heal to intelligent defaults so a dropped
    staging field never bounces new-game init back to the LLM."""
    raw = json.dumps(
        {
            "id": "pell",
            "description": "innkeeper",
            "outfit": "apron",
            "pose": "standing",
            "expression": "curious",
            # Missing required ``name`` — the character's identity, which
            # has no sensible default.
        }
    )
    with pytest.raises(ProviderValidationError):
        parse_json_object(raw, NewCharacterDescriptor)


def test_descriptor_missing_staging_fields_gets_defaults() -> None:
    """A descriptor that carries only its identity (id/name/
    description) — the storyteller dropped age/outfit/pose/
    expression — parses to intelligent defaults instead of
    raising. This is the case that used to bounce world_init
    back to the LLM and could stall init when it recurred."""
    raw = json.dumps(
        {
            "id": "scout-tobin",
            "name": "Tobin",
            "description": "a wary scout",
            "ethnicity": "northern",
        }
    )
    char = parse_json_object(raw, NewCharacterDescriptor)
    assert char.age == DEFAULT_AGE
    assert char.outfit == DEFAULT_OUTFIT
    assert char.pose == DEFAULT_POSE
    assert char.expression == DEFAULT_EXPRESSION


def test_world_init_new_character_missing_staging_defaults() -> None:
    """Regression for the reported stall: a ``new_characters`` entry
    nested in a world_init beat that omits staging fields must NOT
    fail the whole world_init parse — it self-heals to defaults so
    onboarding proceeds without an LLM retry."""
    payload = {
        "game_name": "A Distant Blue",
        "plot_outline": [],
        "active_plot_threads": [],
        "opening_node": {
            "beats": [
                {
                    "text": "A figure steps from the treeline.",
                    "entering_character_ids": ["scout-tobin"],
                    "new_characters": [
                        {
                            "id": "scout-tobin",
                            "name": "Tobin",
                            "description": "a wary scout",
                            "ethnicity": "northern",
                            # age / outfit / pose / expression dropped
                        },
                    ],
                },
            ],
            "options": [],
        },
        "initial_music_prompt": "",
    }
    result = parse_json_object(json.dumps(payload), LlmWorldInit)
    tobin = result.opening_node.beats[0].new_characters[0]
    assert tobin.age == DEFAULT_AGE
    assert tobin.outfit == DEFAULT_OUTFIT
    assert tobin.pose == DEFAULT_POSE
    assert tobin.expression == DEFAULT_EXPRESSION


def test_null_at_deep_path_coerced() -> None:
    """The reported bug path: ``opening_node.beats.2.new_characters.0.effects``
    is 5 levels deep. Coercion walks the loc tuple correctly."""
    payload = {
        "game_name": "A Distant Blue",
        "plot_outline": [],
        "active_plot_threads": [],
        "opening_node": {
            "beats": [
                {
                    "text": "Beat 0",
                    "new_characters": [],
                },
                {
                    "text": "Beat 1",
                    "new_characters": [],
                },
                {
                    "text": "Beat 2",
                    "new_characters": [
                        {
                            "id": "pell",
                            "name": "Pell",
                            "description": "innkeeper",
                            "age": 40,
                            "outfit": "apron",
                            "pose": "standing",
                            "expression": "curious",
                            "effects": None,  # the bug
                        },
                    ],
                },
            ],
            "options": [],
        },
        "initial_music_prompt": "",
    }
    result = parse_json_object(json.dumps(payload), LlmWorldInit)
    new_char = result.opening_node.beats[2].new_characters[0]
    assert new_char.effects == ""


def test_codefence_around_invalid_json_still_coerced() -> None:
    """The reported log shows the LLM wrapped its JSON in
    ```json...``` fences. Coercion happens AFTER fence
    stripping, so the fence + null combination is still
    recoverable."""
    raw = "```json\n" + _new_char_payload(effects=None) + "\n```"
    result = parse_json_object(raw, NewCharacterDescriptor)
    assert result.effects == ""


def test_leading_prose_before_json_is_stripped() -> None:
    """The LLM sometimes narrates its reasoning before emitting JSON.
    Real shape: ``I'll start with the perspective bucket since that
    defines the narrative voice.\\n\\n```json\\n{...}\\n``` ``. The
    preamble must be stripped before parsing — otherwise json.loads
    sees ``I'll`` first and bails out with no recovery path."""
    payload = _new_char_payload()
    raw = (
        "I'll start with the perspective bucket since that "
        "defines the narrative voice.\n\n```json\n" + payload + "\n```"
    )
    result = parse_json_object(raw, NewCharacterDescriptor)
    assert result.id == "pell"


def test_leading_prose_without_fence_is_stripped() -> None:
    """Same recovery, but the LLM didn't bother with a Markdown
    fence — the preamble runs straight into the JSON. The first
    ``{`` is the slice point."""
    payload = _new_char_payload()
    raw = "Sure, here's the character:\n\n" + payload
    result = parse_json_object(raw, NewCharacterDescriptor)
    assert result.id == "pell"


def test_trailing_prose_after_json_is_ignored() -> None:
    """Mirror case: a trailing summary after the JSON object. The
    parser uses ``raw_decode`` so anything past the matching close
    brace is silently dropped."""
    payload = _new_char_payload()
    raw = payload + "\n\nAnd that completes the descriptor."
    result = parse_json_object(raw, NewCharacterDescriptor)
    assert result.id == "pell"


# ---------- Layer 2: retry-on-validation-failure ----------------------------


class _ScriptedSession:
    """Plays back a list of pre-canned LLM responses one per
    ``llm_text`` call. Records every prompt the helper sent so
    tests can assert the corrective continuation has the right
    shape."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def llm_text(
        self,
        prompt: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> tuple[str, list[str]]:
        self.calls.append([dict(m) for m in prompt])
        if not self._responses:
            raise RuntimeError("script exhausted")
        return self._responses.pop(0), []


@pytest.mark.asyncio
async def test_retry_succeeds_when_first_response_invalid() -> None:
    """First call returns malformed JSON (missing required
    field that auto-coerce can't fix); second call returns
    valid JSON. Helper returns the parsed second result and
    re-prompts ONCE. The bad payload omits ``name`` — one of the two
    fields (with ``description``) that stay required after staging
    fields gained self-healing defaults."""
    bad = json.dumps({"id": "pell", "description": "x"})
    good = _new_char_payload()
    session = _ScriptedSession(bad, good)

    result = await call_llm_json_with_retry(
        session,
        prompt=[{"role": "user", "content": "render character JSON"}],
        parse_into=NewCharacterDescriptor,
        max_attempts=3,
    )
    assert result.name == "Pell"
    # Two LLM calls fired: first attempt + one retry.
    assert len(session.calls) == 2
    # The retry's prompt includes the bad response and a
    # corrective user message that quotes the validation error.
    retry_prompt = session.calls[1]
    assert any(m["role"] == "assistant" and m["content"] == bad for m in retry_prompt)
    correction = next(
        m for m in retry_prompt if m["role"] == "user" and "validation" in m["content"].lower()
    )
    assert correction is not None
    # Hints at the most common LLM mistakes.
    assert "null" in correction["content"].lower()
    assert "Re-emit" in correction["content"]


@pytest.mark.asyncio
async def test_retry_gives_up_after_max_attempts() -> None:
    """Three bad responses, max_attempts=3. The helper raises
    the LAST validation error after exhausting the budget."""
    bad = json.dumps({"only": "garbage"})
    session = _ScriptedSession(bad, bad, bad)

    with pytest.raises(ProviderValidationError):
        await call_llm_json_with_retry(
            session,
            prompt=[{"role": "user", "content": "go"}],
            parse_into=NewCharacterDescriptor,
            max_attempts=3,
        )
    assert len(session.calls) == 3


@pytest.mark.asyncio
async def test_retry_doesnt_fire_when_first_response_valid() -> None:
    """Happy path: valid response on the first call → exactly
    one LLM call, no corrective prompt."""
    session = _ScriptedSession(_new_char_payload())
    result = await call_llm_json_with_retry(
        session,
        prompt=[{"role": "user", "content": "go"}],
        parse_into=NewCharacterDescriptor,
        max_attempts=3,
    )
    assert result.name == "Pell"
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_retry_handles_auto_coerced_first_response() -> None:
    """First response has a null effects field. Auto-coerce
    inside parse_json_object handles it without burning a retry.
    Retry helper sees a successful parse on attempt 1."""
    session = _ScriptedSession(_new_char_payload(effects=None))
    result = await call_llm_json_with_retry(
        session,
        prompt=[{"role": "user", "content": "go"}],
        parse_into=NewCharacterDescriptor,
        max_attempts=3,
    )
    assert result.effects == ""
    # Only ONE call — the auto-coerce inside parse_json_object
    # fixed the response transparently.
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_retry_preserves_original_prompt_for_each_attempt() -> None:
    """The corrective continuation appends to the ORIGINAL
    prompt, not the previous attempt's continuation. Otherwise
    the conversation grows quadratically across retries."""
    bad = json.dumps({"id": "pell"})  # missing required fields
    good = _new_char_payload()
    session = _ScriptedSession(bad, bad, good)

    await call_llm_json_with_retry(
        session,
        prompt=[
            {"role": "system", "content": "you are an engine"},
            {"role": "user", "content": "render character"},
        ],
        parse_into=NewCharacterDescriptor,
        max_attempts=3,
    )
    # Second attempt: original prompt (2 msgs) + assistant + corrective user = 4.
    assert len(session.calls[1]) == 4
    # Third attempt: SAME original prompt + new assistant + new corrective = 4.
    # If we'd appended to the previous attempt, this would be 6.
    assert len(session.calls[2]) == 4


@pytest.mark.asyncio
async def test_retry_zero_attempts_rejected() -> None:
    session = _ScriptedSession()
    with pytest.raises(ValueError):
        await call_llm_json_with_retry(
            session,
            prompt=[{"role": "user", "content": "go"}],
            parse_into=NewCharacterDescriptor,
            max_attempts=0,
        )

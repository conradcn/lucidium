"""Coverage for the JSON-repair helper that salvages
LLM-truncated payloads in ``parse_json_object``.

The summarizer LLM call frequently ends up at the max_tokens
ceiling on long playthroughs — the response cuts off mid-string,
mid-pair, or mid-array. Without repair, the entire summarizer
pass is lost (assessment / facts / profile additions all gone).
The repair logic salvages whatever was complete BEFORE the cut
point so the partial pass still updates state.
"""

from __future__ import annotations

import json

import pytest

from lucidium.orchestration.responses import _repair_truncated_json


def _repair_and_parse(broken: str) -> object:
    """Run the repair, parse the result, and return the dict.
    Fails the test if either step fails."""
    repaired = _repair_truncated_json(broken)
    assert repaired is not None, f"repair returned None for: {broken!r}"
    return json.loads(repaired)


def test_unterminated_string_in_value() -> None:
    """Most common LLM truncation: cut landed inside a string
    value. The repair drops the partial pair, keeping the
    fields completed earlier."""
    broken = '{"a": 1, "b": "hello world this is a long stri'
    parsed = _repair_and_parse(broken)
    assert parsed == {"a": 1}


def test_trailing_comma_after_complete_value() -> None:
    """Cut landed right after a comma — drop the trailing
    comma and close the brace."""
    broken = '{"a": 1, "b": 2,'
    parsed = _repair_and_parse(broken)
    assert parsed == {"a": 1, "b": 2}


def test_dangling_key_no_value() -> None:
    """``"key":`` with no value — drop back to the previous
    comma so we keep the well-formed pairs."""
    broken = '{"a": 1, "b":'
    parsed = _repair_and_parse(broken)
    assert parsed == {"a": 1}


def test_nested_array_truncation() -> None:
    """Truncation inside a nested array. Outer object's
    completed entries survive; the partial array is dropped
    along with its key."""
    broken = '{"facts": [1, 2, 3,'
    parsed = _repair_and_parse(broken)
    # Either {"facts": [1, 2, 3]} or {} are acceptable salvage
    # shapes; whichever the implementation picks, the parse
    # must succeed and produce a dict.
    assert isinstance(parsed, dict)


def test_real_summarizer_payload_truncation() -> None:
    """A representative summarizer truncation — the model was
    in the middle of the third character's facts list when
    the cut landed. Earlier-completed fields (assessment,
    direction_signal, first two characters' facts) should
    survive."""
    broken = (
        '{"summarizer_assessment": "The crew has just discovered the '
        'truth.", "direction_signal": "stay_focused", '
        '"new_facts_by_character": '
        '{"alice": [{"id": "f1", "text": "Alice is the captain."}], '
        '"bob": [{"id": "f2", "text": "Bob betrayed the crew."}], '
        '"carol": [{"id": "f3", "text": "Carol is in the middle of saying som'
    )
    parsed = _repair_and_parse(broken)
    assert isinstance(parsed, dict)
    # The first two completed top-level fields must survive.
    assert parsed["summarizer_assessment"] == "The crew has just discovered the truth."
    assert parsed["direction_signal"] == "stay_focused"


def test_unterminated_string_with_escape() -> None:
    """Cut landed mid-string AFTER an escape character — the
    escape state must be tracked correctly so the repair
    doesn't get confused about whether we're still in the
    string."""
    broken = '{"a": 1, "b": "she said \\"hello'
    parsed = _repair_and_parse(broken)
    assert parsed == {"a": 1}


def test_fully_valid_passes_through_unchanged() -> None:
    """Already-valid JSON: repair returns the input verbatim
    so the caller's primary parse path runs identically."""
    valid = '{"a": 1, "b": [2, 3]}'
    repaired = _repair_truncated_json(valid)
    # repaired may equal the input or be None (no work to do);
    # either way the caller's primary json.loads succeeds first
    # and never reaches the repair branch.
    if repaired is not None:
        assert json.loads(repaired) == {"a": 1, "b": [2, 3]}


def test_no_structure_returns_none() -> None:
    """Input that never opened a brace / bracket / saw a comma
    is unrepairable — refuse rather than fabricate."""
    assert _repair_truncated_json('"unterminated string with') is None
    assert _repair_truncated_json("garbage tokens 123") is None


@pytest.mark.parametrize(
    "broken,expected_keys",
    [
        ('{"x": "a", "y": "b', ["x"]),
        ('{"x": "a", "y": [1, 2,', ["x"]),
        ('{"x": [1, 2, 3]', ["x"]),
        ('{"x": {"nested": "value", "broken": "tru', ["x"]),
    ],
)
def test_parametrized_truncations(broken: str, expected_keys: list[str]) -> None:
    """Parametrized smoke set for common truncation shapes —
    the repair must produce a parseable dict whose top-level
    keys contain at least the expected ones."""
    parsed = _repair_and_parse(broken)
    assert isinstance(parsed, dict)
    for key in expected_keys:
        assert key in parsed, f"expected key {key!r} after repairing {broken!r}, got {parsed!r}"


# ---- missing-colon repair (separate from truncation) ---------------


def test_missing_colon_before_array_value() -> None:
    """Real failure shape captured from a summarizer call: the
    LLM dropped both the closing quote and the colon between a
    key and its array value (``"dislikes [],`` instead of
    ``"dislikes": [],``). The whole brace structure is balanced,
    so the truncation repair finds no useful safe boundary —
    the missing-colon repair has to step in BEFORE truncation
    repair runs."""
    broken = (
        "{\n"
        '  "user_profile_additions": {\n'
        '    "likes": ["a", "b"],\n'
        '    "dislikes [],\n'
        '    "notes": []\n'
        "  }\n"
        "}"
    )
    parsed = _repair_and_parse(broken)
    assert parsed["user_profile_additions"]["dislikes"] == []
    assert parsed["user_profile_additions"]["notes"] == []
    assert parsed["user_profile_additions"]["likes"] == ["a", "b"]


def test_missing_colon_before_object_value() -> None:
    """Same shape with ``{`` as the value's opener instead of
    ``[`` — the LLM dropped the close-quote and colon and the
    next token is an object."""
    broken = '{"meta {"author": "x"}, "ok": true}'
    parsed = _repair_and_parse(broken)
    assert parsed["meta"] == {"author": "x"}
    assert parsed["ok"] is True


def test_missing_colon_does_not_corrupt_legit_string_values() -> None:
    """A legitimately quoted value that happens to contain
    bracket characters (e.g. ``"items [a, b]"``) must NOT be
    rewritten — the missing-colon heuristic only fires when
    the suspected key is followed by an OPENING bracket
    BEFORE any closing quote."""
    valid = '{"label": "items [a, b]", "count": 2}'
    # Already valid — repair (if invoked) must not break it.
    repaired = _repair_truncated_json(valid)
    if repaired is not None:
        assert json.loads(repaired) == {"label": "items [a, b]", "count": 2}


# ---- single-element-array wrapper repair ---------------------------


def test_parse_unwraps_single_element_array_for_object_schema() -> None:
    """Real failure shape captured from a Surprise Me LLM call: the
    model wrapped its single object in a ``[...]`` array, e.g.
    ``[{"setting": "...", "genre": "...", ...}]``. Pydantic v2
    rejects with ``Input should be a valid dictionary``. The parser
    should auto-unwrap a one-element array when the target schema
    is an object model."""
    from pydantic import BaseModel

    from lucidium.orchestration.responses import parse_json_object

    class Scenario(BaseModel):
        setting: str
        genre: str

    raw = '[{"setting": "harbor", "genre": "Noir"}]'
    parsed = parse_json_object(raw, Scenario)
    assert parsed.setting == "harbor"
    assert parsed.genre == "Noir"


def test_parse_does_not_unwrap_when_schema_is_a_list() -> None:
    """If the schema legitimately expects a list, don't unwrap.
    This guards against breaking the existing array-typed payloads."""
    from pydantic import BaseModel

    from lucidium.orchestration.responses import parse_json_object

    class WrappedList(BaseModel):
        items: list[str]

    # Object payload with a list field — must validate normally.
    raw = '{"items": ["a", "b"]}'
    parsed = parse_json_object(raw, WrappedList)
    assert parsed.items == ["a", "b"]


def test_parse_does_not_unwrap_multi_element_arrays() -> None:
    """A 2-element array isn't an LLM wrapping mistake — refuse to
    pick one and let the validation error surface as before."""
    import pytest as _pytest
    from pydantic import BaseModel

    from lucidium.api.errors import ProviderValidationError
    from lucidium.orchestration.responses import parse_json_object

    class Scenario(BaseModel):
        setting: str

    raw = '[{"setting": "a"}, {"setting": "b"}]'
    with _pytest.raises(ProviderValidationError):
        parse_json_object(raw, Scenario)


# ---- missing opening brace ----------------------------------------


def test_repair_wraps_object_body_missing_opening_brace() -> None:
    """Real failure shape captured from a summarizer call: the LLM
    emitted a fenced ```...``` block whose body was the object's
    contents only — opening ``{`` was omitted, closing ``}`` was
    present. After fence-strip, json.loads fails with
    ``Extra data`` at the first key. Repair should detect the
    asymmetric-brace shape and wrap with the missing opener."""
    broken = """  "summarizer_assessment": "story not yet begun.",
  "direction_signal": "stay_focused",
  "current_stage_id": "stage-arrival",
  "characters_to_offstage": []
}"""
    parsed = _repair_and_parse(broken)
    assert parsed["summarizer_assessment"] == "story not yet begun."
    assert parsed["direction_signal"] == "stay_focused"
    assert parsed["current_stage_id"] == "stage-arrival"
    assert parsed["characters_to_offstage"] == []


def test_repair_does_not_wrap_already_balanced_object() -> None:
    """Idempotency: if the input already has matching braces, the
    wrap heuristic must NOT add a redundant outer brace."""
    valid = '{"a": 1, "b": [2, 3]}'
    repaired = _repair_truncated_json(valid)
    if repaired is not None:
        assert json.loads(repaired) == {"a": 1, "b": [2, 3]}


# ---- control characters inside strings ---------------------------


def test_parse_tolerates_raw_newlines_inside_string_values() -> None:
    """Real failure shape: surprise_me LLM emitted a JSON object
    whose string values contained literal ``\\n`` / ``\\t`` /
    other ASCII control bytes (0x00-0x1F) instead of the
    json-required escape sequences. Python's ``json.loads`` with
    the default strict=True rejects with ``Invalid control
    character at: line N column M``. parse_json_object should
    fall back to strict=False so the object still parses."""
    from pydantic import BaseModel

    from lucidium.orchestration.responses import parse_json_object

    class Scenario(BaseModel):
        setting: str
        name: str

    # Raw newline inside the setting string — exactly what was
    # captured tonight from a surprise_me run that bricked the
    # whole new-game flow because the handler raised here.
    raw = '{\n  "setting": "A foggy harbor\nwith dim lanterns.",\n  "name": "Wren"\n}'
    parsed = parse_json_object(raw, Scenario)
    assert parsed.name == "Wren"
    assert "foggy harbor" in parsed.setting

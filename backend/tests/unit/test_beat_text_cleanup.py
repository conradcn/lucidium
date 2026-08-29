"""Speaker-slug stripping for LLM-emitted beat text.

The storyteller occasionally leaks a character id slug into the
narrative text (``"dr-mira-sol: She steps into the lamplight..."``)
when it should have set ``speaker_id`` to that id and left the
prose alone. ``strip_speaker_slug`` runs at the handler boundary
to clean that artifact up before the beat ships to the player.

These tests pin both directions:

  * Slugs of every common shape get stripped — bare ``foo-bar:``,
    multi-segment ``dr-mira-sol:``, with digits, with underscores.
  * Real prose that LOOKS slug-adjacent doesn't get stripped —
    capitalised colon prefixes, single-word colons (``"darkness:
    ..."``), quoted dialog, sentences with mid-string colons.
"""

from __future__ import annotations

import pytest

from lucidium.orchestration.beat_text_cleanup import strip_speaker_slug

# ----- positive cases (slug should be stripped) ----------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "dr-mira-sol: She steps into the lamplight, hands raised.",
            "She steps into the lamplight, hands raised.",
        ),
        (
            'mira-quill: "You\'re late."',
            '"You\'re late."',
        ),
        (
            "harbor-master: The man at the dock waves you over.",
            "The man at the dock waves you over.",
        ),
        # Multi-segment slug with digits — happens when the LLM
        # generates a numbered NPC id.
        (
            "guard-2: He doesn't move.",
            "He doesn't move.",
        ),
        # Multi-segment slug with underscores AND dashes.
        (
            'doc_v2-iris: "The instruments don\'t lie."',
            '"The instruments don\'t lie."',
        ),
    ],
)
def test_slug_is_stripped(raw: str, expected: str) -> None:
    assert strip_speaker_slug(raw) == expected


# ----- negative cases (real prose, must not be stripped) -------------------


@pytest.mark.parametrize(
    "text",
    [
        # Capitalised colon prefix — real narrator beat.
        "Note: the door wasn't locked when you came in.",
        # Quoted dialog — slug pattern impossible inside quotes.
        '"Mira-Quill, perhaps?" she suggests, half-smiling.',
        # Single-word colon prefix (no dash) — common in narration.
        "darkness: it pressed in around the lantern's halo.",
        # The colon mid-prose — must not be touched.
        "She read the inscription: a single word, in archaic script.",
        # Empty.
        "",
        # Looks slug-like but is uppercase — assume real prose
        # (e.g. a chapter heading or stylised line).
        "Mira-Quill: stares at the ceiling.",
        # Slug with no whitespace after colon — likely a code
        # reference or url fragment, not a speaker prefix.
        "key-value:42 — the count drifts up each turn.",
    ],
)
def test_real_prose_is_not_stripped(text: str) -> None:
    assert strip_speaker_slug(text) == text


def test_only_first_slug_is_stripped() -> None:
    """If the LLM somehow emits two stacked slugs, only the
    leading one comes off — the second is far less likely to be
    an artifact and more likely intentional prose / dialogue tag."""
    raw = "dr-mira-sol: alex-stone: She turns away."
    assert strip_speaker_slug(raw) == "alex-stone: She turns away."


def test_strip_is_idempotent() -> None:
    """Running the cleaner twice on already-clean text is a no-op."""
    cleaned = "She steps into the lamplight."
    assert strip_speaker_slug(strip_speaker_slug(cleaned)) == cleaned


def test_handles_none_safely() -> None:
    """Empty string in / empty string out — used by callers that
    pass ``beat.text`` which is occasionally empty (Continue beats
    with no narration)."""
    assert strip_speaker_slug("") == ""

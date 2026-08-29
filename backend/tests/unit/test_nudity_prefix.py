"""Coverage for the nudity-detection helper that picks the
clothing-state prefix at the front of every character SDXL prompt.

Two distinct bug shapes get pinned here:

  * ``"clothed"`` was being added when the outfit was effectively
    nude in natural language ("nothing", "no clothing", "bare",
    "unclad", "wearing nothing"). The original regex only caught
    explicit ``\\bnude\\b`` / ``\\bnaked\\b`` and missed the
    storyteller's natural phrasing — Pony XL would then ignore the
    nudity intent and render a clothed body. Pattern set is now
    broadened.
  * Empty / missing outfit used to return ``"clothed"`` —
    ALSO wrong. When the field hasn't been populated yet (mid-
    introduction NPCs, character_changes still in flight), we
    don't actually know the clothing state, and a leading
    ``clothed`` tag is an unjustified claim that biases the body
    prompt. The function now returns ``""`` for empty input.
"""

from __future__ import annotations

import pytest

from lucidium.orchestration.prompts.image_prompts import _nudity_prefix


@pytest.mark.parametrize(
    "outfit, expected",
    [
        # Bare-word forms — must all map to "nude".
        ("nude", "nude"),
        ("Nude", "nude"),
        ("NUDE", "nude"),
        ("naked", "nude"),
        ("nothing", "nude"),
        ("none", "nude"),
        ("no clothing", "nude"),
        ("no clothes", "nude"),
        # Compound forms.
        ("fully nude", "nude"),
        ("stark naked", "nude"),
        ("wearing nothing", "nude"),
        ("bare", "nude"),
        ("unclad", "nude"),
        ("unclothed", "nude"),
        ("undressed", "nude"),
        ("topless", "nude"),
        ("bottomless", "nude"),
        ("bare-chested", "nude"),
        ("fully exposed", "nude"),
        ("in her skin", "nude"),
        ("in nothing but skin", "nude"),
        # Trailing punctuation / surrounding whitespace shouldn't matter.
        ("  nude  ", "nude"),
        ("(nude)", "nude"),
        ("nude.", "nude"),
    ],
)
def test_nudity_phrasings_resolve_to_nude(outfit: str, expected: str) -> None:
    assert _nudity_prefix(outfit) == expected


@pytest.mark.parametrize(
    "outfit",
    [
        "barefoot",
        "bare hands",
        "bare knuckles",
        "barebones tunic",  # bare- prefix on a clothing word
    ],
)
def test_bare_compounds_are_not_misread_as_nude(outfit: str) -> None:
    """``bare`` is a real word that often appears in fully-clothed
    contexts. The negative-lookahead in the regex must keep
    "barefoot" / "bare hands" / "barebones" out of the nudity
    bucket."""
    assert _nudity_prefix(outfit) == "clothed"


@pytest.mark.parametrize(
    "outfit",
    [
        "bikini",
        "lingerie",
        "underwear",
        "thong",
        "swimsuit",
        "bra and panties",
        "sheer dress",
        "transparent gown",
        "negligee",
        "nothing but a towel",
    ],
)
def test_partial_nudity_phrasings_resolve_to_mostly_nude(outfit: str) -> None:
    assert _nudity_prefix(outfit) == "mostly nude"


@pytest.mark.parametrize(
    "outfit",
    [
        "a long charcoal coat",
        "wool dress with lace collar",
        "leather jacket and jeans",
        "kimono",
        "plate armour",
        "noodle-stall apron",
    ],
)
def test_clothed_outfits_resolve_to_clothed(outfit: str) -> None:
    assert _nudity_prefix(outfit) == "clothed"


@pytest.mark.parametrize("outfit", ["", None, "   ", "\t\n "])
def test_empty_outfit_returns_empty_string(outfit) -> None:
    """Empty / whitespace-only outfit MUST NOT default to
    ``"clothed"``. The prior fallback baked an unjustified
    clothing claim into the prompt for every NPC introduced
    without an outfit field — visible as the engine rendering
    a coat over a not-yet-described character."""
    assert _nudity_prefix(outfit) == ""

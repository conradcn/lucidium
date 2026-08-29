"""Verify the inputs to ``_portrait_prompt_hash`` actually
distinguish renders that should differ.

User concern (paraphrased): "pose may not change prompt hash for
a cache hit". The hash dedupes image renders by prompt content —
if pose isn't in the hash, two characters with different poses
collapse to the same on-disk PNG and the second never re-renders.

Spoiler: the hash IS sensitive to pose, expression, outfit,
effects, and lighting. The only collapse case is when two poses
share the same 4-word prefix (``_trim_pose``'s cap) — those
DELIBERATELY produce the same hash because the trimmed prompt
they'd send to SDXL is identical anyway.
"""

from __future__ import annotations

from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.world import WorldState
from lucidium.orchestration.assets import _portrait_prompt_hash


def _make_character(**overrides: object) -> Character:
    base: dict[str, object] = {
        "id": "char-1",
        "is_player": False,
        "name": "Test",
        "description": "A test character.",
        "gender": "female",
        "age": 30,
        "ethnicity": "white",
        "skin": "pale",
        "hair_color": "brown",
        "hairstyle": "long",
        "eye_color": "green",
        "build": "lean",
        "bust": "B",
        "outfit": "charcoal trench coat",
        "pose": "standing tall",
        "expression": "neutral",
        "effects": "",
        "seed": 12345,
        "kind": CharacterKind.human,
    }
    base.update(overrides)
    return Character(**base)


def _make_world() -> WorldState:
    return WorldState(
        game_name="Test",
        setting="A test setting",
        genre="Mystery",
        visual_style="ink-noir",
    )


def test_portrait_hash_changes_with_pose() -> None:
    """Different ``pose`` values that don't share a 4-word prefix
    must produce DIFFERENT hashes — otherwise a pose change
    silently serves a cached image with the wrong gesture."""
    world = _make_world()
    a = _make_character(pose="standing tall")
    b = _make_character(pose="kneeling, hands on knees")
    assert _portrait_prompt_hash(world, a) != _portrait_prompt_hash(world, b)


def test_portrait_hash_changes_with_outfit() -> None:
    world = _make_world()
    a = _make_character(outfit="charcoal trench coat")
    b = _make_character(outfit="rust linen tunic, tan boots")
    assert _portrait_prompt_hash(world, a) != _portrait_prompt_hash(world, b)


def test_portrait_hash_changes_with_expression() -> None:
    world = _make_world()
    a = _make_character(expression="neutral")
    b = _make_character(expression="furious")
    assert _portrait_prompt_hash(world, a) != _portrait_prompt_hash(world, b)


def test_portrait_hash_changes_with_effects() -> None:
    world = _make_world()
    a = _make_character(effects="")
    b = _make_character(effects="blood on left sleeve")
    assert _portrait_prompt_hash(world, a) != _portrait_prompt_hash(world, b)


def test_portrait_hash_changes_with_lighting() -> None:
    """Lighting comes from the active environment, not the
    character — pinned in the hash so a character standing in a
    blue-hour street and the same character in a candlelit room
    get separate cached renders."""
    world = _make_world()
    char = _make_character()
    a = _portrait_prompt_hash(world, char, lighting="harsh midday sun")
    b = _portrait_prompt_hash(world, char, lighting="candlelit gloom, warm amber")
    assert a != b


def test_portrait_hash_changes_with_seed() -> None:
    """Different seeds → different cached files, even with
    identical attributes. Without this, hitting "rerender" on
    the same character would serve the existing PNG."""
    world = _make_world()
    a = _make_character(seed=12345)
    b = _make_character(seed=99999)
    assert _portrait_prompt_hash(world, a) != _portrait_prompt_hash(world, b)


def test_portrait_hash_is_stable_across_calls() -> None:
    """Same inputs → same hash. Cache hit relies on this; if the
    hash had any non-determinism the dedup would always miss."""
    world = _make_world()
    char = _make_character()
    h1 = _portrait_prompt_hash(world, char, lighting="dim candlelight")
    h2 = _portrait_prompt_hash(world, char, lighting="dim candlelight")
    assert h1 == h2


def test_portrait_hash_collapses_poses_sharing_4_word_prefix() -> None:
    """Documented behavior, NOT a regression: ``_trim_pose`` caps
    pose at 4 words, so two long poses whose first 4 words match
    serialize to the same trimmed prompt and therefore the same
    hash. The render they'd produce is also identical (same SDXL
    input), so the cache HIT is correct.

    If a future caller wants per-clause precision, they should
    raise the cap in ``_trim_pose`` rather than rely on hash
    sensitivity to LLM word-order quirks."""
    world = _make_world()
    a = _make_character(pose="kneeling beside the altar, hands clasped")
    b = _make_character(pose="kneeling beside the altar, head bowed")
    assert _portrait_prompt_hash(world, a) == _portrait_prompt_hash(world, b)

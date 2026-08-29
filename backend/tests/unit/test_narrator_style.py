"""Narrator linguistic style is a per-installation Settings
field that the storyteller's prompt embeds verbatim. The
default ``conversational`` produces plain modern English; the
player can flip it to ``literary``, ``hard-boiled noir``,
``pirate``, ``fairy tale``, etc. and the next beat's prose
shifts register accordingly. Pin three things:

  * Default value lands as ``conversational`` on a fresh save.
  * The storyteller prompt embeds the value verbatim.
  * Choice options / structured fields are explicitly held OUT
    of the style restyle so the renderer's button labels and
    the engine's machine-readable fields don't get pirate-
    spoken.
"""

from __future__ import annotations

from lucidium.domain.character import Character
from lucidium.domain.settings import Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.prompts import text_gen


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink",
    )


def _player() -> Character:
    return Character(
        is_player=True,
        name="Iris",
        description="archivist",
        gender="female",
        age=28,
        ethnicity="local",
        skin="fair",
        hair_color="dark",
        hairstyle="braid",
        eye_color="hazel",
        build="slim",
        bust="small",
        outfit="charcoal coat",
        pose="standing",
        expression="alert",
        seed=1,
    )


def test_default_narrator_style_is_conversational() -> None:
    s = Settings()
    assert s.narrator_style == "conversational"


def test_text_gen_embeds_narrator_style_verbatim() -> None:
    msgs = text_gen.build(
        world=_world(),
        history=[],
        on_stage={},
        off_stage={"p": _player()},
        chosen_option_text=None,
        narrator_style="hard-boiled noir",
    )
    text = msgs[-1]["content"]
    assert "NARRATOR STYLE" in text
    assert "'hard-boiled noir'" in text


def test_text_gen_falls_back_to_conversational_for_empty_style() -> None:
    """A whitespace-only or empty narrator_style must fall back
    to ``conversational`` rather than emit ``''`` to the LLM."""
    msgs = text_gen.build(
        world=_world(),
        history=[],
        on_stage={},
        off_stage={"p": _player()},
        chosen_option_text=None,
        narrator_style="   ",
    )
    text = msgs[-1]["content"]
    assert "'conversational'" in text


def test_narrator_style_does_not_apply_to_structured_fields() -> None:
    """The rule must explicitly call out that option text / NPC
    names / structured fields stay neutral. Without that fence
    the LLM happily pirate-speaks the option buttons too,
    which is unreadable."""
    msgs = text_gen.build(
        world=_world(),
        history=[],
        on_stage={},
        off_stage={"p": _player()},
        chosen_option_text=None,
        narrator_style="pirate",
    )
    text = msgs[-1]["content"]
    # The fence: "PROSE ONLY" + named exclusions.
    assert "PROSE ONLY" in text
    assert "option text" in text.lower()
    assert "names" in text.lower()
    # Pose / expression / outfit attribute fields stay neutral —
    # their existing render-friendly rules already make them
    # machine-readable, but pin the explicit exclusion so a
    # future refactor of the narrator rule doesn't accidentally
    # give the LLM permission to pirate-speak the outfit.
    assert "outfit" in text.lower()


def test_default_text_gen_call_uses_conversational() -> None:
    """The text_gen build defaults narrator_style to
    conversational when the caller doesn't pass one — covers
    the legacy callsites that haven't been updated to thread
    the setting through yet."""
    msgs = text_gen.build(
        world=_world(),
        history=[],
        on_stage={},
        off_stage={"p": _player()},
        chosen_option_text=None,
    )
    text = msgs[-1]["content"]
    assert "'conversational'" in text

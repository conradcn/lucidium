"""Pin: every prompt that asks the LLM for an ``outfit`` value
also tells it to include a colour word for each garment.

Without an explicit colour the diffusion model picks one at
random per render — the same character's coat flickers between
black, beige, and dark red across consecutive portraits because
``trench coat`` alone gives the model no anchor. The text_gen
storyteller prompt and the new-game world_init / side-character
prompts all need the same guidance; pinning the rule in tests
catches a future refactor that drops it from any one site.
"""

from __future__ import annotations

from lucidium.domain.character import Character
from lucidium.domain.world import WorldState
from lucidium.orchestration.prompts import interview, text_gen


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


def test_text_gen_prompt_requires_outfit_colour() -> None:
    msgs = text_gen.build(
        world=_world(),
        history=[],
        on_stage={},
        off_stage={},
        chosen_option_text=None,
    )
    user = msgs[-1]["content"]
    assert "outfit lists COLOURED garments" in user, (
        "text_gen prompt must require explicit colour in every "
        "garment of the outfit field — without it character "
        "clothes flicker between renders."
    )
    # The example text covers both good and bad shapes so the
    # LLM has a concrete anchor.
    assert "charcoal trench coat" in user.lower()
    assert "cream wool scarf" in user.lower() or "rust linen" in user.lower()


def test_world_init_prompt_requires_outfit_colour() -> None:
    msgs = interview.world_init(
        setting="harbor",
        genre="Mystery",
        visual_style="ink",
        character_description="archivist",
        name="Iris",
        side_characters=[],
    )
    user = msgs[-1]["content"]
    assert "OUTFIT FORMAT" in user
    assert "colour" in user.lower()


def test_side_character_expansion_requires_outfit_colour() -> None:
    msgs = interview.side_character_expansion(
        one_line="the keeper of the lighthouse",
        setting="harbor",
    )
    user = msgs[-1]["content"]
    assert "colour" in user.lower(), (
        "side_character_expansion must require colour in the "
        "outfit so a player-typed NPC renders consistently."
    )

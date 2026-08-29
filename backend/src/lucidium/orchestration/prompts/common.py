"""Shared prompt fragments referenced by multiple builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ...domain.character import Character
from ...domain.dialog import DialogNode
from ...domain.environment import Environment

SYSTEM_GAME_DIRECTOR = (
    "You are the engine driving a single-player visual novel. You author "
    "tight, novelistic prose and return strictly-valid JSON when the user "
    "asks for structured output. Stay in scene; never break the fourth wall.\n\n"
    "PERSPECTIVE: narrate in SECOND PERSON, present tense. The player is "
    'the protagonist, addressed as "you". Replace third-person references '
    'to the player ("Iris turns", "she lifts the lantern", "his fingers") '
    'with second person ("you turn", "you lift the lantern", "your '
    'fingers"). Other named characters are still he/she/they; only the '
    "player switches. The player's name is set at character creation and "
    "may appear in dialogue when other characters address them, but "
    "narration never refers to the player by name."
)

FORBIDDEN_NAMES: tuple[str, ...] = (
    "Lyra",
    "Elara",
    "Aria",
    "Anya",
    "Kira",
    "Sera",
    "Seraphina",
    "Selene",
    "Lyric",
    "Lyrica",
    "Lyanna",
    "Nyx",
    "Eldrin",
    "Kael",
    "Kaelen",
    "Kaelan",
    "Kaelin",
    "Kaelyn",
    "Kaelith",
    "Lyrael",
    "Lirael",
    "Aiyana",
    "Auralei",
    "Aurelia",
    "Aurora",
    "Astra",
    "Astraea",
    "Caelum",
    "Caelan",
    "Cassia",
    "Cassian",
    "Cyra",
    "Eira",
    "Elowen",
    "Elowyn",
    "Evangeline",
    "Faelan",
    "Isolde",
    "Liora",
    "Lirien",
    "Mira",
    "Mirae",
    "Nyssa",
    "Orin",
    "Rhea",
    "Rowan",
    "Sable",
    "Saoirse",
    "Selia",
    "Silas",
    "Soren",
    "Talia",
    "Thalia",
    "Theron",
    "Vesper",
    "Wren",
    "Yara",
    "Zephyr",
    "Zara",
)

NAMING_TABOO_RULE = (
    "NAMING TABOO: do NOT use any of these overused LLM-default fantasy "
    "names for new characters — they are immediate immersion-breakers "
    "for readers familiar with AI-generated fiction: "
    + ", ".join(FORBIDDEN_NAMES)
    + ". This applies to first names, last names, and any compound "
    "(``Lyra Stonewright``, ``Mira Kell``, etc. all forbidden). "
    "Reach for names that fit the SETTING — a mining town wants "
    "blunt, lived-in names (Hale, Maud, Cully); a stellar empire "
    "wants futurist or transliterated names (Ix-7, Tashir Vodu); a "
    "Victorian mystery wants period names (Edmund, Cordelia). When "
    "in doubt, pick a name a reader could mistake for one from a "
    "real published novel in this genre, NOT one from a generic "
    "fantasy-name-generator. "
)


MATURE_CONTENT_DIRECTIVE = (
    "MATURE-CONTENT MODE: this story may include explicit sexuality, "
    "graphic violence, profanity, and morally complex situations as the "
    "narrative requires. Do not soften, redirect away from, or refuse "
    "content that fits the chosen genre and the player's expressed "
    "direction. Be honest and committed to the story; the player has "
    "opted in."
)


def system_prompt(*, mature_content: bool = False) -> str:
    """Compose the system prompt for any LLM call that drives narrative.

    The mature-content directive (FR-010b / FR-032a) is appended only
    when the setting is enabled. Default behavior is identical to the
    original director persona.
    """
    if mature_content:
        return f"{SYSTEM_GAME_DIRECTOR}\n\n{MATURE_CONTENT_DIRECTIVE}"
    return SYSTEM_GAME_DIRECTOR


def clamp_history(
    nodes: Iterable[DialogNode],
    *,
    char_budget: int,
    environments: Mapping[str, Environment] | None = None,
) -> str:
    """Render committed dialog history, oldest first, clamped to a char budget.

    Inserts ``[LOCATION: ...]`` scene-heading markers between beats
    whenever the location changes — without these, the LLM only
    sees speaker + text and loses track of where each beat
    happens. A character introduced in the harbor and then
    referenced in the noodle stall reads as a continuity drift to
    the model unless it can see the location transition. Markers
    use the environment's ``location_label`` (and ``lighting`` when
    present) so the model has the same shape of information the
    storyteller saw when writing the beat.

    ``environments`` maps environment id → ``Environment`` (typically
    ``game.environments``). When ``None`` or missing keys, the
    marker falls back to the bare location id, and lighting is
    omitted. Backwards compatible: a caller that doesn't pass
    environments gets the legacy speaker-only render.

    Trims from the front when the budget is exceeded; the most
    recent beats matter most for continuity.
    """
    rendered: list[str] = []
    last_location: str | None = None
    for node in nodes:
        if node.text is None:
            continue
        location = (node.location_id or "").strip() or None
        if location and location != last_location:
            label = location
            lighting = ""
            if environments is not None:
                env = environments.get(location)
                if env is not None:
                    label = env.location_label or location
                    lighting = (env.lighting or "").strip()
            if lighting:
                rendered.append(f"[LOCATION: {label} — {lighting}]")
            else:
                rendered.append(f"[LOCATION: {label}]")
            last_location = location
        speaker = node.speaker_id or "narrator"
        rendered.append(f"{speaker}: {node.text}")
    joined = "\n".join(rendered)
    if len(joined) <= char_budget:
        return joined
    excess = len(joined) - char_budget
    return joined[excess:].lstrip("\n")


def render_character_full(character: Character) -> str:
    """Render an on-stage character for the LLM prompt.

    The leading "CANON ATTRIBUTES" label tells the model these
    descriptors are FIXED — when the character is mentioned in
    narration or dialogue tags, the prose must agree with these
    values. Drifting an eye color or outfit is a continuity error.

    Nonhuman characters route through a separate block: the
    structured human-anatomy lines (gender / ethnicity / hair /
    eyes / build / bust) are skipped in favour of the freeform
    ``physical_description`` field, since those fields don't
    apply to dragons / robots / spirits / animals. Outfit / pose /
    expression / effects still apply — even a brass automaton
    can have a slumped pose and a thoughtful expression.
    """
    from ...domain.character import CharacterKind

    effects = character.effects or "(none)"
    decals = character.decals or "(none)"
    facts = "; ".join(f.text for f in character.facts) or "(none)"
    # Pronouns line: only emitted when the player or LLM
    # populated the field. Empty falls through to the prior
    # behaviour where the storyteller picks pronouns from
    # gender / context. Explicit pronouns are non-negotiable
    # canon — the storyteller must use them verbatim.
    pronouns_line = (
        f"  pronouns (USE THESE — do not substitute): {character.pronouns}\n"
        if (character.pronouns or "").strip()
        else ""
    )
    if character.kind == CharacterKind.nonhuman:
        physical = character.physical_description or "(none)"
        return (
            f"CANON ATTRIBUTES for {character.name} (id: {character.id}) — "
            f"NONHUMAN. DO NOT CONTRADICT in narration or description:\n"
            f"  description: {character.description}\n"
            f"{pronouns_line}"
            f"  physical form: {physical}\n"
            f"  age: {character.age}\n"
            f"  outfit / adornment: {character.outfit or '(none)'}\n"
            f"  current pose: {character.pose} | current expression: {character.expression}\n"
            f"  effects (damage/wear/etc): {effects}\n"
            f"  decals (permanent markings: tattoos/scars/brands): {decals}\n"
            f"  established facts: {facts}"
        )
    return (
        f"CANON ATTRIBUTES for {character.name} (id: {character.id}) — "
        f"DO NOT CONTRADICT in narration or description:\n"
        f"  description: {character.description}\n"
        f"{pronouns_line}"
        f"  gender: {character.gender} | age: {character.age} | ethnicity: {character.ethnicity}\n"
        f"  skin: {character.skin} | build: {character.build} | bust: {character.bust}\n"
        f"  hair: {character.hair_color} {character.hairstyle} | eyes: {character.eye_color}\n"
        f"  outfit: {character.outfit}\n"
        f"  current pose: {character.pose} | current expression: {character.expression}\n"
        f"  effects (cuts/bruises/soot/etc): {effects}\n"
        f"  decals (permanent markings: tattoos/scars/brands): {decals}\n"
        f"  established facts: {facts}"
    )


def render_character_brief(character: Character) -> str:
    return f"{character.id}: {character.name} — {character.description}"

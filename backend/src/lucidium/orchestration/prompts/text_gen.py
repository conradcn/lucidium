"""Dialog-node text generation prompt builder."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ...domain.character import Character
from ...domain.dialog import DialogNode
from ...domain.environment import Environment
from ...domain.world import DirectionSignal, WorldState
from .common import (
    NAMING_TABOO_RULE,
    clamp_history,
    render_character_brief,
    render_character_full,
    system_prompt,
)
from .sample_names import render_sample_names

_RESPONSE_SHAPE = """{
  "beats": [
    {
      "text": str,                                     // ONE line, no \\n
      "speaker_id": str|null,
      "entering_character_ids": [str],
      "leaving_character_ids": [str],
      "new_characters": [                              // see NEW CHARACTER RULE
        {
          "id": str,
          "name": str, "description": str,
          // ``kind``: "human" (default) for people; "nonhuman" for
          // creatures, robots, spirits, animals, monsters, etc.
          // See NONHUMAN RULE below for which fields each kind uses.
          "kind": "human"|"nonhuman",
          // Used ONLY when kind="nonhuman"; ignored for humans.
          // Freeform 6-15 word physical form description.
          "physical_description": str,
          // Used ONLY when kind="human"; leave empty strings when
          // kind="nonhuman". age applies to BOTH (a 600-year-old
          // dragon is fine).
          "gender": str, "age": int,
          "ethnicity": str, "skin": str,
          "hair_color": str, "hairstyle": str,
          "eye_color": str,
          "build": str, "bust": str,
          // Apply to BOTH kinds. ``outfit`` covers adornment /
          // armour / wrappings on a nonhuman, or be empty when
          // they wear nothing (a wild beast, a spirit). ``pose``,
          // ``expression``, ``effects``, ``decals`` apply to any
          // subject. ``decals`` = permanent skin markings (tattoos,
          // scars, birthmarks, brands); "" when none.
          "outfit": str, "pose": str, "expression": str,
          "effects": str, "decals": str
        }
      ],
      "location_id": str|null,
      "location_prompt": str|null,
      "location_lighting": str,                        // "" or "few words" — see LIGHTING RULE
      "character_changes": [{"character_id": str, "field": str, "new_value": str}],
      "music_change": str|null                         // see MUSIC RULE — null = no change
    },
    ...
  ],
  "options": [{"id": str, "text": str}]                // empty = Continue
}"""


_TEXT_FORMAT_RULES = (
    "BEAT STRUCTURE: ``beats`` is an ordered list of 2–5 narrative beats. "
    "Each beat becomes its own dialog node and is shown to the player one "
    "at a time. The player presses Continue to walk between beats; only "
    "the LAST beat's ``options`` are presented as choices. Each beat's "
    "``text`` is a SINGLE line (one sentence, occasionally two) with no "
    "carriage returns and no literal \\n inside it. Across beats you may "
    "change speaker, character pose/expression/outfit (via "
    "character_changes), and location. Apply entering/leaving and "
    "character_changes ON the beat where they happen — not all front-loaded "
    "on the first beat. "
    "BEAT MOMENTUM: every beat advances state — a discovery, a reaction, "
    "a tonal shift, a new fact, an entrance. A beat that is purely "
    "atmospheric (mood without movement) is wasted. If a beat doesn't "
    "deliver a small shift, cut it or merge it with a neighbour. The "
    "active plot threads listed above must visibly progress within the "
    "chain — readers should be able to point to which beat advanced "
    "which thread. "
    "BEAT VARIETY: across the chain, beats target different sense channels "
    "(sight, sound, scent, touch, weight) and different action modes "
    "(observation, dialogue, gesture, internal pivot). A chain whose beats "
    "all describe what the eye sees reads flat; mix in a heard sound, a "
    "smelled change, a felt shift in air pressure, a spoken line. "
    "BRANCH PERSISTENCE: when this chain is responding to a player "
    "choice, the choice's consequence echoes for 2+ beats — not just "
    "the first beat. The player's choice changes what someone says, "
    "what is revealed, AND what tone the chain ends on. A choice "
    "whose effect collapses by beat 2 has the same narrative weight "
    "as no choice at all. "
    "CONSEQUENCE VECTOR: a player choice can shift any of: TRUST "
    "(another character's regard for the player), EXPOSURE (what "
    "secret information is now known to whom), POSITION (who is "
    "where, with what), or TIME (what window of opportunity has "
    "closed or opened). Pick the dimension that fits, and let the "
    "chain dramatize THAT dimension's shift across multiple beats. "
    "OPTION DIVERSITY: when you provide options, at least one must "
    "be a wait/observe/refuse/sidestep — not all action verbs. The "
    "non-action option must lead to a different next beat (more "
    "context, a character revealing more), not the same beat reworded. "
    "INTENSITY PRESERVATION: after an intense beat (confrontation, "
    "reveal, time-pressured choice, character in danger), the next "
    "beat must be reaction, escalation, or resolution — never a "
    "tonal pivot to weather, atmosphere, or unrelated setting. "
    "Tension wasted is the chain working against itself. "
    "PROSE STYLE: write CONCRETE, DESCRIPTIVE sentences. Plain "
    "physical action and direct dialogue carry the chain. NO "
    "philosophising about boundaries, frequencies, thresholds, "
    "signal-vs-noise, distance-vs-closeness, what-is-said-vs-what-is-"
    "meant, or any abstract opposition. NO sentences that compare an "
    "ineffable thing to another ineffable thing. NO 'the silence "
    "between heartbeats', 'the frequency at which her smile "
    "registered', 'the threshold where the room ended and the man "
    "began', 'the boundary between memory and what was'. If a "
    "sentence wouldn't survive being read aloud at a kitchen table, "
    "rewrite it. Tell the reader what is on screen and what someone "
    "said or did. Save abstraction for one specific marquee line "
    "(see below). "
    "Forbidden: 'soft click of boots', "
    "'scrape of stone', 'shadow falling', 'salt on the wind', "
    "'silence stretched', 'the air felt heavy' — stock VN atmospherics "
    "that diffuse rather than focus. Also forbidden: any sentence "
    "whose subject is an abstraction ('the boundary', 'the silence', "
    "'the moment', 'the frequency'). Aim for ONE photographable "
    "moment, not five competent ones, and no philosophising. "
    + NAMING_TABOO_RULE
    + "NEW CHARACTER RULE: When a beat introduces or addresses a person OR "
    "named non-human entity (creature, robot, spirit, animal, monster) "
    "who is NOT the PLAYER and NOT in ON-STAGE or OFF-STAGE CHARACTERS, "
    "you MUST emit a full descriptor in ``new_characters`` AND list the "
    "same id in ``entering_character_ids``. Pick a stable, lowercase, "
    "dash-separated id (e.g. ``mira-quill`` or ``ash-the-fox``) and "
    "reuse it for the rest of the chain. The new character's NAME MUST "
    "DIFFER FROM THE PLAYER's name — never introduce an NPC who shares "
    "the player's name; pick something distinct. Never reference a "
    "character by name in narration without either (a) their id already "
    "in PLAYER/ON-STAGE/OFF-STAGE, or (b) a fresh ``new_characters`` "
    "entry — orphan references are dropped silently. "
    'NONHUMAN RULE: ``kind`` defaults to ``"human"``. Switch to '
    '``"nonhuman"`` ONLY for entities that fundamentally do NOT '
    "have human anatomy — a dragon, a four-legged beast, an "
    "automaton built like a clockwork ostrich, a smoke spirit "
    "with no fixed form, a tentacled eldritch shape, a swarm of "
    "ravens speaking with one voice. For these, set ``kind`` to "
    '``"nonhuman"`` and put their physical form in '
    "``physical_description`` (6-15 words covering silhouette, "
    "materials, colours, distinguishing features; e.g. "
    '``"scale-armoured serpent, copper hide, lantern-yellow '
    'eyes, four spiral horns, a brand on its forehead"``). The '
    "human-anatomy fields (``gender``, ``ethnicity``, ``skin``, "
    "``hair_color``, ``hairstyle``, ``eye_color``, ``build``, "
    "``bust``) MUST be empty strings on nonhuman entries — those "
    "fields force a person-shaped image render. "
    "HUMANOID is HUMAN: any character with a person-shaped body "
    'stays ``"human"``, INCLUDING beings the storyteller would '
    "describe with capital-letter awe — Goddesses, gods, angels, "
    "demons, elves, vampires, ghosts in person form, "
    "shapeshifters in their humanoid shape, witches, faeries, "
    "demigods, immortals, demonkin, half-elves, beast-folk that "
    "stand on two legs, AI avatars rendered as people. They have "
    "skin, hair, eyes, a recognisable gender presentation. Fill "
    "the human anatomy fields normally; lean into the supernatural "
    "via ``description``, ``effects`` (a glow, gold-flecked eyes, "
    "smoke trailing from fingertips, wings as outfit detail), and "
    '``visual_style``-driven rendering. Use ``"nonhuman"`` for '
    "shapes a person watching the scene would point at and call "
    "``it`` rather than ``her``/``him``/``they``. A celestial "
    "being who looks like a tall figure with golden eyes is "
    '``"human"``; an orb of light with no anatomy is '
    '``"nonhuman"``. ``age``, ``outfit`` (adornment / armour / '
    "wrappings, or empty), ``pose``, ``expression``, ``effects`` "
    "apply to both kinds. "
    "ON-STAGE PERSISTENCE: characters listed in ON-STAGE CHARACTERS "
    "stay there until they have a NARRATIVE reason to leave the "
    "scene (they walk out, fall unconscious, dismiss themselves, or "
    "die). NEVER add a character to ``leaving_character_ids`` just "
    "because they're not the speaker on this beat or the camera "
    "isn't on them — silence is normal in a multi-character scene. "
    "An on-stage character keeps existing in the player's spatial "
    "model; dropping them mid-conversation reads as a continuity "
    "error and the engine filters them out of the next prompt's "
    "ON-STAGE list, making them effectively invisible. "
    "HARD CONSTRAINT — leaving_character_ids on a given beat MUST be "
    "non-empty ONLY IF that beat's own ``text`` field explicitly "
    "narrates the named character departing the scene (walking out, "
    "losing consciousness, dying, being teleported away, etc.). The "
    "departure must be a *literal action in the prose on this beat*: "
    "if a reader can't point to the sentence in ``text`` that moves "
    "the character out, the id does NOT belong in "
    "leaving_character_ids. Examples that DO qualify: ``Hale slipped "
    "through the side door, leaving Iris alone.``; ``Mira collapsed, "
    "eyes glazing over.``; ``The spirit dissolved into the smoke.`` "
    "Examples that DO NOT qualify (do NOT mark these as leaving): "
    "the camera follows another character into a new room while the "
    "first remains where they were; the speaker turns to face only "
    "one of two listeners; a quiet beat with one speaker and a "
    "watching listener; a chain-internal pause where no one is "
    "narrated to move. When in doubt, leave the field empty — the "
    "next beat can still mark them leaving if narration actually "
    "moves them out then. This rule applies INDEPENDENTLY to every "
    "beat in the chain: do not drop a character on beat 3 just "
    "because they were quiet on beats 1 and 2. "
    "Same constraint applies to entering: the speaker may already "
    "be on stage; you don't need to re-add them. "
    "AUDIBLE MEANS VISIBLE: a beat's ``speaker_id`` must name a "
    "character who is physically present in the scene on that beat. "
    "If they weren't on stage before, the beat's own ``text`` must "
    "bring them in and their id belongs in "
    "``entering_character_ids``. NEVER give a line to someone who is "
    "elsewhere — a voice from another room, a remembered line, or a "
    "character the story has already walked off. Write those as "
    "narration (``speaker_id: null``) instead. Corollary for exits: "
    "a character who speaks their own departure line ('Fine — I'm "
    "leaving.') is still on screen for the beat they say it on; put "
    "their id in that beat's ``leaving_character_ids`` and the "
    "engine keeps them visible for exactly that line, then retires "
    "them. "
    "CHARACTER STATE TRACKING: when a character's pose, expression, "
    "outfit, or effects change within the chain (they sit, they "
    "smile, they shed a coat, their posture stiffens, they bleed "
    "from a fresh cut), emit a ``character_changes`` entry on the "
    "SAME beat where the change happens. The next prompt will see "
    "the updated value as canon — without the character_changes "
    "entry, the engine still believes the character is in their "
    "old state, and the next chain may contradict your narration. "
    "STATE STABILITY (continuity matters more than detail variance): "
    "DO NOT change pose / outfit / expression every beat for tonal "
    "variety. Mutable attributes are CONTINUITY anchors — the player "
    "renders a portrait from them, and a portrait that flips outfit "
    "or pose on every beat reads as a different character every "
    "turn. Only emit a character_change when the BEAT NARRATIVELY "
    "REQUIRES IT — they actually changed clothes, they actually "
    "moved from kneeling to standing, their actual emotional state "
    "shifted because of something on screen. A subtle mood drift "
    "across two beats is NOT a state change; the previous expression "
    "still holds. When in doubt, leave the field alone. "
    "POV: When the pose is sexually explicit, use POV missionary / doggy / or similar to describe it."
    "EFFECTS: a character's ``effects`` field carries visible "
    "physical effects that aren't part of their wardrobe — a fresh "
    "cut, dried blood, soot from a fire, soaked clothing from rain, "
    "dishevelled hair after a fight, restraints. Use ``effects`` "
    "(NOT outfit) when an injury, mess, or wear-and-tear appears on "
    "the character. Outfit stays stable ('charcoal trench coat'); "
    "effects carries the perishable layer ('blood soaking left "
    "sleeve, knuckles split'). Clear effects to the empty string "
    "via character_changes when they wash, change clothes, or "
    "otherwise reset their physical state. ≤ 6 words; same trim "
    "cap as outfit / pose / expression. "
    "DECALS: a character's ``decals`` field carries PERMANENT skin "
    "markings — tattoos, scars, birthmarks, brands, ritual paint. "
    "Unlike ``effects`` (transient injuries/mess) these PERSIST across "
    "outfit / pose / expression changes, so set them ONCE when the "
    "character is introduced (if they have any distinctive markings) "
    "and then leave them — do NOT re-emit or churn them each beat. "
    "Empty string when the character has none. ≤ 6 words unless the "
    "appearance override below lifts the cap. "
    "LIGHTING: when a beat establishes or changes scene lighting, set "
    "``location_lighting`` to a SHORT phrase (3–6 words) describing "
    "the light: 'candlelit gloom, warm amber', 'harsh midday sun, "
    "stark shadows', 'blue hour through tall windows', 'flickering "
    "neon, rain-slicked'. Update it whenever the light meaningfully "
    "shifts (lamps go out, dawn breaks, the player moves to a new "
    "room with different light). Empty string means 'no change since "
    "the last beat in this location'. The renderer feeds this into "
    "BOTH the background prompt AND every on-stage character "
    "portrait, so the faces match the room. "
    "RENDER-FRIENDLY VALUES: ``pose``, ``expression``, ``outfit``, "
    "and ``description`` are concatenated INTO the image-generation "
    "prompt, weighted, verbatim. Hard rule: each of these fields "
    "MUST be ≤ 6 words. The image model uses CLIP, which silently "
    "truncates anything past 77 tokens — every extra word in one "
    "field pushes another field out of the prompt entirely. Pose "
    "describes ONLY the character's body position — limbs, "
    "torso, head angle, hands. Nothing in the surrounding scene, "
    "nothing they're holding or interacting with, no other "
    "characters. Good: ``kneeling, hands on knees``, ``arms "
    "crossed, head tilted``, ``leaning forward``, ``seated, "
    "back straight``, ``hands clasped behind back``. Bad: "
    "``kneeling beside the altar`` (mentions altar — scene "
    "leak), ``holding a brass key`` (object — belongs in the "
    "narrative ``text``, not pose), ``standing in the doorway`` "
    "(location), ``arms around her sister`` (involves another "
    "character). Items the character is holding go in ``effects`` "
    "if they're part of the body image (a clutched ledger, a "
    "drawn dagger). Other characters and props live in ``text``. "
    "Expression says what the face is doing "
    "('grim half-smile'), "
    "outfit lists COLOURED garments. EVERY garment in ``outfit`` "
    "MUST carry a colour or material that implies one — without "
    "it the diffusion model picks a colour at random per render "
    "and the same character flickers between a black and a beige "
    "coat across consecutive portraits. Good: ``charcoal trench "
    "coat, cream wool scarf``, ``rust linen tunic, tan leather "
    "boots``, ``ivory silk blouse, navy trousers``. Bad (no "
    "colour): ``trench coat, scarf``, ``tunic, boots``. "
    "Material-only is OK when the material implies a colour "
    "(``brass plate armour``, ``copper-scaled hauberk``); generic "
    "materials like ``leather`` or ``cotton`` still need an "
    "explicit colour word. AVOID motion verbs in OUTFIT "
    "(``flying``, ``trailing``, ``bunched``) — the renderer "
    "treats them as scene cues and drops the garments. Keep "
    "motion in the narration ``text`` field, never in outfit. "
    "NO 'eyes squeezed shut, lips pressed into a bloodless "
    "line'-style multi-clause descriptions — pick the ONE most "
    "photographable detail. "
    "CHARACTER DESCRIPTION: ``description`` for a new_character is a "
    "SHORT phrase — role + one distinguishing trait. 'tired harbour "
    "constable, kindly', 'wiry caravan scout, pale scar across the "
    "bridge of the nose', 'priest's daughter, quick-tempered'. NOT a "
    "thesis on who they are. NO 'the kind of man who lives in the "
    "spaces between certainty and...', no 'a frequency the room "
    "tunes itself to', no compounding metaphors. If you can't say it "
    "in fifteen words, rewrite it."
)


# Appended to the format rules ONLY when a long-context image model
# (Qwen-Image / Flux / Z-Image) is active. Those models read long
# natural-language prompts rather than CLIP tag-soup, so the ≤ 6-word
# caps above (which exist purely to survive CLIP's 77-token window)
# are counter-productive — they throw away exactly the detail these
# models render well. This block LIFTS the cap for the render-facing
# fields and asks for explicit, concrete descriptions instead.
_DETAILED_APPEARANCE_OVERRIDE = (
    "APPEARANCE OVERRIDE (long-context image model active): the active "
    "renderer reads long natural-language descriptions, NOT CLIP tag-soup. "
    "The ≤ 6-word caps above on ``outfit``, ``pose``, ``expression`` and "
    "``decals`` are LIFTED. Write each as a vivid, explicit, concrete "
    "description — a full phrase or 1–2 sentences: "
    "``outfit`` names every garment with its colour and material, plus cut, "
    "fit, layering, and condition; "
    "``pose`` describes the whole body position — limbs, torso, head angle, "
    "hands, weight distribution — in specific terms (still ONLY the body: no "
    "scene, props, or other characters); "
    "``expression`` describes the exact facial state; "
    "``decals`` describes each permanent marking's design, colour, and "
    "placement. Colours stay explicit (garment-colour stability still "
    "matters) and pose still avoids scene / prop / other-character leakage. "
    "This override changes ONLY the length + detail of these fields; every "
    "other rule above still applies. "
)


def build(
    *,
    world: WorldState,
    history: Iterable[DialogNode],
    on_stage: Mapping[str, Character],
    off_stage: Mapping[str, Character],
    chosen_option_text: str | None,
    free_text_input: str | None = None,
    mature_content: bool = False,
    user_likes: list[str] | None = None,
    user_dislikes: list[str] | None = None,
    user_notes: list[str] | None = None,
    environments: Mapping[str, Environment] | None = None,
    music_enabled: bool = False,
    current_music_prompt: str = "",
    narrator_style: str = "conversational",
    detailed_appearance: bool = False,
) -> list[dict[str, str]]:
    """Build the prompt for the next dialog beat.

    Order is stable so providers' prompt-cache prefixes match across calls
    (Constitution III, Efficiency).

    ``environments`` plumbs the game's environment map through
    ``clamp_history`` so location transitions appear as
    ``[LOCATION: ...]`` markers between beats. Without it the
    history is just speaker + text, and the LLM loses scene-
    heading context across long playthroughs.
    """
    history_chars = world.prompt_history_clamp_chars
    rendered_history = clamp_history(
        history,
        char_budget=history_chars,
        environments=environments,
    )
    on_stage_text = "\n\n".join(render_character_full(c) for c in on_stage.values()) or "(empty)"
    off_stage_text = "\n".join(render_character_brief(c) for c in off_stage.values()) or "(empty)"

    # Single explicit "PLAYER CHARACTER" line so the LLM has the
    # player's id+name in front of it when applying the NEW CHARACTER
    # RULE ("name must differ from the player's"). Without this the
    # prompt only ever sees the player as one entry in off_stage —
    # easy to overlook.
    player = next(
        (c for c in (*on_stage.values(), *off_stage.values()) if c.is_player),
        None,
    )
    player_text = (
        f"id={player.id} name={player.name} — never introduce an NPC "
        f"with this name; never narrate the player by name."
        if player is not None
        else "(unknown — be cautious about name collisions)"
    )

    steering = ""
    if world.player_intent.direction_signal == DirectionSignal.stay_focused:
        steering = (
            "STEERING: Stay focused on the active plot threads; do not introduce side threads."
        )
    elif world.player_intent.direction_signal == DirectionSignal.reintroduce_thread:
        steering = "STEERING: This is a lull. Reintroduce one dropped plot thread naturally."

    if world.active_plot_threads:
        threads_text = "\n".join(f"  - {t.title}: {t.summary}" for t in world.active_plot_threads)
    else:
        threads_text = "  (none — establish or surface one this turn)"
    # CURRENT STAGE: the slice of the broader plot outline the
    # summarizer has selected as active. The storyteller never sees
    # the full multi-stage plan — that would tempt it to dump beats
    # from late stages into the current scene. The summarizer holds
    # the map; this prompt only carries the bit of map under the
    # player's feet right now.
    stage = world.current_stage()
    if stage is not None:
        stage_text = f"{stage.title} — goal: {stage.goal}\n  {stage.summary}"
    else:
        stage_text = "(unspecified — improvise; the summarizer will pick up the thread)"
    # Cross-save player profile, learned by the summarizer over many
    # sessions and editable by the player. Lean on this for tone /
    # pacing / style choices — never call it out in narration.
    profile_lines: list[str] = []
    for label, items in (
        ("LIKES", user_likes or []),
        ("DISLIKES", user_dislikes or []),
        ("NOTES", user_notes or []),
    ):
        if items:
            profile_lines.append(f"  {label}: " + "; ".join(items))
    profile_block = (
        "\n".join(profile_lines)
        if profile_lines
        else "  (no profile yet — observe and let the summarizer learn)"
    )

    # Music rule: STATIC instructions only, no embedded
    # ``current_music_prompt``. The current track's prompt is
    # emitted as its own line in the volatile section below; that
    # keeps this rule block byte-stable across turns so the
    # provider-side prompt cache can match it (was previously
    # rebuilt per turn whenever the storyteller swapped the
    # background score).
    music_rule = (
        "MUSIC RULE: A background instrumental track plays under the "
        "game (its current prompt is given below as ``CURRENT MUSIC "
        "PROMPT``). When the scene's tonal direction shifts hard "
        "enough that the existing track stops fitting (action breaks "
        "out, the tone tips into dread, a tender beat arrives, the "
        "player moves to a categorically different setting), emit a "
        "``music_change`` on the beat where the shift happens. "
        "Your music_change string is fed STRAIGHT INTO the music "
        "diffusion model — there is no second LM rewriting it. "
        "Be specific and rich: lead with the emotional MOOD "
        "(melancholic, tense, hopeful, dread-soaked, weightless, "
        "tender, etc. — score what the player FEELS, not what "
        "they see). Then layer on tempo (e.g. ``slow``, ``60 "
        "BPM``, ``mid-tempo``), a key/mode hint when distinctive "
        "(e.g. ``minor key``, ``F# major``, ``dorian``), time "
        "signature when it matters (e.g. ``3/4 time``, ``5/4``), "
        "the specific instrument layers (sustained strings, "
        "soft piano, low drone, sub bass, plucked harp, choir "
        "pad, distant brass, tape hiss), and one or two mix / "
        "production descriptors (warm, dry, reverb-soaked, "
        "lo-fi, analog, digital, intimate). Do NOT describe "
        "the visual style or art direction; that's the "
        "renderer's job, not the music's. The track MUST be "
        "seamlessly loopable: include a ``seamless loop`` tag, "
        "favour sustained textures / drones / ostinati / "
        "static-mood beds, and avoid hard intros, cold "
        "endings, or build-and-release arcs that telegraph a "
        "finale. Comma-separated tags only — no full sentences. "
        "Aim for 8-15 tags. INSTRUMENTAL (no vocals, no "
        "lyrics). Example: ``melancholic, contemplative, "
        "sustained strings, soft piano, low drone, slow, 60 "
        "BPM, minor key, 4/4 time, warm reverb, intimate, "
        "seamless loop``. Most beats do NOT warrant a change — "
        "leave music_change null. Frequent music swaps are "
        "jarring; aim for 0-1 changes per chain, never more."
        if music_enabled
        else "MUSIC RULE: Background music generation is DISABLED. "
        "Always set music_change to null. Engine-side it is "
        "ignored, but emitting non-null values still wastes "
        "output tokens."
    )

    # Narrator style — fed verbatim into the prompt as a
    # stylistic directive on the prose. The default
    # ``conversational`` keeps the storyteller in plain modern
    # English, which the LLM produces by default; an explicit
    # value (``literary``, ``hard-boiled noir``, ``pirate``,
    # ``fairy tale``, etc.) turns the dial. Only the narrative
    # ``text`` of each beat is affected — choice options, NPC
    # names, and structured fields stay neutral.
    style_text = (narrator_style or "").strip() or "conversational"
    narrator_rule = (
        "NARRATOR STYLE: write the narrative ``text`` of each beat in a "
        f"{style_text!r} register. Apply the style to PROSE ONLY — option "
        "text, character names, location ids, and the pose / expression / "
        "outfit attribute fields stay neutral and machine-readable. The "
        "style colours diction and rhythm; it does NOT change the BEAT "
        "STRUCTURE rules, the NEW CHARACTER RULE, the LIGHTING rule, or any "
        "of the structured-field constraints above. ``conversational`` "
        "(the default) means plain modern English."
    )

    response_shape_block = (
        "Return JSON matching this exact shape (no commentary, no Markdown fence):\n"
        + _RESPONSE_SHAPE
        + "\n"
        + _TEXT_FORMAT_RULES
        + ("\n" + _DETAILED_APPEARANCE_OVERRIDE if detailed_appearance else "")
        + "\n"
        + narrator_rule
        + "\n"
        + music_rule
        + "\nRules: Provide 0–4 short options OR an empty list to mean Continue. "
        "When you DO provide options, each must lead to a different "
        "consequence — different next beats, different revealed facts, "
        "different relationships shifted. Three options that all end "
        "with the player examining the same object are decorative; "
        "options should diverge by intent (approach vs. observe vs. "
        "leave; trust vs. challenge vs. deflect; act now vs. wait vs. "
        "warn another). "
        "Only emit character_changes for characters currently or newly on stage. "
        "Set location_prompt only when the location actually changes."
    )

    # PROMPT ORDER (cache-friendliness):
    #
    # The user message is split into a STABLE preamble followed by a
    # VOLATILE state block. The provider-side prompt cache (DeepSeek's
    # automatic context cache, surfaced through OpenRouter for the
    # deepseek-* models) matches identical PREFIXES at byte
    # granularity — the cache walks left-to-right and gives up at the
    # first byte that diverges from the cached version.
    #
    # The earlier ordering put GAME header → CURRENT STAGE →
    # SUMMARIZER ASSESSMENT (changes every turn) right at the front,
    # so the cache prefix died at the assessment line and the big
    # static rule blocks at the END (RESPONSE_SHAPE / TEXT_FORMAT_RULES
    # / narrator_rule / music_rule / closing Rules block) couldn't
    # benefit. Reordering so all stable bytes lead lets the cache
    # cover the whole rule preamble — typically the longest section
    # of the prompt — and only the volatile tail has to be processed
    # fresh each turn. ``current_music_prompt`` was extracted out of
    # ``music_rule`` and into its own volatile line so the music
    # rule's instruction body stays byte-stable across changes to the
    # current track.
    stable_preamble = [
        # Engine-stable response shape and rules. Identical for every
        # storyteller call across the whole game (modulo the
        # narrator_style + music_enabled toggles, which the player
        # sets in Settings and rarely flips).
        response_shape_block,
        # Game-stable identity. ``world.game_name`` / setting /
        # genre / visual_style are written once at world_init and
        # never mutate during play.
        f"GAME: {world.game_name} | setting: {world.setting} | genre: {world.genre} | "
        f"visual style: {world.visual_style}",
        # Player identity is also game-stable. Lifted to the
        # preamble (was buried mid-message) so the NEW CHARACTER
        # RULE's "name must differ from the player's" clause has
        # immediate context.
        "PLAYER CHARACTER: " + player_text,
    ]

    volatile_state: list[str] = [
        f"CURRENT STAGE (write the next chain to advance THIS, not the whole game):\n  {stage_text}",
        "ACTIVE PLOT THREADS (must visibly progress in this chain):\n" + threads_text,
        steering,
        # Current music prompt — extracted from music_rule so the
        # rule itself stays cacheable. Only emitted when music is
        # enabled (otherwise the music rule already says generation
        # is disabled and there's no track to reference).
        f"CURRENT MUSIC PROMPT: {current_music_prompt or '(none)'}" if music_enabled else "",
        # Cross-save player profile — changes when the summarizer
        # learns new tags. Below we explain how the storyteller
        # should USE the profile; the data block follows.
        "PLAYER PROFILE — use this to STEER what you author. Pick "
        "scene types, character archetypes, and pacing that fit the "
        "LIKES; avoid the DISLIKES. Aim to surface something this "
        "player will find genuinely interesting, not just acceptable:\n" + profile_block,
        f"SUMMARIZER ASSESSMENT: {world.summarizer_assessment or '(none)'}",
        "ON-STAGE CHARACTERS:\n" + on_stage_text,
        "OFF-STAGE CHARACTERS (base description only):\n" + off_stage_text,
        # Random sample of real names refreshed each turn — sits
        # JUST BEFORE the history block so the LLM has a fresh slate
        # of grounded names to reach for when ``new_characters`` is
        # populated in the response. Pairs with NAMING_TABOO_RULE
        # (the "don't") by giving an explicit "do" list. Volatile by
        # design (different sample per turn); placed here so it joins
        # the already-volatile tail rather than poisoning the stable
        # preamble's prefix cache.
        render_sample_names(),
        "RECENT HISTORY (oldest first):\n" + (rendered_history or "(empty)"),
    ]

    if free_text_input is not None:
        volatile_state.append(
            "PLAYER WROTE: " + free_text_input + "\n"
            "Respond as a natural continuation. Treat this input as canonical."
        )
    elif chosen_option_text is not None:
        volatile_state.append(
            "PLAYER JUST CHOSE (past tense — gesture has happened): " + chosen_option_text
        )
    else:
        volatile_state.append("PLAYER PRESSED CONTINUE.")

    user_prompt_parts = [*stable_preamble, *volatile_state]

    return [
        {"role": "system", "content": system_prompt(mature_content=mature_content)},
        {"role": "user", "content": "\n\n".join(p for p in user_prompt_parts if p)},
    ]

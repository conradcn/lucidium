"""Stable Diffusion / ComfyUI prompt builders for portraits and backgrounds."""

from __future__ import annotations

import re

from ...domain.character import Character, CharacterKind, age_band
from ...domain.environment import Environment
from ...domain.world import WorldState

# Outfit substrings (case-insensitive, word-boundary anchored) that
# imply the character is nude or mostly-nude. SDXL will happily render
# clothing over a "naked" descriptor unless we lift it to the very
# front of the prompt with high attention weight — late tokens get
# drowned out by the wardrobe priors baked into the checkpoint.
#
# Two earlier bugs worth pinning here:
#
#   * The list used to fail-closed on natural-language nudity
#     phrasings the storyteller actually emits ("wearing nothing",
#     "bare", "unclad", "no clothes / no clothing"); the prompt then
#     defaulted to ``"clothed"`` and rendered the character clothed.
#     The patterns below cover those forms.
#   * ``_nudity_prefix("")`` used to return ``"clothed"``, which is
#     WORSE than empty — when we don't know, we shouldn't make a
#     claim. The function now returns ``""`` for empty / None.
_NUDE_PATTERNS: tuple[str, ...] = (
    r"\bnude\b",
    r"\bnaked\b",
    r"\btopless\b",
    r"\bbottomless\b",
    r"\bbare[- ]?(?:chested|breasted|breasts?|skinned)\b",
    # ``bare`` as a STANDALONE word — "she stands bare" / "bare
    # under the lantern light". Excludes "barefoot" and similar
    # via the negative-lookahead suffix list.
    r"\bbare\b(?!\s*(?:foot|hand|knuckle|hill|bone|tree|wire))",
    r"\bin (?:the )?(?:nude|nothing|nothing but skin)\b",
    r"\bwearing (?:nothing|no clothing|no clothes)\b",
    r"\bno (?:clothes|clothing|garments|covering)\b",
    r"\bunclothed\b",
    r"\bunclad\b",
    r"\bundressed\b",
    r"\bstark naked\b",
    r"\bbirthday suit\b",
    r"\bin (?:her|his|their) skin\b",
    r"\bfully exposed\b",
    r"\bcompletely exposed\b",
)
_NUDE_RE = re.compile("|".join(_NUDE_PATTERNS), re.IGNORECASE)

_MOSTLY_NUDE_PATTERNS: tuple[str, ...] = (
    r"\blingerie\b",
    r"\bunderwear\b",
    r"\bbra\b",
    r"\bpanties\b",
    r"\bthong\b",
    r"\bbikini\b",
    r"\bswimsuit\b",
    r"\bg[- ]?string\b",
    r"\bsheer\b",
    r"\bsee[- ]?through\b",
    r"\btransparent (?:dress|gown|top|robe)\b",
    r"\bnegligee\b",
    r"\bbabydoll\b",
    r"\btowel(?: only)?\b",
    r"\bnothing but a towel\b",
)
_MOSTLY_NUDE_RE = re.compile("|".join(_MOSTLY_NUDE_PATTERNS), re.IGNORECASE)


def is_nudity_outfit(outfit: str) -> bool:
    """Public predicate: would ``_nudity_prefix(outfit)`` emit a
    nudity tag (``"nude"`` or ``"mostly nude"``) for this outfit?

    Lifted out for the safety-side detection path that bumps an
    under-18 character's age before any SDXL prompt is built. Don't
    inline the prefix-substring check — keep this aligned with
    whatever the prefix function decides.
    """
    return _nudity_prefix(outfit) in {"nude", "mostly nude"}


def _nudity_prefix(outfit: str) -> str:
    """Return a plain clothing-state tag for the front of the prompt.

    Four shapes:
      * ``"nude"`` — outfit substring matched a nudity pattern.
      * ``"mostly nude"`` — outfit substring matched the partial-
        nudity pattern (lingerie, towel, bikini).
      * ``"clothed"`` — outfit is non-empty AND matched neither
        nude pattern. The leading ``clothed`` tag exists for SDXL
        checkpoints (Pony XL in particular) trained heavily on
        NSFW art, which silently slip into nudity unless told
        otherwise — even when the rest of the prompt names a coat
        / dress / armour. Non-Pony checkpoints render clothed
        bodies anyway, so the extra word is a no-op for them.
      * ``""`` — outfit is missing / empty / whitespace-only. We
        DON'T know the clothing state and shouldn't claim either
        way. Earlier this path returned ``"clothed"``, which was
        actively wrong on characters whose outfit field hadn't
        been populated yet (e.g. mid-introduction NPCs).

    No ``(text:weight)`` wrapper — plain bare token. Earlier
    ``(nude:1.5)`` shapes over-pushed in qualitative eval and
    distorted the body composition.
    """
    text = (outfit or "").strip()
    if not text:
        return ""
    # Bare-word equality short-circuit. Storytellers occasionally set
    # the whole outfit field to a single word like "nothing" / "none"
    # when the character is naked. Word-boundary regex won't catch
    # ``"nothing"`` alone (it'd false-positive on phrases like
    # "nothing of note") but exact-match is unambiguous.
    if text.lower() in {"nothing", "none", "no clothing", "no clothes"}:
        return "nude"
    if _NUDE_RE.search(text):
        return "nude"
    if _MOSTLY_NUDE_RE.search(text):
        return "mostly nude"
    return "clothed"


def _trim_attr(value: str, max_words: int = 8) -> str:
    """Cap an LLM-supplied attribute (pose / expression / outfit /
    description) to ``max_words``. The storyteller prompt asks for
    ≤ 6 words but LLMs occasionally ignore length rules and emit a
    long multi-clause phrase. Truncating here keeps the eventual
    image prompt under CLIP's 77-token window even when the
    storyteller misbehaves. Splits on whitespace, joins back, and
    drops any trailing punctuation introduced by the cut."""
    text = (value or "").strip()
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;.: ")


def _trim_pose(value: str) -> str:
    """Pose-specific trim. Cap tighter than the generic ``_trim_attr``
    (4 words instead of 8) because pose runs at 1.4 weight and a
    long verbose pose dominates SDXL's body composition — over-
    anchoring the gesture pushes outfit out of the actually-rendered
    region. Combined with the reordering below (outfit BEFORE pose
    in the prompt), this protects clothing against being lost to
    CLIP truncation OR over-weighted gesture detail."""
    return _trim_attr(value, max_words=4)


_SMILE_WORDS: tuple[str, ...] = (
    "smile",
    "smiling",
    "smirk",
    "smirking",
    "grin",
    "grinning",
    "laugh",
    "laughing",
    "happy",
    "joyful",
    "cheerful",
    "amused",
    "delighted",
    "beaming",
    "playful",
    "soft expression",
    "warm expression",
)


def _is_smile_expression(expression: str) -> bool:
    """``True`` when the expression text already contains a smile-
    related cue. Used to gate the anti-smile negative prompt: SDXL's
    prior for adult portraits is a smiling face, so unless the
    storyteller explicitly asked for one we shove smile tokens into
    the negative to suppress it."""
    text = (expression or "").lower()
    return any(word in text for word in _SMILE_WORDS)


def _normalize_visual_style(visual_style: str) -> str:
    """Strip whitespace; otherwise pass the visual style through
    UNCHANGED. Earlier this helper trimmed to the first comma chunk
    + a 6-word cap as defence against CLIP's 77-token window, but
    that stripped useful style guidance ("anime, vibrant palette,
    sharp linework, soft ambient occlusion" → just "anime") and
    the user noticed the rendered images had stopped reflecting
    the chosen style. With ``compel`` chunked encoding in the
    embedded backend and ComfyUI's native long-prompt handling,
    the style block can be the full string the world ships."""
    return (visual_style or "").strip()


def _human_identity_tag(age_text: str, gender: str) -> str:
    """``"thirty-year-old woman"`` → kept; ``"thirty-year-old "``
    or ``"-year-old woman"`` → drop the partial. Both halves have
    to be present for the tag to be meaningful."""
    age_text = age_text.strip()
    gender = (gender or "").strip()
    if not age_text or not gender:
        return ""
    return f"{age_text}-year-old {gender}"


def _hair_tag(hair_color: str, hairstyle: str) -> str:
    color = (hair_color or "").strip()
    style = (hairstyle or "").strip()
    if not color and not style:
        return ""
    return f"{(color + ' ' + style).strip()} hair"


def _body_tag(skin: str, build: str, bust: str) -> str:
    """Body block — only emit the components the LLM actually
    populated so we don't paste ``" skin,  build, bust"`` into
    the prompt for partially-described characters."""
    parts: list[str] = []
    skin = (skin or "").strip()
    build = (build or "").strip()
    bust = (bust or "").strip()
    if skin:
        parts.append(f"{skin} skin")
    if build:
        parts.append(f"{build} build")
    if bust:
        parts.append(f"{bust} bust")
    return ", ".join(parts)


def _portrait_prompt_nonhuman(
    *,
    character: Character,
    style: str,
    pose: str,
    outfit: str,
    expression: str,
    effects: str,
    decals: str = "",
    lighting: str,
) -> str:
    """Build the SDXL positive prompt for a NONHUMAN character.

    Differences from the human path:
      * No anatomy tags (gender / ethnicity / hair / eye / skin /
        build / bust). Those would either be empty or contradict
        the actual subject.
      * The freeform ``physical_description`` carries the visual
        identity. We trim it but cap less aggressively than the
        6-word attributes — a dragon's description legitimately
        wants 10-15 words to nail silhouette, hide, eyes, scale,
        distinguishing features, and any narrative damage.
      * The ``(human portrait)`` framing tags don't apply — many
        nonhuman subjects aren't humanoid at all. We swap in a
        species-agnostic full-frame instruction so SDXL still
        renders the WHOLE creature and not a head crop.
      * Nudity gating doesn't apply — a robot is not "naked",
        a smoke spirit doesn't wear clothes. Outfit only emits
        when present.
    """
    # Looser word cap on the freeform anatomy — see docstring.
    description = _trim_attr(character.physical_description, max_words=20)
    # Fallback to the narrative ``description`` when the
    # storyteller didn't fill ``physical_description``. Better to
    # hand SDXL the prose blurb than nothing at all.
    if not description:
        description = _trim_attr(character.description, max_words=20)
    parts = [
        # Framing FIRST — same reasoning as the human path
        # (early tokens weight higher in CLIP), but worded for
        # arbitrary subject geometry. The "single subject"
        # constraint matters more here because nonhuman scenes
        # are higher variance and SDXL likes to add scenery.
        "(full subject in frame, head to feet/base:1.5)",
        "(centered, entire form fully visible:1.4)",
        "single subject, no companions",
        # Subject identity — the freeform description rides
        # at moderate weight so the storyteller's words stay
        # legible without crushing the rest of the prompt.
        f"({description}:1.3)" if description else "",
        # Expression / pose work for nonhuman subjects too —
        # a dragon can have a "wary" expression, an automaton
        # can have a "slumped" pose. Optional, gated on
        # non-empty content.
        expression,
        # Outfit before pose — same rationale as the human path:
        # the wardrobe tag is identity, ship it inside the early
        # weight window so it isn't dropped to a verbose pose or
        # CLIP truncation. Pose runs at the same 1.3 as outfit
        # (was once 1.4 here too) so neither dominates the body
        # composition.
        f"(wearing {outfit}:1.3)" if outfit else "",
        f"(pose: {pose}:1.3)" if pose else "",
        f"(effects: {effects}:1.2)" if effects else "",
        f"(markings: {decals}:1.1)" if decals else "",
        style,
        f"{lighting}" if lighting else "",
    ]
    return ", ".join(p for p in parts if p)


def _portrait_prompt_long_context(*, world: WorldState, character: Character, lighting: str) -> str:
    """Natural-language portrait prompt for long-context text encoders
    (Qwen-Image, Flux, Z-Image).

    These models read PROSE, not the Danbooru-style weighted tag soup
    SDXL/Pony expects. Two concrete consequences drive this builder:

      * ``(tag:1.3)`` attention-weight syntax is NOT parsed — Qwen /
        Flux feed it to a T5 / Qwen2.5-VL text encoder verbatim, so the
        parentheses and numbers pollute the description. We emit plain
        sentences with no weight wrappers.
      * There is no 77-token CLIP window, so the aggressive ``_trim_pose``
        (4 words) / ``_trim_attr`` (8 words) caps that protect the SDXL
        path actively HURT here — they throw away exactly the detail
        these models are good at. Pose and clothing are described in
        full, explicit sentences instead.

    Structure follows the published Qwen-Image / Flux prompting guides:
    state the subject first, then wardrobe and pose, then style and
    lighting; keep it concrete; skip quality-booster spam (``masterpiece,
    8k, best quality`` waste tokens on these models).
    """
    style = _normalize_visual_style(world.visual_style)
    clauses: list[str] = []

    if character.kind == CharacterKind.nonhuman:
        subject = (character.physical_description or character.description or "").strip()
        clauses.append(
            f"A full-body illustration of {subject}, shown in full from head to base."
            if subject
            else "A full-body illustration of the subject, shown in full."
        )
    else:
        identity = _human_identity_tag(age_band(character.age), character.gender)
        head = identity or "person"
        ethnicity = (character.ethnicity or "").strip()
        subject = f"{ethnicity} {head}".strip() if ethnicity else head
        clauses.append(f"A full-body portrait of a {subject}, shown from head to toe.")
        appearance: list[str] = []
        hair = _hair_tag(character.hair_color, character.hairstyle)
        if hair:
            appearance.append(hair)
        if (character.eye_color or "").strip():
            appearance.append(f"{character.eye_color.strip()} eyes")
        body = _body_tag(character.skin, character.build, character.bust)
        if body:
            appearance.append(body)
        if appearance:
            clauses.append("They have " + ", ".join(appearance) + ".")

    # Clothing — explicit and fully detailed (NOT trimmed). The nudity
    # tables still lead the decision so a naked descriptor wins over the
    # checkpoint's clothing prior, but phrased as plain language.
    outfit = (character.outfit or "").strip()
    nudity = _nudity_prefix(character.outfit)
    if nudity == "nude":
        clauses.append("They are completely nude, wearing no clothing at all.")
    elif nudity == "mostly nude" and outfit:
        clauses.append(f"They are wearing only {outfit}, described in detail and clearly visible.")
    elif outfit:
        clauses.append(
            f"They are dressed in {outfit}. The clothing is rendered in full "
            f"detail and clearly visible on the figure."
        )

    # Pose — explicit and fully detailed (NOT trimmed).
    pose = (character.pose or "").strip()
    if pose:
        clauses.append(
            f"Their pose: {pose}. The entire body stays visible in this pose, "
            f"framed from head to feet."
        )

    expression = (character.expression or "").strip()
    if expression:
        clauses.append(f"Their facial expression is {expression}.")

    decals = (character.decals or "").strip()
    if decals:
        clauses.append(f"Permanent skin markings: {decals}, clearly visible.")

    effects = (character.effects or "").strip()
    if effects:
        clauses.append(f"Visible on them: {effects}.")

    if style:
        clauses.append(f"Rendered in this art style: {style}.")
    if lighting:
        clauses.append(f"Lighting: {lighting}.")
    clauses.append("A single subject, no other people in the frame.")
    return " ".join(c for c in clauses if c)


def portrait_prompt(
    *,
    world: WorldState,
    character: Character,
    lighting: str = "",
    long_context: bool = False,
) -> str:
    """Build the positive image prompt for a character portrait.

    ``long_context=True`` routes to :func:`_portrait_prompt_long_context`
    (natural-language prose for Qwen / Flux / Z-Image). The default is the
    SDXL / Pony weighted-tag builder below.

    Two pipelines:

      * ``CharacterKind.human`` (default) — uses the structured
        anatomy fields (age band, gender, ethnicity, hair, eyes,
        skin, build, bust). Token order matters because SDXL's
        CLIP truncates past 77 tokens and weights earlier tokens
        higher; identity / expression / pose lands first so it
        survives the cut.
      * ``CharacterKind.nonhuman`` — substitutes the freeform
        ``physical_description`` for the anatomy block. Most
        nonhuman subjects (dragons, robots, ghosts, monsters,
        animals) sit far enough outside SDXL's "human portrait"
        prior that nailing them with structured tags is worse
        than handing the model the storyteller's own prose.
        The framing / pose / expression / outfit / effects
        scaffolding still applies — a brass automaton can have
        a ``thoughtful`` expression and a ``slumped`` pose just
        as a person can.
    """
    if long_context:
        return _portrait_prompt_long_context(world=world, character=character, lighting=lighting)
    style = _normalize_visual_style(world.visual_style)
    # Defensive cap on the LLM-supplied free-text fields so a
    # storyteller that ignores the ≤6-words rule can't push the
    # weighted tags into 30+ tokens each. Pose is capped tighter
    # (4 words vs 8) — a verbose pose at 1.4 weight steamrolls the
    # body composition and squeezes outfit out of the rendered
    # region; ``_trim_pose`` keeps the gesture concrete and short.
    pose = _trim_pose(character.pose)
    outfit = _trim_attr(character.outfit)
    expression = _trim_attr(character.expression)
    effects = _trim_attr(character.effects)
    decals = _trim_attr(character.decals)

    if character.kind == CharacterKind.nonhuman:
        return _portrait_prompt_nonhuman(
            character=character,
            style=style,
            pose=pose,
            outfit=outfit,
            expression=expression,
            effects=effects,
            decals=decals,
            lighting=lighting,
        )

    nudity = _nudity_prefix(character.outfit)
    parts = [
        # Nudity tag still leads — it has to win over the
        # checkpoint's clothing prior, which is the strongest
        # baked-in prior in SDXL. Empty string when clothed.
        nudity,
        # Framing tags MUST come before subject identity. SD weighs
        # earlier tokens more heavily, and the checkpoint's prior
        # for "human portrait" is overwhelmingly head-and-shoulders.
        # Three weighted reinforcements keep the body in the frame
        # even on a scene with strong face cues.
        "(full body shot from head to toe:1.5)",
        # Expression has to land in the MAIN prompt — the SDXL
        # prior for an adult portrait paints a smiling face during
        # the body pass, and a downstream FaceDetailer inpaint can
        # rarely override what the body composition already
        # committed to. NO weight multiplier: weighting expression
        # past 1.0 distorted the face geometry on multi-step SDXL
        # checkpoints (the model over-amplified the adjective and
        # the eyes/mouth deformed). The bare adjective at 1× is
        # enough — combined with the anti-smile negative below —
        # to override the smiling-face default.
        expression,
        # Subject identity. Cheap tags, high impact on character
        # consistency. Each tag is gated on the underlying field
        # being non-empty — nonhuman-branched characters now
        # default empty for these fields, but a save edited from
        # human → nonhuman could leave partial entries here, and
        # an empty value would emit stray fragments like
        # ``" hair"`` or ``"thirty-year-old "``.
        _human_identity_tag(age_band(character.age), character.gender),
        character.ethnicity,
        _hair_tag(character.hair_color, character.hairstyle),
        f"{character.eye_color} eyes" if character.eye_color else "",
        # OUTFIT BEFORE POSE. Two reasons: (a) outfit is identity
        # — the player's named character should be wearing what
        # the LLM said even if CLIP later truncates the tail, so
        # ship the wardrobe tag inside the high-impact early window;
        # (b) putting pose's heavy 1.3 weight first squeezed outfit
        # off the rendered figure ("clothing falls off the end").
        # Pose dropped to 1.3 (was 1.4) so it doesn't dominate the
        # body composition at the expense of the wardrobe.
        f"(wearing {outfit}:1.3)" if outfit else "",
        f"(pose: {pose}:1.3)" if pose else "",
        "wide framing, full-length figure, single subject",
        # Effects (cuts / bruises / blood / soot / soaked etc) sit
        # at moderate weight — strong enough to actually appear in
        # the render, weak enough not to distort body composition
        # or override the wardrobe. Empty string is a no-op so a
        # character without injuries doesn't get a (effects: :1.2)
        # tag with empty content.
        f"(effects: {effects}:1.2)" if effects else "",
        # Permanent skin markings (tattoos / scars / birthmarks). Modest
        # weight — present enough to render but not so strong it distorts
        # the body. Empty string is a no-op.
        f"(skin markings: {decals}:1.1)" if decals else "",
        # Build / bust / skin late — slow-moving identity bits that
        # the seed pins anyway.
        _body_tag(character.skin, character.build, character.bust),
        # Visual style passes through unchanged (compel handles
        # length on the embedded path; ComfyUI's CLIPTextEncode
        # chunks natively). Style mid-prompt rather than at the
        # very back so its contribution stays meaningful.
        style,
        # Lighting last — tints colour balance without overriding
        # identity. Mirrors the nonhuman path. Drives the portrait
        # prompt-hash apart for the same character standing in
        # different rooms; without this here, two beats whose only
        # difference was room lighting collapsed to the same
        # cached PNG and the second beat showed the first beat's
        # render. Empty string is a no-op.
        lighting if lighting else "",
    ]
    return ", ".join(p for p in parts if p)


def portrait_face_prompt(*, character: Character) -> str:
    """The face-detail refinement prompt fed to the FaceDetailer
    node in the ComfyUI workflow (via the ``wildcard`` slot — see
    workflows/character.json node 12). FaceDetailer uses this
    string for its face-only inpaint pass, so this is where the
    expression lives — keeping it out of the main prompt
    simplifies main-pass adherence (the model focuses on body
    composition instead of competing face / body instructions).

    Identity anchors (eye colour, age band, ethnicity, gender) are
    duplicated from the main prompt so the inpaint matches; without
    them the FaceDetailer can drift the face's identity slightly
    against the body's.

    Plain text — NO ``(text:weight)`` syntax. Impact-Pack's
    FaceDetailer feeds the wildcard string through its dynamic /
    wildcard processor BEFORE handing it to CLIPTextEncode, and
    that processor treats parentheses as wildcard branching syntax
    (``(option1|option2|option3)``). The colons inside our weight
    syntax (``(expression: ...:1.3)``) made the wildcard parser
    error or silently emit an empty wildcard substitution, which
    then bypassed the face-detail pass entirely — the user reported
    "FaceDetailer doesn't seem to be triggering". Plain comma-
    separated tags pass through cleanly.

    Embedded backends without a face-detail pass receive this
    string and append it to the positive prompt at low priority
    so expression-related tokens still influence the base render
    without dominating composition.
    """
    expression = _trim_attr(character.expression)
    if character.kind == CharacterKind.nonhuman:
        # FaceDetailer is fundamentally a HUMAN-face inpaint pass:
        # the bbox detector targets human face proportions and the
        # inpaint prior is heavily biased toward stock-photo human
        # features. Running it on a dragon's snout or a robot's
        # face plate routinely smears the geometry into something
        # vaguely human. For nonhuman subjects we emit a string
        # short enough that ComfyUI's wildcard parser stays happy
        # but light enough that even if FaceDetailer fires anyway
        # the inpaint barely shifts the original render.
        description = _trim_attr(character.physical_description, max_words=10)
        parts = [
            expression,
            description,
            "detailed, sharp focus",
        ]
        return ", ".join(p for p in parts if p)

    parts = [
        # Bare adjective at 1× weight. Earlier ``(angry:1.3)``
        # over-amplified the expression on the FaceDetailer pass
        # and distorted face geometry — eyes and mouth deformed
        # because the model treated the weight as "more of this
        # adjective" rather than "match this adjective". 1× is
        # enough because FaceDetailer's inpaint already focuses
        # the model's attention on the face region; weight stacks
        # on top of that focus.
        expression,
        "detailed face, sharp focus",
        f"{character.eye_color} eyes" if character.eye_color else "",
        _human_identity_with_ethnicity(
            age_band(character.age),
            character.gender,
            character.ethnicity,
        ),
    ]
    return ", ".join(p for p in parts if p)


def _human_identity_with_ethnicity(
    age_text: str,
    gender: str,
    ethnicity: str,
) -> str:
    """Same shape the original face prompt assembled, but only
    emits the bits actually present so partial / nonhuman edits
    don't leave stray punctuation."""
    base = _human_identity_tag(age_text, gender)
    ethnicity = (ethnicity or "").strip()
    if base and ethnicity:
        return f"{base}, {ethnicity}"
    return base or ethnicity


def portrait_negative_extras(*, character: Character) -> str:
    """Extras appended to the boilerplate negative prompt in the workflow.

    Anti-smile (HUMAN ONLY): SDXL checkpoints trained on stock
    photography paint a smile by default for adult portraits
    regardless of expression cues elsewhere in the prompt. When
    the character's actual expression isn't smile-related, push
    the smile vocabulary into the negative so the body pass
    doesn't render one. Skipped when the expression is empty (no
    opinion) or already smile-flavoured.

    Nonhuman characters skip the anti-smile block — a dragon
    isn't going to be misread as smiling and the negative
    vocabulary doesn't help SDXL render non-human geometry. We
    DO push a "human, person" negative for nonhuman subjects so
    the checkpoint's overwhelming person-portrait prior doesn't
    leak humanoid features into the render.
    """
    if character.kind == CharacterKind.nonhuman:
        return "human, person, humanoid face, anthropomorphic"
    expression = (character.expression or "").strip()
    if not expression or _is_smile_expression(expression):
        return "child, childlike, loli"
    return "smiling, smile, grinning, grin, happy expression, cheerful, child, childlike, loli"


def portrait_negative_prompt() -> str:
    return "low quality, blurry, deformed, extra limbs, watermark, text"


def background_prompt(
    *, world: WorldState, environment: Environment, long_context: bool = False
) -> str:
    """Build the positive image prompt for a scene background.

    Style now passes through unchanged (compel handles length on
    the embedded path; ComfyUI chunks natively). Style leads so
    the scene's visual treatment is committed to before the
    location detail tints it.

    ``long_context=True`` emits a plain-language scene description for
    Qwen / Flux / Z-Image instead of the comma-tag SDXL form — same
    reasoning as the portrait path (prose over weighted tags).
    """
    style = _normalize_visual_style(world.visual_style)
    if long_context:
        loc = (environment.location_label or "").strip()
        desc = (environment.prompt or "").strip()
        subject = desc or loc or "the scene"
        clauses = [f"A wide establishing shot of {subject}."]
        if loc and desc:
            clauses.append(f"Location: {loc}.")
        if (environment.lighting or "").strip():
            clauses.append(f"Lighting: {environment.lighting.strip()}.")
        if style:
            clauses.append(f"Rendered in this art style: {style}.")
        clauses.append("No people in the frame.")
        return " ".join(c for c in clauses if c)
    parts = [
        # Style first so the visual treatment is committed to
        # before the location-specific text fights for attention.
        style,
        "masterpiece, wide establishing shot",
        f"{environment.location_label}",
        f"{environment.prompt}",
        f"{environment.lighting}" if environment.lighting else "",
        "no people in frame",
    ]
    return ", ".join(p for p in parts if p)


def background_negative_prompt() -> str:
    return "low quality, blurry, watermark, text, people, characters"

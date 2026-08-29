"""Prompts for the New Game interview and post-confirmation world init.

The first two interview steps (Setting and Genre) use the hard-coded
defaults below — they don't need an LLM round trip, and that lets the
backend kick off the character-description LLM call (Step 4) the moment
the user has picked a setting, in parallel with the user thinking about
Genre and Visual Style. By the time the user reaches Step 4 the options
are usually already in hand.
"""

from __future__ import annotations

from typing import Final

from .common import NAMING_TABOO_RULE, SYSTEM_GAME_DIRECTOR, system_prompt

# 30 evocative settings; ``new_game_start_handler`` surfaces 5 random
# choices to the user (matches the original "5 of 30" UX) without any
# LLM call.
HARDCODED_SETTING_OPTIONS: Final[tuple[str, ...]] = (
    "A forgotten subway platform at midnight",
    "A perfumery where scents conjure ghosts",
    "A junkyard of vintage spacecraft at dusk",
    "A hotel room frozen in the 1920s",
    "A neon-drenched alley after closing time",
    "A stone harbor at dawn",
    "A monastery library during a snowstorm",
    "A high-altitude weather station in the Andes",
    "An abandoned amusement park at twilight",
    "A submarine canyon teeming with bioluminescent life",
    "A deep-space station orbiting a gas giant",
    "A pre-Cambrian seabed alive with strange shapes",
    "A 1960s diner on an empty desert highway",
    "A floating university amid storm clouds",
    "A mining colony on a tidally-locked moon",
    "An apothecary in 19th-century New Orleans",
    "A lighthouse on a haunted coastline",
    "A quaint English village hiding ancient secrets",
    "A samurai-era teahouse in old Kyoto",
    "A Renaissance Florence apothecary",
    "An art-deco airship over the polar ice",
    "A post-apocalyptic farmstead reclaiming the suburbs",
    "An underground crypt beneath Constantinople",
    "A caravan crossing the steppe under a meteor shower",
    "A pirate cove in the golden age of sail",
    "A research base buried under Antarctic ice",
    "A Tibetan monastery during a meteor shower",
    "A solar-punk arcology with vertical farms",
    "A 1980s arcade in a dying mall",
    "A train carriage crossing Siberia at night",
)


# Tone choices — what the storytelling FEELS like, not what world
# the game is set in. The world is settled by Setting; this picks
# the emotional register the engine writes in. Surfaced in full
# (no random sub-sampling).
HARDCODED_GENRE_OPTIONS: Final[tuple[str, ...]] = (
    "Adventure",
    "Power fantasy",
    "Coming of age",
    "Slice of life",
    "Tragedy",
    "Romance",
    "Comedy",
    "Mystery",
    "Thriller",
    "Horror",
    "Noir",
    "Wholesome",
    "Bittersweet",
    "Heroic",
    "Existential",
    "Dark academia",
    "Cozy",
)


# Visual styles bucketed into four flavour categories. The player
# never sees the full set — at runtime the Setting handler samples
# ONE entry from each bucket so the four options on screen always
# span "looks like a photograph", "looks like a cartoon", "looks
# like a painting", and "looks like nothing else". Each entry is a
# Stable Diffusion prompt fragment ready to concatenate into a
# ComfyUI positive prompt; no LLM call ever fires for visual styles.
VISUAL_STYLES_HYPERREALISTIC: Final[tuple[str, ...]] = (
    "film noir black and white, deep shadows, cigarette smoke, 35mm grain, high contrast",
    "soft summer photography, hazy bokeh, warm golden light, fine grain, nostalgic palette",
    "cinematic sci-fi concept art, deep teal and amber, soft rim light, atmospheric haze",
    "stark winter cinematography, deep blue moonlight, soft lamp glow, sparse frost, fine film grain",
    "high-contrast black and white photography, vertical fluorescent shafts, deep blacks, wet sheen",
    "documentary 35mm photography, natural light, neutral palette, subtle vignette, fine grain",
    "amber-tungsten interior photography, low light, sodium-vapor warmth, gentle motion blur, grain",
    "wide-format landscape photography, deep depth of field, soft volumetric mist, painterly highlights",
    "kodachrome 1970s color photography, warm reds, saturated greens, slight halation, fine grain",
    "polaroid SX-70 instant photo, washed-out tones, soft vignette, faint chemical bloom, square framing",
)

VISUAL_STYLES_CARTOONY: Final[tuple[str, ...]] = (
    "cel-shaded anime, vivid palette, sharp linework, soft ambient occlusion, bright daylight",
    "retro pulp magazine cover, halftone dots, lurid neon colors, bold contour lines, glossy",
    "Saturday morning cartoon, bold flat colors, thick outlines, simple shading, cheerful",
    "ligne claire bande dessinée, clean black inking, flat European pastels, precise shapes",
    "90s shonen anime, screentone shading, dramatic speed lines, saturated palette, sharp linework",
    "modern Pixar render, soft global illumination, plush shapes, warm color script, cinematic lighting",
    "newspaper comic strip, brushed black ink, halftone dots, three-color spot palette, flat shapes",
    "early-2000s cartoon network, angular silhouettes, flat saturated palette, thick outlines, energetic",
    "studio ghibli watercolor backdrop, hand-painted clouds, soft greens, gentle skies, nostalgic",
    "12-bit pixel art illustration, dithered shading, limited palette, crisp geometric forms",
)

VISUAL_STYLES_ARTSY: Final[tuple[str, ...]] = (
    "baroque oil painting, chiaroscuro, dramatic god rays, rich gold and crimson, painterly brushwork",
    "soft watercolor sketch, loose brushwork, warm pastel palette, faint paper texture, dreamy",
    "monochrome ink wash painting, sumi-e brush strokes, rice paper texture, deep blacks, contemplative",
    "ukiyo-e woodblock print, soft mineral pigments, flat shading, delicate lines, cherry blossom palette",
    "Renaissance fresco, warm ochre palette, soft volumetric light, classical composition, painterly",
    "art nouveau lithograph, sinuous lines, muted jewel tones, ornate decorative borders, flat shading",
    "impressionist oil painting, broken color, dappled light, loose brushwork, soft pastel atmosphere",
    "charcoal life drawing, smudged shading, deep blacks on cream paper, expressive linework",
    "stained glass illustration, leaded contour lines, jewel-bright translucent panels, ornate composition",
    "tempera icon painting, gold leaf background, flattened perspective, mineral pigments, devotional stillness",
)

VISUAL_STYLES_ODDBALL: Final[tuple[str, ...]] = (
    "vaporwave glitch aesthetic, magenta and cyan gradients, scanlines, distorted geometry, dream-logic",
    "collage assemblage, torn paper edges, mixed media, hand-typed labels, vintage ephemera, eccentric",
    "lo-fi VHS analogue horror, chromatic aberration, tape distortion, washed greens, uneasy dread",
    "risograph print, registration mis-alignment, two-color spot inks, grainy paper, lo-fi charm",
    "claymation stop-motion still, visible fingerprints in clay, soft tungsten lighting, handmade textures",
    "blacklight rave poster, fluorescent neon palette, jagged shapes, psychedelic geometry, throbbing energy",
    "soviet constructivist poster, bold red and black blocks, stenciled type, harsh angular shapes",
    "X-ray photograph aesthetic, ghostly translucent forms, blue-green glow on black, anatomical eeriness",
    "low-poly PS1 era 3D, jittery vertices, blurry textures, dithered fog, eerie geometric shapes",
    "wet plate collodion tintype, aged silver tones, vignette, ghost-like blur, antique distortion",
)

# Public iterable used by tests / type-checkers. The four buckets
# remain individually addressable above so future tweaks can swap
# entries inside one flavour without touching the others.
VISUAL_STYLE_BUCKETS: Final[dict[str, tuple[str, ...]]] = {
    "hyperrealistic": VISUAL_STYLES_HYPERREALISTIC,
    "cartoony": VISUAL_STYLES_CARTOONY,
    "artsy": VISUAL_STYLES_ARTSY,
    "oddball": VISUAL_STYLES_ODDBALL,
}


def character_description_options(*, setting: str) -> list[dict[str, str]]:
    """Step-4 prompt builder.

    Depends ONLY on setting so the backend can speculatively kick off
    this LLM call the moment the user picks a setting (i.e., when Step 2
    is presented). Genre and visual style — formerly inputs here —
    don't materially shape the protagonist's personality, so dropping
    them is a free win for latency.
    """
    user = (
        f"Setting: {setting}\n\n"
        "Propose 6 protagonist character descriptions that fit this setting. "
        "Each is one sentence (≤30 words) capturing personality and a distinctive "
        'trait. Return JSON: {"options": [str, ...]} with 6 entries.'
    )
    return [
        {"role": "system", "content": SYSTEM_GAME_DIRECTOR},
        {"role": "user", "content": user},
    ]


def name_options(*, setting: str, character_description: str) -> list[dict[str, str]]:
    user = (
        f"Setting: {setting}\nCharacter description: {character_description}\n\n"
        "Propose 8 first-name+last-name pairs that fit. Return JSON: "
        '{"options": [str, ...]} with 8 entries. ' + NAMING_TABOO_RULE
    )
    return [
        {"role": "system", "content": SYSTEM_GAME_DIRECTOR},
        {"role": "user", "content": user},
    ]


def side_character_expansion(*, one_line: str, setting: str) -> list[dict[str, str]]:
    user = (
        f"Setting: {setting}\nOne-line side character description: {one_line!r}\n\n"
        "Flesh this into a full character record. Return JSON matching:\n"
        '{"name":str,"description":str,"gender":str,"age":int,"ethnicity":str,'
        '"skin":str,"hair_color":str,"hairstyle":str,"eye_color":str,"build":str,'
        '"bust":str,"outfit":str,"pose":str,"expression":str,"effects":str,"decals":str}\n'
        "OUTFIT FORMAT: every garment MUST include a colour (or "
        "a material that implies one) — e.g. ``charcoal trench "
        "coat, cream wool scarf`` not ``trench coat, scarf``. "
        "Without an explicit colour the image model picks one at "
        "random per render and the same character's clothes "
        "flicker between portraits. ≤ 6 words total. " + NAMING_TABOO_RULE
    )
    return [
        {"role": "system", "content": SYSTEM_GAME_DIRECTOR},
        {"role": "user", "content": user},
    ]


def surprise_me_scenario(
    *,
    visual_style: str,
    likes: list[str],
    dislikes: list[str],
    notes: list[str],
) -> list[dict[str, str]]:
    """Build the prompt that synthesises a complete interview answer
    set ("Setting / Genre / Character description / Name") from the
    player's cross-save profile.

    The visual style is NOT generated — the caller passes whatever
    style the player's most recent save used (or the first hardcoded
    option on a fresh install) so the dream-guide / preview imagery
    stays in the same family the player has already been seeing.
    """

    def _bullet(items: list[str]) -> str:
        if not items:
            return "  (none recorded yet)"
        return "\n".join(f"  - {item}" for item in items[:20])

    user = (
        "The player just clicked SURPRISE ME on the start screen. "
        "Construct a complete starting scenario tailored to what we "
        "know about this player — they want a fresh story they will "
        "actually enjoy, not a generic one. Use the LIKES to bias "
        "what kind of setting / tone / protagonist gets chosen; use "
        "the DISLIKES as hard constraints (do NOT lean into anything "
        "from that list); the NOTES are softer observations that may "
        "or may not apply.\n\n"
        f"VISUAL STYLE (already chosen — your scenario must FIT this "
        f"visual style; do not pick a setting whose tone fights it):\n"
        f"  {visual_style}\n\n"
        f"LIKES:\n{_bullet(likes)}\n\n"
        f"DISLIKES:\n{_bullet(dislikes)}\n\n"
        f"NOTES:\n{_bullet(notes)}\n\n"
        "Pick a SETTING — a one-sentence place + time-of-day phrase, "
        "the same shape as ``A stone harbor at dawn`` or ``A neon "
        "alley after closing time``. Avoid recycling the literal "
        "examples here unless they're a perfect fit.\n\n"
        "Pick a GENRE — one of: Power fantasy, Coming of age, Slice "
        "of life, Tragedy, Romance, Comedy, Mystery, Thriller, "
        "Horror, Noir, Wholesome, Bittersweet, Heroic, Existential, "
        "Dark academia, Cozy. (Pick exactly one; not a hyphenation.)\n\n"
        "Author a CHARACTER DESCRIPTION — second person, ≤ 30 words, "
        "the shape the interview's character_description step would "
        "have produced (e.g. 'You are a tired harbour constable, "
        "carrying a coat that's seen better winters and a sketchbook "
        "you don't show anyone.'). Concrete, specific, with one "
        "evocative detail.\n\n"
        "Pick a NAME — a single name (not first + last) that fits "
        "the setting and the character description. " + NAMING_TABOO_RULE + "\n\n"
        "Return ONLY a JSON object on a single line, no Markdown "
        "fence, exactly:\n"
        '{"setting": str, "genre": str, "character_description": str, "name": str}'
    )

    return [
        {"role": "system", "content": system_prompt(mature_content=False)},
        {"role": "user", "content": user},
    ]


# Appended to world_init ONLY when a long-context image model
# (Qwen / Flux / Z-Image) is active — those render long prose, so the
# ≤ 6-word outfit cap that protects SDXL's CLIP window is lifted for the
# opening cast. Mirrors ``text_gen._DETAILED_APPEARANCE_OVERRIDE``.
_WORLD_INIT_APPEARANCE_OVERRIDE = (
    "APPEARANCE OVERRIDE (long-context image model active): the active "
    "renderer reads long natural-language descriptions. The ≤ 6-word cap on "
    "``outfit`` is LIFTED — write ``outfit``, ``pose`` and ``expression`` for "
    "the player_character AND every new_characters entry as vivid, explicit, "
    "concrete descriptions (a full phrase or 1–2 sentences), keeping every "
    "garment's colour explicit and ``pose`` restricted to the body itself. "
    "Describe each character's ``decals`` markings explicitly (design, "
    "colour, placement).\n"
)


def world_init(
    *,
    setting: str,
    genre: str,
    visual_style: str,
    character_description: str,
    name: str,
    side_characters: list[str],
    mature_content: bool = False,
    music_enabled: bool = False,
    detailed_appearance: bool = False,
) -> list[dict[str, str]]:
    side_block = "\n".join(f"- {s}" for s in side_characters) or "(none)"
    user = (
        f"Setting: {setting}\nGenre: {genre}\nVisual style: {visual_style}\n"
        f"Player character: {name} — {character_description}\n"
        f"Side characters present at start:\n{side_block}\n\n"
        "Return JSON matching:\n"
        '{"game_name": str,'
        '"plot_outline": ['
        '{"id":str,"title":str,"goal":str,"summary":str}'
        "],"
        '"active_plot_threads": [{"id":str,"title":str,"summary":str,'
        '"last_referenced_node_id": null}],'
        '"opening_node": {'
        '"beats": [{'
        '"text": str, "speaker_id": str|null, '
        '"entering_character_ids": [str], "leaving_character_ids": [str], '
        '"new_characters": [{"id":str,"name":str,"description":str,'
        '"gender":str,"age":int,"ethnicity":str,"skin":str,'
        '"hair_color":str,"hairstyle":str,"eye_color":str,"build":str,'
        '"bust":str,"outfit":str,"pose":str,"expression":str,"effects":str,"decals":str}], '
        '"location_id": str|null, "location_prompt": str|null, '
        '"character_changes": []'
        "}], "
        '"options": [{"id": str, "text": str}]'
        "},"
        '"player_character": {"name":str,"description":str,"gender":str,"age":int,'
        '"ethnicity":str,"skin":str,"hair_color":str,"hairstyle":str,'
        '"eye_color":str,"build":str,"bust":str,"outfit":str,"pose":str,'
        '"expression":str,"decals":str},'
        '"initial_music_prompt": str}\n'
        "The opening_node MUST contain 2–5 beats. Each beat is a SINGLE line "
        "(one sentence, occasionally two) with no carriage returns and no "
        "literal \\n inside it; the engine renders each beat as its own "
        "dialog node and walks the player through them one at a time. Only "
        "the last beat's ``options`` are presented as choices (empty list = "
        "Continue). "
        "If a beat introduces a character besides the player or the named "
        "side characters above, you MUST emit a full descriptor in that "
        "beat's ``new_characters`` AND list its id in "
        "``entering_character_ids``. Use a stable lowercase dash-separated "
        "id (e.g. ``mira-quill``).\n"
        + NAMING_TABOO_RULE
        + "OUTFIT FORMAT: every garment in any ``outfit`` field "
        "(player_character AND every new_characters entry) MUST "
        "include a colour or a material that implies one — "
        "``charcoal trench coat, cream wool scarf``, ``rust "
        "linen tunic, tan leather boots``, ``ivory silk blouse, "
        "navy trousers``. Without an explicit colour the image "
        "model picks one at random per render and the same "
        "character's clothes flicker between portraits. Bare "
        "``trench coat, scarf`` is wrong; ``leather`` and "
        "``cotton`` still need an explicit colour word. Keep "
        "outfit ≤ 6 words.\n"
        "DECALS: set each character's ``decals`` field to their PERMANENT "
        "skin markings — tattoos, scars, birthmarks, brands — if they have "
        'any distinctive ones, otherwise "". These persist for the whole '
        "game; do not put transient injuries here.\n"
        + (_WORLD_INIT_APPEARANCE_OVERRIDE if detailed_appearance else "")
        + "PLOT OUTLINE: Provide 3–6 stages forming the broad arc of the "
        "game from setup to resolution. Each stage gets a short ``title`` "
        "(2–6 words), a one-sentence ``goal`` (what this stage must "
        "accomplish for the story to progress), and a 1–3 sentence "
        "``summary`` of the beats / shape. Use stable lowercase "
        "dash-separated ids (e.g. ``stage-keeper-vanishes``). The "
        "storyteller is shown ONLY the currently-active stage at any "
        "given turn — do NOT pack the outline with details that need "
        "to bleed across stages; let each stage stand on its own."
        + (
            "\nINITIAL MUSIC PROMPT: emit an instrumental music "
            "prompt under ``initial_music_prompt`` describing the "
            "track that should play under the opening scenes. Your "
            "string is fed STRAIGHT INTO the music diffusion model "
            "— there is no second LM rewriting or expanding it, so "
            "be rich and specific. Lead with the EMOTIONAL MOOD "
            "(melancholic, tense, hopeful, dread-soaked, "
            "weightless, tender, etc.) — score what the player "
            "FEELS, not what they see. Then layer on tempo "
            "(e.g. ``slow``, ``60 BPM``, ``mid-tempo``), a "
            "key/mode hint when distinctive (``minor key``, "
            "``F# major``, ``dorian``), time signature when it "
            "matters (``3/4 time``, ``5/4``), specific "
            "instrument layers (sustained strings, soft piano, "
            "low drone, sub bass, plucked harp, choir pad, "
            "distant brass, tape hiss), and one or two mix / "
            "production descriptors (warm, dry, reverb-soaked, "
            "lo-fi, analog, digital, intimate). Do NOT "
            "describe the visual style or art direction; that "
            "belongs to the renderer, not the score. The track "
            "MUST be seamlessly loopable: include a ``seamless "
            "loop`` tag, favour sustained textures / drones / "
            "ostinati / static-mood beds, and AVOID hard "
            "intros, cold endings, or build-and-release arcs "
            "that telegraph a finale — the engine plays this on "
            "a loop under reading prose. Comma-separated tags "
            "only — no full sentences. Aim for 8-15 tags. "
            "INSTRUMENTAL (no vocals, no lyrics). Example: "
            "``melancholic, contemplative, sustained strings, "
            "soft piano, low drone, slow, 60 BPM, minor key, "
            "4/4 time, warm reverb, intimate, seamless loop``."
            if music_enabled
            else "\n``initial_music_prompt`` should be the empty "
            "string — music generation is disabled in this session."
        )
    )
    return [
        {"role": "system", "content": system_prompt(mature_content=mature_content)},
        {"role": "user", "content": user},
    ]

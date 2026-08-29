"""Source of truth for the main-menu carousel pairs.

The renderer cycles through these (slug, background, guide) pairs on
the start screen. The regen script (``backend/scripts/regen_main_menu.py``)
imports the table to render the bundled PNGs once at build time; the
runtime imports it to re-render the same dream-guide character in the
player-chosen visual style during the New Game interview.

Splitting the data + prompt builders out of the script lets the live
preview pipeline avoid string-duplication of long character body /
face descriptions. Edit the prompts here and both paths pick them up.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MenuCombo:
    slug: str
    setting: str
    genre: str
    visual_style: str
    location_label: str
    location_prompt: str
    guide_body: str
    guide_face: str
    title_color: str
    title_font: str
    title_letter_spacing: str = "0.18em"
    title_weight: str = "600"
    # Stroke + drop-shadow color used to separate the title from
    # the background. Default is a near-black warm tint that works
    # for the majority of (light-coloured) titles. Pairs whose
    # ``title_color`` is itself dark must override this to a light
    # tint — without that the stroke merges into the title and the
    # word collapses against the background. Any CSS color value
    # (hex, rgb(), rgba()) is valid; if alpha < 1 it lets the bg
    # bleed through slightly which softens the effect.
    title_shade_color: str = "rgba(8, 6, 4, 0.85)"


COMBOS: list[MenuCombo] = [
    MenuCombo(
        slug="harbor-noir",
        setting="A stone harbor at dawn",
        genre="Mystery",
        visual_style=(
            "film noir black and white, deep shadows, cigarette smoke, 35mm grain, high contrast"
        ),
        location_label="stone harbor at dawn",
        location_prompt=(
            "stone harbor before sunrise, sea-fog clinging to weathered "
            "pilings, gulls circling far overhead, lone lantern on a "
            "wet pier, slick cobblestones"
        ),
        guide_body=(
            "lone private detective in a long charcoal trench coat "
            "with belted waist, broad-brimmed fedora pulled low, hands "
            "deep in coat pockets, smoldering cigarette held between "
            "thumb and forefinger, woolen scarf, polished oxford shoes "
            "wet from the dock, mid-thirties, lean build, standing "
            "still, weight on one hip, full body frontal"
        ),
        guide_face=(
            "tired stubble-shadowed jaw, eyes in deep shadow under hat "
            "brim, faint cigarette ember underlighting cheekbones"
        ),
        title_color="#f6f3ee",
        title_font='"Playfair Display", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.32em",
    ),
    MenuCombo(
        slug="airship-baroque",
        setting="An art-deco airship over the polar ice",
        genre="Power fantasy",
        visual_style=(
            "baroque oil painting, chiaroscuro, dramatic god rays, "
            "rich gold and crimson, painterly brushwork"
        ),
        location_label="art-deco airship gondola",
        location_prompt=(
            "interior of an art-deco airship gondola at sunrise, "
            "polished walnut floorboards taking the foreground, "
            "brass railings and frosted portholes along the walls, "
            "ice fields visible through the windows in the distance"
        ),
        guide_body=(
            "imperious airship captain in a deep crimson high-collared "
            "coat trimmed in gold braid and brass buttons, peaked "
            "officer's cap with golden insignia, white silk cravat, "
            "polished black gloves with one hand resting on the brass "
            "hilt of an ornate dress sabre, knee-high black boots "
            "buffed to a mirror shine, late forties, regal posture, "
            "chest forward, full body, slight three-quarter pose"
        ),
        guide_face=(
            "weathered patrician face, neat silvering temples, sharp "
            "nose, severe pale-blue eyes, faint scar across one brow, "
            "trimmed grey moustache"
        ),
        title_color="#fff3d0",
        title_font='"Cinzel", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.2em",
        title_weight="700",
    ),
    MenuCombo(
        slug="diner-watercolor",
        setting="A 1960s diner on an empty desert highway",
        genre="Coming of age",
        visual_style=(
            "soft watercolor sketch, loose brushwork, warm pastel "
            "palette, faint paper texture, dreamy"
        ),
        location_label="1960s desert diner",
        location_prompt=(
            "interior of a vintage 1960s diner at golden hour seen "
            "from the entry, checkerboard linoleum floor stretching "
            "into the foreground, red vinyl booths to either side, "
            "neon sign glowing through the door at the back"
        ),
        guide_body=(
            "young 1960s diner waitress in a mint-green pleated "
            "uniform dress with white pinafore apron, name tag, "
            "white nylons and white nurse-style flats, blonde "
            "hair pinned up under a tiny matching cap, holding a "
            "glass coffee pot half full of dark brew in one hand and "
            "a small order pad in the other, late teens, slim, leaning "
            "slightly with one foot ahead, full body, frontal pose, "
            "bright open expression"
        ),
        guide_face=(
            "freckled cheeks, hazel eyes, light pink lipstick, soft "
            "smile, blonde wisps escaping the cap"
        ),
        title_color="#fae4de",
        title_font="tahoma",
        title_letter_spacing="0.06em",
        title_weight="700",
        title_shade_color="rgba(166, 0, 0, 0.2)",
    ),
    MenuCombo(
        slug="arcology-anime",
        setting="A solar-punk arcology with vertical farms",
        genre="Wholesome",
        visual_style=(
            "highly detailed anime, vivid palette, sharp linework, soft "
            "ambient occlusion, bright daylight, optimistic"
        ),
        location_label="vertical-farm atrium",
        location_prompt=(
            "ground-level promenade of a solar-punk arcology atrium, "
            "polished bamboo decking taking the foreground, terraced "
            "hydroponic gardens rising on either side and cascading "
            "in the distance, sunlit bridges far overhead, lush greenery"
        ),
        guide_body=(
            "cheerful young horticulturist in olive-green canvas short shorts "
            ", wide-brimmed straw "
            "sunhat tied under the chin with a green ribbon, leather "
            "tool belt with seed pouches and small pruning shears, "
            "dirt-smudged knees, brown work gloves, woven basket "
            "balanced on one hip filled with leafy herbs and small "
            "tomatoes, early twenties, athletic build, full body, "
            "warm grin"
        ),
        guide_face=(
            "bronzed sunkissed cheeks, large warm brown eyes, broad "
            "smile, freckles across the nose, soft black bob just "
            "visible under the sunhat"
        ),
        title_color="#39af39",
        title_font='"Quicksand", "Trebuchet MS", sans-serif',
        title_letter_spacing="0.22em",
        title_weight="500",
    ),
    MenuCombo(
        slug="monastery-ink",
        setting="A monastery library during a snowstorm",
        genre="Dark academia",
        visual_style=(
            "monochrome ink wash painting, sumi-e brush strokes, "
            "rice paper texture, deep blacks, contemplative"
        ),
        location_label="monastery scriptorium",
        location_prompt=(
            "high-vaulted monastery scriptorium during a blizzard, snow "
            "drifting past tall lancet windows, candlelit oak desks "
            "covered in open codices, ironwood shelves vanishing into "
            "shadow, hush of falling snow"
        ),
        guide_body=(
            "monochrome ink wash sumi-e painting of a monk in heavy woolen robes "
            "but not over the eyes, sandals just visible at the hem, "
            "illuminated manuscript cradled in one arm "
            "open to a page, late fifties, slight frame, "
            "contemplative stillness, full body slight three-quarter. Monochromatic image"
        ),
        guide_face=(
            "deeply lined ascetic face, sunken cheeks, kind quiet eyes "
            "behind small wire spectacles, neat grey-white beard"
        ),
        title_color="#e7e7e7",
        title_font='"Cormorant Garamond", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.28em",
    ),
    MenuCombo(
        slug="hotel-1920s-bittersweet",
        setting="A hotel room frozen in the 1920s",
        genre="Bittersweet",
        visual_style=(
            "soft summer photography, hazy bokeh, warm golden light, "
            "fine grain, nostalgic 1920s palette"
        ),
        location_label="hotel suite, late afternoon",
        location_prompt=(
            "1920s hotel suite at late afternoon viewed from the "
            "doorway, polished hardwood floor with a small persian "
            "rug taking the foreground, four-poster bed and side "
            "table receded against the back wall, sun cutting through "
            "lace curtains, dust motes in the air"
        ),
        guide_body=(
            "1920s flapper in a dropped-waist champagne silk dress "
            "covered in fine glass beadwork that catches the light, "
            "knee-length hem with handkerchief points, long pearl rope "
            "necklace looped twice, satin T-strap heels, single white "
            "feather pinned in a black bobbed wig, long black silk "
            "gloves to the elbow, holding a slim gold cigarette holder "
            "loosely, late twenties, melancholy half-turned pose, "
            "weight on back foot, full body"
        ),
        guide_face=(
            "porcelain skin, smoky kohl-rimmed eyes, dark cupid's-bow "
            "lips, faint distant gaze, subtle eyeshadow"
        ),
        title_color="#f3d9b8",
        title_font='"Cormorant Garamond", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.18em",
        title_weight="500",
    ),
    MenuCombo(
        slug="moon-mining-existential",
        setting="A mining colony on a tidally-locked moon",
        genre="Existential",
        visual_style=(
            "cinematic sci-fi concept art, deep teal and amber, soft "
            "rim light, vast scale, lonely, atmospheric haze"
        ),
        location_label="tidally-locked terminator",
        location_prompt=(
            "mining gantries silhouetted against the eternal twilight "
            "of a tidally-locked moon, distant gas-giant looming over "
            "the horizon, mineral spires casting long shadows, "
            "industrial floodlights piercing thin atmosphere"
        ),
        guide_body=(
            "weathered mining engineer in a battered slate-grey EVA "
            "suit scuffed by mineral dust, removable hard helmet "
            "cradled under one arm with a cracked visor, life-support "
            "harness across the chest with glowing teal status lights, "
            "thick gloves clipped to the belt, heavy magnetic boots, "
            "small chest patch reading colony designation, late "
            "forties, broad shoulders, weary upright stance, full body"
        ),
        guide_face=(
            "weather-cured face, deep crow's feet, salt-and-pepper "
            "stubble, faintly amber-lit cheekbones, tired but steady "
            "grey eyes"
        ),
        title_color="#bcd9dd",
        title_font='"Cormorant Garamond", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.34em",
        title_weight="400",
    ),
    MenuCombo(
        slug="arcade-1980s-pulp",
        setting="A 1980s arcade in a dying mall",
        genre="Slice of life",
        visual_style=(
            "retro pulp magazine cover, halftone dots, lurid neon "
            "colors, bold contour lines, glossy"
        ),
        location_label="1980s arcade hall",
        location_prompt=(
            "interior of a 1980s arcade in a dying mall, rows of CRT "
            "cabinets glowing magenta and cyan, threadbare carpet "
            "patterned with shapes, vending machine humming in the "
            "corner, faint cigarette smoke"
        ),
        guide_body=(
            "teenage arcade champion girl in a faded denim jacket plastered "
            "with band patches over a hot-pink graphic tee, acid-wash "
            "skinny jeans rolled at the cuff, white high-top sneakers, "
            "neon-cyan plastic digital watch, walkman clipped to the "
            "belt with orange foam headphones around the neck, holding "
            "a roll of arcade tokens loosely, late teens, lean, "
            "casual loose-limbed posture, full body, smug confident "
            "smirk"
        ),
        guide_face=(
            "long bleached-blond hair, dark eyebrows, sharp cheekbones, "
            "pierced ear, faint freckles, smug crooked grin"
        ),
        title_color="#5be5ff",
        title_font='"Bebas Neue", "Impact", sans-serif',
        title_letter_spacing="0.22em",
        title_weight="400",
    ),
    MenuCombo(
        slug="siberia-train-tragedy",
        setting="A train carriage crossing Siberia at night",
        genre="Tragedy",
        visual_style=(
            "stark winter cinematography, deep blue moonlight, soft "
            "lamp glow, sparse frost, fine film grain, quiet"
        ),
        location_label="overnight carriage corridor",
        location_prompt=(
            "narrow corridor of a Trans-Siberian sleeper carriage at "
            "midnight, ribbed wooden floor running into the foreground, "
            "compartment doors and frost-blooming windows along one "
            "side, single brass lamp casting amber pools, dark fir "
            "forests sweeping past beyond the windows"
        ),
        guide_body=(
            "grim train conductor in a heavy charcoal greatcoat with "
            "tarnished brass buttons down the front, thick black "
            "astrakhan fur cap with a small enameled rail badge, "
            "leather-gloved hand holding a brass kerosene lantern at "
            "shoulder height casting amber pools, dark woolen scarf, "
            "snow-dusted boots, late fifties, broad solid build, "
            "stoic upright stance, full body frontal"
        ),
        guide_face=(
            "wind-roughened ruddy cheeks, ice-blue eyes, thick "
            "salt-and-pepper moustache, deep furrow between brows"
        ),
        title_color="#d8e3ee",
        title_font='"Playfair Display", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.3em",
        title_weight="400",
    ),
    MenuCombo(
        slug="kyoto-teahouse-romance",
        setting="A samurai-era teahouse in old Kyoto",
        genre="Romance",
        visual_style=(
            "ukiyo-e woodblock print, soft mineral pigments, flat "
            "shading, delicate lines, cherry blossom palette"
        ),
        location_label="teahouse veranda",
        location_prompt=(
            "engawa veranda of an old Kyoto teahouse at dusk, polished "
            "wooden boards and a stretch of clean tatami in the "
            "foreground center, low lacquered table pushed to one side, "
            "paper lanterns glowing soft pink in the eaves, sakura "
            "petals drifting, distant pagoda silhouette"
        ),
        guide_body=(
            "young geisha in an elaborate cherry-blossom-patterned "
            "pale-pink furisode kimono with long flowing sleeves, "
            "deep crimson silk obi tied in an elaborate bow at the "
            "back, white tabi socks and lacquered black geta sandals, "
            "ornate kanzashi hairpins of dangling silver flowers, "
            "holding a closed bamboo-and-paper umbrella with painted "
            "blossoms loosely at her side, mid-twenties, demure pose, "
            "weight balanced, full body frontal"
        ),
        guide_face=(
            "porcelain shironuri makeup, vivid red lower lip, jet-black "
            "swept-up hair, kohl-lined eyes, serene expression"
        ),
        title_color="#f5e8d0",
        title_font='"Cormorant Garamond", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.16em",
        title_weight="600",
    ),
    MenuCombo(
        slug="subway-horror",
        setting="A forgotten subway platform at midnight",
        genre="Horror",
        visual_style=(
            "high-contrast black and white photography, vertical "
            "fluorescent shafts, deep blacks, chiaroscuro, grain, "
            "wet sheen"
        ),
        location_label="abandoned platform",
        location_prompt=(
            "abandoned underground subway platform at midnight, "
            "tile sweating condensation, single fluorescent strip "
            "flickering, mosaic of cracked enamel signage, water "
            "pooling on rails, empty"
        ),
        guide_body=(
            "stark figure in a long ankle-length oilskin black "
            "raincoat, hood pulled forward leaving the upper face in "
            "deep shadow, thick rubber gloves, holding a heavy "
            "industrial flashlight angled downward casting a hard "
            "white pool, the lit area revealing wet boots and a "
            "rust-stained canvas duffel bag, indistinct gender, "
            "tense ready stance, full body"
        ),
        guide_face=(
            "lower jaw and pale chin barely visible under hood, "
            "single sharp glint of an eye in deep shadow, thin tense "
            "mouth"
        ),
        title_color="#ffffff",
        title_font='"Playfair Display", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.36em",
        title_weight="700",
    ),
    MenuCombo(
        slug="florence-fresco-heroic",
        setting="A Renaissance Florence apothecary",
        genre="Heroic",
        visual_style=(
            "Renaissance fresco, warm ochre palette, soft volumetric "
            "light, classical composition, painterly"
        ),
        location_label="apothecary nave",
        location_prompt=(
            "Renaissance Florentine apothecary, vaulted ceiling "
            "dusted with frescoes, walnut shelves crammed with "
            "ceramic apothecary jars, mortar and pestle on a marble "
            "counter, sun cutting through high arched windows, motes "
            "of dust"
        ),
        guide_body=(
            "Renaissance scholar-physician in floor-length deep "
            "burgundy velvet robes trimmed with thin gold embroidery, "
            "matching velvet biretta cap, soft leather slippers "
            "peeking from the hem, large leather-bound medical codex "
            "tucked under one arm, the other hand holding a "
            "blown-glass alembic with a curl of pale steam, broad "
            "shoulders, late forties, dignified upright posture, "
            "full body slight three-quarter"
        ),
        guide_face=(
            "warm olive complexion, neatly trimmed dark beard, alert "
            "intelligent dark eyes, slight smile lines, faint "
            "ink-smudge on the temple"
        ),
        title_color="#e6c97a",
        title_font='"Cinzel", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.24em",
        title_weight="700",
        title_shade_color="rgba(0, 0, 0, 0.4)",
    ),
    # ----- expansion batch (24 total) -----------------------------------
    MenuCombo(
        slug="perfumery-thriller",
        setting="A perfumery where scents conjure ghosts",
        genre="Thriller",
        visual_style=(
            "art-deco poster illustration, sharp angular lines, gold and "
            "midnight-blue palette, geometric shadows, ornate"
        ),
        location_label="ghost-haunted perfumery",
        location_prompt=(
            "art-nouveau perfumery interior at dusk, glass atomisers in "
            "ranks on lacquered shelves, faint coloured vapour drifting "
            "from open bottles, polished marble counter, no occupants"
        ),
        guide_body=(
            "detailed elegant Parisian perfumer in a fitted black silk gown with "
            "gold-embroidered cuffs, long satin gloves, lace choker, "
            "neat chignon held by a jet pin, holding a slim crystal "
            "atomiser between thumb and forefinger, late thirties, "
            "poised contrapposto stance, full body, slight three-quarter"
        ),
        guide_face=(
            "porcelain skin, sharp dark eyes, neat arched brows, "
            "deep wine lipstick, single beauty mark by the lip"
        ),
        title_color="#d8b16a",
        title_font='"Cinzel", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.26em",
        title_weight="600",
    ),
    MenuCombo(
        slug="junkyard-spacecraft-noir",
        setting="A junkyard of vintage spacecraft at dusk",
        genre="Noir",
        visual_style=(
            "retro sci-fi noir paperback cover, washed sepia, halftone "
            "shadows, hand-painted cracks of light, 1960s pulp finish"
        ),
        location_label="vintage spacecraft junkyard",
        location_prompt=(
            "eye-level view of wasteland with scrap fuselages and "
            "skeletal solar arrays only along the distant horizon, "
            "no spacecraft in the foreground, no vehicles in the "
            "immediate frame, deep-dusk sky, lone sodium floodlight "
            "casting yellow pools against the far hulks"
        ),
        guide_body=(
            "former scrap-pilot in a battered tan flight jacket scarred "
            "by acid burns, oil-stained jumpsuit, tool harness across "
            "the hip, leather flying gloves, scuffed steel-toed boots, "
            "holding a worn brass calliper measuring an old hull plate, "
            "early forties, lean and wiry, tired upright stance, full body"
        ),
        guide_face=(
            "weathered cheekbones, salt-and-pepper short cut, narrowed "
            "amber eyes, faint scar across the left brow, three-day stubble"
        ),
        title_color="#0c1a35",
        title_font='"Bebas Neue", "Impact", sans-serif',
        title_letter_spacing="0.28em",
        title_weight="400",
        title_shade_color="rgba(248, 240, 218, 0.9)",
    ),
    MenuCombo(
        slug="neon-alley-comedy",
        setting="A neon-drenched alley after closing time",
        genre="Comedy",
        visual_style=(
            "vibrant cyberpunk illustration, flat-color shading, bold "
            "magenta and cyan rim light, clean linework, playful tone"
        ),
        location_label="cyberpunk alley after hours",
        location_prompt=(
            "narrow rain-slick alley between neon-signed shopfronts, "
            "magenta and cyan light pooling on wet pavement, steam from "
            "a vent, holographic ramen sign flickering, no occupants"
        ),
        guide_body=(
            "cheerful courier in a cropped iridescent windbreaker, mesh "
            "tank top, short shorts with reflective trim, neon-pink "
            "sneakers, oversized data-glove on one hand, holographic "
            "messenger bag slung crosswise blinking with delivery codes, "
            "early twenties, springy weight-on-toes pose, full body, "
            "bright open expression"
        ),
        guide_face=(
            "warm tan complexion, mismatched cybernetic eye glowing soft "
            "cyan, freckles across nose, lopsided grin, undercut buzz with "
            "a long magenta-dyed forelock"
        ),
        title_color="#FF00FF",
        title_font='"Quicksand", "Trebuchet MS", sans-serif',
        title_letter_spacing="0.18em",
        title_weight="500",
    ),
    MenuCombo(
        slug="andes-weather-cozy",
        setting="A high-altitude weather station in the Andes",
        genre="Cozy",
        visual_style=(
            "warm cozy oil painting, thick brushwork, golden hour glow, "
            "muted palette, gentle highlights, hand-painted feel"
        ),
        location_label="Andean weather station common room",
        location_prompt=(
            "interior of a wood-panelled mountain weather station "
            "common room seen from the doorway, broad pine-plank floor "
            "with a worn rug taking the foreground center, cast-iron "
            "stove and writing desk receded against the side walls, "
            "frost on the windows, snow-capped peaks beyond the glass"
        ),
        guide_body=(
            "nerdy Andean meteorologist woman in a tight oatmeal sweater "
            "tucked into wool trousers, sheepskin slippers, woven "
            "alpaca scarf in earthy reds and browns looped at the neck, "
            "leather-bound logbook tucked under one arm, late forties, "
            "warm relaxed stance, full body"
        ),
        guide_face=(
            "ruddy cheeks chapped by wind, kind dark brown eyes with "
            "deep crow's feet, salt-and-pepper braid down one shoulder, "
            "calm smile"
        ),
        title_color="#fff3d8",
        title_font='"Cormorant Garamond", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.16em",
        title_weight="500",
    ),
    MenuCombo(
        slug="amusement-park-thriller",
        setting="An abandoned amusement park at twilight",
        genre="Thriller",
        visual_style=(
            "80s slasher movie still, deep magenta and cyan lighting, "
            "soft fog, lens halation, vintage 35mm grain, ominous"
        ),
        location_label="abandoned amusement park at twilight",
        location_prompt=(
            "rusted carousel and toppled ferris wheel silhouetted against "
            "a magenta-cyan twilight sky, cracked asphalt strewn with "
            "dead leaves, sodium lamps flickering on, paper streamers "
            "lifting in the wind"
        ),
        guide_body=(
            "wary park-security guard in a faded burgundy windbreaker "
            "over a navy uniform shirt, brass nameplate, dark slacks, "
            "scuffed black duty boots, heavy MagLite gripped low and "
            "ready, walkie-talkie clipped to a worn belt, mid-thirties, "
            "weight on back foot, alert tense stance, full body"
        ),
        guide_face=(
            "olive complexion, square jaw, neat dark beard, intense brown "
            "eyes, faint sheen of sweat on the brow, mouth set in a hard line"
        ),
        title_color="#ff79b8",
        title_font='"Bebas Neue", "Impact", sans-serif',
        title_letter_spacing="0.32em",
        title_weight="400",
    ),
    MenuCombo(
        slug="bioluminescent-canyon-existential",
        setting="A submarine canyon teeming with bioluminescent life",
        genre="Existential",
        visual_style=(
            "deep-sea cinematic concept art, electric teal and indigo "
            "luminance, soft volumetric haze, vast scale, contemplative"
        ),
        location_label="bioluminescent submarine canyon ledge",
        location_prompt=(
            "wide flat pale-sand seabed terrace filling the entire "
            "foreground, completely empty seafloor, glowing jellyfish "
            "drifting only high above, luminescent kelp clinging to "
            "canyon walls far in the distance fading into deeper blue, "
            "particles drifting in the water column, no creatures in "
            "the immediate foreground"
        ),
        guide_body=(
            "deep-sea diver in an articulated atmospheric pressure suit "
            "of dark steel and glass-fronted helmet, faint internal blue "
            "running lights along the joints, tethered umbilical trailing "
            "behind, gloved hand reaching slowly into the dark, mid-forties, "
            "weightless steady float, full body slight three-quarter"
        ),
        guide_face=(
            "behind the broad curved helmet glass: a pensive face faintly "
            "lit teal from within, short greying hair, focused grey eyes, "
            "thin pressed lips"
        ),
        title_color="#88e3d8",
        title_font='"Cormorant Garamond", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.34em",
        title_weight="400",
    ),
    MenuCombo(
        slug="precambrian-seabed-wholesome",
        setting="A pre-Cambrian seabed alive with strange shapes",
        genre="Wholesome",
        visual_style=(
            "Victorian scientific illustration, hand-tinted plate, fine "
            "stippled linework, soft warm sepia, pale watercolour wash"
        ),
        location_label="pre-Cambrian seabed plain",
        location_prompt=(
            "flat sandy seabed plain stretching wide and empty across "
            "the entire foreground, strange ribbed Ediacaran creatures "
            "and feathery fronds visible only in the far middle "
            "distance and along the sides, soft golden light filtering "
            "from above the water, gently swaying fronds in the "
            "background, no creatures in the immediate foreground"
        ),
        guide_body=(
            "earnest paleobiologist-illustrator in a brass-trimmed canvas "
            "diving smock buttoned up to a high collar, leather "
            "cinch-belt, knee-length wading breeches, tall wading boots, "
            "weatherproof sketchbook open in one hand and a slim brass "
            "stylus in the other, late twenties, eager forward-leaning "
            "pose, full body"
        ),
        guide_face=(
            "freckled fair skin, large curious hazel eyes behind round "
            "wire spectacles, untidy strawberry-blonde hair tied back "
            "with a leather thong, delighted half-smile"
        ),
        title_color="#3a2614",
        title_font='"Cormorant Garamond", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.18em",
        title_weight="500",
        title_shade_color="rgba(248, 240, 218, 0.9)",
    ),
    MenuCombo(
        slug="floating-university-mystery",
        setting="A floating university amid storm clouds",
        genre="Mystery",
        visual_style=(
            "art-nouveau illustrated plate, sinuous gilded linework, "
            "deep teal and brass palette, ornate frames, painterly clouds"
        ),
        location_label="floating university faculty cloister",
        location_prompt=(
            "vast stone cloister floor in the foreground center, "
            "polished flagstones receding toward arched openings, "
            "brass-railed balconies along the sides looking out into "
            "towering thunderheads and curtains of rain, lit braziers "
            "in iron sconces, leather-bound tomes stacked on a marble "
            "bench against one wall"
        ),
        guide_body=(
            "hovering witch professor with flowing blue robes "
            "robes embroidered with constellations in fine gold thread, "
            "a brass-gilt astrolabe on a chain at the chest, leather "
            "satchel of scrolls slung across the body, late twenties, contemplative "
            "stillness, full body slight three-quarter"
        ),
        guide_face=(
            "warm bronze complexion, dark intelligent eyes, neat goatee, "
            "spectacles low on the nose, ink smudge on one temple"
        ),
        title_color="#f4d488",
        title_font='"Cinzel", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.22em",
        title_weight="600",
    ),
    MenuCombo(
        slug="new-orleans-apothecary-noir",
        setting="An apothecary in 19th-century New Orleans",
        genre="Noir",
        visual_style=(
            "moody gothic ink illustration, deep blacks and warm sepia "
            "highlights, fine cross-hatched shadows, gas-lamp glow"
        ),
        location_label="19th-century New Orleans apothecary",
        location_prompt=(
            "cluttered 19th-century New Orleans apothecary at night, "
            "shelves of amber bottles labelled in copperplate, mortar "
            "and pestle on a polished walnut counter, gas lamp casting "
            "warm pools, faint cigarillo smoke, wrought-iron balcony "
            "shadows on the floor"
        ),
        guide_body=(
            "Creole apothecary in a long charcoal frock coat over a high "
            "lace collar and silk waistcoat, narrow black trousers, "
            "polished spats over leather shoes, silver-headed cane held "
            "loosely, antique pocket-watch chain across the waistcoat, "
            "late thirties, watchful upright stance, full body"
        ),
        guide_face=(
            "warm bronze complexion, neat trimmed goatee, sharp dark "
            "eyes, faint scar bisecting one eyebrow, single brass earring"
        ),
        title_color="#0c0c0c",
        title_font='"Playfair Display", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.24em",
        title_weight="600",
        title_shade_color="rgba(248, 240, 218, 0.9)",
    ),
    MenuCombo(
        slug="haunted-lighthouse-horror",
        setting="A lighthouse on a haunted coastline",
        genre="Horror",
        visual_style=(
            "gothic gas-lamp horror illustration, deep blue-black sea, "
            "single sweeping beam of pale light, fine cross-hatched fog"
        ),
        location_label="moorland approach to a haunted lighthouse",
        location_prompt=(
            "windswept coastal moor at night, wide flat path of "
            "trampled grass and wet earth running into the foreground "
            "center, hag-stone walls along the sides, a single "
            "lighthouse tower visible only as a small silhouette far "
            "in the distance ahead with its lamp a tiny bright dot, "
            "dark fog rolling over heather, no cliffs in the front "
            "half, no buildings near the camera, faint lantern light"
        ),
        guide_body=(
            "weathered fisherman in a heavy oilskin coat over a "
            "thick wool jumper, knee-high rubber sea-boots crusted with "
            "salt, leather logbook strapped to a worn belt, late fifties, broad solid "
            "frame, stoic upright stance, full body frontal"
        ),
        guide_face=(
            "wind-cured ruddy face, deep weather lines, frost-grey full "
            "beard, pale grey-blue eyes that see something just beyond "
            "the camera, mouth set"
        ),
        title_color="#f5f8ff",
        title_font='"Playfair Display", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.34em",
        title_weight="700",
    ),
    MenuCombo(
        slug="english-village-cozy",
        setting="A quaint English village hiding ancient secrets",
        genre="Cozy",
        visual_style=(
            "soft pastoral watercolour, gentle washes, warm afternoon "
            "light, hand-drawn outlines, storybook charm"
        ),
        location_label="English village high street at teatime",
        location_prompt=(
            "narrow cobbled high street of a Cotswold village in late "
            "afternoon, honey-stone cottages with climbing roses, an old "
            "red post-box, the spire of a Norman church beyond, a tabby "
            "cat asleep on a low stone wall"
        ),
        guide_body=(
            "amiable village librarian in a soft heather-grey cardigan "
            "buttoned over a cream blouse, calf-length tweed skirt with "
            "subtle moss-green check, woollen tights, low brogues, a "
            "leather satchel of returned books slung at the hip, knitted "
            "shawl folded over one arm, sixties, comfortable stance, full body"
        ),
        guide_face=(
            "kind English face, soft wrinkles around the eyes, warm hazel "
            "eyes behind half-moon spectacles, silver bob neatly pinned, "
            "gentle smile"
        ),
        title_color="#000000",
        title_font='"Caveat", "Comic Sans MS", cursive',
        title_letter_spacing="0.08em",
        title_weight="700",
        title_shade_color="rgba(248, 240, 218, 0.5)",
    ),
    MenuCombo(
        slug="throne-hall-high-fantasy",
        setting="A high castle's throne hall under house banners",
        genre="High fantasy",
        visual_style=(
            "highly detailed realistic high-fantasy oil painting, sweeping "
            "atmospheric perspective, warm torch and shaft-light through "
            "stained glass, rich jewel-tone palette, ornate masterwork detail"
        ),
        location_label="throne hall under banners",
        location_prompt=(
            "vast vaulted great hall of a high-fantasy stone castle, "
            "polished marble flagstone floor stretching wide and empty "
            "across the foreground center, twin rows of carved stone "
            "columns receding toward a distant raised dais and gilded "
            "throne, embroidered house banners hanging from the rafters, "
            "tall stained-glass windows on either side casting coloured "
            "shafts of late-afternoon light, motes of dust"
        ),
        guide_body=(
            "noble paladin in finely articulated polished steel plate "
            "over a quilted gambeson, surcoat in deep sapphire emblazoned "
            "with a silver sun, bear-fur trimmed cloak clasped at the "
            "shoulder with an ornate silver brooch, gauntleted hand "
            "resting on the pommel of a longsword sheathed at the hip, "
            "elaborate kite shield strapped across the back, mid-thirties, "
            "broad shoulders, dignified upright stance, full body slight "
            "three-quarter"
        ),
        guide_face=(
            "noble bearing, fair complexion lightly weathered, neat "
            "short-cropped chestnut hair, calm steel-grey eyes, narrow "
            "trimmed beard, faint pale scar across the jaw"
        ),
        title_color="#e6c97a",
        title_font='"Cinzel", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.24em",
        title_weight="700",
    ),
    MenuCombo(
        slug="ruined-cathedral-dark-fantasy",
        setting="A ruined cathedral where the dead don't stay buried",
        genre="Dark fantasy",
        visual_style=(
            "grim dark-fantasy oil painting, desaturated muted palette, "
            "deep shadows, painterly realism, ominous low fog, low-key "
            "lighting, fine atmospheric detail"
        ),
        location_label="shattered cathedral nave",
        location_prompt=(
            "interior of a long-abandoned gothic cathedral nave, broken "
            "flagstone floor with tendrils of grey mist crawling across "
            "the foreground center, fallen pews and rotting tapestries "
            "to either side, half-collapsed apse in the distance with "
            "wan grey daylight stabbing through ribs of broken vaulting, "
            "ravens perched on shattered statuary, no figures"
        ),
        guide_body=(
            "hardened femme-fatale witch-hunter in a long oilskin greatcoat over a "
            "dull chainmail shirt, leather pauldrons darkened with old "
            "blood, broad-brimmed black hat shadowing the upper face, "
            "ammunition bandolier across the chest, twin flintlock "
            "pistols holstered at the hip, silver-banded crossbow gripped "
            "low and ready in one gloved hand, knee-high mud-caked "
            "boots, late thirties, lean dangerous build, weight on back "
            "foot, full body slight three-quarter"
        ),
        guide_face=(
            "gaunt pale face under hat brim, pale grey eyes catching "
            "the dim light, hard pressed mouth, three-day stubble, faint "
            "pale scar across the cheekbone"
        ),
        title_color="#cdbfa8",
        title_font='"Cormorant Garamond", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.32em",
        title_weight="400",
    ),
    MenuCombo(
        slug="convenience-store-anime-slice-of-life",
        setting="A late-night convenience store on a quiet city corner",
        genre="Slice of life",
        visual_style=(
            "modern slice-of-life anime, soft cel shading, gentle "
            "cinematic light, clean linework, warm pastel palette, "
            "atmospheric, highly detailed"
        ),
        location_label="convenience store interior at midnight",
        location_prompt=(
            "interior of a brightly-lit Japanese convenience store after "
            "midnight, polished tile floor stretching wide and empty "
            "across the foreground center, neat aisles of snacks and "
            "canned drinks receding to the back wall, fluorescent ceiling "
            "lights casting cool white pools, magazine rack along one "
            "wall, coffee machine humming next to a rain-streaked "
            "window looking out onto a rain-slick neon street"
        ),
        guide_body=(
            "warmly cheerful late-shift anime girl clerk in a fitted navy work "
            "apron over a white button-down shirt with rolled sleeves, "
            "slim black trousers, clean white sneakers, name tag pinned "
            "to the apron, holding a small handheld scanner loosely at "
            "the hip, early twenties, slight slim build, relaxed friendly "
            "stance with weight on one leg, head tilted slightly, full "
            "body, bright open expression"
        ),
        guide_face=(
            "warm fair skin, large bright dark eyes, soft round cheeks, "
            "gentle close-mouthed smile, neat tousled black hair with one "
            "stubborn cowlick, a single small bandage across one cheek"
        ),
        title_color="#bfd9f7",
        title_font='"Quicksand", "Trebuchet MS", sans-serif',
        title_letter_spacing="0.18em",
        title_weight="500",
    ),
    MenuCombo(
        slug="steppe-caravan-heroic",
        setting="A caravan crossing the steppe under a meteor shower",
        genre="Heroic",
        visual_style=(
            "epic painterly fantasy illustration, dramatic skies, sweeping "
            "atmospheric perspective, warm golden firelight, rich palette"
        ),
        location_label="steppe caravan camp at meteor-fall",
        location_prompt=(
            "wide flat grassy steppe in the foreground center "
            "completely empty of animals or wagons, low pebbles and "
            "wind-bent grass, a tiny camel caravan visible only on "
            "the distant horizon with low fires, white streaks of "
            "meteors raking the violet sky overhead, no animals in "
            "the front half of the frame"
        ),
        guide_body=(
            "steppe-caravan captain in a knee-length deep-crimson coat "
            "trimmed with grey wolf fur, embroidered sash at the waist "
            "holding a curved sabre and an ornate flintlock, leather "
            "riding breeches tucked into tall felt boots, gloved hand "
            "resting on the sabre hilt, late thirties, commanding upright "
            "stance, full body slight three-quarter"
        ),
        guide_face=(
            "high cheekbones, weather-burnished bronze skin, sharp dark "
            "eyes, a single thin scar across one cheek, neat black "
            "moustache, iron-grey at the temples"
        ),
        title_color="#2a1a08",
        title_font='"Cinzel", "Iowan Old Style", Georgia, serif',
        title_letter_spacing="0.24em",
        title_weight="700",
        title_shade_color="rgba(248, 240, 218, 0.9)",
    ),
]


_BY_SLUG: dict[str, MenuCombo] = {c.slug: c for c in COMBOS}


def combo_for_slug(slug: str) -> MenuCombo | None:
    return _BY_SLUG.get(slug)


# ---------------------------------------------------------------------------
# Prompt builders shared with regen_main_menu.py and the runtime preview
# pipeline. Re-rendering at runtime reuses ``preview_*`` variants that take
# (visual_style, location_label, location_prompt, ...) directly so the same
# character can be re-rendered against the player's chosen visual style.
# ---------------------------------------------------------------------------


def background_positive_for(
    *,
    visual_style: str,
    location_label: str,
    location_prompt: str,
    setting: str,
    genre: str,
) -> str:
    # Backgrounds host a dream-guide figure composited over the
    # lower-center. Without explicit framing tags SDXL composes the
    # scene "to the camera" — putting tables, beds, vehicles, or
    # close-up objects right where the figure needs to stand. Lead
    # with the framing contract so the model commits to a stage-set
    # composition before reading the scene description.
    return (
        "masterpiece, wide establishing shot, eye-level camera, "
        "stage-set composition, clear open foreground center, "
        "empty floor in the foreground large enough to fit a "
        "standing person, no foreground objects blocking center, "
        f"{visual_style}, "
        f"{location_label}, {location_prompt}, "
        f"setting: {setting}, genre: {genre}, "
        "no people in frame, no figures, no occupants, atmospheric"
    )


def guide_positive_for(*, visual_style: str, guide_body: str) -> str:
    return (
        f"{visual_style}, masterpiece, full body, standing, "
        f"centered, head and feet visible, single subject, "
        f"{guide_body}, soft rim light, atmospheric, detailed"
    )


def background_negative() -> str:
    return "low quality, blurry, watermark, text, people, characters, cropped, out of frame"


def guide_negative_extras() -> str:
    return (
        "extra people, crowd, multiple subjects, anime mascot, "
        "cartoon mascot, deformed hands, blurry, low quality, watermark"
    )


# ---------------------------------------------------------------------------
# Static-script convenience wrappers (kept for the regen_main_menu script).
# ---------------------------------------------------------------------------


def background_positive(combo: MenuCombo) -> str:
    return background_positive_for(
        visual_style=combo.visual_style,
        location_label=combo.location_label,
        location_prompt=combo.location_prompt,
        setting=combo.setting,
        genre=combo.genre,
    )


def guide_positive(combo: MenuCombo) -> str:
    return guide_positive_for(
        visual_style=combo.visual_style,
        guide_body=combo.guide_body,
    )


def guide_face(combo: MenuCombo) -> str:
    return combo.guide_face


__all__ = [
    "COMBOS",
    "MenuCombo",
    "background_negative",
    "background_positive",
    "background_positive_for",
    "combo_for_slug",
    "guide_face",
    "guide_negative_extras",
    "guide_positive",
    "guide_positive_for",
]

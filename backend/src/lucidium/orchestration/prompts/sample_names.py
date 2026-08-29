"""Bundled real-person name pool sampled into the storyteller prompt.

The LLM, left to its own devices, defaults to the same handful of
fantasy-flavoured names every time (Lyra / Elara / Aria / Kael —
see ``common.FORBIDDEN_NAMES``). ``NAMING_TABOO_RULE`` tells it
which names NOT to use; this module gives it a positive pool of
names it CAN reach for. The combination is more reliable than the
taboo alone — telling an LLM "not X" without offering "but Y"
often pushes it to a different cliché instead of grounded names.

Source mix:
  * ``CENSUS_NAMES`` — common US-census first names sampled across
    decades (1920s grandparents to 2020s kids). Boring on purpose:
    these are the names you'd encounter on a real apartment buzzer.
  * ``ODDBALL_NAMES`` — less-common but historically-real names
    (Maud, Phineas, Hortense). Seed a few of these alongside the
    boring pile so the LLM doesn't only ever propose a Tom or a Sarah.

Both lists are deliberately culled of any name on
``common.FORBIDDEN_NAMES``; covered by ``test_sample_names.py``.

A render is sampled randomly per turn (~12 names by default) so the
LLM sees a different slate each call — keeps repeat playthroughs
from filling up with the same first ten names from the pool.
"""

from __future__ import annotations

import random

CENSUS_NAMES: tuple[str, ...] = (
    # Mid-century American (grandparent generation): boring,
    # unmistakably-real names. Useful for any setting that needs
    # someone who reads as ordinary.
    "Mary",
    "Patricia",
    "Linda",
    "Barbara",
    "Susan",
    "Margaret",
    "Dorothy",
    "Helen",
    "Betty",
    "Ruth",
    "Nancy",
    "Carol",
    "Janet",
    "Joyce",
    "Joan",
    "Doris",
    "Gloria",
    "Eleanor",
    "Frances",
    "James",
    "John",
    "Robert",
    "William",
    "David",
    "Richard",
    "Charles",
    "Joseph",
    "Thomas",
    "Donald",
    "George",
    "Paul",
    "Frank",
    "Walter",
    "Carl",
    "Edward",
    "Henry",
    "Harold",
    "Albert",
    "Arthur",
    "Raymond",
    # Late-century (parent generation): still common, slightly
    # newer texture.
    "Jennifer",
    "Lisa",
    "Michelle",
    "Amanda",
    "Christine",
    "Heather",
    "Stephanie",
    "Diane",
    "Julie",
    "Brenda",
    "Kimberly",
    "Catherine",
    "Christopher",
    "Daniel",
    "Matthew",
    "Anthony",
    "Mark",
    "Steven",
    "Andrew",
    "Kenneth",
    "Brian",
    "Kevin",
    "Jason",
    "Jeffrey",
    "Gregory",
    "Patrick",
    # Contemporary kids — for younger characters or fresh-from-the-
    # academy types.
    "Emma",
    "Olivia",
    "Charlotte",
    "Amelia",
    "Sophia",
    "Isabella",
    "Evelyn",
    "Harper",
    "Luna",
    "Camila",
    "Ella",
    "Mila",
    "Nora",
    "Eloise",
    "Hazel",
    "Iris",
    "Violet",
    "Liam",
    "Noah",
    "Oliver",
    "Elijah",
    "Lucas",
    "Mason",
    "Logan",
    "Ethan",
    "Aiden",
    "Jackson",
    "Owen",
    "Asher",
    "Wyatt",
    "Levi",
    "Theodore",
    "Julian",
    "Leo",
)

ODDBALL_NAMES: tuple[str, ...] = (
    # Historically-real but uncommon names. Use these to break up
    # the boring pile so every new character isn't a Sarah or a
    # Michael. Mostly Victorian / Edwardian / mid-century-Americana
    # — names that wear well in mystery, period, or working-class
    # settings without veering into fantasy-default territory.
    "Maud",
    "Beulah",
    "Hester",
    "Cordelia",
    "Hortense",
    "Magdalena",
    "Tilda",
    "Esmé",
    "Clementine",
    "Winifred",
    "Phineas",
    "Bartholomew",
    "Ignatius",
    "Hale",
    "Cully",
    "Wilbur",
    "Clement",
    "Cyrus",
    "Lemuel",
    "Atticus",
)


def render_sample_names(
    count: int = 12,
    *,
    rng: random.Random | None = None,
) -> str:
    """Pick ``count`` names from the combined pool and format them as
    a single prompt block ready to drop into the storyteller user
    message.

    ``rng`` is injectable so tests can pin the sample without
    monkey-patching the module-global ``random``. Production calls
    pass ``None`` and get a fresh random sample per turn.
    """
    # ``random.sample`` is the bound method of the module-global Random
    # instance, so falling back to it keeps the historical behaviour
    # (including responsiveness to ``random.seed``) exactly.
    sample = rng.sample if rng is not None else random.sample
    pool = list(CENSUS_NAMES) + list(ODDBALL_NAMES)
    k = max(1, min(count, len(pool)))
    sampled = sample(pool, k)
    return (
        "NEW CHARACTER NAMES — if you introduce a brand-new character in "
        "this chain, pick a first name from this list (or use it as the "
        "first name and invent a fitting surname). These are real "
        "people-names from a bundled census + oddballs pool, refreshed "
        "each turn, included so you don't default to the same handful "
        "of LLM-flavoured names. Stay inside this list unless the "
        "setting makes a name from it impossible (e.g. a culture-locked "
        "setting where these names wouldn't fit at all):\n" + ", ".join(sampled)
    )

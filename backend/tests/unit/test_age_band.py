from __future__ import annotations

import pytest

from lucidium.domain.character import age_band


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        # Floored at 18. Sub-18 ages render as the explicit numeric
        # band ``"eighteen"`` — NOT ``"teenage"`` (which SDXL's
        # training distribution couples with visually-minor
        # features) and NOT ``"young adult"`` (which earlier
        # versions used; the explicit decade word produces sharper
        # adult-portrait conditioning).
        (0, "eighteen"),
        (12, "eighteen"),
        (13, "eighteen"),
        (17, "eighteen"),
        (18, "eighteen"),
        (19, "eighteen"),
        (20, "twenty"),
        (29, "twenty"),
        (35, "thirty"),
        (47, "forty"),
        (60, "sixty"),
        (99, "ninety"),
        (100, "hundred"),
        (140, "hundred"),
    ],
)
def test_age_band_maps_decade_to_word(age: int, expected: str) -> None:
    assert age_band(age) == expected


def test_age_band_floors_sub_18_to_eighteen() -> None:
    """No SDXL image prompt should describe a minor — the floor
    inside ``age_band`` is the chokepoint. Storage of the
    canonical integer is unaffected (tested separately).

    Output is the literal decade word ``"eighteen"``, NOT
    ``"teenage"`` (which still pulls visually-young features even
    when the numeric age in the rest of the prompt is adult) and
    NOT ``"young adult"`` (an earlier shape that lacked an
    explicit numeric anchor).
    """
    for under_age in (0, 5, 8, 12, 13, 15, 17):
        assert age_band(under_age) == "eighteen", under_age


def test_age_band_never_emits_teenage_or_child_or_minor() -> None:
    """Hard pin: those words must never appear as outputs of the
    band function — they're the SDXL-vocabulary words most
    strongly coupled to under-18 visuals."""
    for age in range(0, 130):
        result = age_band(age)
        assert result != "teenage", age
        assert result != "child", age
        assert result != "minor", age


def test_age_band_rejects_negative() -> None:
    with pytest.raises(ValueError):
        age_band(-1)

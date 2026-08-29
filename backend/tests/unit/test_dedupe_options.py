"""LLM-emitted options lists must not contain duplicates.

The new-game interview's name step in particular is prone to
the LLM repeating the same name with different trailing
punctuation / whitespace, or returning ``\"Mira\"`` and
``\"mira\"`` as separate entries. The renderer paints those as
separate buttons — visually broken. The backend's
``_dedupe_options`` helper filters out case-insensitive
duplicates at the parse boundary so every downstream path
(sync return, async emit, the surface-random sampler) sees a
clean list.
"""

from __future__ import annotations

from lucidium.api.handlers import _dedupe_options


def test_drops_exact_duplicates() -> None:
    out = _dedupe_options(["Iris", "Iris", "Hale"])
    assert out == ["Iris", "Hale"]


def test_drops_case_insensitive_duplicates_keeping_first_casing() -> None:
    out = _dedupe_options(["Mira", "MIRA", "mira"])
    assert out == ["Mira"]


def test_drops_whitespace_only_entries() -> None:
    out = _dedupe_options(["Iris", "  ", "", "Hale"])
    assert out == ["Iris", "Hale"]


def test_trims_surrounding_whitespace_when_deduping() -> None:
    out = _dedupe_options(["Iris", " Iris ", "iris"])
    assert out == ["Iris"]


def test_preserves_distinct_names() -> None:
    out = _dedupe_options(["Iris Vale", "Iris Quill", "Mira"])
    assert out == ["Iris Vale", "Iris Quill", "Mira"]


def test_empty_input_returns_empty() -> None:
    assert _dedupe_options([]) == []

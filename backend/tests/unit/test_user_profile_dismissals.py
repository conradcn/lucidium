"""Sticky-dismissal behaviour for the cross-save user profile.

When the player edits the "Inferred by the engine" section in the
Settings UI and removes an entry, the next summarizer pass MUST NOT
re-add the same entry just because the LLM noticed the same pattern
again. The mechanism: the player-facing settings handler diffs the
old vs new ``summarizer_*`` lists and appends the removed entries
to ``dismissed_*``; the summarizer's ``extend_user_profile`` merge
function reads ``dismissed_*`` into its dedup ``seen`` set so the
addition is rejected.

These tests exercise the merge step directly — the settings-handler
end of the contract is exercised via the full settings update test
elsewhere; here we just pin "if dismissed contains X, X never lands
in summarizer_X again no matter how many times the LLM re-emits it."
"""

from __future__ import annotations

from lucidium.domain.settings import UserProfile
from lucidium.orchestration.summarizer import _merge_profile_additions as extend_user_profile


def test_dismissed_entry_not_re_added_by_summarizer() -> None:
    """A like the player previously deleted (so it sits in
    ``dismissed_likes``) must NOT come back when the summarizer's
    next pass emits the same string."""
    profile = UserProfile(
        likes=["fast pacing"],
        summarizer_likes=["dialog-driven scenes"],
        dismissed_likes=["spooky atmosphere"],
    )
    additions = {
        "likes": ["spooky atmosphere", "morally grey characters"],
        "dislikes": [],
        "notes": [],
    }
    next_profile = extend_user_profile(current=profile, additions=additions)

    assert "morally grey characters" in next_profile.summarizer_likes, (
        "novel additions should still land in the summarizer bucket"
    )
    assert "spooky atmosphere" not in next_profile.summarizer_likes, (
        "an entry on the dismissed list must not be re-added by the "
        "summarizer pass — that's the whole point of dismissed_*"
    )
    assert "fast pacing" in next_profile.likes, "player-typed entries pass through unchanged"
    assert next_profile.dismissed_likes == ["spooky atmosphere"], (
        "dismissed list must persist across summarizer passes"
    )


def test_dismissal_is_case_insensitive() -> None:
    """LLM phrasing wobbles in case / spacing. A dismissed
    'Spooky Atmosphere' must still block a re-emission of
    'spooky atmosphere' (or 'SPOOKY ATMOSPHERE')."""
    profile = UserProfile(
        dismissed_dislikes=["Slow Burn Mysteries"],
    )
    additions = {
        "likes": [],
        "dislikes": ["slow burn mysteries", "SLOW BURN MYSTERIES"],
        "notes": [],
    }
    next_profile = extend_user_profile(current=profile, additions=additions)
    assert next_profile.summarizer_dislikes == [], (
        "case variants of a dismissed entry must all be rejected"
    )


def test_existing_summarizer_entries_still_dedup_against_player() -> None:
    """The dismissed-list addition is layered on top of the existing
    dedup against player-typed entries — both must still work."""
    profile = UserProfile(
        notes=["likes pacing breaks"],
        summarizer_notes=["responds to slow reveals"],
        dismissed_notes=["heavy exposition"],
    )
    additions = {
        "likes": [],
        "dislikes": [],
        "notes": [
            "likes pacing breaks",  # already in player bucket
            "responds to slow reveals",  # already in summarizer bucket
            "heavy exposition",  # in dismissed
            "tracks small character beats",  # actually new
        ],
    }
    next_profile = extend_user_profile(current=profile, additions=additions)
    # Only the genuinely-new entry lands.
    assert next_profile.summarizer_notes == [
        "responds to slow reveals",
        "tracks small character beats",
    ], next_profile.summarizer_notes

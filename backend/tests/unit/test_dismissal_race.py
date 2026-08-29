"""Regression test for the dismissal-vs-summarizer race.

Bug shape: the player opens Settings, deletes an entry from a
``summarizer_*`` bucket, clicks Save. The settings handler
correctly removes the entry and appends it to the ``dismissed_*``
list. Then a summarizer task that was IN FLIGHT when the player
clicked Save returns. Pre-fix the summarizer applied a
profile-update derived from a snapshot taken BEFORE the user's
deletion, overwriting the freshly-saved settings with stale
state — the deleted entry came back, the player's edits felt
ignored.

The fix: the summarizer-task apply step re-reads
``session.settings.user_profile`` (live, post-user-edit) and
re-merges the LLM's additions against THAT, instead of using
the snapshot-derived ``application.user_profile``.

This test mocks the LLM (no live calls) and exercises the merge
helper directly with a "user dismissed entry while summarizer
was running" sequence, asserting:

  * The dismissed entry stays out of ``summarizer_*``.
  * The dismissed list is preserved.
  * Genuine NEW additions still land.
"""

from __future__ import annotations

from lucidium.domain.settings import UserProfile
from lucidium.orchestration.summarizer import _merge_profile_additions


def test_dismissed_entry_does_not_resurrect_after_concurrent_settings_edit() -> None:
    """The merge function — operating against the LIVE post-edit
    profile (which the handler now passes in after the fix) — must
    keep the dismissed entry out even if the LLM proposed adding
    it back during the same turn."""
    # Simulate the live profile after the user dismissed "haunted
    # houses" mid-summarizer-task.
    live_profile = UserProfile(
        summarizer_likes=["slow-burn investigation"],
        dismissed_likes=["haunted houses"],
    )
    # The summarizer's LLM call (which started before the dismissal)
    # proposed adding the now-dismissed entry back, plus a genuinely
    # new tag.
    additions = {
        "likes": ["haunted houses", "morally grey mentors"],
        "dislikes": [],
        "notes": [],
    }
    merged = _merge_profile_additions(
        current=live_profile,
        additions=additions,
    )
    assert "haunted houses" not in merged.summarizer_likes, (
        "dismissed entry resurrected by the post-LLM merge — the "
        "merge function isn't reading dismissed_* into its dedup set"
    )
    assert "morally grey mentors" in merged.summarizer_likes, (
        "genuine new tag was rejected — the merge over-suppressed"
    )
    assert "slow-burn investigation" in merged.summarizer_likes, "existing summarizer tag dropped"
    assert merged.dismissed_likes == ["haunted houses"], "dismissed list lost across the merge"


def test_repeated_summarizer_passes_keep_dismissals() -> None:
    """A long run where the LLM keeps re-emitting the dismissed
    pattern still doesn't resurrect it. Each merge takes the
    previous merged profile as its ``current`` — dismissed_*
    must persist across the chain."""
    profile = UserProfile(
        summarizer_likes=[],
        dismissed_dislikes=["tonal whiplash"],
    )
    for _ in range(5):
        profile = _merge_profile_additions(
            current=profile,
            additions={
                "likes": [],
                "dislikes": ["tonal whiplash"],  # LLM re-emits
                "notes": [],
            },
        )
    assert "tonal whiplash" not in profile.summarizer_dislikes
    assert profile.dismissed_dislikes == ["tonal whiplash"]


def test_dismissal_works_when_player_edits_during_in_flight_summarizer() -> None:
    """End-to-end shape of the bug: the merge happens AFTER the
    user's settings update has already been written. The merge's
    ``current`` is the post-update profile; the LLM additions
    are what the in-flight task computed against the pre-update
    profile."""
    pre_update_profile = UserProfile(
        summarizer_likes=["A", "B"],
        dismissed_likes=[],
    )
    # User dismisses "A" via settings — backend appends to
    # dismissed and removes from summarizer_likes. (This is what
    # ``settings_update_handler`` does; here we simulate the
    # post-update state.)
    post_update_profile = UserProfile(
        summarizer_likes=["B"],
        dismissed_likes=["A"],
    )
    # Summarizer's LLM call proposed adding "C" (new) and
    # accidentally re-proposing "A" because it was working from
    # the pre-update snapshot.
    additions = {
        "likes": ["A", "C"],
        "dislikes": [],
        "notes": [],
    }
    # Pre-fix: the handler used apply_summary's output (computed
    # against ``pre_update_profile``), which would land
    # ``["A", "B", "C"]``. Post-fix: the handler re-runs the merge
    # against ``post_update_profile``.
    stale_merged = _merge_profile_additions(
        current=pre_update_profile,
        additions=additions,
    )
    assert stale_merged.summarizer_likes == ["A", "B", "C"], (
        "sanity: merging against the pre-update snapshot is what reintroduces 'A'"
    )
    merged = _merge_profile_additions(
        current=post_update_profile,
        additions=additions,
    )
    assert "A" not in merged.summarizer_likes, (
        "race regression: the player's dismissal of 'A' lost to a stale summarizer snapshot"
    )
    assert "B" in merged.summarizer_likes
    assert "C" in merged.summarizer_likes
    assert merged.dismissed_likes == ["A"]

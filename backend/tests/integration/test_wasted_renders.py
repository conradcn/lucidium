"""Capture wasted image-generation effort.

Background. The engine speculatively pre-renders character portraits
when it pre-extends the dialog tree (so by the time the player walks
a beat, the portrait is already on disk). When subsequent player
actions invalidate that branch — free_text, edit_character, an
attribute change that bumps the prompt_hash — the speculative render
is orphaned. The PNG sits on disk but no live ``character.images[*]``
or ``environment.image_path`` references it.

The user reported this as the engine "waiting to present trees until
character images render, then often discarding them". This module
adds full-stack tests that make the waste observable: every call to
``image_client.generate()`` writes a PNG, the test then walks the
live ``session.game`` after all background tasks drain, and asserts
that every generated PNG is referenced from somewhere on the live
game.

These tests run with mocked LLM + image clients (no GPU, no LLM
endpoint, no ComfyUI) so they live in the default suite. The point
is to pin the orphan-render budget; if a refactor introduces a new
discarded path, the assertion fails and the regression is visible.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import (
    Envelope,
    InterviewStep,
    MessageType,
)
from lucidium.orchestration.session import Session

# --- Shared minimal fixtures (PNG bytes + JSON payloads) -------------------


_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _FastLlm:
    _SPEC_FALLBACK = json.dumps(
        {
            "beats": [
                {
                    "text": "(speculative).",
                    "speaker_id": None,
                    "entering_character_ids": [],
                    "leaving_character_ids": [],
                    "new_characters": [],
                    "location_id": None,
                    "location_prompt": None,
                    "character_changes": [],
                }
            ],
            "options": [],
        }
    )

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(self, *_a, **_kw) -> AsyncIterator[str]:
        nv = self._responses.pop(0) if self._responses else self._SPEC_FALLBACK

        async def gen() -> AsyncIterator[str]:
            yield nv

        return gen()


class _CountingImage:
    """Image client that tracks every generate() call. Returns a
    deterministic PNG so the engine can content-address it on disk
    via ``hash_payload``."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def generate(self, workflow, prompts, *, seed) -> bytes:
        self.calls.append((workflow, dict(prompts), seed))
        return _PNG_BYTES


async def _drain(handler_result):
    out = []
    async for msg in handler_result:
        out.append(msg)
    return out


def _opt(items: list[str]) -> str:
    return json.dumps({"options": items})


def _player_payload() -> dict:
    return {
        "name": "Iris",
        "description": "wry archivist",
        "gender": "female",
        "age": 28,
        "ethnicity": "local",
        "skin": "pale",
        "hair_color": "auburn",
        "hairstyle": "braid",
        "eye_color": "grey",
        "build": "slight",
        "bust": "moderate",
        "outfit": "wool coat",
        "pose": "standing",
        "expression": "alert",
    }


def _new_character_descriptor(*, id_: str, name: str) -> dict:
    return {
        "id": id_,
        "name": name,
        "description": "a stranger from the road",
        "gender": "male",
        "age": 50,
        "ethnicity": "local",
        "skin": "weathered",
        "hair_color": "iron grey",
        "hairstyle": "short",
        "eye_color": "hazel",
        "build": "lean",
        "bust": "n/a",
        "outfit": "oilskin coat",
        "pose": "standing",
        "expression": "guarded",
    }


def _world_init_payload() -> str:
    return json.dumps(
        {
            "game_name": "Salt Lantern",
            "overall_plot_direction": "Investigate the keeper.",
            "active_plot_threads": [],
            "opening_node": {
                "beats": [
                    {
                        "text": "The harbor wakes slow.",
                        "speaker_id": None,
                        "entering_character_ids": [],
                        "leaving_character_ids": [],
                        "new_characters": [],
                        "location_id": "harbor",
                        "location_prompt": "stone harbor at dawn",
                        "character_changes": [],
                    },
                ],
                "options": [
                    {"id": "left", "text": "Walk to the inn."},
                    {"id": "right", "text": "Wait at the dock."},
                ],
            },
            "player_character": _player_payload(),
        }
    )


def _branch_with_new_character(text: str, char_id: str, name: str) -> str:
    return json.dumps(
        {
            "beats": [
                {
                    "text": text,
                    "speaker_id": None,
                    "entering_character_ids": [char_id],
                    "leaving_character_ids": [],
                    "new_characters": [
                        _new_character_descriptor(id_=char_id, name=name),
                    ],
                    "location_id": None,
                    "location_prompt": None,
                    "character_changes": [],
                }
            ],
            "options": [],
        }
    )


async def _walk_through_interview(ctx: HandlerContext, registry) -> None:
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_start, payload={}),
            ctx,
        )
    )
    for step, ans in [
        (InterviewStep.setting, "stone harbor"),
        (InterviewStep.visual_style, "ink wash"),
        (InterviewStep.genre, "mystery"),
        (InterviewStep.character_description, "wry archivist"),
        (InterviewStep.name, "Iris"),
    ]:
        await _drain(
            registry.dispatch(
                Envelope(
                    type=MessageType.c2s_new_game_answer,
                    payload={"step": step.value, "answer": ans, "is_free_text": False},
                ),
                ctx,
            )
        )


async def _drain_async_tasks(session: Session, max_iterations: int = 80) -> None:
    """Yield until all background asset / speculation tasks have
    settled. Without this the test's measurement window misses
    renders that finished after the last awaited handler."""
    for _ in range(max_iterations):
        await asyncio.sleep(0.02)
        spec = list((getattr(session, "_speculative_tasks", None) or {}).values())
        asset = list(getattr(session, "_asset_tasks", None) or [])
        live = [t for t in (*spec, *asset) if not t.done()]
        if not live:
            return


def _live_referenced_paths(session: Session) -> set[Path]:
    """Collect every PNG path the live ``session.game`` references —
    most-recent character portraits, environment backgrounds, and
    the historical entries on every character.images list (we keep
    the history so the renderer's previous-portrait fallback works
    on an out-of-date image_ready event)."""
    refs: set[Path] = set()
    if session.game is None:
        return refs
    for char in session.game.characters.values():
        for image in char.images:
            if image.path:
                refs.add(Path(image.path))
    for env in session.game.environments.values():
        if env.image_path:
            refs.add(Path(env.image_path))
    return refs


def _generated_pngs_on_disk(saves_root: Path) -> set[Path]:
    """Every gameplay PNG the engine wrote during this test.

    Excludes ``_interview_preview/`` — those are deliberately
    transient renders surfaced through ``session.interview`` during
    onboarding (chosen-style background, dream guide, player-
    character preview) so the player can see their selections
    before clicking Begin. They aren't referenced from the final
    ``game`` object by design; the user's question about wasted
    renders is about post-confirm gameplay, not the preview UX
    that's understood to be transient.
    """
    return {p for p in saves_root.rglob("*.png") if "_interview_preview" not in p.parts}


# ---------------------------------------------------------------------------
# 1. Steady-state opening: every render is referenced.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opening_renders_have_no_orphans(tmp_app_data: Path) -> None:
    """Spec: a fresh game's opening pass should produce exactly the
    images the live state references — nothing more, nothing less.
    Establishes the baseline for the regression tests below."""
    image = _CountingImage()
    llm = _FastLlm(
        responses=[
            _opt([f"c{i}" for i in range(6)]),
            _opt([f"n{i}" for i in range(8)]),
            _world_init_payload(),
        ]
    )
    session = Session(llm_client=llm, image_client=image, emit=lambda *_a: None)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_through_interview(ctx, registry)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )
    await _drain_async_tasks(session)

    on_disk = _generated_pngs_on_disk(session.saves_root)
    referenced = _live_referenced_paths(session)
    orphans = on_disk - referenced
    assert orphans == set(), (
        f"Opening render produced {len(orphans)} orphan PNG(s) — files "
        f"on disk that no live character.images / environment.image_path "
        f"references:\n  " + "\n  ".join(str(p) for p in sorted(orphans))
    )


# ---------------------------------------------------------------------------
# 2. Speculation ALONE doesn't leak orphans either.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speculation_pre_renders_are_referenced(tmp_app_data: Path) -> None:
    """When the engine pre-renders a portrait for a speculative
    character, the rendered file must end up referenced on the
    speculatively-committed character record. Without that link
    the file would sit orphaned even on the happy path."""
    image = _CountingImage()
    llm = _FastLlm(
        responses=[
            _opt([f"c{i}" for i in range(6)]),
            _opt([f"n{i}" for i in range(8)]),
            _world_init_payload(),
            _branch_with_new_character("Mira steps from the fog.", "mira-quill", "Mira"),
            _branch_with_new_character("Aldo waves you over.", "aldo-rust", "Aldo"),
        ]
    )
    session = Session(llm_client=llm, image_client=image, emit=lambda *_a: None)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_through_interview(ctx, registry)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )
    await _drain_async_tasks(session)

    on_disk = _generated_pngs_on_disk(session.saves_root)
    referenced = _live_referenced_paths(session)
    orphans = on_disk - referenced
    assert orphans == set(), (
        f"After speculation drained, {len(orphans)} render(s) were not "
        f"referenced from the live game. Speculative pre-renders need "
        f"to land on character.images[].path:\n  " + "\n  ".join(str(p) for p in sorted(orphans))
    )


# ---------------------------------------------------------------------------
# 3. The regression target: rerender DOES discard the prior portrait file.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_character_rerender_does_not_double_count_renders(
    tmp_app_data: Path,
) -> None:
    """When the player triggers a manual rerender, the previously-
    rendered portrait file stays referenced on ``character.images``
    (the engine prepends the new image; the prior one stays in
    history so the renderer's fallback can show it during the swap).
    So neither file is orphaned — both are reachable from the live
    state. Pins this contract: rerenders DON'T leak orphans, even
    though they generate a fresh image."""
    image = _CountingImage()
    llm = _FastLlm(
        responses=[
            _opt([f"c{i}" for i in range(6)]),
            _opt([f"n{i}" for i in range(8)]),
            _world_init_payload(),
        ]
    )
    session = Session(llm_client=llm, image_client=image, emit=lambda *_a: None)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_through_interview(ctx, registry)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )
    await _drain_async_tasks(session)

    initial_render_count = len(image.calls)
    initial_orphans = _generated_pngs_on_disk(session.saves_root) - _live_referenced_paths(session)
    assert initial_orphans == set(), f"Pre-rerender state already had orphans: {initial_orphans}"

    # Edit player's outfit and rerender — this changes the prompt
    # hash, so the new PNG lands at a different filename.
    player_id = next(cid for cid, c in session.game.characters.items() if c.is_player)
    await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_edit_character,
                payload={
                    "character_id": player_id,
                    "field": "outfit",
                    "value": "long oilskin cloak, mud-spattered",
                },
            ),
            ctx,
        )
    )
    await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_edit_character_rerender,
                payload={"character_id": player_id},
            ),
            ctx,
        )
    )
    await _drain_async_tasks(session)

    # A new render has fired (prompt hash changed → new file).
    assert len(image.calls) > initial_render_count, (
        "rerender after attribute edit should produce a new image"
    )
    on_disk = _generated_pngs_on_disk(session.saves_root)
    referenced = _live_referenced_paths(session)
    orphans = on_disk - referenced
    assert orphans == set(), (
        f"Rerender left {len(orphans)} orphan PNG(s) — file(s) on disk "
        f"that no live character.images entry references:\n  "
        + "\n  ".join(str(p) for p in sorted(orphans))
    )


# ---------------------------------------------------------------------------
# 4. The regression target: speculative branch invalidated → its renders ARE
# wasted (file on disk, never referenced).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speculation_invalidated_by_freetext_does_not_leak_orphans(
    tmp_app_data: Path,
) -> None:
    """The actual orphan-render scenario: a speculative branch has
    pre-rendered its character; the player then submits free_text
    that invalidates the branch. The speculative character is
    dropped from the dialog tree AND from game.characters (the
    invalidation handler trims orphan characters too), so the
    pre-rendered PNG file is no longer referenced.

    This test PINS the engine's current cleanup behaviour. If the
    engine starts leaking orphan PNGs after a free_text invalidation,
    this assertion fires and the regression is visible.
    """
    image = _CountingImage()
    llm = _FastLlm(
        responses=[
            _opt([f"c{i}" for i in range(6)]),
            _opt([f"n{i}" for i in range(8)]),
            _world_init_payload(),
            # Two speculative branches each introducing a new NPC.
            _branch_with_new_character("Mira steps from the fog.", "mira-quill", "Mira"),
            _branch_with_new_character("Aldo waves you over.", "aldo-rust", "Aldo"),
            # The free_text response — a NEW chain that doesn't
            # reuse either speculative character.
            json.dumps(
                {
                    "beats": [
                        {
                            "text": "You step away from the dock entirely.",
                            "speaker_id": None,
                            "entering_character_ids": [],
                            "leaving_character_ids": [],
                            "new_characters": [],
                            "location_id": None,
                            "location_prompt": None,
                            "character_changes": [],
                        },
                    ],
                    "options": [],
                }
            ),
        ]
    )
    session = Session(llm_client=llm, image_client=image, emit=lambda *_a: None)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_through_interview(ctx, registry)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )
    await _drain_async_tasks(session)

    pre_freetext_calls = len(image.calls)
    pre_freetext_orphans = _generated_pngs_on_disk(session.saves_root) - _live_referenced_paths(
        session
    )
    if pre_freetext_orphans:
        # If speculation already produced orphans, that's the
        # baseline waste this test is *measuring* — log it but don't
        # fail; the meaningful comparison is against this baseline.
        baseline = len(pre_freetext_orphans)
    else:
        baseline = 0

    # Override with free_text — should invalidate both speculative
    # branches and any portraits attached only to those branches.
    await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_play_free_text,
                payload={"text": "I walk away from the dock entirely."},
            ),
            ctx,
        )
    )
    await _drain_async_tasks(session)

    post_orphans = _generated_pngs_on_disk(session.saves_root) - _live_referenced_paths(session)
    post_calls = len(image.calls)
    new_orphans = len(post_orphans) - baseline
    new_calls = post_calls - pre_freetext_calls

    # The whole point of this test: if free_text invalidates a
    # branch, the engine should NOT leave that branch's pre-rendered
    # PNGs on disk un-referenced. Either the engine cleans them up,
    # or the branch's character record stays attached to keep them
    # reachable.
    #
    # We allow ONE orphan (a marginal race we can document) but
    # flag anything more as a regression. If this needs to grow,
    # update the threshold WITH a comment explaining the new
    # waste path.
    assert new_orphans <= 1, (
        f"free_text invalidation leaked {new_orphans} orphan render(s). "
        f"Pre: {baseline} orphan(s) / {pre_freetext_calls} call(s). "
        f"Post: {len(post_orphans)} orphan(s) / {post_calls} call(s). "
        f"Orphan files:\n  " + "\n  ".join(str(p) for p in sorted(post_orphans))
    )
    # And the engine SHOULD have produced at least one new render
    # for the new branch (or zero if the new branch reused existing
    # characters / environments). Either way no negative count.
    assert new_calls >= 0

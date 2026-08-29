"""Message handlers.

A handler runs against a ``HandlerContext`` (session + emit channel) and
yields outbound messages. Streaming text is emitted incrementally; state
changes are committed to disk before downstream events fire.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from pydantic import BaseModel, ValidationError

from ..background import spawn as _spawn_background
from ..config import PROTOCOL_VERSION, WORLD_INIT_DEADLINE_S
from ..domain.character import Character
from ..domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
    NewCharacterDescriptor,
)
from ..domain.environment import Environment
from ..domain.game import Game
from ..domain.world import WorldState
from ..orchestration import obsolescence
from ..orchestration.beat_text_cleanup import strip_speaker_slug
from ..orchestration.prompts import interview as interview_prompts
from ..orchestration.prompts import text_gen as text_gen_prompts
from ..orchestration.responses import (
    LlmCharacterPayload,
    LlmDialogPayload,
    LlmFactsConsolidation,
    LlmOptionList,
    LlmProfileConsolidation,
    LlmRetconBeatBatch,
    LlmRetconCharacterUpdates,
    LlmRetconResult,  # noqa: F401 -- kept for any out-of-tree consumers
    LlmRetconWorldUpdates,
    LlmSummaryResult,
    LlmSurpriseMeScenario,
    LlmWorldInit,
    call_llm_json_with_retry,
    parse_json_object,
)
from ..orchestration.session import (
    InterviewState,
    Session,
    empty_world,
    make_environment,
    random_seed,
)
from ..persistence import save_store
from .errors import (
    NotFoundError,
    ProviderUnreachableError,
    ProviderValidationError,
    SchemaError,
    new_correlation_id,
)
from .messages import (
    C2S_PAYLOAD_BY_TYPE,
    C2SEditCharacter,
    C2SEditCharacterDismiss,
    C2SEditCharacterRerender,
    C2SEditCharacterShow,
    C2SEditEnvironment,
    C2SEditEnvironmentApply,
    C2SEditEnvironmentRerender,
    C2SEditHistory,
    C2SEditHistoryDelete,
    C2SEditHistoryRetcon,
    C2SEditWorld,
    C2SNewGameAddSideCharacter,
    C2SNewGameAnswer,
    C2SNewGameConfirm,
    C2SNewGameDeleteSideCharacter,
    C2SNewGameEditSideCharacter,
    C2SNewGameGoBack,
    C2SNewGameStart,
    C2SNewGameSurpriseMe,
    C2SPlayAdvance,
    C2SPlayCancel,
    C2SPlayFreeText,
    C2SSavesDelete,
    C2SSavesLoad,
    C2SSavesRename,
    C2SSettingsUpdate,
    Envelope,
    ImageKind,
    InterviewStep,
    MessageType,
    PatchOp,
    S2CHello,
    S2CImageReady,
    S2CMusicReady,
    S2CPlayCancelled,
    S2CSavesList,
    S2CStateFull,
    S2CStatePatch,
    S2CTextComplete,
    S2CTextStreaming,
    SaveSummaryPayload,
)

if TYPE_CHECKING:
    from ..orchestration.assets import GeneratedAsset
    from ..orchestration.streaming_dialog_parser import StreamingDialogParser

_log = logging.getLogger(__name__)


@dataclass
class HandlerContext:
    session: Session


OutboundMessage = tuple[MessageType, BaseModel]
HandlerResult = AsyncIterator[OutboundMessage]
Handler = Callable[[BaseModel, HandlerContext], Awaitable[HandlerResult]]


def _redacted_validation_message(message_type: str, exc: BaseException) -> str:
    """Describe a payload validation failure without echoing the payload.

    ``str(ValidationError)`` embeds ``input_value`` for every failing
    field, which would round-trip whatever the client (or a corrupt
    save) sent — including free text and paths — back over the socket.
    We keep only ``loc`` and ``type``, which is what a caller needs to
    fix the shape, and log the full error against a correlation id.
    """
    ref = new_correlation_id()
    _log.error("payload validation failed for %s [ref %s]: %r", message_type, ref, exc)
    if isinstance(exc, ValidationError):
        details = ", ".join(
            f"{'.'.join(str(part) for part in err['loc']) or '<root>'} ({err['type']})"
            for err in exc.errors()
        )
    else:
        details = type(exc).__name__
    return f"invalid payload for {message_type}: {details} (ref: {ref})"


class HandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[MessageType, Handler] = {}

    def register(self, message_type: MessageType, handler: Handler) -> None:
        self._handlers[message_type] = handler

    async def dispatch(
        self, envelope: Envelope, ctx: HandlerContext
    ) -> AsyncIterator[OutboundMessage]:
        payload_type = C2S_PAYLOAD_BY_TYPE.get(envelope.type)
        if payload_type is None:
            raise SchemaError(f"unsupported message type {envelope.type.value!r}")
        try:
            validated = payload_type.model_validate(envelope.payload)
        except Exception as exc:
            raise SchemaError(_redacted_validation_message(envelope.type.value, exc)) from exc

        handler = self._handlers.get(envelope.type)
        if handler is None:
            raise NotFoundError(f"no handler registered for {envelope.type.value}")

        async for outbound in await handler(validated, ctx):
            yield outbound


# ---------- helpers ----------


def _full_state_message(session: Session) -> S2CStateFull:
    if session.game is None:
        msg = "cannot emit s2c/state/full without an installed game"
        raise SchemaError(msg)
    return S2CStateFull(game=session.game, settings=session.settings)


def _save_summary_payloads(saves_root: Path) -> list[SaveSummaryPayload]:
    return [
        SaveSummaryPayload(
            id=s.id,
            name=s.name,
            last_played_at=s.last_played_at.isoformat(),
            created_at=s.created_at.isoformat(),
            schema_version=s.schema_version,
            summary=s.summary,
            corrupt=s.corrupt,
        )
        for s in save_store.list_saves(saves_root)
    ]


async def _emit_text(node_id: str, chunks: list[str]) -> AsyncIterator[OutboundMessage]:
    for chunk in chunks:
        yield (MessageType.s2c_text_streaming, S2CTextStreaming(node_id=node_id, delta=chunk))
    yield (MessageType.s2c_text_complete, S2CTextComplete(node_id=node_id))


# ---------- hello / saves ----------


async def hello_handler(_payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    has_save = save_store.most_recent_save_id(ctx.session.saves_root) is not None
    response = S2CHello(protocol_version=PROTOCOL_VERSION, has_save=has_save)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_hello, response)

    return gen()


async def saves_list_handler(_payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    payloads = _save_summary_payloads(ctx.session.saves_root)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_saves_list, S2CSavesList(saves=payloads))

    return gen()


async def saves_continue_handler(_payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    save_id = save_store.most_recent_save_id(ctx.session.saves_root)
    if save_id is None:
        raise NotFoundError("no saves available")
    game = save_store.load_save(save_id, root=ctx.session.saves_root)
    ctx.session.install_game(game)
    # Same backfill pass as saves_load: kick off ensure_assets so
    # any character / environment that was committed without a
    # finished render gets one.
    _spawn_async_assets(ctx.session)
    # Backfill background music if the save loaded with music
    # enabled but no track on disk.
    _ensure_music_on_load(ctx.session)
    # Pre-generate the next beat (or a speculation per option) so the
    # player's first Continue/choice on a freshly-loaded run lands as
    # a tree-walk instead of stalling on a synchronous LLM call.
    _ensure_speculative_branches(ctx.session)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_state_full, _full_state_message(ctx.session))

    return gen()


async def saves_load_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SSavesLoad)
    game = save_store.load_save(payload.save_id, root=ctx.session.saves_root)
    ctx.session.install_game(game)
    # Saves can carry characters whose ``images`` list is empty —
    # either the run was committed before the asset pipeline finished
    # rendering them, or the player edited an attribute (outfit /
    # pose) since the last render so the existing entries are stale
    # for the current ``_portrait_prompt_hash``. ``ensure_assets``
    # is idempotent: it skips characters whose latest hash is
    # already on disk and only generates the missing ones, so this
    # background pass costs nothing on a fully-rendered save and
    # rescues a partially-rendered one. Same hook the play handlers
    # use after a turn lands.
    _spawn_async_assets(ctx.session)
    _ensure_music_on_load(ctx.session)
    # Pre-generate the next beat (or a speculation per option) so
    # the first Continue/choice on a freshly-loaded run is a
    # tree-walk rather than a synchronous LLM round-trip.
    _ensure_speculative_branches(ctx.session)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_state_full, _full_state_message(ctx.session))

    return gen()


async def saves_rename_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SSavesRename)
    save_store.rename_save(payload.save_id, payload.new_name, root=ctx.session.saves_root)
    payloads = _save_summary_payloads(ctx.session.saves_root)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_saves_list, S2CSavesList(saves=payloads))

    return gen()


async def saves_delete_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SSavesDelete)
    save_store.delete_save(payload.save_id, root=ctx.session.saves_root)
    payloads = _save_summary_payloads(ctx.session.saves_root)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_saves_list, S2CSavesList(saves=payloads))

    return gen()


# ---------- new game ----------


def _surface_random(options: list[str], *, count: int) -> list[str]:
    if len(options) <= count:
        return list(options)
    return random.sample(options, count)


async def new_game_start_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SNewGameStart)
    # Step 1 (Setting) uses hard-coded options; no LLM round trip
    # required. FR-022: bundled white-room + dream-guide imagery is
    # exposed up front. This handler also RESETS the session's
    # interview state and cancels any in-flight prefetch task —
    # without that, a user who returns to the start screen and picks
    # New Game a second time would race two onboardings against each
    # other.
    from ..config import bundled_dream_guide_path, bundled_white_room_path
    from ..orchestration.session import InterviewState

    _cancel_pending_prefetch(ctx.session)
    # Cancel any speculative-branch tasks tied to the previous game.
    # The Continue → Menu → New Game round-trip would otherwise
    # leave them running in the background, racing the new
    # interview's LLM calls and potentially mutating the new game
    # with stale speculation. Each task is keyed by node-id of the
    # OLD game's tree, so they're irrelevant once we clear
    # ``session.game`` below.
    spec_tasks = ctx.session._speculative_tasks
    for task in spec_tasks.values():
        if not task.done():
            task.cancel()
    ctx.session._speculative_tasks = {}

    # Clear the previously-loaded game (if any). Without this, the
    # renderer's ``loading``-phase auto-route bounces back into the
    # main view on the loaded save's ``current_node_id`` the
    # MOMENT the player clicks Begin — flashing the OLD save's
    # content while the new ``world_init`` is still running. The
    # new game gets installed by ``new_game_confirm_handler`` once
    # the LLM finishes; until then the renderer's ``game`` is null
    # so the loading screen waits cleanly. A separate state-patch
    # below propagates the same null to the renderer's store.
    ctx.session.game = None

    options = list(interview_prompts.HARDCODED_SETTING_OPTIONS)
    surfaced = _surface_random(options, count=5)

    dream_guide = str(bundled_dream_guide_path())
    white_room = str(bundled_white_room_path())

    ctx.session.interview = InterviewState(
        setting_options=options,
        dream_guide_image_path=dream_guide,
        white_room_image_path=white_room,
        preview_character_slug=payload.preview_character_slug,
    )

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    # Null out the renderer's game store so the
                    # loading-phase auto-route stops finding a
                    # current_node_id when the player clicks
                    # Begin. See the comment on session.game = None
                    # above for the full bounce-back symptom.
                    PatchOp(op="replace", path="/game", value=None),
                    # Replacing the entire ``/interview`` subtree
                    # atomically clears any partial answers from a
                    # prior aborted onboarding before installing the
                    # fresh options.
                    PatchOp(
                        op="replace",
                        path="/interview",
                        value={
                            "step": "setting",
                            "setting_options": surfaced,
                            "dream_guide_image_path": dream_guide,
                            "white_room_image_path": white_room,
                            "preview_character_slug": payload.preview_character_slug,
                        },
                    ),
                ]
            ),
        )

    return gen()


_NEXT_STEP_AFTER: dict[InterviewStep, str] = {
    # Reordered (FR-027b): Setting → Visual Style → Genre →
    # Character → Name. Setting + Visual Style up front lets the
    # backend kick off the chosen-style background render the moment
    # they're known; Character moves later so its portrait render
    # can replace the dream guide right before Review.
    InterviewStep.setting: "visual_style",
    InterviewStep.visual_style: "genre",
    InterviewStep.genre: "character_description",
    InterviewStep.character_description: "name",
}


async def _prefetch_character_description(session: Session, setting: str) -> list[str]:
    """Background LLM call kicked off when Step 2 (Genre) is presented.

    By the time the user picks Genre and Visual Style, this task is
    typically done; the Step-4 handler just awaits its result instead
    of starting a fresh LLM call from scratch.
    """
    raw, _chunks = await session.llm_text(
        interview_prompts.character_description_options(setting=setting)
    )
    return parse_json_object(raw, LlmOptionList).options


def _consume_or_schedule_emit(
    session: Session,
    *,
    attr: str,
    field_name: str,
    fallback: Callable[[], list[dict[str, str]]] | None,
    surface_count: int = 5,
) -> list[str]:
    """Non-blocking options retrieval for an interview step transition.

    Three cases:
      1. The prefetch task is already done — return its result inline
         (typical happy path: prefetch finished while player was reading).
      2. The prefetch task is in flight — return empty options NOW so
         the answer handler can transition the player to the next step
         immediately, and schedule a fire-and-emit follow-up that
         publishes the options via ``s2c/state/patch`` when ready.
      3. No prefetch was scheduled — start one from ``fallback`` and
         schedule the same fire-and-emit follow-up. Used as a guard
         against any path that forgot to prefetch upstream.

    The contract is that the caller NEVER blocks on the LLM. Slow
    providers extend the time-to-options-appearing, but they never
    delay the visible step transition.
    """
    task: asyncio.Task[list[str]] | None = getattr(session, attr, None)
    if task is None and fallback is not None:
        task = asyncio.create_task(_run_options_fallback(session, fallback))
        setattr(session, attr, task)
    if task is None:
        return []
    if task.done() and not task.cancelled():
        try:
            options = task.result()
            setattr(session, attr, None)
            return options
        except Exception:
            _log.warning("prefetch %s failed; emitting empty", attr, exc_info=True)
            setattr(session, attr, None)
            return []
    # Still in flight — fire-and-emit when the task lands. The slot
    # is cleared inside ``_await_task_and_emit`` so subsequent answers
    # don't see a stale task pointer.
    _spawn_background(
        _await_task_and_emit(
            session,
            attr=attr,
            field_name=field_name,
            surface_count=surface_count,
        ),
        label=f"options-emit:{attr}",
    )
    return []


async def _run_options_fallback(
    session: Session,
    fallback: Callable[[], list[dict[str, str]]],
) -> list[str]:
    raw, _chunks = await session.llm_text(fallback())
    options = parse_json_object(raw, LlmOptionList).options
    return _dedupe_options(options)


def _dedupe_options(options: list[str]) -> list[str]:
    """Drop case-insensitive duplicates from an options list,
    preserving the first occurrence's casing.

    LLMs sometimes produce duplicate suggestions — most often
    on the name step where ``\"Mira\"``, ``\"Mira Quill\"``,
    ``\"Mira Vale\"`` arrive in the same payload, or where the
    same name shows up twice with different trailing
    whitespace. The renderer paints them as separate buttons,
    which reads as a bug. Filter once at the parse boundary so
    every downstream consumer (sync return, async emit, the
    surface-random sampler) sees a clean list.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw_option in options:
        text = (raw_option or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


async def _await_task_and_emit(
    session: Session,
    *,
    attr: str,
    field_name: str,
    surface_count: int,
) -> None:
    """Await an in-flight prefetch task and emit its result via
    ``s2c/state/patch`` so the renderer's options grid populates as
    soon as the LLM returns. Best-effort; failures clear the slot."""
    import time

    task = getattr(session, attr, None)
    if task is None:
        return
    started = time.monotonic()
    try:
        options = await task
    except asyncio.CancelledError:
        return
    except Exception:
        _log.warning("async prefetch %s failed", attr, exc_info=True)
        setattr(session, attr, None)
        return
    setattr(session, attr, None)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    surfaced = _surface_random(options, count=surface_count)
    _log.info(
        "async options ready field=%s in %d ms (n=%d)",
        field_name,
        elapsed_ms,
        len(surfaced),
    )
    try:
        session.interview = session.interview.model_copy(update={field_name: surfaced})
    except Exception:
        _log.warning("async options persist failed for %s", field_name, exc_info=True)
    if session.emit is None:
        return
    try:
        session.emit(
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path=f"/interview/{field_name}",
                        value=surfaced,
                    )
                ]
            ),
        )
    except Exception:
        _log.warning("async options emit failed for %s", field_name, exc_info=True)


async def _prefetch_world_init(
    session: Session,
    *,
    setting: str,
    genre: str,
    visual_style: str,
    character_description: str,
    character_name: str,
    side_characters: list[str],
) -> str:
    """Background LLM call kicked off the moment the player reaches the
    Review step. By the time they actually click Begin, the longest
    LLM call of the whole onboarding is usually done.
    """
    from ..orchestration.assets import _long_context_image_model

    raw, _chunks = await session.llm_text(
        interview_prompts.world_init(
            setting=setting,
            genre=genre,
            visual_style=visual_style,
            character_description=character_description,
            name=character_name,
            side_characters=side_characters,
            mature_content=session.settings.mature_content,
            music_enabled=session.settings.music.enabled,
            detailed_appearance=_long_context_image_model(session),
        )
    )
    return raw


def _cancel_pending_prefetch(session: Session) -> None:
    """Cancel any in-flight speculative LLM tasks. Called whenever the
    interview is reset (e.g. user clicks New Game a second time)."""
    for attr in (
        "_char_desc_task",
        "_world_init_task",
        "_preview_bg_task",
        "_preview_guide_task",
        "_pc_portrait_task",
        "_name_options_task",
    ):
        task = getattr(session, attr, None)
        if task is not None and not task.done():
            task.cancel()
        setattr(session, attr, None)


async def _preview_render_background(
    session: Session,
    *,
    setting: str,
    genre: str,
    visual_style: str,
    slug: str,
) -> None:
    """Render the player-chosen setting in the player-chosen visual
    style and emit ``s2c/state/patch`` so the renderer can swap the
    cycling menu carousel to a frozen preview frame.

    Best-effort: failures are logged and dropped — the renderer keeps
    cycling menu pairs and the New Game flow proceeds normally.
    """
    import time

    from ..orchestration.menu_combos import (
        background_positive_for,
        combo_for_slug,
    )
    from ..persistence.atomic import atomic_write_bytes

    started = time.monotonic()
    combo = combo_for_slug(slug)
    # The player's chosen SETTING is the environment they picked, so it
    # must drive the preview background. The carousel ``slug`` is a
    # character-menu pairing whose baked-in location is unrelated to that
    # choice — using its combo location here rendered a backdrop that
    # didn't match the environment the player selected (the reported bug).
    # Prefer the setting; fall back to the combo location only when the
    # player left the setting blank (e.g. the surprise-me path).
    location_label = setting or (combo.location_label if combo else "")
    location_prompt = setting or (combo.location_prompt if combo else "")
    positive = background_positive_for(
        visual_style=visual_style,
        location_label=location_label,
        location_prompt=location_prompt,
        setting=setting,
        genre=genre,
    )
    target_dir = session.saves_root / "_interview_preview"
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"bg|{slug}|{setting}|{genre}|{visual_style}".encode()).hexdigest()[:24]
    target = target_dir / f"{digest}.png"

    try:
        if not target.exists():
            client = session._image_factory()
            payload = await client.generate(
                session.settings.image.background_workflow,
                {"positive_prompt": positive},
                seed=abs(hash(digest)) % (2**63),
            )
            atomic_write_bytes(target, payload)
    except asyncio.CancelledError:
        raise
    except Exception:
        _log.warning(
            "preview background render failed (slug=%s vs=%r)",
            slug,
            visual_style,
            exc_info=True,
        )
        return

    elapsed_ms = int((time.monotonic() - started) * 1000)
    _log.info(
        "preview background rendered slug=%s in %d ms -> %s",
        slug,
        elapsed_ms,
        target.name,
    )
    if session.emit is None:
        return
    try:
        session.interview = session.interview.model_copy(
            update={"preview_background_path": str(target)}
        )
        session.emit(
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path="/interview/preview_background_path",
                        value=str(target),
                    )
                ]
            ),
        )
    except Exception:
        _log.warning("preview background emit failed", exc_info=True)


async def _preview_render_pc_portrait(
    session: Session,
    *,
    character_description: str,
    visual_style: str,
) -> None:
    """Render a portrait of the player character — using their chosen
    character_description in the chosen visual_style — and emit
    ``s2c/state/patch`` on ``/interview/pc_portrait_path``. The
    renderer swaps the dream-guide layer for this portrait when it
    arrives, so the player sees their own avatar by the time they
    pick a name and reach Review.
    """
    import time

    from ..orchestration.menu_combos import (
        guide_negative_extras,
        guide_positive_for,
    )
    from ..orchestration.portrait_post import crop_to_figure
    from ..persistence.atomic import atomic_write_bytes

    if not character_description or not visual_style:
        return

    started = time.monotonic()
    positive = guide_positive_for(visual_style=visual_style, guide_body=character_description)
    target_dir = session.saves_root / "_interview_preview"
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"pc|{character_description}|{visual_style}".encode()).hexdigest()[:24]
    target = target_dir / f"{digest}.png"

    try:
        if not target.exists():
            client = session._image_factory()
            payload = await client.generate(
                session.settings.image.portrait_workflow,
                {
                    "positive_prompt": positive,
                    "face_prompt": character_description,
                    "negative_extras": guide_negative_extras(),
                },
                seed=abs(hash(digest)) % (2**63),
            )
            atomic_write_bytes(target, crop_to_figure(payload))
    except asyncio.CancelledError:
        raise
    except Exception:
        _log.warning(
            "preview pc-portrait render failed (vs=%r desc=%r)",
            visual_style,
            character_description[:60],
            exc_info=True,
        )
        return

    elapsed_ms = int((time.monotonic() - started) * 1000)
    _log.info("preview pc-portrait rendered in %d ms -> %s", elapsed_ms, target.name)
    if session.emit is None:
        return
    try:
        session.interview = session.interview.model_copy(update={"pc_portrait_path": str(target)})
        session.emit(
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path="/interview/pc_portrait_path",
                        value=str(target),
                    )
                ]
            ),
        )
    except Exception:
        _log.warning("preview pc-portrait emit failed", exc_info=True)


async def _preview_render_character(
    session: Session,
    *,
    slug: str,
    visual_style: str,
) -> None:
    """Re-render the menu-pair character (body+face from
    ``menu_combos``) in the player-chosen visual style. Same dream-guide
    figure across the entire onboarding flow — only the aesthetic
    shifts to match the player's pick.
    """
    import time

    from ..orchestration.menu_combos import (
        combo_for_slug,
        guide_negative_extras,
        guide_positive_for,
    )
    from ..orchestration.portrait_post import crop_to_figure
    from ..persistence.atomic import atomic_write_bytes

    combo = combo_for_slug(slug)
    if combo is None:
        # No matching menu pair — nothing to re-render. Keep cycling
        # the original carousel guide.
        return

    started = time.monotonic()
    positive = guide_positive_for(visual_style=visual_style, guide_body=combo.guide_body)
    target_dir = session.saves_root / "_interview_preview"
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"guide|{slug}|{visual_style}".encode()).hexdigest()[:24]
    target = target_dir / f"{digest}.png"

    try:
        if not target.exists():
            client = session._image_factory()
            payload = await client.generate(
                session.settings.image.portrait_workflow,
                {
                    "positive_prompt": positive,
                    "face_prompt": combo.guide_face,
                    "negative_extras": guide_negative_extras(),
                },
                seed=abs(hash(digest)) % (2**63),
            )
            atomic_write_bytes(target, crop_to_figure(payload))
    except asyncio.CancelledError:
        raise
    except Exception:
        _log.warning(
            "preview character render failed (slug=%s vs=%r)",
            slug,
            visual_style,
            exc_info=True,
        )
        return

    elapsed_ms = int((time.monotonic() - started) * 1000)
    _log.info(
        "preview character rendered slug=%s in %d ms -> %s",
        slug,
        elapsed_ms,
        target.name,
    )
    if session.emit is None:
        return
    try:
        session.interview = session.interview.model_copy(update={"preview_guide_path": str(target)})
        session.emit(
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path="/interview/preview_guide_path",
                        value=str(target),
                    )
                ]
            ),
        )
    except Exception:
        _log.warning("preview character emit failed", exc_info=True)


async def new_game_answer_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SNewGameAnswer)
    interview = ctx.session.interview
    update: dict[str, Any] = {}
    next_options: list[str] = []
    next_field: str = ""
    surfaced: list[str] | None = None

    if payload.step is InterviewStep.setting:
        update["setting"] = payload.answer
        # Step 2 (Visual Style) is hard-coded — no LLM round-trip.
        # Sample ONE entry from each of the four flavour buckets
        # (hyperrealistic / cartoony / artsy / oddball) so the four
        # on-screen options always span every aesthetic family. The
        # player never sees the raw 40-entry table.
        ctx.session._char_desc_task = asyncio.create_task(
            _prefetch_character_description(ctx.session, payload.answer)
        )
        next_field = "visual_style_options"
        next_options = [
            random.choice(bucket) for bucket in interview_prompts.VISUAL_STYLE_BUCKETS.values()
        ]

    elif payload.step is InterviewStep.visual_style:
        update["visual_style"] = payload.answer
        # The player has now picked Setting + Visual Style. Kick off
        # the chosen-style background render in parallel; when the PNG
        # lands the renderer hides the cycling dream guide and shows
        # the player's chosen-style backdrop. Genre is hard-coded —
        # no LLM round-trip — so the next step appears instantly.
        ctx.session._preview_bg_task = asyncio.create_task(
            _preview_render_background(
                ctx.session,
                setting=interview.setting,
                genre=interview.genre or "",
                visual_style=payload.answer,
                slug=interview.preview_character_slug,
            )
        )
        next_field = "genre_options"
        next_options = list(interview_prompts.HARDCODED_GENRE_OPTIONS)

    elif payload.step is InterviewStep.genre:
        update["genre"] = payload.answer
        # Step 4 (Character Description). Same non-blocking pattern as
        # Setting → Visual Style: surface prefetched options inline if
        # ready, otherwise transition with empty options and emit via
        # state/patch when the LLM returns.
        next_field = "character_description_options"
        next_options = _consume_or_schedule_emit(
            ctx.session,
            attr="_char_desc_task",
            field_name="character_description_options",
            fallback=lambda: interview_prompts.character_description_options(
                setting=interview.setting
            ),
        )

    elif payload.step is InterviewStep.character_description:
        update["character_description"] = payload.answer
        # The player character is NOT rendered before the game starts:
        # the PC is the camera lens, never an on-stage figure, and
        # rendering a full portrait during the interview burned a render
        # slot on an image that's never shown in play. The interview
        # keeps the dream-guide / chosen-style background preview instead.
        # Transition to the Name step IMMEDIATELY with empty options; the
        # ``_consume_or_schedule_emit`` path below starts the name-options
        # LLM fetch and emits the result via state/patch when it lands.
        next_field = "name_options"
        char_desc_for_name = payload.answer
        next_options = _consume_or_schedule_emit(
            ctx.session,
            attr="_name_options_task",
            field_name="name_options",
            fallback=lambda: interview_prompts.name_options(
                setting=interview.setting,
                character_description=char_desc_for_name,
            ),
        )

    elif payload.step is InterviewStep.name:
        update["character_name"] = payload.answer
        # Combined Identity step: payload.pronouns carries the
        # player's chosen pronouns alongside the name. Empty string
        # is allowed (means "no preference"), so we always overwrite
        # rather than gating on truthiness.
        update["pronouns"] = payload.pronouns
        ctx.session.interview = interview.model_copy(update=update)
        # Kick off the world_init LLM call in the background NOW —
        # while the player is reading the Review screen. The confirm
        # handler will await this task instead of starting a fresh
        # call. The world_init prompt is the single longest LLM call
        # in onboarding, so this prefetch shaves the perceived wait
        # all the way down to "as long as the user lingers on Review".
        ctx.session._world_init_task = asyncio.create_task(
            _prefetch_world_init(
                ctx.session,
                setting=ctx.session.interview.setting,
                genre=ctx.session.interview.genre,
                visual_style=ctx.session.interview.visual_style,
                character_description=ctx.session.interview.character_description,
                character_name=payload.answer,
                side_characters=ctx.session.interview.side_character_descriptions,
            )
        )

        async def gen_done() -> AsyncIterator[OutboundMessage]:
            yield (
                MessageType.s2c_state_patch,
                S2CStatePatch(
                    ops=[
                        PatchOp(
                            op="replace", path="/interview/character_name", value=payload.answer
                        ),
                        PatchOp(op="replace", path="/interview/pronouns", value=payload.pronouns),
                        PatchOp(op="replace", path="/interview/step", value="confirm"),
                    ]
                ),
            )

        return gen_done()
    else:  # pragma: no cover -- exhaustive
        raise SchemaError(f"unhandled interview step: {payload.step}")

    update[next_field] = next_options
    ctx.session.interview = interview.model_copy(update=update)

    if surfaced is None:
        surfaced = list(next_options)

    next_step_name = _NEXT_STEP_AFTER[payload.step]

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace", path=f"/interview/{payload.step.value}", value=payload.answer
                    ),
                    PatchOp(op="replace", path=f"/interview/{next_field}", value=surfaced),
                    PatchOp(op="replace", path="/interview/step", value=next_step_name),
                ]
            ),
        )

    return gen()


async def new_game_edit_review_handler(
    payload: BaseModel,
    ctx: HandlerContext,
) -> HandlerResult:
    """Edit a previously-answered interview field from the
    Review step. Update the named field in-place, cancel the
    in-flight world_init prefetch (which was launched against
    the OLD values when the player answered Name), and fire a
    fresh prefetch with the new values so the next confirm
    consumes a prefetch that matches what's on screen.

    Without the prefetch swap, an edit on Review would silently
    fall through: world_init would complete with the prior
    setting/genre/etc, the confirm handler would consume that
    stale result, and the player's edits would have no effect
    on the actual game state.

    Allow-listed to the five answered-step keys so a stray
    patch can't write arbitrary fields like ``side_characters``
    or ``preview_*_path``.
    """
    from ..api.messages import C2SNewGameEditReview

    assert isinstance(payload, C2SNewGameEditReview)
    valid_fields = {
        "setting",
        "visual_style",
        "genre",
        "character_description",
        "character_name",
        "pronouns",
    }
    if payload.field not in valid_fields:
        raise SchemaError(
            f"edit_review: unknown field {payload.field!r}; valid: {sorted(valid_fields)}"
        )
    interview = ctx.session.interview
    update = {payload.field: payload.value}
    ctx.session.interview = interview.model_copy(update=update)

    # Cancel the running prefetch (if any) and start a fresh one
    # with the corrected values. Best-effort: a CancelledError on
    # the old task is normal; anything else gets logged but not
    # propagated. Pronouns aren't part of the world_init prompt
    # (they're applied to the player Character at confirm time), so
    # an edit that only touches pronouns doesn't need a re-fire.
    new_iv = ctx.session.interview
    pronouns_only = payload.field == "pronouns"
    old_task = ctx.session._world_init_task
    if old_task is not None and not old_task.done() and not pronouns_only:
        old_task.cancel()
    # Only re-fire the prefetch if the player has finished the
    # name step at least once — otherwise there's no prefetch
    # to refresh, and edits are just state updates.
    if (new_iv.character_name or "").strip() and not pronouns_only:
        ctx.session._world_init_task = asyncio.create_task(
            _prefetch_world_init(
                ctx.session,
                setting=new_iv.setting,
                genre=new_iv.genre,
                visual_style=new_iv.visual_style,
                character_description=new_iv.character_description,
                character_name=new_iv.character_name,
                side_characters=new_iv.side_character_descriptions,
            )
        )

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path=f"/interview/{payload.field}",
                        value=payload.value,
                    ),
                ]
            ),
        )

    return gen()


async def new_game_add_side_character_handler(
    payload: BaseModel, ctx: HandlerContext
) -> HandlerResult:
    assert isinstance(payload, C2SNewGameAddSideCharacter)
    # Side-character add is now non-blocking: the description goes
    # straight into the interview state, and the LLM expansion runs
    # in parallel during confirm/world_init. This keeps the Add
    # button responsive — the player isn't stuck waiting for an
    # extra round-trip per NPC. Stubs are detected at confirm time
    # by their empty ``outfit`` field (real expansions always fill
    # outfit, pose, and expression).
    description = payload.description.strip()
    if not description:
        raise SchemaError("description cannot be empty")
    stub = Character(
        is_player=False,
        name=description,
        description=description,
        outfit="",
        pose="",
        expression="",
        age=18,
        seed=random_seed(),
    )
    ctx.session.interview.side_character_descriptions.append(description)
    ctx.session.interview.side_characters.append(stub)
    _log.info(
        "add_side_character: stub appended id=%s desc=%r (now %d entries; LLM expansion deferred to confirm)",
        stub.id,
        description,
        len(ctx.session.interview.side_characters),
    )

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="add",
                        path=f"/interview/side_characters/{stub.id}",
                        value={"id": stub.id, "name": stub.name},
                    )
                ]
            ),
        )

    return gen()


async def new_game_edit_side_character_handler(
    payload: BaseModel, ctx: HandlerContext
) -> HandlerResult:
    """Rename a side-character stub the player added earlier on the
    Review step. We mutate the stub in place (preserving its id and
    seed so the renderer's existing patch keys stay valid) and emit
    a ``replace`` op for the renderer. Expansion still happens at
    confirm time, so editing here is cheap — no LLM round-trip.
    """
    assert isinstance(payload, C2SNewGameEditSideCharacter)
    description = payload.description.strip()
    if not description:
        raise SchemaError("description cannot be empty")
    sides = ctx.session.interview.side_characters
    descs = ctx.session.interview.side_character_descriptions
    target_idx: int | None = None
    for i, side in enumerate(sides):
        if side.id == payload.character_id:
            target_idx = i
            break
    if target_idx is None:
        raise NotFoundError(f"unknown side character {payload.character_id!r}")
    updated = sides[target_idx].model_copy(
        update={"name": description, "description": description},
    )
    sides[target_idx] = updated
    # Keep the parallel description list in sync — confirm-time
    # expansion reads from this list when re-running the LLM stub
    # expansion. Without keeping it aligned, an edit would be
    # rendered to the player but lost the moment they pressed Begin.
    if target_idx < len(descs):
        descs[target_idx] = description
    _log.info(
        "edit_side_character: id=%s renamed to %r",
        payload.character_id,
        description,
    )

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path=f"/interview/side_characters/{updated.id}",
                        value={"id": updated.id, "name": updated.name},
                    )
                ]
            ),
        )

    return gen()


async def new_game_delete_side_character_handler(
    payload: BaseModel, ctx: HandlerContext
) -> HandlerResult:
    """Remove a side-character stub the player added earlier on the
    Review step. Drops the entry from both ``side_characters`` and
    the parallel ``side_character_descriptions`` list, then emits a
    ``remove`` op so the renderer drops its row too.
    """
    assert isinstance(payload, C2SNewGameDeleteSideCharacter)
    sides = ctx.session.interview.side_characters
    descs = ctx.session.interview.side_character_descriptions
    target_idx: int | None = None
    for i, side in enumerate(sides):
        if side.id == payload.character_id:
            target_idx = i
            break
    if target_idx is None:
        raise NotFoundError(f"unknown side character {payload.character_id!r}")
    removed = sides.pop(target_idx)
    if target_idx < len(descs):
        descs.pop(target_idx)
    _log.info(
        "delete_side_character: id=%s name=%r (now %d entries)",
        removed.id,
        removed.name,
        len(sides),
    )

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="remove",
                        path=f"/interview/side_characters/{removed.id}",
                    )
                ]
            ),
        )

    return gen()


async def _expand_side_character_stub(
    *,
    session: Session,
    stub: Character,
    description: str,
    setting: str,
) -> Character:
    """Run the LLM expansion that turns a one-liner into a fully
    populated NPC. Used at confirm time to fill in the stubs the
    add-side-character handler appended without blocking. The
    stub's id and seed are preserved so the patch keys the
    frontend already received still resolve."""
    from ..orchestration.responses import LlmCharacterPayload

    raw, _chunks = await session.llm_text(
        interview_prompts.side_character_expansion(
            one_line=description,
            setting=setting,
        )
    )
    try:
        expanded = parse_json_object(raw, LlmCharacterPayload)
    except Exception:
        _log.warning(
            "side-character expansion failed validation for %r; raw[:300]=%r — keeping stub as-is",
            description,
            raw[:300],
        )
        return stub
    return stub.model_copy(
        update={
            "name": expanded.name,
            "description": expanded.description,
            "gender": expanded.gender,
            "pronouns": getattr(expanded, "pronouns", "") or "",
            "age": expanded.age,
            "ethnicity": expanded.ethnicity,
            "skin": expanded.skin,
            "hair_color": expanded.hair_color,
            "hairstyle": expanded.hairstyle,
            "eye_color": expanded.eye_color,
            "build": expanded.build,
            "bust": expanded.bust,
            "outfit": expanded.outfit,
            "pose": expanded.pose,
            "expression": expanded.expression,
        }
    )


async def _acquire_world_init(
    session: Session,
    init_prompt: list[dict[str, str]],
) -> LlmWorldInit:
    """Resolve the ``world_init`` LLM response for a new game.

    Prefers the prefetch kicked off when the player answered the Name
    step — by the time they press Begin the longest onboarding call is
    usually already done. Falls back to a fresh retry-wrapped call when
    the prefetch was cancelled, never started (e.g. a save resumed
    mid-interview), or came back invalid.

    Split out of the confirm handler so the WHOLE acquisition — prefetch
    await, prefetch-invalid retry, and the no-prefetch path — runs under
    a single wall-clock deadline in the caller. Each of those internally
    stacks ``call_llm_json_with_retry`` × ``llm_text`` × ``complete``
    retries, so without an outer bound a wedged provider could freeze the
    new-game flow for many minutes.
    """
    raw: str | None = None
    task = session._world_init_task
    if task is not None:
        try:
            raw = await task
        except asyncio.CancelledError:
            # Awaiting a task auto-propagates our cancellation to it, so
            # ``task.cancelled()`` is True in BOTH the deadline case and
            # the independently-cancelled case and can't tell them
            # apart. Ask instead whether WE were asked to cancel: a
            # positive ``cancelling()`` count means the outer deadline
            # is aborting us — propagate so it surfaces as a clean
            # TimeoutError. Zero means the prefetch was cancelled out
            # from under us — fall back to a fresh call rather than
            # crashing the whole new-game flow with a stray cancel.
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0:
                raise
            raw = None
        finally:
            # Never leak the prefetch: cancel it if it somehow outlived
            # the await, and drop the handle either way.
            if not task.done():
                task.cancel()
            session._world_init_task = None
    if raw is not None:
        # Try to parse the prefetch result. If it's invalid (and not
        # auto-coerceable), fall through to the retry helper which
        # re-prompts with a corrective hint. The prefetch is one
        # expensive call; a transient validation error must never
        # silently nuke the whole new-game flow.
        try:
            return parse_json_object(raw, LlmWorldInit)
        except ProviderValidationError as exc:
            _log.warning(
                "world_init prefetch failed validation; retrying with corrective hint: %s",
                exc,
            )
            return await call_llm_json_with_retry(
                session,
                init_prompt,
                parse_into=LlmWorldInit,
                max_attempts=2,
            )
    # No prefetch (cancelled / never started). Fire the call now, with
    # retry from the start.
    return await call_llm_json_with_retry(
        session,
        init_prompt,
        parse_into=LlmWorldInit,
        max_attempts=3,
    )


async def new_game_confirm_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SNewGameConfirm)
    interview = ctx.session.interview
    # Build the world_init prompt once; the acquisition helper reuses it
    # for both the prefetch reuse path and the retry-on-validation-
    # failure path (the retry helper threads a corrective continuation
    # onto the original prompt).
    from ..orchestration.assets import _long_context_image_model

    init_prompt = interview_prompts.world_init(
        setting=interview.setting,
        genre=interview.genre,
        visual_style=interview.visual_style,
        character_description=interview.character_description,
        name=interview.character_name,
        side_characters=interview.side_character_descriptions,
        mature_content=ctx.session.settings.mature_content,
        music_enabled=ctx.session.settings.music.enabled,
        detailed_appearance=_long_context_image_model(ctx.session),
    )
    # Bound the whole acquisition by a wall-clock deadline. The retry
    # layers underneath (validation × llm_text × complete, each able to
    # wait out the 300 s httpx read timeout) can otherwise stack into a
    # multi-minute freeze on a wedged provider. On expiry we surface a
    # recoverable error so the player just presses Begin again rather
    # than watching an indefinite spinner. ``_acquire_world_init`` owns
    # cancelling the in-flight prefetch task when the deadline fires.
    try:
        init = await asyncio.wait_for(
            _acquire_world_init(ctx.session, init_prompt),
            timeout=WORLD_INIT_DEADLINE_S,
        )
    except TimeoutError as exc:
        raise ProviderUnreachableError(
            "World generation timed out — the story engine didn't "
            "respond in time. Check your model/connection in Settings "
            "and press Begin to try again."
        ) from exc

    world = empty_world(
        setting=interview.setting, genre=interview.genre, visual_style=interview.visual_style
    )
    # Materialise the structured plot outline. The LLM may emit the
    # new ``plot_outline`` shape, the legacy ``overall_plot_direction``
    # string, or both — synthesise a single-stage outline from the
    # legacy string when the structured field is empty so downstream
    # code (text_gen's "current stage" slice) always has something
    # to point at.
    plot_outline = list(init.plot_outline)
    if not plot_outline and init.overall_plot_direction:
        from ..domain.world import PlotStage as _PlotStage

        plot_outline = [
            _PlotStage(
                title="Opening arc",
                goal=init.overall_plot_direction,
                summary=init.overall_plot_direction,
            )
        ]
    current_stage_id = plot_outline[0].id if plot_outline else None
    # Modern LLM responses send a structured ``plot_outline`` and
    # leave ``overall_plot_direction`` blank. Derive a single-string
    # synopsis from the outline so the StoryPanel UI doesn't show an
    # empty field — the storyteller's prompt also still references
    # this string for legacy reasons.
    overall = init.overall_plot_direction.strip()
    if not overall and plot_outline:
        overall = " → ".join(stage.goal for stage in plot_outline if stage.goal)
    # Seed the summarizer assessment with the opening setup so the UI
    # has SOMETHING from turn 1. The summarizer will overwrite this
    # the first time it runs after a turn settles.
    seed_assessment = ""
    if plot_outline:
        first = plot_outline[0]
        seed_assessment = (
            f"Opening stage '{first.title}': {first.summary or first.goal}. "
            f"Story has not yet begun — first turn will establish baseline."
        )
    world = world.model_copy(
        update={
            "game_name": init.game_name,
            "overall_plot_direction": overall,
            "plot_outline": plot_outline,
            "current_stage_id": current_stage_id,
            "active_plot_threads": init.active_plot_threads,
            "summarizer_assessment": seed_assessment,
        }
    )

    pc_descriptor = init.player_character or _from_interview_player(interview)
    player_character = Character(
        is_player=True,
        name=pc_descriptor.name,
        description=pc_descriptor.description,
        gender=pc_descriptor.gender,
        # Player-chosen pronouns from the Identity step win over
        # whatever the world_init LLM put on the descriptor; this is
        # the player's call, not the model's. Falls back to the
        # descriptor when the player didn't pick anything (legacy
        # saves / surprise_me path).
        pronouns=interview.pronouns or getattr(pc_descriptor, "pronouns", "") or "",
        age=pc_descriptor.age,
        ethnicity=pc_descriptor.ethnicity,
        skin=pc_descriptor.skin,
        hair_color=pc_descriptor.hair_color,
        hairstyle=pc_descriptor.hairstyle,
        eye_color=pc_descriptor.eye_color,
        build=pc_descriptor.build,
        bust=pc_descriptor.bust,
        outfit=pc_descriptor.outfit,
        pose=pc_descriptor.pose,
        expression=pc_descriptor.expression,
        effects=getattr(pc_descriptor, "effects", "") or "",
        decals=getattr(pc_descriptor, "decals", "") or "",
        seed=random_seed(),
    )

    characters = {player_character.id: player_character}
    # Expand any side-character stubs left by the deferred-add path.
    # Stubs are detected by an empty ``outfit`` field — the LLM
    # expansion always fills it. Running in parallel keeps the
    # confirm latency near a single round-trip even with multiple
    # side characters queued up.
    # ``strict=False``: the two lists are maintained independently by the
    # interview flow and may legitimately be ragged; truncating to the
    # shorter one is the established behaviour here.
    pairs = list(
        zip(
            interview.side_characters,
            interview.side_character_descriptions,
            strict=False,
        )
    )
    stubs = [(s, d) for s, d in pairs if not s.outfit]
    expanded_pairs: list[tuple[Character, Character]] = []
    if stubs:
        expansions = await asyncio.gather(
            *(
                _expand_side_character_stub(
                    session=ctx.session,
                    stub=s,
                    description=d,
                    setting=interview.setting,
                )
                for s, d in stubs
            )
        )
        expanded_pairs = list(zip([s for s, _ in stubs], expansions, strict=True))
    expanded_by_stub_id = {stub.id: exp for stub, exp in expanded_pairs}
    for side in interview.side_characters:
        resolved = expanded_by_stub_id.get(side.id, side)
        characters[resolved.id] = resolved

    if not init.opening_node.beats:
        raise SchemaError("world_init returned no opening beats")

    # Materialize any character descriptors the LLM emitted in any
    # opening beat — same shape as in-game beats — so on-stage ids
    # can resolve cleanly when the chain is built below. Skip any
    # descriptor that would collide with the player character (same
    # id or same name) so the player never has a same-name shadow.
    for beat in init.opening_node.beats:
        for descriptor in beat.new_characters:
            if descriptor.id in characters:
                continue
            if _collides_with_player(descriptor, characters):
                _log.warning(
                    "dropping NPC descriptor %r (id=%s) — collides with player name",
                    descriptor.name,
                    descriptor.id,
                )
                continue
            characters[descriptor.id] = _materialize_new_character(descriptor)

    # Build the opening chain — one node per beat. Per FR-034a the
    # player is NOT a stage actor; we filter them out of every
    # beat's entering_character_ids.
    def _without_player(ids: list[str]) -> list[str]:
        return [cid for cid in ids if cid not in characters or not characters[cid].is_player]

    opening_premise = ctx.session.compute_premise_hash(parent_id=None, chosen_option_id=None)
    opening_chain: list[DialogNode] = []
    cur_parent_id: str | None = None
    for index, beat in enumerate(init.opening_node.beats):
        is_last = index == len(init.opening_node.beats) - 1
        node = DialogNode(
            parent_id=cur_parent_id,
            chosen_option_id=None,
            speaker_id=beat.speaker_id,
            text=strip_speaker_slug(beat.text),
            options=init.opening_node.options if is_last else [],
            entering_character_ids=_without_player(beat.entering_character_ids),
            leaving_character_ids=beat.leaving_character_ids,
            new_characters=beat.new_characters,
            location_id=beat.location_id,
            location_prompt=beat.location_prompt,
            location_lighting=beat.location_lighting,
            character_changes=beat.character_changes,
            state=DialogNodeState.committed,
            premise_hash=opening_premise,
        )
        opening_chain.append(node)
        cur_parent_id = node.id

    first_beat = opening_chain[0]
    on_stage_ids: list[str] = list(first_beat.entering_character_ids)
    # An opening beat that starts mid-line ("Took you long enough,"
    # she says) names a speaker the LLM often forgets to also list as
    # entering. Speaking is presence: put them on stage so the very
    # first frame never shows a name tag with nobody behind it.
    # ``Session.install_game`` enforces the same rule for every later
    # beat — see ``domain.game.with_speaker_on_stage``.
    if (
        first_beat.speaker_id
        and first_beat.speaker_id in characters
        and not characters[first_beat.speaker_id].is_player
        and first_beat.speaker_id not in on_stage_ids
    ):
        on_stage_ids.append(first_beat.speaker_id)

    # Promote the first beat's location_prompt into a real Environment
    # entry so ensure_assets can produce a background. Subsequent
    # beats may carry additional location_prompt values; they get
    # promoted lazily when the player walks to them.
    environments: dict[str, Environment] = {}
    if first_beat.location_prompt:
        env = make_environment(
            label=first_beat.location_id or "opening",
            prompt=first_beat.location_prompt,
        )
        env = env.model_copy(update={"lighting": first_beat.location_lighting})
        environments[env.id] = env
        first_beat = first_beat.model_copy(update={"location_id": env.id})
        opening_chain[0] = first_beat

    nodes_dict = {n.id: n for n in opening_chain}
    tree = DialogTree(
        nodes=nodes_dict,
        root_id=first_beat.id,
        committed_path=[first_beat.id],
    )

    game = Game(
        world=world,
        characters=characters,
        dialog_tree=tree,
        environments=environments,
        current_node_id=first_beat.id,
        on_stage=on_stage_ids,
    )
    ctx.session.install_game(game)
    await ctx.session.commit()

    # First-frame assets are AWAITED so the opening's first three lines
    # paint COMPLETE: the initial backdrop plus the portrait of every
    # NPC visible across the first up-to-three beats (the lines the
    # player reads before anything could have streamed in). Everything
    # past the first frame — NPCs a later beat introduces, scenes the
    # player hasn't walked to — is left to the background render path
    # (each Continue spawns assets for the beat it lands on), so a long
    # opening still paints fast. On a single GPU these renders are
    # serial; awaiting ONLY the first frame keeps the wait bounded to
    # the backdrop + the handful of faces on screen at the start, not
    # the whole cast. ensure_portraits_for runs the minor-nudity safety
    # pass before building any prompt.
    first_frame_char_ids: list[str] = []
    _ff_seen: set[str] = set()
    _ff_active: set[str] = set()

    def _want_first_frame_face(cid: str) -> bool:
        return cid not in _ff_seen and cid in characters and not characters[cid].is_player

    for node in opening_chain[:3]:
        _ff_active |= set(node.entering_character_ids)
        # The beat's speaker is on screen for that beat whether or not
        # the LLM listed them as entering, and stays on screen even if
        # this beat also walks them out (deferred exit — see
        # ``_apply_node``). Either way their face is needed for the
        # first frame, so they're checked BEFORE the leaving pass.
        if node.speaker_id and _want_first_frame_face(node.speaker_id):
            _ff_active.add(node.speaker_id)
            _ff_seen.add(node.speaker_id)
            first_frame_char_ids.append(node.speaker_id)
        _ff_active -= {cid for cid in node.leaving_character_ids if cid != node.speaker_id}
        for cid in node.entering_character_ids:
            if cid in _ff_active and _want_first_frame_face(cid):
                _ff_seen.add(cid)
                first_frame_char_ids.append(cid)

    from ..orchestration.assets import (
        _resolve_active_environment_id,
        ensure_backgrounds_for,
        ensure_portraits_for,
    )

    first_frame_assets: list[GeneratedAsset] = []
    try:
        _image_client = ctx.session._image_factory()
        first_frame_assets += await ensure_portraits_for(
            session=ctx.session,
            image_client=_image_client,
            character_ids=first_frame_char_ids,
            saves_root=ctx.session.saves_root,
        )
        # ``install_game`` above guarantees a game is installed; the
        # cast is purely for the type checker (this whole block is
        # inside the swallow-and-log try).
        _active_env_id = _resolve_active_environment_id(cast("Game", ctx.session.game))
        first_frame_assets += await ensure_backgrounds_for(
            session=ctx.session,
            image_client=_image_client,
            environment_ids=[_active_env_id] if _active_env_id else [],
            saves_root=ctx.session.saves_root,
        )
    except Exception as exc:
        _log.exception("opening first-frame asset generation failed; continuing with placeholders")
        _emit_render_failure_notice(ctx.session, exc, kind="portrait/background")
    await ctx.session.commit()

    # Everything past the first frame streams in the background — the
    # same render-scheduler path every in-game turn takes. At the
    # opening node this is a no-op once the first frame already covered
    # the on-stage cast; it stays as a safety net so an on-stage face
    # whose first-frame render FAILED (or any on-stage entity the
    # first-frame set missed) is still queued rather than left blank.
    _spawn_async_assets(ctx.session)
    # Kick off speculative branches for the opening node's options so
    # the player's first real choice walks an existing chain instead
    # of waiting on a fresh LLM call. Same chain-tail speculation as
    # the storyteller path: speculate the first option of the LAST
    # opening beat at commit time so the next chain's LLM call has
    # the player's reading window through the opening beats as a
    # head start. One option only (not all) to avoid saturating the
    # LLM scheduler — see the equivalent comment in
    # ``_generate_and_walk_chain``.
    _ensure_speculative_branches(ctx.session)
    if opening_chain and opening_chain[-1].id != first_beat.id and opening_chain[-1].options:
        _ensure_speculative_branches(
            ctx.session,
            target_node_id=opening_chain[-1].id,
            options_subset=[opening_chain[-1].options[0].id],
        )
    # Music gen is SLOW (ACE-Step: 30-90 s) and never gates the
    # opening — the renderer's BackgroundMusic component starts silent
    # and crossfades the track in once s2c/music/ready arrives. Spawn
    # it as a background task (the exact path an in-game music_change
    # takes via _spawn_async_music) instead of AWAITING it here: the
    # prior inline await blocked the whole confirm response — and thus
    # the first painted beat — on a full music render, so a player who
    # enabled music watched a multi-minute spinner before the opening
    # appeared. The task emits s2c/music/ready + a state/full when the
    # track lands.
    music_prompt = (init.initial_music_prompt or "").strip()
    if music_prompt and ctx.session.settings.music.enabled:
        _spawn_async_music(ctx.session, music_prompt)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_state_full, _full_state_message(ctx.session))
        # image_ready for the first-frame assets we awaited above — the
        # backdrop + first-three-lines faces are on disk by now, so the
        # renderer can present them immediately rather than waiting for a
        # background push. Later-beat assets arrive via session.emit as
        # the render scheduler completes them.
        for asset in first_frame_assets:
            yield (
                MessageType.s2c_image_ready,
                S2CImageReady(
                    kind=ImageKind.portrait if asset.kind == "portrait" else ImageKind.background,
                    target_id=asset.target_id,
                    image_path=str(asset.image_path),
                    image_id=asset.prompt_hash,
                ),
            )

    return gen()


async def new_game_go_back_handler(
    payload: BaseModel,
    ctx: HandlerContext,
) -> HandlerResult:
    """Abandon the new-game interview and return to the main menu.

    The Review screen's ``Go back`` button now wipes the entire
    interview rather than rewinding step-by-step — the player wanted
    a clean exit, not a partial undo. The renderer transitions to
    the main menu optimistically (without waiting on us); our job
    here is to:

      * cancel every in-flight prefetch the interview spawned —
        world_init, character-description options, name options,
        preview background, preview guide, PC portrait — so they
        don't burn LLM / image credits on a session the player has
        walked away from;
      * reset ``session.interview`` to a fresh ``InterviewState`` so
        the next New Game click starts from a clean slate (the
        renderer's reset also kicks in, but server-side state is the
        source of truth for the next ``c2s/new_game/start``).

    No outbound message: the renderer doesn't need a confirmation
    state-patch — it has already navigated away. Returning an empty
    generator keeps the dispatcher's contract intact.
    """
    assert isinstance(payload, C2SNewGameGoBack)
    from ..orchestration.session import InterviewState

    # Cancel every in-flight task this interview spawned. The Session
    # exposes them as private ``_<task>_task`` attrs (see new_game_*
    # handlers above); collect by name so a future task hook gets
    # picked up automatically if it follows the same pattern.
    cancellable_attrs = (
        "_world_init_task",
        "_char_desc_task",
        "_name_options_task",
        "_preview_bg_task",
        "_preview_guide_task",
        "_pc_portrait_task",
    )
    for attr in cancellable_attrs:
        task = getattr(ctx.session, attr, None)
        if task is not None and not task.done():
            task.cancel()
        setattr(ctx.session, attr, None)

    # Fresh interview state. The renderer is already navigating to
    # the main menu so we don't need to push a state-patch with the
    # blank fields — but keeping server-side state clean prevents a
    # half-filled InterviewState from leaking into the next session
    # via ``c2s/new_game/start``.
    ctx.session.interview = InterviewState()

    async def gen() -> AsyncIterator[OutboundMessage]:
        return
        yield  # pragma: no cover -- unreachable; keeps async-gen typing

    return gen()


async def new_game_surprise_me_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    """Skip the entire interview: ask the LLM to author a complete
    starting scenario tailored to the player's cross-save profile,
    re-use the most recent save's visual style, then dive straight
    into ``new_game_confirm_handler`` to do the heavy world_init +
    opening-chain build.

    Two LLM calls run sequentially: the surprise-me scenario call
    (small — four short fields) followed by world_init (the usual
    long one). The renderer transitions to the loading screen as
    soon as it sends the request and stays there until the engine
    returns the full state — same UX as Continue.
    """
    from ..orchestration.session import InterviewState
    from ..persistence import save_store

    assert isinstance(payload, C2SNewGameSurpriseMe)

    _cancel_pending_prefetch(ctx.session)

    # Cross-save profile (set & curated by player + summarizer over
    # past sessions). ``merged_*`` deduplicates between the player-
    # entered and summarizer-inferred buckets.
    profile = ctx.session.settings.user_profile
    # Surprise-me feeds the new-game seed scenario; gating at the
    # surface threshold keeps speculative below-threshold tags
    # from steering brand-new saves toward genres the engine
    # only TENTATIVELY thinks the player likes.
    likes = profile.surfaced_likes()
    dislikes = profile.surfaced_dislikes()
    notes = profile.surfaced_notes()

    # Re-use the most recent save's visual style. ``list_saves``
    # returns summaries sorted most-recent-first; the visual style
    # itself lives on world.visual_style and isn't in the summary,
    # so we load the actual save to read it. Falls back to the
    # first hardcoded visual style on a fresh install with no
    # saves yet — same shape the interview's first option would be.
    visual_style = ""
    try:
        saves = save_store.list_saves(root=ctx.session.saves_root)
    except Exception:
        saves = []
    if saves:
        try:
            latest_game = save_store.load_save(saves[0].id, root=ctx.session.saves_root)
            visual_style = (latest_game.world.visual_style or "").strip()
        except Exception:
            _log.warning(
                "surprise-me: failed to read latest save %s for visual style",
                saves[0].id,
                exc_info=True,
            )
    if not visual_style:
        visual_style = interview_prompts.VISUAL_STYLES_HYPERREALISTIC[0]

    scenario_prompt = interview_prompts.surprise_me_scenario(
        visual_style=visual_style,
        likes=likes,
        dislikes=dislikes,
        notes=notes,
    )
    raw, _ = await ctx.session.llm_text(scenario_prompt, max_tokens=2048)
    scenario = parse_json_object(raw, LlmSurpriseMeScenario)

    # Pre-fill the InterviewState so new_game_confirm_handler can
    # read every field the standard onboarding flow would have set.
    # No prefetch tasks land on the session — surprise-me runs
    # world_init synchronously below.
    ctx.session.interview = InterviewState(
        setting=scenario.setting,
        visual_style=visual_style,
        genre=scenario.genre,
        character_description=scenario.character_description,
        character_name=scenario.name,
        side_character_descriptions=[],
        side_characters=[],
    )

    return await new_game_confirm_handler(C2SNewGameConfirm(), ctx)


def _from_interview_player(interview: InterviewState) -> LlmCharacterPayload:
    """Synthesize a minimal player character payload when world_init omits it."""
    from ..orchestration.responses import LlmCharacterPayload

    return LlmCharacterPayload(
        name=interview.character_name or "Protagonist",
        description=interview.character_description or "the protagonist",
        gender="unspecified",
        age=25,
        ethnicity="unspecified",
        skin="unspecified",
        hair_color="brown",
        hairstyle="medium",
        eye_color="brown",
        build="average",
        bust="unspecified",
        outfit="travel clothes",
        pose="standing",
        expression="alert",
    )


# ---------- play ----------


async def _generate_beat_chain(
    ctx: HandlerContext,
    *,
    parent: DialogNode,
    chosen_option_id: str | None,
    chosen_option_text: str | None,
    free_text: str | None = None,
) -> tuple[list[DialogNode], list[str]]:
    return await _generate_beat_chain_for_session(
        ctx.session,
        parent=parent,
        chosen_option_id=chosen_option_id,
        chosen_option_text=chosen_option_text,
        free_text=free_text,
        stream=True,
    )


async def _generate_beat_chain_for_session(
    session: Session,
    *,
    parent: DialogNode,
    chosen_option_id: str | None,
    chosen_option_text: str | None,
    free_text: str | None = None,
    stream: bool = True,
    on_beat_streamed: Callable[[DialogNode, bool], Awaitable[None]] | None = None,
) -> tuple[list[DialogNode], list[str]]:
    """Run an LLM call and return a chain of committed beat nodes.

    The first beat carries ``chosen_option_id`` (so the obsolescence
    layer can match the user's chosen branch later); subsequent
    beats have None. Only the LAST beat carries options. All beats
    share the same premise hash — they were authored under the same
    parent state.

    Streaming mode: when ``stream=True``, the function consumes the
    LLM response chunk-by-chunk through ``StreamingDialogParser``
    so beats can be materialised as ``DialogNode`` instances the
    moment their closing brace arrives — instead of waiting for the
    full LLM payload to finish generating. The ``on_beat_streamed``
    callback (when supplied) fires per beat with ``(node,
    is_first)`` so the foreground play-advance flow can install
    the first beat into the live game and emit text + state
    patches BEFORE the rest of the LLM response finishes
    generating. The callback receives provisional nodes — the
    LAST beat's ``options`` aren't known until the full response
    parses, so the function still returns a fully-formed chain
    where the last node carries the parsed options. Callers that
    don't pass a callback get the same return shape as the
    non-streaming path.
    """
    from ..domain.dialog import GenerationMetadata

    assert session.game is not None
    game = session.game

    history = obsolescence.committed_path(game)
    # Removed characters (dead or player-dismissed) are filtered out
    # of both maps so the storyteller never sees them and can't try
    # to re-summon them. ``Session.undo_stack`` snapshots Game before
    # each turn, so undoing past a death restores ``removed=False``.
    on_stage = {
        cid: game.characters[cid] for cid in game.on_stage if not game.characters[cid].removed
    }
    off_stage = {
        cid: ch
        for cid, ch in game.characters.items()
        if cid not in game.on_stage and not ch.removed
    }

    profile = session.settings.user_profile
    from ..orchestration.assets import _long_context_image_model

    prompt = text_gen_prompts.build(
        world=game.world,
        history=history,
        on_stage=on_stage,
        off_stage=off_stage,
        chosen_option_text=chosen_option_text,
        free_text_input=free_text,
        mature_content=session.settings.mature_content,
        # Merged view: player-typed entries + summarizer-inferred,
        # deduplicated. The storyteller doesn't need to know which
        # was which — both feed scene/character selection equally.
        # surfaced_* gates summarizer-inferred tags behind the
        # 1.0 strength threshold so a single noisy inference
        # can't contaminate the storyteller's scene/character
        # selection. Player-typed entries always pass through.
        user_likes=profile.surfaced_likes(),
        user_dislikes=profile.surfaced_dislikes(),
        user_notes=profile.surfaced_notes(),
        environments=game.environments,
        music_enabled=session.settings.music.enabled,
        current_music_prompt=game.world.music_prompt,
        narrator_style=session.settings.narrator_style,
        # Long-context image models (Qwen / Flux / Z-Image) render rich
        # natural-language prompts, so ask the storyteller for detailed
        # pose / outfit / decals instead of the CLIP-capped ≤6-word form.
        detailed_appearance=_long_context_image_model(session),
    )
    parent_premise = session.compute_premise_hash(
        parent_id=parent.id, chosen_option_id=chosen_option_id
    )
    metadata = GenerationMetadata(
        model=session.settings.llm.model,
        seed_parameters={"temperature": session.settings.llm.temperature},
    )

    def _build_node(
        beat: Any,  # LlmBeat — Any to avoid circular import
        cur_parent_id: str,
        is_first: bool,
        options: list[Any],
    ) -> DialogNode:
        return DialogNode(
            parent_id=cur_parent_id,
            chosen_option_id=chosen_option_id if is_first else None,
            speaker_id=beat.speaker_id,
            text=strip_speaker_slug(beat.text),
            options=options,
            entering_character_ids=beat.entering_character_ids,
            leaving_character_ids=beat.leaving_character_ids,
            new_characters=beat.new_characters,
            location_id=beat.location_id,
            location_prompt=beat.location_prompt,
            location_lighting=beat.location_lighting,
            character_changes=beat.character_changes,
            state=DialogNodeState.committed,
            premise_hash=parent_premise,
            generation_metadata=metadata,
        )

    chain: list[DialogNode] = []
    cur_parent_id = parent.id
    music_change: str | None = None

    if stream:
        # Streaming path: consume chunks through the incremental
        # parser, materialising each DialogNode as its beat closes.
        # The on_beat_streamed callback (if any) fires per beat
        # with the provisional node — useful for the foreground
        # flow to start emitting text + state patches before the
        # whole response finishes.
        from ..orchestration.streaming_dialog_parser import StreamingDialogParser

        parser = StreamingDialogParser()
        chunks: list[str] = []
        async for chunk in session.llm_stream_chunks(prompt):
            chunks.append(chunk)
            for beat in parser.feed(chunk):
                # Provisional: options=[]. The LAST beat will get
                # its options field replaced once the stream ends
                # and finalize() reveals them.
                node = _build_node(
                    beat,
                    cur_parent_id,
                    is_first=(len(chain) == 0),
                    options=[],
                )
                chain.append(node)
                cur_parent_id = node.id
                if beat.music_change and beat.music_change.strip():
                    music_change = beat.music_change.strip()
                if on_beat_streamed is not None:
                    await on_beat_streamed(node, len(chain) == 1)
        payload = parser.finalize()
        # If the parser's whole-buffer fallback found beats the
        # incremental path missed (e.g. fenced JSON that didn't
        # surface a clean "beats": [ marker until late), use those
        # to top up the chain.
        if len(chain) < len(payload.beats):
            for beat in payload.beats[len(chain) :]:
                node = _build_node(
                    beat,
                    cur_parent_id,
                    is_first=(len(chain) == 0),
                    options=[],
                )
                chain.append(node)
                cur_parent_id = node.id
                if beat.music_change and beat.music_change.strip():
                    music_change = beat.music_change.strip()
                if on_beat_streamed is not None:
                    await on_beat_streamed(node, len(chain) == 1)
        if not chain:
            raise SchemaError("LLM returned no beats")
        # Replace the last node's options with the parsed ones.
        # The streaming-emitted node had empty options (since we
        # didn't know yet); the canonical chain returned to the
        # caller carries the real options on the last beat.
        last_idx = len(chain) - 1
        chain[last_idx] = chain[last_idx].model_copy(
            update={"options": list(payload.options)},
        )
    else:
        # Non-streaming path: collect the full response then parse
        # in one shot. Used by speculation where there's no UX win
        # from per-beat emission.
        raw, chunks = await session.llm_text(prompt, stream=False)
        payload = parse_json_object(raw, LlmDialogPayload)
        if not payload.beats:
            raise SchemaError("LLM returned no beats")
        for index, beat in enumerate(payload.beats):
            is_first = index == 0
            is_last = index == len(payload.beats) - 1
            node = _build_node(
                beat,
                cur_parent_id,
                is_first=is_first,
                options=payload.options if is_last else [],
            )
            chain.append(node)
            cur_parent_id = node.id
            if beat.music_change and beat.music_change.strip():
                music_change = beat.music_change.strip()

    # Stuff music_change onto the metadata of the LAST node — the
    # caller scrapes it back to fire the music render task.
    # (DialogNode itself doesn't carry a music field; it's
    # session-level state on WorldState. Persisting on the metadata
    # dict would survive save/reload but isn't needed because the
    # current music_path lives on World.)
    if music_change and chain:
        last = chain[-1]
        last.generation_metadata.seed_parameters["music_change"] = music_change
    return chain, chunks


def _add_beats_to_tree(game: Game, beats: list[DialogNode]) -> Game:
    """Place every beat in dialog_tree.nodes WITHOUT extending
    committed_path — those nodes are pre-generated but the player
    hasn't walked them yet. ``_apply_node`` walks them one at a time."""
    if not beats:
        return game
    nodes = dict(game.dialog_tree.nodes)
    for node in beats:
        nodes[node.id] = node
    new_tree = game.dialog_tree.model_copy(update={"nodes": nodes})
    return game.model_copy(update={"dialog_tree": new_tree})


def _speculation_key(parent_id: str, option_id: str | None) -> str:
    return f"{parent_id}|{option_id or '__continue__'}"


async def _speculate_branch(
    session: Session,
    *,
    parent_id: str,
    option_id: str | None,
    option_text: str | None,
) -> None:
    """Background pre-generation of one branch of the dialog tree.

    Speculatively runs the LLM call for ``parent + option`` so when
    the player actually picks that option the next chain is already
    sitting in the tree and ``play_advance_handler`` walks it without
    a fresh LLM round-trip. Drops out cleanly if:
      * a chain has already been added under the same key,
      * the parent has been invalidated,
      * the game has been replaced.
    """
    spec_label = f"parent={parent_id[-6:]} option={(option_id or '__continue__')[-12:]}"
    if session.game is None:
        _spec_log.info("speculation skip: no game (%s)", spec_label)
        return
    parent = session.game.dialog_tree.nodes.get(parent_id)
    if parent is None:
        _spec_log.info("speculation skip: parent missing from tree (%s)", spec_label)
        return
    # Bail if an earlier task already won the race.
    if (
        obsolescence.first_valid_child(session.game, parent_id, chosen_option_id=option_id)
        is not None
    ):
        _spec_log.info(
            "speculation skip: child already exists pre-LLM (%s)",
            spec_label,
        )
        return
    _spec_log.info("speculation start: LLM call (%s)", spec_label)
    session.latency.record(
        "llm_start",
        parent_id=parent_id,
        option_id=option_id,
        source="speculation",
    )

    # Per-beat installation via the streaming callback: each beat is
    # placed into the dialog tree the moment its closing brace lands
    # in the LLM stream. ``_await_speculation`` polls ``first_valid_child``
    # so play_advance walks into beat N as soon as beat N is installed,
    # without waiting for the rest of the chain to generate. This is
    # the same incremental pattern the foreground drain uses; without
    # it the player saw "Continue 1 streams, Continue 2+ wait for the
    # full LLM response".
    speculative_char_ids: list[str] = []
    lost_race = False
    install_lock = session._ensure_play_lock()

    async def _install_streamed(node: DialogNode, is_first: bool) -> None:
        nonlocal lost_race
        if lost_race:
            return
        newly_materialised: list[str] = []
        materialised_chars: dict[str, Character] = {}
        async with install_lock:
            if session.game is None:
                lost_race = True
                return
            # First-beat race-check: a foreground drain or a competing
            # speculation may have installed a child under the same
            # (parent, option) while our LLM call was running. Discard
            # the rest of the chain in that case so the player's tree
            # doesn't gain a second sibling chain.
            if (
                is_first
                and obsolescence.first_valid_child(
                    session.game,
                    parent_id,
                    chosen_option_id=option_id,
                )
                is not None
            ):
                _spec_log.info(
                    "speculation discard: child appeared mid-stream (%s)",
                    spec_label,
                )
                lost_race = True
                return
            chars = dict(session.game.characters)
            for descriptor in node.new_characters:
                if descriptor.id in chars:
                    continue
                if _collides_with_player(descriptor, chars):
                    _spec_log.warning(
                        "speculative beat dropped NPC %r (id=%s) — collides with player",
                        descriptor.name,
                        descriptor.id,
                    )
                    continue
                materialised = _materialize_new_character(descriptor)
                chars[descriptor.id] = materialised
                materialised_chars[descriptor.id] = materialised
                newly_materialised.append(descriptor.id)
            new_game = _add_beats_to_tree(session.game, [node])
            if newly_materialised:
                new_game = new_game.model_copy(update={"characters": chars})
                speculative_char_ids.extend(newly_materialised)
            session.install_game(new_game)
            session.latency.record(
                "beat_ready",
                node_id=node.id,
                parent_id=node.parent_id,
                option_id=node.chosen_option_id,
                source="speculation",
            )
        # Push the freshly-installed speculative beat to the renderer so
        # its dialog_tree mirror gains the walkable child immediately —
        # the same fix the foreground drain applies per install. The
        # speculation task only mutated server-side state via
        # ``install_game``; without this emit the renderer doesn't learn
        # the next beat exists until its own next Continue walks to it.
        # Until then ``hasContinueChild`` in the InteractionPanel is
        # false, so the Continue glyph keeps spinning even though the
        # beat the player could walk to is already sitting in the tree —
        # the "spinner persists even though a next line is ready" bug.
        # Targeted add-patch (not a full-state emit) so the in-progress
        # typewriter for the current beat is undisturbed; any characters
        # this beat materialised ride along first so the mirrored node
        # never dangles a speaker the renderer hasn't seen yet.
        if session.emit is not None:
            ops = [
                PatchOp(
                    op="add",
                    path=f"/characters/{cid}",
                    value=mat.model_dump(mode="json"),
                )
                for cid, mat in materialised_chars.items()
            ]
            ops.append(
                PatchOp(
                    op="add",
                    path=f"/dialog_tree/nodes/{node.id}",
                    value=node.model_dump(mode="json"),
                )
            )
            session.emit(
                MessageType.s2c_state_patch,
                S2CStatePatch(ops=ops),
            )

    try:
        chain, _chunks = await _generate_beat_chain_for_session(
            session,
            parent=parent,
            chosen_option_id=option_id,
            chosen_option_text=option_text,
            stream=True,
            on_beat_streamed=_install_streamed,
        )
    except asyncio.CancelledError:
        _spec_log.info("speculation cancelled mid-LLM (%s)", spec_label)
        raise
    except RuntimeError as exc:
        # Loop teardown during streaming raises RuntimeErrors from
        # httpx's stream-aclose path ("no running event loop", sniffio
        # ``unknown async library``). Log as info — the task is being
        # cancelled by the harness, not failing on its own merits.
        msg = str(exc).lower()
        if "event loop" in msg or "async library" in msg or "current_async_library" in msg:
            _spec_log.info(
                "speculation aborted during loop teardown (%s)",
                spec_label,
            )
            return
        _spec_log.warning(
            "speculative branch generation failed (parent=%s option=%s); skipping",
            parent_id,
            option_id,
            exc_info=True,
        )
        return
    except Exception:
        _spec_log.warning(
            "speculative branch generation failed (parent=%s option=%s); skipping",
            parent_id,
            option_id,
            exc_info=True,
        )
        return
    if session.game is None:
        _spec_log.info(
            "speculation discard: game went away post-LLM (%s)",
            spec_label,
        )
        return
    if lost_race or not chain:
        return

    # Final pass: the last beat was installed via the streaming
    # callback with options=[] (the parser didn't know them yet). Now
    # the stream has ended and ``chain[-1]`` carries the parsed
    # options — replace the in-tree copy so the player sees the right
    # choices when they walk to it.
    async with install_lock:
        if session.game is None:
            return
        nodes_updated = dict(session.game.dialog_tree.nodes)
        nodes_updated[chain[-1].id] = chain[-1]
        new_tree = session.game.dialog_tree.model_copy(
            update={"nodes": nodes_updated},
        )
        session.install_game(session.game.model_copy(update={"dialog_tree": new_tree}))
        await session.commit()
    # The tail beat was pushed to the renderer earlier with options=[]
    # (the parser hadn't parsed them yet when ``_install_streamed`` ran).
    # Replace the mirror's copy with the options-bearing version so a
    # player who walks to the speculative tail sees the real choices
    # instead of a Continue glyph until the next full-state sync — the
    # same finalize the foreground drain performs.
    if session.emit is not None:
        session.emit(
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path=f"/dialog_tree/nodes/{chain[-1].id}",
                        value=chain[-1].model_dump(mode="json"),
                    )
                ]
            ),
        )
    _spec_log.info(
        "speculation installed: %d beats (%s)",
        len(chain),
        spec_label,
    )
    # Pre-render portraits for speculative characters in the
    # background. By the time the player walks the speculative chain,
    # the prompt-hash file is on disk and ``ensure_assets`` skips
    # regeneration. If pre-render hasn't finished by walk time, the
    # foreground async-asset task will pick it up and the renderer
    # keeps showing the previous on-stage portrait until the new one
    # arrives.
    if speculative_char_ids:
        _spawn_speculative_portrait_render(session, speculative_char_ids)


def _spawn_speculative_portrait_render(
    session: Session,
    character_ids: list[str],
) -> None:
    """Queue speculative-character portraits on the redraw scheduler's
    lower tier: every stale on-screen entity renders before any of
    these, and if the branch is never picked the invalidation pass
    drops the request before it starts. The foreground walk's own
    ``on_turn`` still catches any portrait that didn't pre-render."""
    session.render_scheduler.request_speculative(character_ids)


# Maximum number of beats the engine will pre-generate ahead of the
# player, counted as the TOTAL non-invalidated descendants of the
# current node (across all branches — not just the longest path).
# A fan-out tree under an options node accumulates beats quickly, so
# this cap covers the whole speculative subtree. Once the count
# reaches the cap, ``_ensure_speculative_branches`` stops spawning
# new chains; as the player walks forward the subtree shrinks and
# the cap re-allows speculation.
SPECULATIVE_DEPTH_CAP = 10


def _descendant_count(game: Game, root_id: str) -> int:
    """Count of non-invalidated descendants of ``root_id`` across all
    branches (excluding ``root_id`` itself). Used as the gate for
    ``_ensure_speculative_branches`` so the engine stops speculating
    when the prefetch buffer is already big enough."""
    children: dict[str, list[str]] = {}
    for node in game.dialog_tree.nodes.values():
        parent = node.parent_id
        if parent is None:
            continue
        if node.state == DialogNodeState.invalidated:
            continue
        children.setdefault(parent, []).append(node.id)

    total = 0
    stack = [root_id]
    visited: set[str] = set()
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        for child in children.get(nid, []):
            total += 1
            stack.append(child)
    return total


def _prune_done_speculations(session: Session) -> None:
    """Drop completed speculation tasks from the tracking dict.

    ``_speculative_tasks`` exists only to (a) dedupe spawns — gated on
    ``not task.done()`` — and (b) let the await / cancel paths find
    tasks that are still IN FLIGHT. A task that has finished already
    installed its beats into the dialog tree, so every future walk
    reads the tree, not the task; its dict entry is dead weight.

    Before this prune the dict grew one entry per option per turn and
    never shrank (the 200-turn soak harness measured 169 entries with
    essentially all of them ``done()``). That turned every per-commit
    ``_ensure_speculative_branches`` / ``_cancel_speculative_descendants``
    scan into an O(session-length) walk and pinned every finished
    ``Task`` (and whatever it closed over) for the life of the save.

    Each done task's result/exception is retrieved here so a
    speculation that raised doesn't later surface as an "exception was
    never retrieved" warning — ``_speculate_branch`` already logs its
    own failures, so we only need to clear the flag.
    """
    tasks: dict[str, asyncio.Task[None]] = session._speculative_tasks
    if not tasks:
        return
    for key, task in list(tasks.items()):
        if not task.done():
            continue
        if not task.cancelled():
            task.exception()  # retrieve so it isn't reported as unhandled
        tasks.pop(key, None)
    session._speculative_tasks = tasks


def _ensure_speculative_branches(
    session: Session,
    *,
    target_node_id: str | None = None,
    options_subset: list[str] | None = None,
) -> None:
    """Spawn one background task per current-node option that doesn't
    already have a valid child or in-flight task. Called after every
    commit that lands the player on a node with explicit options
    AND on save load — see ``saves_load_handler`` — so the player
    doesn't have to wait through a fresh LLM call the first time
    they press Continue on a freshly-loaded run.

    Capped at ``SPECULATIVE_DEPTH_CAP`` total non-invalidated
    descendants of the current node (across all branches) so a
    slow-clicking player doesn't cause the engine to stack
    chain-on-chain into an arbitrarily large prefetch. As the player
    walks, the subtree shrinks and the cap re-allows new
    speculations.

    ``target_node_id`` overrides the default ``current_node_id``
    so the chain-commit path can speculate from the option-bearing
    tail beat instead of beat 1 (where the player just landed).
    ``options_subset`` (when provided) restricts speculation to
    those option ids — used to fire only ONE speculative call from
    the chain tail (the most-likely-picked first option) instead of
    one per option, which iter 1 showed saturates the LLM scheduler
    and slows everything down.
    """
    if session.game is None:
        return
    # Drop finished speculations before doing anything else. This runs
    # on every commit, so the tracking dict stays bounded to in-flight
    # work instead of accumulating one dead entry per option per turn
    # for the life of the save (see ``_prune_done_speculations``).
    _prune_done_speculations(session)
    if target_node_id is None:
        if session.game.current_node_id is None:
            return
        target_node_id = session.game.current_node_id
    node = session.game.dialog_tree.nodes.get(target_node_id)
    if node is None:
        return
    if _descendant_count(session.game, node.id) >= SPECULATIVE_DEPTH_CAP:
        return
    tasks: dict[str, asyncio.Task[None]] = session._speculative_tasks
    if node.options:
        # Options node: prep one speculation per choice so whichever
        # the player picks lands as a tree-walk rather than an LLM
        # round-trip.
        target_options = node.options
        if options_subset is not None:
            target_options = [o for o in node.options if o.id in options_subset]
        for option in target_options:
            key = _speculation_key(node.id, option.id)
            existing = tasks.get(key)
            if existing is not None and not existing.done():
                continue
            if (
                obsolescence.first_valid_child(session.game, node.id, chosen_option_id=option.id)
                is not None
            ):
                continue
            _spec_log.info(
                "speculation spawn: parent=%s option=%s key=%s",
                node.id[-6:],
                option.id[-12:],
                key,
            )
            session.latency.record(
                "speculation_spawn",
                parent_id=node.id,
                option_id=option.id,
            )
            tasks[key] = asyncio.create_task(
                _speculate_branch(
                    session,
                    parent_id=node.id,
                    option_id=option.id,
                    option_text=option.text,
                )
            )
    else:
        # Continue node: only one path forward (option_id=None). Most
        # of the time a multi-beat chain has already been queued from
        # the same generation that produced this beat, so
        # ``first_valid_child`` returns a hit and we no-op. The case
        # this branch matters for is save load: a save committed
        # mid-chain doesn't necessarily carry the speculative beat
        # that came after, so without this prep the next Continue
        # would land a synchronous LLM call.
        key = _speculation_key(node.id, None)
        existing = tasks.get(key)
        already_running = existing is not None and not existing.done()
        already_walked = (
            obsolescence.first_valid_child(session.game, node.id, chosen_option_id=None) is not None
        )
        if not already_running and not already_walked:
            session.latency.record(
                "speculation_spawn",
                parent_id=node.id,
                option_id=None,
            )
            tasks[key] = asyncio.create_task(
                _speculate_branch(
                    session,
                    parent_id=node.id,
                    option_id=None,
                    option_text=None,
                )
            )
    session._speculative_tasks = tasks


def _cancel_speculative_descendants(
    session: Session,
    *,
    parent_id: str,
    also_invalidated_ids: list[str] | None = None,
) -> None:
    """Cancel speculative tasks rooted at ``parent_id`` — used by free-
    text invalidation to drop work that became obsolete.

    ``also_invalidated_ids`` extends the cancellation to tasks whose
    parent is ANY id in the list — needed when free-text invalidates
    a whole descendant subtree, not just the player's immediate
    current node. Without this, a chain-tail speculation rooted at
    the OLD chain's last beat (downstream of the player's current
    position when F fires) would keep running and waste a 30 s LLM
    call on a branch the player can no longer reach. Symptom seen
    in instrumented audit: ``speculation installed: 5 beats`` for
    an invalidated SPFT31 right after the player took a free-text
    detour through HR16MJ.
    """
    tasks: dict[str, asyncio.Task[None]] = session._speculative_tasks
    prefixes = {f"{parent_id}|"}
    if also_invalidated_ids:
        prefixes.update(f"{nid}|" for nid in also_invalidated_ids)
    cancelled = 0
    for key, task in list(tasks.items()):
        if not any(key.startswith(p) for p in prefixes):
            continue
        if not task.done():
            task.cancel()
            cancelled += 1
        tasks.pop(key, None)
    if cancelled:
        _spec_log.info(
            "speculation cancel: dropped %d in-flight tasks rooted at "
            "parent=%s or %d invalidated descendant(s)",
            cancelled,
            parent_id[-6:],
            len(also_invalidated_ids or []),
        )
    session._speculative_tasks = tasks


async def _await_speculation(session: Session, *, parent_id: str, option_id: str | None) -> None:
    """Wait until a speculative task for this (parent, option) has
    produced a walkable child — NOT necessarily until the whole task
    is done.

    Speculation installs each beat into the dialog tree as its
    closing brace lands in the LLM stream. The player only cares
    about the FIRST beat under their (parent, option) — that's the
    one play_advance walks into. Returning as soon as that beat
    appears lets the player walk on the per-beat budget instead of
    the full-chain budget. The task keeps running in the background
    and installs the rest of the chain for future continues.
    """
    import time

    tasks: dict[str, asyncio.Task[None]] = session._speculative_tasks
    key = _speculation_key(parent_id, option_id)
    task = tasks.get(key)
    label = f"parent={parent_id[-6:]} option={(option_id or '__continue__')[-12:]}"
    if task is None:
        _spec_log.info(
            "speculation await: NO TASK at key=%s — will fall back to fresh "
            "gen if no in-tree child (%s)",
            key,
            label,
        )
        return
    if task.done():
        _spec_log.info(
            "speculation await: task already done (%s)",
            label,
        )
        tasks.pop(key, None)
        session._speculative_tasks = tasks
        return
    started = time.monotonic()
    _spec_log.info("speculation await: WAITING for in-flight task (%s)", label)
    session.latency.record(
        "speculation_await_start",
        parent_id=parent_id,
        option_id=option_id,
    )
    poll_interval = 0.02
    try:
        while True:
            game = session.game
            if (
                game is not None
                and obsolescence.first_valid_child(
                    game,
                    parent_id,
                    chosen_option_id=option_id,
                )
                is not None
            ):
                return
            if task.done():
                return
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=poll_interval,
                )
            except TimeoutError:
                # Speculation still running; loop to re-check the
                # first_valid_child guard.
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                # Speculation task raised; surface nothing — the
                # caller will fall back to fresh generation.
                return
    except asyncio.CancelledError:
        _spec_log.info(
            "speculation await: cancelled while waiting (%s)",
            label,
        )
        return
    finally:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _spec_log.info(
            "speculation await: returned after %d ms (%s)",
            elapsed_ms,
            label,
        )
        session.latency.record(
            "speculation_await_end",
            parent_id=parent_id,
            option_id=option_id,
            elapsed_ms=elapsed_ms,
        )
        # Only forget the task once it's actually finished — the
        # background task may still be installing later beats that
        # downstream continues need.
        if task.done():
            tasks.pop(key, None)
            session._speculative_tasks = tasks


_spec_log = logging.getLogger(__name__).getChild("speculation")


def _recent_scene_change(game: Game, window: int = 3) -> bool:
    """True when the active location changed within the last ``window``
    committed beats.

    Used to gate the summarizer's ``characters_to_offstage`` pass: bulk
    offstaging is only trusted when the player has actually moved scenes
    (the summarizer's job (b)). Same-scene exits are handled per-beat by
    the storyteller's ``leaving_character_ids``, so blocking summarizer
    offstaging while the scene is unchanged stops quiet-but-present
    characters from being pruned just for going a few beats without a
    line — the reported disappearing-character bug.

    The location is resolved per committed node the same way
    ``assets._resolve_active_environment_id`` does — carry the last
    non-empty ``location_id`` forward across beats that don't set one.
    """
    path = game.dialog_tree.committed_path
    nodes = game.dialog_tree.nodes
    resolved: list[str | None] = []
    current: str | None = None
    for nid in path:
        node = nodes.get(nid)
        if node is not None and node.location_id:
            current = node.location_id
        resolved.append(current)
    if len(resolved) < 2:
        return False
    recent = resolved[-(window + 1) :]
    return len({loc for loc in recent if loc is not None}) > 1


def _apply_node(game: Game, node: DialogNode) -> Game:
    nodes = dict(game.dialog_tree.nodes)
    nodes[node.id] = node
    new_path = [*game.dialog_tree.committed_path, node.id]
    new_tree = game.dialog_tree.model_copy(update={"nodes": nodes, "committed_path": new_path})

    # Create Characters for any descriptor the LLM emitted in this
    # beat — must happen BEFORE on_stage is computed, otherwise the
    # entering id is silently dropped (line below filters by membership).
    new_chars = dict(game.characters)
    for descriptor in node.new_characters:
        if descriptor.id in new_chars:
            continue
        if _collides_with_player(descriptor, new_chars):
            _log.warning(
                "_apply_node dropped NPC descriptor %r (id=%s) — collides with player",
                descriptor.name,
                descriptor.id,
            )
            continue
        new_chars[descriptor.id] = _materialize_new_character(descriptor)

    new_on_stage = list(game.on_stage)
    # DEFERRED EXIT. A character who speaks their own exit line
    # ("Fine — I'm leaving.") arrives with their id in that beat's
    # ``leaving_character_ids``. Removing them there would leave the
    # player reading a name tag over an empty stage, so they stay
    # visible for the beat they speak and are retired here, on the
    # beat after. Only the previous beat's SPEAKER is drained — every
    # other id in that beat's leaving list already left at the beat
    # itself, and re-applying the whole list here would silently undo
    # a character the player put back on stage from the cast panel in
    # the meantime. The drain runs BEFORE this beat's entering list so
    # a character who walks straight back in still wins.
    prev_node = (
        game.dialog_tree.nodes.get(game.current_node_id)
        if game.current_node_id is not None
        else None
    )
    if (
        prev_node is not None
        and prev_node.speaker_id
        and prev_node.speaker_id in prev_node.leaving_character_ids
    ):
        new_on_stage = [cid for cid in new_on_stage if cid != prev_node.speaker_id]
    # Treat the BEAT'S SPEAKER as an implicit "entering" id when the
    # storyteller forgets to include them in ``entering_character_ids``.
    # The user's reported symptom: a character speaks but has no
    # portrait visible because the LLM didn't list them in entering.
    # Speaking is unambiguous evidence the character is present, so
    # we surface them without waiting for the LLM to remember the
    # bookkeeping. The player character (lens) is excluded — they
    # never appear on stage by design.
    implicit_entering: list[str] = []
    if node.speaker_id and node.speaker_id in new_chars:
        speaker = new_chars[node.speaker_id]
        if not speaker.is_player and not speaker.removed:
            implicit_entering.append(node.speaker_id)
    for cid in [*node.entering_character_ids, *implicit_entering]:
        if cid in new_chars and cid not in new_on_stage:
            # The player character is the camera lens, not an
            # on-stage figure — they never get a portrait rendered
            # and they don't belong in the on_stage list. The
            # ``implicit_entering`` branch already enforces this for
            # the speaker fallback; the explicit
            # ``entering_character_ids`` branch needs the same gate so
            # an LLM beat that wrongly lists the PC's id doesn't push
            # them on-stage and trigger ``ensure_assets`` to render
            # them later.
            if new_chars[cid].is_player:
                continue
            # Removed characters (player-dismissed or LLM-killed) stay
            # off-stage even if a later beat tries to bring them back.
            # Undo restores ``removed=False`` from the snapshot, so the
            # gate naturally lifts when the player rolls past the death.
            if new_chars[cid].removed:
                continue
            new_on_stage.append(cid)
    # The beat's own speaker is exempt from its leaving list — see
    # DEFERRED EXIT above; the next beat retires them.
    leaving_now = {cid for cid in node.leaving_character_ids if cid != node.speaker_id}
    new_on_stage = [cid for cid in new_on_stage if cid not in leaving_now]
    new_on_stage = [cid for cid in new_on_stage if not new_chars[cid].removed]

    for change in node.character_changes:
        if change.character_id not in new_chars:
            continue
        existing = new_chars[change.character_id]
        new_chars[change.character_id] = existing.model_copy(
            update={change.field.value: change.new_value}
        )

    return game.model_copy(
        update={
            "dialog_tree": new_tree,
            "current_node_id": node.id,
            "on_stage": new_on_stage,
            "characters": new_chars,
        }
    )


def _collides_with_player(
    descriptor: NewCharacterDescriptor, characters: dict[str, Character]
) -> bool:
    """True if a freshly-introduced NPC descriptor would shadow the
    player character — same id or same case-folded name.

    The LLM occasionally invents an NPC who happens to share the
    player's name; the prompt explicitly forbids it but defense in
    depth: drop the descriptor at materialization so the on-stage
    roster never contains two characters answering to the same name.
    The orphaned ``entering_character_ids`` reference is silently
    ignored by ``_apply_node``'s membership check.
    """
    player = next(
        (c for c in characters.values() if getattr(c, "is_player", False)),
        None,
    )
    if player is None:
        return False
    if descriptor.id == player.id:
        return True
    desc_name = (descriptor.name or "").strip().casefold()
    player_name = (player.name or "").strip().casefold()
    return bool(desc_name) and desc_name == player_name


def _materialize_new_character(descriptor: NewCharacterDescriptor) -> Character:
    """Turn an LLM-emitted ``NewCharacterDescriptor`` into a full
    Character — same shape as the player character, with a fresh seed
    so portrait generation produces a stable likeness for them.
    """
    return Character(
        id=descriptor.id,
        is_player=False,
        name=descriptor.name,
        description=descriptor.description,
        kind=descriptor.kind,
        physical_description=descriptor.physical_description,
        gender=descriptor.gender,
        # Pronouns flow through if the LLM populated them on
        # the descriptor; fall back to empty (= storyteller
        # derives from gender) so legacy LLM responses without
        # the field still parse.
        pronouns=getattr(descriptor, "pronouns", "") or "",
        age=descriptor.age,
        ethnicity=descriptor.ethnicity,
        skin=descriptor.skin,
        hair_color=descriptor.hair_color,
        hairstyle=descriptor.hairstyle,
        eye_color=descriptor.eye_color,
        build=descriptor.build,
        bust=descriptor.bust,
        outfit=descriptor.outfit,
        pose=descriptor.pose,
        expression=descriptor.expression,
        effects=getattr(descriptor, "effects", "") or "",
        decals=getattr(descriptor, "decals", "") or "",
        seed=random_seed(),
    )


async def _emit_walk(
    ctx: HandlerContext,
    target: DialogNode,
    *,
    text_chunks: list[str] | None = None,
) -> AsyncIterator[OutboundMessage]:
    """Common emission sequence for landing on ``target``: state/patch
    with the new current_node_id, the typewriter stream for the beat
    text, a full state snapshot, and any image-ready events."""
    chunks = text_chunks if text_chunks is not None else [target.text or ""]

    ctx.session.latency.record(
        "displayed",
        node_id=target.id,
        parent_id=target.parent_id,
        option_id=target.chosen_option_id,
    )
    yield (
        MessageType.s2c_state_patch,
        S2CStatePatch(ops=[PatchOp(op="replace", path="/current_node_id", value=target.id)]),
    )
    async for message in _emit_text(target.id, chunks):
        yield message
    yield (MessageType.s2c_state_full, _full_state_message(ctx.session))


async def _emit_assets(
    assets: list[GeneratedAsset],
) -> AsyncIterator[OutboundMessage]:
    for asset in assets:
        yield (
            MessageType.s2c_image_ready,
            S2CImageReady(
                kind=ImageKind.portrait if asset.kind == "portrait" else ImageKind.background,
                target_id=asset.target_id,
                image_path=str(asset.image_path),
                image_id=asset.prompt_hash,
            ),
        )


async def _consolidate_facts_safe(session: Session) -> None:
    """Run a per-character facts consolidation pass when any
    character's facts list has crossed the soft threshold. Merges
    near-duplicates, drops ephemeral observations, and keeps the
    list under ``FACTS_PER_CHARACTER_CAP - 5`` entries so the next
    few summarizer additions don't immediately re-trigger
    consolidation.

    Best-effort and silent on failure — the character keeps its
    full list and the hard cap continues to clamp it on the next
    summarizer pass.
    """
    from ..orchestration import summarizer
    from ..orchestration.prompts import facts_consolidate as facts_prompts

    if session.game is None:
        return
    facts_by_id = {cid: list(ch.facts) for cid, ch in session.game.characters.items()}
    over_cap = summarizer.facts_need_consolidation(facts_by_id)
    if not over_cap:
        return

    facts_arg: dict[str, list[tuple[str, str]]] = {}
    briefs: dict[str, str] = {}
    for cid in over_cap:
        char = session.game.characters[cid]
        facts_arg[cid] = [(f.text, f.confidence.value) for f in char.facts]
        briefs[cid] = f"{char.name} — {char.description}"

    try:
        prompt = facts_prompts.build(
            facts_by_character=facts_arg,
            character_briefs=briefs,
        )
        raw, _ = await session.llm_text(prompt)
        result = parse_json_object(raw, LlmFactsConsolidation)
    except asyncio.CancelledError:
        raise
    except Exception:
        _spec_log.warning("facts consolidation failed", exc_info=True)
        return

    # Re-read session.game after the await — a turn may have
    # replaced it. Apply only to characters that still exist.
    if session.game is None:
        return
    new_chars = dict(session.game.characters)
    changed = False
    for cid in over_cap:
        if cid not in new_chars:
            continue
        char = new_chars[cid]
        entries = result.facts_by_character.get(cid)
        if not entries:
            continue
        new_facts = summarizer.consolidate_facts_for_character(
            existing=list(char.facts),
            consolidated_texts=[(e.text, e.confidence) for e in entries],
        )
        new_chars[cid] = char.model_copy(update={"facts": new_facts})
        changed = True
    if not changed:
        return
    session.install_game(session.game.model_copy(update={"characters": new_chars}))
    try:
        await session.commit()
        if session.emit is not None:
            session.emit(MessageType.s2c_state_full, _full_state_message(session))
    except Exception:
        _spec_log.warning("post-facts-consolidation emit failed", exc_info=True)


async def _consolidate_profile_safe(session: Session) -> None:
    """Run a profile-consolidation pass when any bucket has hit
    ``PROFILE_BUCKET_CAP``. Merges near-duplicates and drops
    storyline-specific tags so the cross-save profile stays a stable
    signal of taste, not a log of one playthrough's events."""
    from ..orchestration import summarizer
    from ..orchestration.prompts import profile_consolidate as profile_prompts
    from ..persistence import settings_store

    profile = session.settings.user_profile
    if not summarizer.profile_needs_consolidation(profile):
        return
    try:
        # Consolidation only sees the SUMMARIZER buckets — player-
        # entered entries are sacrosanct and never go through the
        # consolidator's merge / drop rules.
        prompt = profile_prompts.build(
            likes=list(profile.summarizer_likes),
            dislikes=list(profile.summarizer_dislikes),
            notes=list(profile.summarizer_notes),
        )
        raw, _ = await session.llm_text(prompt)
        result = parse_json_object(raw, LlmProfileConsolidation)
    except asyncio.CancelledError:
        raise
    except Exception:
        _spec_log.warning("profile consolidation failed", exc_info=True)
        return

    new_profile = summarizer.consolidate_profile(
        profile=profile,
        consolidated={
            "likes": list(result.likes),
            "dislikes": list(result.dislikes),
            "notes": list(result.notes),
        },
    )
    session.settings = session.settings.model_copy(update={"user_profile": new_profile})
    try:
        settings_store.save_settings(session.settings)
    except Exception:
        _spec_log.warning("user profile persist failed after consolidation", exc_info=True)
    if session.emit is not None and session.game is not None:
        try:
            session.emit(MessageType.s2c_state_full, _full_state_message(session))
        except Exception:
            _spec_log.warning("post-consolidation emit failed", exc_info=True)


def _spawn_async_summarizer(session: Session) -> None:
    """Kick off a background summarizer pass so ``WorldState`` fields
    that depend on it (``summarizer_assessment``, plot-stage pointer,
    character facts, cross-save user profile) keep updating as the
    story progresses. Runs after each turn — the LLM call is cheap
    relative to the storyteller pass and never blocks the player."""
    if session.emit is None or session.game is None:
        return
    task = asyncio.create_task(_async_summarizer_task(session))
    tasks: list[asyncio.Task[None]] = session._summarizer_tasks
    tasks = [t for t in tasks if not t.done()]
    tasks.append(task)
    session._summarizer_tasks = tasks


async def _async_summarizer_task(session: Session) -> None:
    from ..orchestration import obsolescence, summarizer
    from ..orchestration.prompts import world_refresh as world_refresh_prompts
    from ..persistence import settings_store

    try:
        if session.game is None:
            return
        game = session.game
        history = obsolescence.committed_path(game)
        history_list = list(history)
        if not history_list:
            return
        typed_quotes = summarizer.collect_player_typed_quotes(history_list)
        # Player CHOICES (menu picks + typed input) feed the
        # profile-inference step in isolation from the rest of the
        # history. The summarizer must not infer taste from
        # engine-authored events the player didn't choose; passing
        # the choices block separately lets the prompt enforce
        # that constraint.
        player_choices = summarizer.collect_player_choices(history_list)
        profile = session.settings.user_profile
        # Stage roster (excluding the player) feeds the summarizer's
        # idle-cleanup job. Names go alongside the ids so the LLM
        # can match speakers in the rendered history to the
        # currently-staged ids without a second lookup.
        player_id = next(
            (cid for cid, ch in game.characters.items() if ch.is_player),
            None,
        )
        char_names = {cid: ch.name for cid, ch in game.characters.items()}
        prompt = world_refresh_prompts.build(
            world=game.world,
            history=history_list,
            player_typed_quotes=typed_quotes,
            player_choices=player_choices,
            # Summarizer sees ONLY surfaced tags so it doesn't
            # treat speculative one-off inferences as confirmed
            # player taste. Below-threshold tags stay tracked
            # internally — if the summarizer re-infers the same
            # tag, ``_merge_profile_additions`` bumps its
            # strength rather than appending a duplicate. That's
            # the "batched together" mechanic: multiple
            # independent observations of a trait accumulate
            # until the strength crosses the surface threshold.
            user_likes=profile.surfaced_likes(),
            user_dislikes=profile.surfaced_dislikes(),
            user_notes=profile.surfaced_notes(),
            environments=game.environments,
            on_stage_ids=list(game.on_stage),
            player_character_id=player_id,
            character_names=char_names,
        )
        # Summarizer responses commonly run long: the assessment
        # summary plus per-character fact additions, pruned ids,
        # and user_profile additions easily fill 2-3 KB. The
        # default max_tokens cap (1024) truncates mid-array on a
        # mature save and the parse fails. 16384 covers most of
        # what frontier models actually return on a mature save;
        # the parser also runs a JSON-repair pass on truncated
        # output (see ``_repair_truncated_json``) so even when
        # the cap is hit, the pre-cut portion of the response
        # still applies instead of being lost wholesale.
        raw, _ = await session.llm_text(prompt, max_tokens=16384)
        result = parse_json_object(raw, LlmSummaryResult)
        application = summarizer.apply_summary(
            summary=result,
            character_facts={cid: list(ch.facts) for cid, ch in game.characters.items()},
            current_stage_id=game.world.current_stage_id,
            plot_outline=list(game.world.plot_outline),
            user_profile=profile,
            on_stage=list(game.on_stage),
            player_character_id=player_id,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        _spec_log.warning("async summarizer failed", exc_info=True)
        return

    # Game may have shifted while the LLM was thinking. Re-read and
    # apply against whatever the live state is now.
    if session.game is None:
        return
    game = session.game
    new_chars = dict(game.characters)
    for cid, facts in application.character_facts.items():
        existing = new_chars.get(cid)
        if existing is None:
            continue
        new_chars[cid] = existing.model_copy(update={"facts": facts})
    world_update: dict[str, object] = {
        "summarizer_assessment": application.summarizer_assessment,
        "direction_signal": application.direction_signal,
    }
    if application.current_stage_id is not None:
        world_update["current_stage_id"] = application.current_stage_id
    if application.revised_outline is not None:
        world_update["plot_outline"] = application.revised_outline
    new_world = game.world.model_copy(update=world_update)
    # Character cleanup from the summarizer. We re-validate against the
    # LIVE on_stage list (the LLM saw a stale snapshot), so a character
    # who left mid-flight via a normal beat doesn't get double-removed
    # and a character who walked back on stage mid-flight isn't kicked
    # off again.
    #
    # BUT: the reported bug was characters vanishing while still present
    # and interacting, just because they'd been quiet for a few beats —
    # the summarizer over-eagerly offstaged them despite the prompt
    # telling it not to idle-prune. Negative instructions to an LLM
    # aren't reliable, so we add a deterministic backstop: only honour
    # summarizer offstaging when the scene has actually moved recently.
    # A same-scene departure (someone says goodbye and walks out) is
    # already handled per-beat by the storyteller's
    # ``leaving_character_ids``; the summarizer's job (b) — "the player
    # left them behind on a scene change" — is the one that needs this
    # cross-beat pass, and it is exactly the case a location change
    # detects. When the scene hasn't changed, a quiet on-stage character
    # STAYS on stage.
    game_update: dict[str, object] = {"world": new_world, "characters": new_chars}
    if application.characters_to_offstage and _recent_scene_change(game):
        live_on_stage = list(game.on_stage)
        offstage_set = set(application.characters_to_offstage)
        new_on_stage = [cid for cid in live_on_stage if cid not in offstage_set]
        if len(new_on_stage) != len(live_on_stage):
            game_update["on_stage"] = new_on_stage
    new_game = game.model_copy(update=game_update)
    session.install_game(new_game)
    # ``Settings.learn_user_profile`` is the master toggle for
    # cross-save player-profile learning. When False, discard the
    # summarizer's ``user_profile`` additions entirely — the
    # ``summarizer_*`` buckets stop growing, the consolidation
    # pass doesn't fire, and the player-typed buckets stay
    # whatever the player set them to via Settings. We still
    # accepted the LLM's other outputs (facts, stage, drift) above;
    # only the profile-mutation step is gated.
    if session.settings.learn_user_profile:
        # CRITICAL race-fix. ``application.user_profile`` was
        # computed against the profile snapshotted at task start —
        # but during the LLM round-trip the player may have
        # opened Settings and deleted entries from the
        # ``summarizer_*`` lists. Writing the snapshot-derived
        # profile back wholesale would resurrect those deletions
        # AND wipe the freshly-added ``dismissed_*`` entries the
        # settings handler appended. Instead, re-merge the LLM's
        # additions against the LIVE
        # ``session.settings.user_profile`` (which has whatever
        # the user did mid-flight, including dismissals). The
        # merge function reads ``dismissed_*`` into its dedup
        # set, so any tag the player removed during this window
        # is rejected from the additions and stays gone.
        live_profile = session.settings.user_profile
        merged_profile = summarizer._merge_profile_additions(
            current=live_profile,
            additions={
                "likes": result.user_profile_additions.likes,
                "dislikes": result.user_profile_additions.dislikes,
                "notes": result.user_profile_additions.notes,
            },
        )
        session.settings = session.settings.model_copy(update={"user_profile": merged_profile})
        try:
            settings_store.save_settings(session.settings)
        except Exception:
            _spec_log.warning("user profile persist failed", exc_info=True)
        if summarizer.profile_needs_consolidation(merged_profile):
            await _consolidate_profile_safe(session)
    # Facts cleanup: if any character's accumulated facts list has
    # grown past ``FACTS_CONSOLIDATION_THRESHOLD``, run the LLM-driven
    # condensation pass before the hard ``FACTS_PER_CHARACTER_CAP``
    # silently drops oldest entries. Cheap when no character is over
    # the threshold (the helper short-circuits without an LLM call).
    await _consolidate_facts_safe(session)
    try:
        await session.commit()
        # ``session.emit`` is guaranteed non-None by the guard in
        # ``_spawn_async_summarizer`` that gates this task's creation; a
        # teardown that clears it mid-flight raises into the except below.
        session.emit(  # type: ignore[misc]
            MessageType.s2c_state_full, _full_state_message(session)
        )
    except Exception:
        _spec_log.warning("summarizer emit failed", exc_info=True)


def _spawn_async_music(session: Session, prompt: str) -> None:
    """Render a fresh ACE-Step background-music track and emit
    ``s2c/music/ready`` when it lands. Best-effort and silent on
    failure — a music render that crashes (server down, prompt
    rejected, OOM on the audio model) leaves the previous track
    playing and surfaces a single warning log line. The next
    music_change attempt re-fires; we don't retry the failed one
    automatically.

    The task tracks itself on ``session._music_tasks`` so a
    session teardown can cancel any in-flight render."""
    if session.emit is None:
        return
    task = asyncio.create_task(_async_music_task(session, prompt))
    tasks: list[asyncio.Task[None]] = session._music_tasks
    tasks = [t for t in tasks if not t.done()]
    tasks.append(task)
    session._music_tasks = tasks


async def _async_music_task(session: Session, prompt: str) -> None:
    music_client = session.music_client()
    if music_client is None:
        return
    from ..orchestration.assets import ensure_music_for_world

    try:
        asset = await ensure_music_for_world(
            session=session,
            music_client=music_client,
            prompt=prompt,
            saves_root=session.saves_root,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        _spec_log.warning("async music generation failed", exc_info=True)
        return
    if asset is None or session.emit is None or session.game is None:
        return
    # Persist the new music_path / music_prompt fields on the live
    # World; ensure_music_for_world already mutated session.game,
    # but commit + emit need an explicit kick.
    try:
        await session.commit()
    except Exception:
        _spec_log.warning("async music commit failed", exc_info=True)
    try:
        session.emit(
            MessageType.s2c_music_ready,
            S2CMusicReady(
                audio_path=str(asset.image_path),
                music_id=asset.prompt_hash,
                prompt=session.game.world.music_prompt,
            ),
        )
        # Push a state/full so the renderer's stored
        # game.world.music_path stays in lockstep with the
        # music_ready event the audio component reacted to.
        session.emit(
            MessageType.s2c_state_full,
            _full_state_message(session),
        )
    except Exception:
        _spec_log.warning("async music emit failed", exc_info=True)


def _ensure_music_on_load(session: Session) -> None:
    """Backfill the world's background music when the loaded save
    has music enabled but no playable track on disk.

    Triggers a music render in three cases:
      * ``music_path`` is empty (save predates music gen, or the
        world was committed before the first track finished).
      * ``music_path`` is set but the file is missing on disk
        (save copied to a new machine without the audio file, or
        the file was manually deleted).
      * Either of the above is true AND we can derive a prompt —
        we prefer ``world.music_prompt`` (last prompt the LLM
        emitted) and fall back to a generic prompt built from the
        world's setting / genre when the field is empty (e.g. a
        save committed when music was off, now reloaded with
        music on).

    No-ops cleanly when:
      * ``Settings.music.enabled`` is False (don't surprise the
        player by rendering music after they turned it off).
      * No music client is available (settings disabled the
        backend or construction failed).
      * The track already exists on disk for the current prompt.
    """
    if session.game is None or session.emit is None:
        return
    if not session.settings.music.enabled:
        return
    world = session.game.world
    music_path = world.music_path or ""
    have_music = bool(music_path) and Path(music_path).exists()
    if have_music:
        return
    prompt = (world.music_prompt or "").strip()
    if not prompt:
        prompt = _fallback_music_prompt(world)
    if not prompt:
        return
    _spawn_async_music(session, prompt)


def _fallback_music_prompt(world: WorldState) -> str:
    """Synthesize a generic seamless-loop ambient prompt from the
    world's tone fields when the save lacks an explicit
    ``music_prompt``. Pulls genre + setting (NOT visual_style —
    the music scores what the player FEELS, not what they SEE).
    Empty when the world is too sparse to seed anything
    meaningful."""
    setting = (getattr(world, "setting", "") or "").strip()
    genre = (getattr(world, "genre", "") or "").strip()
    bits = [b for b in (genre, setting) if b]
    if not bits:
        return ""
    return (
        "ambient instrumental, sustained pads, soft piano, "
        "low drone, slow, mid-tempo, minor key, warm reverb, "
        "intimate, seamless loop, no vocals; " + ", ".join(bits)
    )


def _spawn_async_assets(session: Session) -> None:
    """Kick off ``ensure_assets`` in the background and push the
    resulting state/full + image_ready events to the renderer through
    ``session.emit`` when each batch finishes.

    The foreground walk handler returns IMMEDIATELY (so Continue is
    never blocked on image generation); the background task arrives
    later with whatever portraits / backgrounds the new node needed.
    The renderer's ``character.images[0]`` is the most-recent image,
    so until the new portrait lands the previous one stays on stage —
    "show stale until it completes" without any frontend changes.
    """
    # The per-entity redraw scheduler owns the turn: bump every stale
    # on-screen entity's staleness score and let its pump render the
    # highest scorer (characters before the background, one at a time).
    # It runs even with no emit channel — renders still mutate game
    # state for the next save/state_full; only the image_ready push is
    # skipped when ``session.emit`` is None. The scheduler registers its
    # pump on ``session._asset_tasks`` so existing drains still see it.
    session.render_scheduler.on_turn()


async def _walk_existing_child(
    ctx: HandlerContext, target: DialogNode
) -> AsyncIterator[OutboundMessage]:
    """Walk to a pre-generated beat already in the dialog tree. No
    LLM call — just apply the node's state changes (on_stage,
    character_changes), emit the standard advance sequence, and
    THEN persist + spawn background work.

    Ordering rationale: ``commit()`` is a disk write (now on a worker
    thread, but still awaited before we return) and
    the speculation/asset spawns are bookkeeping that the player's
    next click is at least a beat-read away from caring about. The
    foreground ``_generate_and_walk_chain`` path already commits
    *after* its first beat emits (because the commit lives at the end
    of the stream-finalise block); matching that order here removes
    the disk-write from the click->state_patch critical path. The
    latency audit measured this delay at ~20 ms on selected turns
    and the invariant test was failing for that reason.
    """
    assert ctx.session.game is not None
    new_game = _apply_node(ctx.session.game, target)
    new_game = _ensure_environment_for_node(new_game, target)
    ctx.session.install_game(new_game)
    async for message in _emit_walk(ctx, target):
        yield message
    # Post-emit: persist + schedule background work. Asset / music /
    # speculation tasks are ``asyncio.create_task`` calls so they
    # don't block; ``commit()`` runs its save_store write on a thread.
    await ctx.session.commit()
    _spawn_async_assets(ctx.session)
    _spawn_async_summarizer(ctx.session)
    _ensure_speculative_branches(ctx.session)


class _BuildNode(Protocol):
    """Shape of the node-builder closure returned by
    ``_foreground_chain_node_builder``."""

    def __call__(self, beat: Any, cur_parent_id: str, *, is_first: bool) -> DialogNode: ...


def _foreground_chain_node_builder(
    *,
    chosen_option_id: str | None,
    parent_premise: str,
    metadata: Any,
) -> _BuildNode:
    """Closure-builder for the foreground chain. Shared between the
    initial-emit phase (under play_lock) and the background drain
    phase (post-lock) so they construct nodes identically.
    """

    def _build(beat: Any, cur_parent_id: str, *, is_first: bool) -> DialogNode:
        return DialogNode(
            parent_id=cur_parent_id,
            chosen_option_id=chosen_option_id if is_first else None,
            speaker_id=beat.speaker_id,
            text=strip_speaker_slug(beat.text),
            options=[],
            entering_character_ids=beat.entering_character_ids,
            leaving_character_ids=beat.leaving_character_ids,
            new_characters=beat.new_characters,
            location_id=beat.location_id,
            location_prompt=beat.location_prompt,
            location_lighting=beat.location_lighting,
            character_changes=beat.character_changes,
            state=DialogNodeState.committed,
            premise_hash=parent_premise,
            generation_metadata=metadata,
        )

    return _build


async def _drain_remaining_foreground_chain(
    *,
    session: Session,
    stream_iter: AsyncIterator[str],
    parser: StreamingDialogParser,
    chain: list[DialogNode],
    cur_parent_id: str,
    build_node: _BuildNode,
    pending_beats: list[Any],
    music_change: str | None,
) -> None:
    """Background drain of an in-flight foreground LLM chain.

    Phase 1 of ``_generate_and_walk_chain`` parses + emits beat 1 to
    the player and then schedules this task. We continue reading the
    LLM stream, install each remaining beat into the tree, then
    finalise (patch tail options, commit, spawn speculation / assets /
    music). Acquires ``session.play_lock`` BRIEFLY per install — long
    enough to read + transform + install_game atomically against
    concurrent play handlers, but not long enough to make a Continue
    click wait the full chain duration.

    Cancellable: free-text invalidation cancels in-flight drains since
    the unwalked-descendant invalidation just wiped the chain we were
    growing. Crashes are logged + swallowed (best-effort: at worst
    the player walks via a fresh LLM round-trip on the next Continue).
    """
    play_lock = session._ensure_play_lock()
    try:

        async def _install(node: DialogNode) -> bool:
            """Acquire play_lock briefly, install ``node`` into the
            tree. Returns False if the session's game was wiped while
            we were waiting (drain should abort)."""
            async with play_lock:
                if session.game is None:
                    return False
                tree_updated = _add_beats_to_tree(session.game, [node])
                session.install_game(tree_updated)
            # Push the freshly-installed beat to the renderer so its
            # dialog_tree mirror gains the walkable child immediately.
            # The drain only mutates server-side state via
            # ``install_game``; without this emit the renderer doesn't
            # learn the next beat exists until its own next Continue
            # walks to it. Until then ``hasContinueChild`` in the
            # InteractionPanel is false, so the Continue glyph keeps
            # spinning even though the beat the player could walk to is
            # already sitting in the tree — the "spinner stays spinning
            # while the model streams, even with a next node ready" bug.
            # Targeted add-patch (not a full-state emit) so the
            # in-progress typewriter for the current beat is undisturbed.
            if session.emit is not None:
                session.emit(
                    MessageType.s2c_state_patch,
                    S2CStatePatch(
                        ops=[
                            PatchOp(
                                op="add",
                                path=f"/dialog_tree/nodes/{node.id}",
                                value=node.model_dump(mode="json"),
                            )
                        ]
                    ),
                )
            return True

        # 1) Drain any beats the initial chunk-loop parsed but didn't
        #    have time to install before exiting (rare: the chunk that
        #    closed beat 1 might also contain beat 2's opening brace).
        for beat in pending_beats:
            node = build_node(beat, cur_parent_id, is_first=False)
            chain.append(node)
            cur_parent_id = node.id
            if beat.music_change and beat.music_change.strip():
                music_change = beat.music_change.strip()
            session.latency.record(
                "beat_ready",
                node_id=node.id,
                parent_id=node.parent_id,
                option_id=node.chosen_option_id,
                source="foreground_drain",
            )
            if not await _install(node):
                return

        # 2) Continue reading the LLM stream.
        async for chunk in stream_iter:
            for beat in parser.feed(chunk):
                node = build_node(beat, cur_parent_id, is_first=False)
                chain.append(node)
                cur_parent_id = node.id
                if beat.music_change and beat.music_change.strip():
                    music_change = beat.music_change.strip()
                session.latency.record(
                    "beat_ready",
                    node_id=node.id,
                    parent_id=node.parent_id,
                    option_id=node.chosen_option_id,
                    source="foreground_drain",
                )
                if not await _install(node):
                    return

        # 3) Stream done — finalise parser (catches any beats the
        #    incremental path missed, e.g. fenced JSON).
        payload = parser.finalize()
        if len(chain) < len(payload.beats):
            for beat in payload.beats[len(chain) :]:
                node = build_node(beat, cur_parent_id, is_first=False)
                chain.append(node)
                cur_parent_id = node.id
                if beat.music_change and beat.music_change.strip():
                    music_change = beat.music_change.strip()
                session.latency.record(
                    "beat_ready",
                    node_id=node.id,
                    parent_id=node.parent_id,
                    option_id=node.chosen_option_id,
                    source="foreground_drain",
                )
                if not await _install(node):
                    return

        if not chain:
            _log.warning("foreground drain: empty chain at finalise — nothing to commit")
            return

        # 4) Patch tail options into the in-memory chain + the live
        #    tree, commit, spawn post-chain background work.
        async with play_lock:
            if session.game is None:
                return
            last_idx = len(chain) - 1
            last_with_options = chain[last_idx].model_copy(
                update={"options": list(payload.options)},
            )
            chain[last_idx] = last_with_options
            nodes_updated = dict(session.game.dialog_tree.nodes)
            nodes_updated[last_with_options.id] = last_with_options
            new_tree = session.game.dialog_tree.model_copy(
                update={"nodes": nodes_updated},
            )
            session.install_game(session.game.model_copy(update={"dialog_tree": new_tree}))
            # The tail beat was streamed to the renderer earlier with
            # empty options (we didn't know them until finalize). Push
            # the options-bearing version so the renderer's mirror
            # matches the tree — otherwise the player would land on the
            # tail via a tree-walk showing a Continue glyph instead of
            # the real choices until the next full-state sync.
            if session.emit is not None:
                session.emit(
                    MessageType.s2c_state_patch,
                    S2CStatePatch(
                        ops=[
                            PatchOp(
                                op="replace",
                                path=f"/dialog_tree/nodes/{last_with_options.id}",
                                value=last_with_options.model_dump(mode="json"),
                            )
                        ]
                    ),
                )
            if music_change:
                last_with_options.generation_metadata.seed_parameters["music_change"] = music_change

            await session.commit()
            _spawn_async_assets(session)
            _spawn_async_summarizer(session)
            _ensure_speculative_branches(session)
            tail = chain[-1]
            if tail.id != session.game.current_node_id:
                if tail.options:
                    _ensure_speculative_branches(
                        session,
                        target_node_id=tail.id,
                        options_subset=[tail.options[0].id],
                    )
                else:
                    # Tail has no options — likely an LLM data-quality
                    # blip; the player will hit Continue past the
                    # nominal end of the chain. Speculate that Continue
                    # so the click isn't a fresh foreground gen.
                    _ensure_speculative_branches(
                        session,
                        target_node_id=tail.id,
                    )
            if music_change:
                _spawn_async_music(session, music_change)
    except asyncio.CancelledError:
        _log.info("foreground drain cancelled")
        raise
    except Exception:
        _log.warning("foreground drain failed", exc_info=True)
    finally:
        try:
            cur_task = asyncio.current_task()
        except RuntimeError:
            cur_task = None
        if cur_task is not None and session._foreground_stream_task is cur_task:
            session._foreground_stream_task = None


async def _generate_and_walk_chain(
    ctx: HandlerContext,
    *,
    parent: DialogNode,
    chosen_option_id: str | None,
    chosen_option_text: str | None,
    free_text: str | None = None,
) -> AsyncIterator[OutboundMessage]:
    """Generate a fresh chain of beats from the LLM and walk to the
    first one.

    Streaming-aware: drives the LLM stream + ``StreamingDialogParser``
    so the FIRST beat's state-patch + text-streaming + state-full
    messages emit as soon as that beat's closing brace arrives. The
    caller invokes us under ``play_lock``; after we emit beat 1 we
    spawn a background drain task that parses the rest of the chain
    into the tree (acquiring play_lock briefly per install) and
    then RETURN — which lets the caller's ``async with play_lock:``
    release. Subsequent play_advance dispatches can then walk into
    already-streamed beats without paying for the full chain
    duration; the prior architecture held the lock until the whole
    chain finalised so a Continue clicked 1 s after beat 1 had to
    wait the full ~30 s stream just to walk to beat 2 that the parser
    had already produced.
    """
    from ..domain.dialog import GenerationMetadata
    from ..orchestration.streaming_dialog_parser import StreamingDialogParser

    session = ctx.session
    assert session.game is not None
    game = session.game

    history = obsolescence.committed_path(game)
    on_stage = {
        cid: game.characters[cid] for cid in game.on_stage if not game.characters[cid].removed
    }
    off_stage = {
        cid: ch
        for cid, ch in game.characters.items()
        if cid not in game.on_stage and not ch.removed
    }
    profile = session.settings.user_profile
    from ..orchestration.assets import _long_context_image_model

    prompt = text_gen_prompts.build(
        world=game.world,
        history=history,
        on_stage=on_stage,
        off_stage=off_stage,
        chosen_option_text=chosen_option_text,
        free_text_input=free_text,
        mature_content=session.settings.mature_content,
        user_likes=profile.surfaced_likes(),
        user_dislikes=profile.surfaced_dislikes(),
        user_notes=profile.surfaced_notes(),
        environments=game.environments,
        music_enabled=session.settings.music.enabled,
        current_music_prompt=game.world.music_prompt,
        narrator_style=session.settings.narrator_style,
        detailed_appearance=_long_context_image_model(session),
    )

    parent_premise = session.compute_premise_hash(
        parent_id=parent.id,
        chosen_option_id=chosen_option_id,
    )
    metadata = GenerationMetadata(
        model=session.settings.llm.model,
        seed_parameters={"temperature": session.settings.llm.temperature},
    )

    build_node = _foreground_chain_node_builder(
        chosen_option_id=chosen_option_id,
        parent_premise=parent_premise,
        metadata=metadata,
    )

    parser = StreamingDialogParser()
    chain: list[DialogNode] = []
    cur_parent_id = parent.id
    music_change: str | None = None
    walked_first = False
    pending_beats: list[Any] = []

    session.latency.record(
        "llm_start",
        parent_id=parent.id,
        option_id=chosen_option_id,
        source="foreground",
    )

    stream = session.llm_stream_chunks(prompt)
    stream_iter = stream.__aiter__()

    # Phase 1 — under caller's play_lock: pull chunks until beat 1
    # parses, install + emit it, then schedule the drain task and
    # return. Subsequent beats from the SAME chunk (rare but possible)
    # are stashed in ``pending_beats`` so the drain installs them
    # first thing.
    while not walked_first:
        try:
            chunk = await stream_iter.__anext__()
        except StopAsyncIteration:
            break
        for beat in parser.feed(chunk):
            is_first = not chain
            node = build_node(beat, cur_parent_id, is_first=is_first)
            if is_first:
                chain.append(node)
                cur_parent_id = node.id
                if beat.music_change and beat.music_change.strip():
                    music_change = beat.music_change.strip()
                session.latency.record(
                    "beat_ready",
                    node_id=node.id,
                    parent_id=node.parent_id,
                    option_id=node.chosen_option_id,
                    source="foreground",
                )
                applied = _apply_node(session.game, node)
                applied = _ensure_environment_for_node(applied, node)
                session.install_game(applied)
                walked_first = True
                async for message in _emit_walk(ctx, node):
                    yield message
            else:
                # Beat parsed from the same chunk as beat 1 — defer
                # to the drain so the play_lock can release ASAP.
                pending_beats.append(beat)

    if walked_first:
        # Cancel any prior in-flight drain — a fresh chain supersedes
        # whatever the previous one was building.
        prior = session._foreground_stream_task
        if prior is not None and not prior.done():
            prior.cancel()
        drain = asyncio.create_task(
            _drain_remaining_foreground_chain(
                session=session,
                stream_iter=stream_iter,
                parser=parser,
                chain=chain,
                cur_parent_id=cur_parent_id,
                build_node=build_node,
                pending_beats=pending_beats,
                music_change=music_change,
            )
        )
        session._foreground_stream_task = drain
        return

    # Phase 1b — stream ended without yielding beat 1 via the
    # incremental parser. Try ``parser.finalize()`` (some providers
    # only fence the JSON at end-of-stream). If we still have nothing,
    # the LLM failed to produce content.
    payload = parser.finalize()
    if not payload.beats:
        raise SchemaError("LLM returned no beats")
    # Process beats inline — no remaining stream to drain.
    for beat in payload.beats:
        is_first = not chain
        node = build_node(beat, cur_parent_id, is_first=is_first)
        chain.append(node)
        cur_parent_id = node.id
        if beat.music_change and beat.music_change.strip():
            music_change = beat.music_change.strip()
        session.latency.record(
            "beat_ready",
            node_id=node.id,
            parent_id=node.parent_id,
            option_id=node.chosen_option_id,
            source="foreground_finalize",
        )
        if not walked_first:
            applied = _apply_node(session.game, node)
            applied = _ensure_environment_for_node(applied, node)
            session.install_game(applied)
            walked_first = True
            async for message in _emit_walk(ctx, node):
                yield message
        else:
            tree_updated = _add_beats_to_tree(session.game, [node])
            session.install_game(tree_updated)

    if not chain:
        raise SchemaError("LLM returned no beats")

    # Inline finalize (no drain — stream had no more chunks).
    last_idx = len(chain) - 1
    last_with_options = chain[last_idx].model_copy(
        update={"options": list(payload.options)},
    )
    chain[last_idx] = last_with_options
    nodes_updated = dict(session.game.dialog_tree.nodes)
    nodes_updated[last_with_options.id] = last_with_options
    new_tree = session.game.dialog_tree.model_copy(
        update={"nodes": nodes_updated},
    )
    session.install_game(session.game.model_copy(update={"dialog_tree": new_tree}))
    if music_change:
        last_with_options.generation_metadata.seed_parameters["music_change"] = music_change

    # ``_emit_walk`` above sent a full state that only carried beat 1
    # (the rest of the chain was added to the tree afterward). Refresh
    # the renderer with the complete tree so its dialog_tree mirror has
    # the walkable children + the tail's options — same reason the
    # streaming drain pushes per-beat patches: without it the Continue
    # glyph stays a spinner even though the next beat is already in the
    # tree.
    if len(chain) > 1 or payload.options:
        yield (MessageType.s2c_state_full, _full_state_message(ctx.session))

    await session.commit()
    _spawn_async_assets(session)
    _spawn_async_summarizer(session)
    _ensure_speculative_branches(session)
    tail = chain[-1]
    if tail.id != session.game.current_node_id and tail.options:
        _ensure_speculative_branches(
            session,
            target_node_id=tail.id,
            options_subset=[tail.options[0].id],
        )
    if music_change:
        _spawn_async_music(session, music_change)


_UNDO_STACK_MAX = 30


def _push_undo_snapshot(session: Session) -> None:
    """Capture the current Game so a later ``c2s/play/undo`` can
    restore it. Called at the top of every play handler that
    mutates committed state. Bounded length: when the stack
    exceeds ``_UNDO_STACK_MAX``, the OLDEST entry falls off — we
    keep the recent N turns reachable, not the entire run."""
    if session.game is None:
        return
    session.undo_stack.append(session.game.model_copy(deep=True))
    if len(session.undo_stack) > _UNDO_STACK_MAX:
        session.undo_stack = session.undo_stack[-_UNDO_STACK_MAX:]


# Max time play_advance will wait for an in-flight foreground drain
# to install the next beat. The drain installs each beat as soon as
# the LLM parser emits it; this poll budget bounds the wait so that
# if the stream genuinely has no next beat for this parent (e.g.
# we're at the chain tail), we fall through to fresh generation
# instead of hanging.
_FOREGROUND_DRAIN_POLL_BUDGET_S = 60.0
_FOREGROUND_DRAIN_POLL_INTERVAL_S = 0.02


async def _wait_for_foreground_drain_child(
    session: Session,
    *,
    parent_id: str,
    chosen_option_id: str | None,
) -> bool:
    """Yield while an in-flight foreground drain task installs the
    next beat under ``parent_id``. Returns ``True`` when a matching
    child appears (or was already in the tree), ``False`` when the
    drain finishes without producing one (caller should fall through
    to a fresh generation).

    MUST be called with ``session.play_lock`` released — the drain
    re-acquires the lock briefly to install each beat, so holding
    it here deadlocks.
    """
    import time as _time

    task = session._foreground_stream_task
    if task is None or task.done():
        return False
    deadline = _time.monotonic() + _FOREGROUND_DRAIN_POLL_BUDGET_S
    while _time.monotonic() < deadline:
        await asyncio.sleep(_FOREGROUND_DRAIN_POLL_INTERVAL_S)
        game = session.game
        if game is None:
            return False
        if (
            obsolescence.first_valid_child(
                game,
                parent_id,
                chosen_option_id=chosen_option_id,
            )
            is not None
        ):
            return True
        if task.done():
            # Drain finished without producing a child for this
            # (parent, option) — the chain's tail was reached.
            return (
                obsolescence.first_valid_child(
                    game,
                    parent_id,
                    chosen_option_id=chosen_option_id,
                )
                is not None
            )
    _log.warning(
        "foreground drain wait timed out after %.0fs (parent=%s option=%s)",
        _FOREGROUND_DRAIN_POLL_BUDGET_S,
        parent_id[-6:],
        (chosen_option_id or "__continue__")[-12:],
    )
    return False


def _mark_foreground(session: Session) -> None:
    """Register the running dispatch task as the session's foreground
    generation so ``c2s/play/cancel`` has a handle to cancel.

    Only the play handlers that can BLOCK the player on an LLM call
    mark themselves — undo and the edit handlers are fast and local,
    and a Stop click should never be able to interrupt them halfway
    through a tree mutation."""
    try:
        session._foreground_task = asyncio.current_task()
    except RuntimeError:  # pragma: no cover -- no running loop
        session._foreground_task = None


def _clear_foreground(session: Session) -> None:
    """Deregister, but only if we're still the registered task — a
    newer turn may have already claimed the slot while this one was
    unwinding."""
    try:
        current = asyncio.current_task()
    except RuntimeError:  # pragma: no cover -- no running loop
        current = None
    if session._foreground_task is current:
        session._foreground_task = None


async def play_cancel_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    """Stop whatever generation the player is waiting on.

    Cancelling the dispatch task unwinds the play handler through its
    ``finally``, which releases the play lock, and — for an embedded
    render already in a worker thread — sets the per-render abort event
    so the denoise loop stops instead of running to completion while
    still holding ``gpu_lock``.
    """
    assert isinstance(payload, C2SPlayCancel)
    cancelled = ctx.session.cancel_foreground()

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield MessageType.s2c_play_cancelled, S2CPlayCancelled(cancelled=cancelled)

    return gen()


async def play_advance_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SPlayAdvance)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    play_lock = ctx.session._ensure_play_lock()

    # Record the click at handler entry (BEFORE acquiring the play lock)
    # so lock contention from a still-finalising prior turn doesn't bias
    # the click timestamp. parent_id read here is the node the player
    # actually clicked from; same node will be re-read under the lock
    # for the mutation logic.
    _click_parent_id = ctx.session.game.current_node_id
    ctx.session.latency.record(
        "click",
        parent_id=_click_parent_id,
        option_id=payload.option_id,
        kind="advance",
    )

    async def gen() -> AsyncIterator[OutboundMessage]:
        # Manual lock management so we can RELEASE it while awaiting
        # external work (speculation tasks, in-flight foreground
        # drain). The original ``async with play_lock`` implementation
        # held the lock for the entire handler, which serialised play
        # handlers safely but also blocked a Continue click from
        # walking into a beat the in-flight foreground drain had
        # already parsed — the click would have to wait for the
        # whole chain to finalise.
        _mark_foreground(ctx.session)
        await play_lock.acquire()
        lock_held = True
        try:
            if ctx.session.game is None:
                raise NotFoundError("no game loaded")
            game = ctx.session.game
            if game.current_node_id is None:
                raise NotFoundError("no current node")
            parent = game.dialog_tree.nodes[game.current_node_id]
            _push_undo_snapshot(ctx.session)

            chosen_option_id = payload.option_id
            chosen_option_text: str | None = None
            if chosen_option_id is not None:
                for option in parent.options:
                    if option.id == chosen_option_id:
                        chosen_option_text = option.text
                        break
                if chosen_option_text is None:
                    raise NotFoundError(f"unknown option {chosen_option_id!r} on current node")

            advance_label = (
                f"parent={parent.id[-6:]} option={(chosen_option_id or '__continue__')[-12:]}"
            )

            existing_id = obsolescence.first_valid_child(
                game, parent.id, chosen_option_id=chosen_option_id
            )
            if existing_id is not None:
                existing = game.dialog_tree.nodes[existing_id]
                _spec_log.info(
                    "advance HIT: in-tree child %s — walking without LLM (%s)",
                    existing.id[-6:],
                    advance_label,
                )
                async for message in _walk_existing_child(ctx, existing):
                    yield message
                return

            _spec_log.info(
                "advance MISS: no in-tree child — checking speculation (%s)",
                advance_label,
            )
            # Release lock so speculation tasks (which don't take the
            # lock today) and the foreground drain (which acquires it
            # briefly per install) can run. Re-acquire afterwards.
            play_lock.release()
            lock_held = False
            await _await_speculation(
                ctx.session,
                parent_id=parent.id,
                option_id=chosen_option_id,
            )
            await play_lock.acquire()
            lock_held = True

            game = ctx.session.game
            if game is None or game.current_node_id is None:
                raise NotFoundError("game state went away during speculation await")
            existing_id = obsolescence.first_valid_child(
                game,
                parent.id,
                chosen_option_id=chosen_option_id,
            )
            if existing_id is not None:
                existing = game.dialog_tree.nodes[existing_id]
                _spec_log.info(
                    "advance: speculation paid off — walking %s (%s)",
                    existing.id[-6:],
                    advance_label,
                )
                async for message in _walk_existing_child(ctx, existing):
                    yield message
                return

            # Still nothing in the tree. Before kicking off a fresh
            # LLM call, check if there's an in-flight foreground drain
            # that's currently producing beats — if its next install
            # would land here, waiting for it is much cheaper than
            # racing a second LLM call.
            drain_task = ctx.session._foreground_stream_task
            if drain_task is not None and not drain_task.done():
                _spec_log.info(
                    "advance: in-flight foreground drain — waiting for next beat (%s)",
                    advance_label,
                )
                play_lock.release()
                lock_held = False
                drain_paid_off = await _wait_for_foreground_drain_child(
                    ctx.session,
                    parent_id=parent.id,
                    chosen_option_id=chosen_option_id,
                )
                await play_lock.acquire()
                lock_held = True
                if drain_paid_off:
                    game = ctx.session.game
                    if game is None or game.current_node_id is None:
                        raise NotFoundError("game state went away during drain wait")
                    existing_id = obsolescence.first_valid_child(
                        game,
                        parent.id,
                        chosen_option_id=chosen_option_id,
                    )
                    if existing_id is not None:
                        existing = game.dialog_tree.nodes[existing_id]
                        _spec_log.info(
                            "advance: drain paid off — walking %s (%s)",
                            existing.id[-6:],
                            advance_label,
                        )
                        async for message in _walk_existing_child(ctx, existing):
                            yield message
                        return

            # Final pre-fresh-gen check: drain's finalise block (when
            # the drained chain's tail had no options) spawns a
            # Continue speculation for the tail. That speculation
            # didn't exist when we did our first ``_await_speculation``
            # above; if it just-launched while we were polling the
            # drain, joining it now is cheaper than starting a fresh
            # competing LLM call. ``_await_speculation`` is cheap
            # when no task exists at the key (returns immediately).
            play_lock.release()
            lock_held = False
            await _await_speculation(
                ctx.session,
                parent_id=parent.id,
                option_id=chosen_option_id,
            )
            await play_lock.acquire()
            lock_held = True
            game = ctx.session.game
            if game is None or game.current_node_id is None:
                raise NotFoundError("game state went away during late-speculation await")
            existing_id = obsolescence.first_valid_child(
                game,
                parent.id,
                chosen_option_id=chosen_option_id,
            )
            if existing_id is not None:
                existing = game.dialog_tree.nodes[existing_id]
                _spec_log.info(
                    "advance: late speculation paid off — walking %s (%s)",
                    existing.id[-6:],
                    advance_label,
                )
                async for message in _walk_existing_child(ctx, existing):
                    yield message
                return

            _spec_log.info(
                "advance: speculation MISS — falling back to fresh foreground gen (%s)",
                advance_label,
            )
            async for message in _generate_and_walk_chain(
                ctx,
                parent=parent,
                chosen_option_id=chosen_option_id,
                chosen_option_text=chosen_option_text,
            ):
                yield message
        finally:
            if lock_held:
                play_lock.release()
            _clear_foreground(ctx.session)

    return gen()


async def play_free_text_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SPlayFreeText)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")

    play_lock = ctx.session._ensure_play_lock()

    # See play_advance_handler for why click is recorded outside the lock.
    ctx.session.latency.record(
        "click",
        parent_id=ctx.session.game.current_node_id,
        option_id=None,
        kind="free_text",
    )

    async def gen() -> AsyncIterator[OutboundMessage]:
        # Serialize against play_advance / play_undo. Without this
        # lock, a free-text submitted while play_advance was awaiting
        # speculation produced a "skip backwards" — both handlers
        # raced to install_game and whichever finished last won,
        # which could be the older premise.
        #
        # Manual acquire/release (rather than ``async with``) so
        # ``_generate_and_walk_chain`` can hand the lock back early
        # via the drain-task pattern (see its docstring).
        _mark_foreground(ctx.session)
        await play_lock.acquire()
        lock_held = True
        try:
            if ctx.session.game is None:
                raise NotFoundError("no game loaded")
            game = ctx.session.game
            if game.current_node_id is None:
                raise NotFoundError("no current node")
            parent = game.dialog_tree.nodes[game.current_node_id]
            _push_undo_snapshot(ctx.session)

            invalidated = obsolescence.invalidate_unwalked_descendants(
                game,
                parent.id,
            )
            _log.info(
                "free_text invalidated %d unwalked descendants",
                len(invalidated),
            )
            ctx.session.invalidate_speculation_from(parent.id)
            _cancel_speculative_descendants(
                ctx.session,
                parent_id=parent.id,
                also_invalidated_ids=invalidated,
            )
            # Cancel any in-flight foreground drain — the chain it was
            # extending is now obsolete (the player's free-text turn
            # supersedes whatever the previous turn was streaming).
            prior_stream = ctx.session._foreground_stream_task
            if prior_stream is not None and not prior_stream.done():
                prior_stream.cancel()

            async for message in _generate_and_walk_chain(
                ctx,
                parent=parent,
                chosen_option_id=None,
                chosen_option_text=None,
                free_text=payload.text,
            ):
                yield message
        finally:
            if lock_held:
                play_lock.release()
            _clear_foreground(ctx.session)

    return gen()


def _ensure_environment_for_node(game: Game, node: DialogNode) -> Game:
    """Promote ``node.location_prompt`` into the environments map.

    If a new node introduces a location with a fresh prompt, attach an
    ``Environment`` entry so the asset generator can produce a
    background for it. Reuses an existing entry when the prompt hash
    matches; refreshes the existing entry's ``lighting`` when this
    beat overrides it (so a "the lamps go out" beat updates the
    stored lighting that drives both background and portrait
    prompts).
    """
    from ..orchestration.session import make_environment

    if not node.location_prompt:
        return game
    label = node.location_id or "scene"
    candidate = make_environment(label=label, prompt=node.location_prompt)
    candidate = candidate.model_copy(update={"lighting": node.location_lighting})
    for existing in game.environments.values():
        if existing.prompt_hash == candidate.prompt_hash:
            updates: dict[str, str] = {}
            # Refresh stored lighting only when this beat actually
            # carries a new descriptor; an empty string means "no
            # change, keep what was set last time the env was used."
            if node.location_lighting and node.location_lighting != existing.lighting:
                updates["lighting"] = node.location_lighting
            new_envs = dict(game.environments)
            if updates:
                new_envs[existing.id] = existing.model_copy(update=updates)
            new_node = node.model_copy(update={"location_id": existing.id})
            new_nodes = dict(game.dialog_tree.nodes)
            new_nodes[node.id] = new_node
            new_tree = game.dialog_tree.model_copy(update={"nodes": new_nodes})
            game_updates: dict[str, object] = {"dialog_tree": new_tree}
            if updates:
                game_updates["environments"] = new_envs
            return game.model_copy(update=game_updates)
    new_envs = dict(game.environments)
    new_envs[candidate.id] = candidate
    new_node = node.model_copy(update={"location_id": candidate.id})
    new_nodes = dict(game.dialog_tree.nodes)
    new_nodes[node.id] = new_node
    new_tree = game.dialog_tree.model_copy(update={"nodes": new_nodes})
    return game.model_copy(update={"dialog_tree": new_tree, "environments": new_envs})


def _emit_render_failure_notice(session: Session, exc: Exception, *, kind: str) -> None:
    """Push a player-facing modal explaining a swallowed render
    failure. Called from the catch sites in ``_generate_assets_safe``
    and the speculative pre-render task so a hidden OOM / pipeline
    import error / stale-music-model symptom doesn't manifest as
    "the engine just skipped my image."

    Body is short and copy-paste friendly so the player can
    forward it to a bug report. ``kind`` distinguishes the
    foreground portrait/background pass from the speculative
    pre-render in the modal title.
    """
    if session.emit is None:
        return
    from ..api.errors import ProviderUnreachableError
    from .messages import MessageType, NoticeKind, S2CNotice

    summary = str(exc) or exc.__class__.__name__
    # Trim runaway tracebacks that some libraries embed in their
    # ``str(exc)`` (diffusers does this when an import fails). The
    # full text is in the log; the modal just needs the headline.
    if len(summary) > 480:
        summary = summary[:480].rstrip() + "…"

    # OOM is by far the most common cause of this path on a real
    # game session — call it out by name and tell the player what
    # to try next, rather than just dumping the exception string.
    if "out of memory" in summary.lower() or (
        "cuda" in summary.lower() and "memory" in summary.lower()
    ):
        body = (
            "The image render ran out of GPU memory. Try one of: "
            "(a) turn off background music in Settings → Audio, "
            "(b) close other GPU apps and retry, "
            "(c) switch to a smaller checkpoint. "
            f"Diagnostic: {summary}"
        )
    elif isinstance(exc, ProviderUnreachableError):
        body = (
            "The image render couldn't reach its backend. Check "
            "Settings → Image to verify your model is selected, or "
            f"open the console for the full trace. Diagnostic: {summary}"
        )
    else:
        body = (
            f"The image render failed and was skipped. The scene "
            "will keep using the previous portrait/background. "
            f"Diagnostic: {summary}"
        )

    try:
        session.emit(
            MessageType.s2c_notice,
            S2CNotice(
                title=f"Image render skipped ({kind})",
                body=body,
                kind=NoticeKind.warning,
            ),
        )
    except Exception:
        _log.warning("failed to emit render-failure notice", exc_info=True)


# ---------- undo ----------


async def play_undo_handler(_payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    """Pop the most recent advance off the session's undo stack and
    install the prior Game state. Pushes the CURRENT Game onto a
    redo stack would be the symmetric op; not implemented yet —
    undo only goes one direction for now."""
    if not ctx.session.undo_stack:
        raise NotFoundError("nothing to undo")
    play_lock = ctx.session._ensure_play_lock()

    async def gen() -> AsyncIterator[OutboundMessage]:
        async with play_lock:
            if not ctx.session.undo_stack:
                raise NotFoundError("nothing to undo")
            prior = ctx.session.undo_stack.pop()
            ctx.session.install_game(prior)
            await ctx.session.commit()
            if prior.current_node_id is not None:
                invalidated = ctx.session.invalidate_speculation_from(
                    prior.current_node_id,
                )
                _cancel_speculative_descendants(
                    ctx.session,
                    parent_id=prior.current_node_id,
                    also_invalidated_ids=invalidated,
                )
            _ensure_speculative_branches(ctx.session)
            # Re-evaluate staleness for the restored state. Normally an
            # undo snapshot is self-consistent (its images match its
            # attributes), so this is a no-op; but if the snapshot was
            # captured while a render was still in flight, the restored
            # on-screen entity is stale and nothing else would redraw it.
            # Every other state-changing path notifies the scheduler —
            # undo must too, so it always renders whatever is stale.
            ctx.session.render_scheduler.on_turn()
            yield (MessageType.s2c_state_full, _full_state_message(ctx.session))

    return gen()


# ---------- edits ----------


# Player-editable fields, per model. Anything not listed here is
# either derived (``prompt_hash``), identity (``id``), engine-owned
# (``plot_outline``, ``images``, ``facts``), or a filesystem path
# (``image_path``, ``music_path``) that must never be settable from
# a client frame. Mirrors the allow-list the interview edit path
# already enforces in ``new_game_edit_review_handler``.
_EDITABLE_WORLD_FIELDS = frozenset(
    {
        "game_name",
        "setting",
        "genre",
        "visual_style",
        "overall_plot_direction",
        "summarizer_assessment",
        "prompt_history_clamp_chars",
    }
)

_EDITABLE_CHARACTER_FIELDS = frozenset(
    {
        "name",
        "description",
        "kind",
        "physical_description",
        "gender",
        "pronouns",
        "age",
        "ethnicity",
        "skin",
        "hair_color",
        "hairstyle",
        "eye_color",
        "build",
        "bust",
        "outfit",
        "pose",
        "expression",
        "effects",
        "decals",
        "seed",
    }
)

_EDITABLE_ENVIRONMENT_FIELDS = frozenset(
    {
        "location_label",
        "prompt",
        "lighting",
        "seed",
    }
)


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _validated_update(
    model: _ModelT,
    field: str,
    value: Any,
    allowed: frozenset[str],
    label: str,
) -> _ModelT:
    """Return a copy of ``model`` with ``field`` set to ``value``,
    running the model's own validators.

    ``model_copy(update=...)`` writes straight into ``__dict__``
    with no validation, so a single edit frame could persist a
    ``game.json`` that fails to re-validate on load — an
    unrecoverable save. Round-tripping through ``model_validate``
    means a bad value is rejected as a schema error before anything
    is installed or committed.
    """
    if field not in allowed:
        raise SchemaError(f"{label}: field {field!r} is not editable; valid: {sorted(allowed)}")
    data = model.model_dump()
    data[field] = value
    try:
        return type(model).model_validate(data)
    except ValidationError as exc:
        raise SchemaError(f"{label}: invalid value for {field!r}: {exc}") from exc


def _patch_world(world: WorldState, field: str, value: Any) -> WorldState:
    return _validated_update(
        world,
        field,
        value,
        _EDITABLE_WORLD_FIELDS,
        "edit_world",
    )


async def edit_world_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SEditWorld)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    new_world = _patch_world(ctx.session.game.world, payload.field, payload.value)
    new_game = ctx.session.game.model_copy(update={"world": new_world})
    if new_game.current_node_id:
        ctx.session.install_game(new_game)
        ctx.session.invalidate_speculation_from(new_game.current_node_id)
    else:
        ctx.session.install_game(new_game)
    await ctx.session.commit()

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[PatchOp(op="replace", path=f"/world/{payload.field}", value=payload.value)]
            ),
        )

    return gen()


async def edit_character_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SEditCharacter)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    from ..domain.dialog import CharacterAttributeField, CharacterChange

    char = ctx.session.game.characters.get(payload.character_id)
    if char is None:
        raise NotFoundError(f"unknown character {payload.character_id}")
    updated = _validated_update(
        char,
        payload.field,
        payload.value,
        _EDITABLE_CHARACTER_FIELDS,
        "edit_character",
    )
    new_chars = dict(ctx.session.game.characters)
    new_chars[char.id] = updated

    # Stamp the edit onto the current beat's character_changes so
    # the change persists across replay/retcon/summarizer passes.
    # Without this, the next beat's _apply_node would re-derive
    # state from the stored character_changes (which still
    # reflected the LLM's pre-edit choice) and overwrite the
    # player's edit a few frames later. Only fields that map onto
    # the dialog-level attribute enum can ride this path —
    # ``seed`` / ``images`` etc. live outside the dialog state
    # machine and don't need recording.
    new_tree = ctx.session.game.dialog_tree
    current_id = ctx.session.game.current_node_id
    try:
        attr_field = CharacterAttributeField(payload.field)
    except ValueError:
        attr_field = None
    extra_ops: list[PatchOp] = []
    if attr_field is not None and current_id is not None and current_id in new_tree.nodes:
        current_node = new_tree.nodes[current_id]
        kept = [
            change
            for change in current_node.character_changes
            if not (change.character_id == char.id and change.field == attr_field)
        ]
        kept.append(
            CharacterChange(
                character_id=char.id,
                field=attr_field,
                new_value=str(payload.value),
            )
        )
        new_node = current_node.model_copy(update={"character_changes": kept})
        new_nodes = dict(new_tree.nodes)
        new_nodes[current_id] = new_node
        new_tree = new_tree.model_copy(update={"nodes": new_nodes})
        extra_ops.append(
            PatchOp(
                op="replace",
                path=f"/dialog_tree/nodes/{current_id}/character_changes",
                value=[c.model_dump() for c in kept],
            )
        )
    new_game = ctx.session.game.model_copy(
        update={"characters": new_chars, "dialog_tree": new_tree},
    )
    ctx.session.install_game(new_game)
    if new_game.current_node_id:
        ctx.session.invalidate_speculation_from(new_game.current_node_id)
    await ctx.session.commit()

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path=f"/characters/{char.id}/{payload.field}",
                        value=payload.value,
                    ),
                    *extra_ops,
                ]
            ),
        )

    return gen()


async def edit_character_dismiss_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    """Mark a character ``removed=True`` and drop them from on_stage.
    Captures an undo snapshot so the action is reversible. The
    storyteller / summarizer prompts filter removed characters out,
    so the LLM stops re-summoning them on subsequent turns."""
    assert isinstance(payload, C2SEditCharacterDismiss)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    game = ctx.session.game
    char = game.characters.get(payload.character_id)
    if char is None:
        raise NotFoundError(f"unknown character {payload.character_id}")
    if char.is_player:
        raise SchemaError("cannot dismiss the player character")
    _push_undo_snapshot(ctx.session)
    updated = char.model_copy(
        update={"removed": True, "removed_reason": payload.reason or "dismissed"}
    )
    new_chars = dict(game.characters)
    new_chars[char.id] = updated
    new_on_stage = [cid for cid in game.on_stage if cid != char.id]
    new_game = game.model_copy(update={"characters": new_chars, "on_stage": new_on_stage})
    ctx.session.install_game(new_game)
    await ctx.session.commit()

    async def gen() -> AsyncIterator[OutboundMessage]:
        # Targeted ops, not a full state push: dismissing one character
        # changes three scalars, and shipping the whole game to say so
        # costs ~1 MB on the wire at a thousand-node tree.
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path=f"/characters/{char.id}/removed",
                        value=True,
                    ),
                    PatchOp(
                        op="replace",
                        path=f"/characters/{char.id}/removed_reason",
                        value=updated.removed_reason,
                    ),
                    PatchOp(op="replace", path="/on_stage", value=new_on_stage),
                ]
            ),
        )

    return gen()


async def edit_character_show_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    """Inverse of dismiss — clear ``removed`` and put the character
    back on stage. Captures an undo snapshot so the action is
    reversible. Useful when the storyteller drops a character the
    player still wants present, or when the player changes their
    mind about a dismissal."""
    assert isinstance(payload, C2SEditCharacterShow)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    game = ctx.session.game
    char = game.characters.get(payload.character_id)
    if char is None:
        raise NotFoundError(f"unknown character {payload.character_id}")
    _push_undo_snapshot(ctx.session)
    updated = char.model_copy(update={"removed": False, "removed_reason": ""})
    new_chars = dict(game.characters)
    new_chars[char.id] = updated
    new_on_stage = list(game.on_stage)
    if not char.is_player and char.id not in new_on_stage:
        new_on_stage.append(char.id)
    new_game = game.model_copy(update={"characters": new_chars, "on_stage": new_on_stage})
    ctx.session.install_game(new_game)
    await ctx.session.commit()

    async def gen() -> AsyncIterator[OutboundMessage]:
        # Mirror of dismiss — three targeted ops rather than the whole
        # game state. See ``edit_character_dismiss_handler``.
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path=f"/characters/{char.id}/removed",
                        value=False,
                    ),
                    PatchOp(
                        op="replace",
                        path=f"/characters/{char.id}/removed_reason",
                        value="",
                    ),
                    PatchOp(op="replace", path="/on_stage", value=new_on_stage),
                ]
            ),
        )

    return gen()


async def edit_character_rerender_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    """Re-run the portrait pipeline for one character against its
    CURRENT attributes. Used after the player edits a field
    (outfit / pose / hair / etc.) and wants to apply the change
    immediately without waiting for the next turn's
    ``ensure_assets`` pass.

    Does NOT bump the seed — seed is part of character identity and
    a stable seed keeps the face consistent across renders. To get
    a different draw of the same character, the player edits the
    seed itself via the Cast tab.

    The pipeline is idempotent: if the character's attributes
    haven't changed since the existing image was rendered (same
    prompt_hash), no work happens — the on-disk file is already
    current. The new image (when generated) is appended to
    ``character.images``; the renderer swaps to ``images[0]`` so
    the freshest render is always on screen.
    """
    assert isinstance(payload, C2SEditCharacterRerender)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    char = ctx.session.game.characters.get(payload.character_id)
    if char is None:
        raise NotFoundError(f"unknown character {payload.character_id}")

    # Explicit player request: hand it to the redraw scheduler with the
    # +1000 boost so it jumps ahead of any accumulated turn staleness and
    # renders next. ``request_redraw`` carries the PC authorisation (the
    # one legitimate path that refreshes the camera-lens character) and
    # the fresh portrait arrives over the emit channel via state_full +
    # image_ready when the render lands.
    ctx.session.render_scheduler.request_redraw(char.id)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_state_full, _full_state_message(ctx.session))

    return gen()


async def edit_history_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SEditHistory)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    nodes = dict(ctx.session.game.dialog_tree.nodes)
    if payload.node_id not in nodes:
        raise NotFoundError(f"unknown node {payload.node_id}")
    edited = nodes[payload.node_id].model_copy(update={"text": payload.new_text})
    nodes[payload.node_id] = edited
    new_tree = ctx.session.game.dialog_tree.model_copy(update={"nodes": nodes})
    new_game = ctx.session.game.model_copy(update={"dialog_tree": new_tree})
    ctx.session.install_game(new_game)
    if new_game.current_node_id:
        ctx.session.invalidate_speculation_from(new_game.current_node_id)
    await ctx.session.commit()

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path=f"/dialog_tree/nodes/{payload.node_id}/text",
                        value=payload.new_text,
                    )
                ]
            ),
        )

    return gen()


async def edit_history_delete_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SEditHistoryDelete)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    game = ctx.session.game
    tree = game.dialog_tree
    node_id = payload.node_id
    if node_id not in tree.nodes:
        raise NotFoundError(f"unknown node {node_id}")
    if node_id == tree.root_id:
        raise SchemaError("cannot delete the root beat")
    if node_id == game.current_node_id:
        raise SchemaError("cannot delete the current beat")
    committed = list(tree.committed_path)
    if node_id not in committed:
        raise SchemaError("can only delete beats on the committed path")
    idx = committed.index(node_id)
    deleted = tree.nodes[node_id]
    parent_id = deleted.parent_id

    new_committed = committed[:idx] + committed[idx + 1 :]
    new_nodes = dict(tree.nodes)
    # Splice: rewrite the parent_id of the next committed beat (if any)
    # to skip over the deleted node so the chain stays linked.
    if idx < len(committed) - 1:
        successor_id = committed[idx + 1]
        successor = new_nodes[successor_id]
        new_nodes[successor_id] = successor.model_copy(update={"parent_id": parent_id})
    # Remove the deleted node from the tree entirely.
    del new_nodes[node_id]

    new_tree = tree.model_copy(update={"nodes": new_nodes, "committed_path": new_committed})
    new_game = game.model_copy(update={"dialog_tree": new_tree})
    ctx.session.install_game(new_game)
    # Anything speculatively rendered after the deletion point is now
    # premised on a vanished beat — invalidate everything past the
    # splice point so fresh speculation resumes from the new tail.
    if new_committed:
        ctx.session.invalidate_speculation_from(new_committed[-1])
    await ctx.session.commit()

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_full,
            S2CStateFull(game=new_game, settings=ctx.session.settings),
        )

    return gen()


async def apply_global_retcon(
    ctx: HandlerContext,
    instructions: str,
    *,
    push_undo: bool = True,
) -> tuple[int, int]:
    """Run the LLM-driven retcon pass against the live game.

    Walks the committed history and every character's mutable state,
    asking the LLM to rewrite whichever pieces need to change to fit
    ``instructions``. Returns ``(rewritten_beats, character_updates)``.

    Extracted from the c2s/edit/history/retcon handler so the
    safety-side path (under-18 + nudity → bump age and rewrite past
    age references) can reuse the same machinery without duplicating
    the batched-call orchestration. Set ``push_undo=False`` when the
    caller has already pushed a snapshot or doesn't want one (e.g.
    a chained safety retcon issued mid-asset-generation).
    """
    from ..domain.dialog import CharacterAttributeField
    from ..orchestration.prompts import retcon as retcon_prompts

    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    if not instructions.strip():
        raise SchemaError("retcon instructions are required")

    game = ctx.session.game
    committed = list(game.dialog_tree.committed_path)
    nodes = dict(game.dialog_tree.nodes)
    committed_history = [nodes[nid] for nid in committed if nid in nodes]

    if push_undo:
        _push_undo_snapshot(ctx.session)

    # BATCHED RETCON: instead of one giant LLM call (which truncates
    # against non-frontier models' short output windows — the user
    # reported "rewrite doesn't seem to rewrite all lines"), fan out
    # N small calls in parallel:
    #
    #   * one call per ``RETCON_BATCH_SIZE`` slice of committed beats;
    #   * one call for character_updates;
    #   * one call for world_updates.
    #
    # Each call's output stays well under the default 1024-token cap,
    # so cheap models parse cleanly. asyncio.gather runs them
    # concurrently, so wall-clock time is roughly 1 LLM round-trip
    # instead of N + 2.
    batches = retcon_prompts.chunk_beats(committed_history)

    async def _run_beat_batch(batch: list[DialogNode]) -> LlmRetconBeatBatch:
        if not batch:
            return LlmRetconBeatBatch()
        beat_prompt = retcon_prompts.build_beat_batch(
            world=game.world,
            instructions=instructions,
            committed_history=committed_history,
            batch=batch,
            characters=dict(game.characters),
        )
        # 4096 leaves headroom for verbose rewrites of long beats
        # without blowing past the smaller models' caps. Each batch
        # is at most ~4 beats so the typical response is ~600-800
        # tokens; this cap exists for safety, not norm.
        beat_raw, _ = await ctx.session.llm_text(beat_prompt, max_tokens=4096)
        return parse_json_object(beat_raw, LlmRetconBeatBatch)

    async def _run_character_updates() -> LlmRetconCharacterUpdates:
        char_prompt = retcon_prompts.build_character_updates(
            world=game.world,
            instructions=instructions,
            characters=dict(game.characters),
        )
        char_raw, _ = await ctx.session.llm_text(char_prompt, max_tokens=2048)
        return parse_json_object(char_raw, LlmRetconCharacterUpdates)

    async def _run_world_updates() -> LlmRetconWorldUpdates:
        world_prompt = retcon_prompts.build_world_updates(
            world=game.world,
            instructions=instructions,
        )
        # World updates are at most two short strings — even a
        # 512-token cap is generous.
        world_raw, _ = await ctx.session.llm_text(world_prompt, max_tokens=1024)
        return parse_json_object(world_raw, LlmRetconWorldUpdates)

    import asyncio as _asyncio

    # Heterogeneous by design: the beat batches come first and the
    # character + world coroutines are appended last, which is exactly
    # what the ``results[:-2]`` / ``results[-2]`` / ``results[-1]``
    # unpacking below relies on.
    pending: list[
        Coroutine[
            Any,
            Any,
            LlmRetconBeatBatch | LlmRetconCharacterUpdates | LlmRetconWorldUpdates,
        ]
    ] = [_run_beat_batch(batch) for batch in batches]
    pending.append(_run_character_updates())
    pending.append(_run_world_updates())
    results = await _asyncio.gather(*pending, return_exceptions=True)

    # The last two entries are character + world results; everything
    # before is a beat batch. Failures degrade gracefully — one
    # batch erroring shouldn't lose the others' rewrites.
    beat_results: list[LlmRetconBeatBatch] = []
    char_result: LlmRetconCharacterUpdates | None = None
    world_result: LlmRetconWorldUpdates | None = None
    failures: list[str] = []
    for idx, res in enumerate(results[:-2]):
        if isinstance(res, Exception):
            failures.append(f"beat batch {idx}: {res}")
        elif isinstance(res, LlmRetconBeatBatch):
            beat_results.append(res)
    if isinstance(results[-2], Exception):
        failures.append(f"character updates: {results[-2]}")
    elif isinstance(results[-2], LlmRetconCharacterUpdates):
        char_result = results[-2]
    if isinstance(results[-1], Exception):
        failures.append(f"world updates: {results[-1]}")
    elif isinstance(results[-1], LlmRetconWorldUpdates):
        world_result = results[-1]
    if failures:
        _log.warning(
            "retcon partial failure (continuing with what landed): %s",
            "; ".join(failures),
        )

    committed_set = set(committed)
    rewritten = 0
    for batch_result in beat_results:
        for rewrite in batch_result.rewritten_beats:
            existing = nodes.get(rewrite.node_id)
            if existing is None or rewrite.node_id not in committed_set:
                continue
            update: dict[str, object] = {"text": rewrite.text}
            if rewrite.speaker_id is not None:
                update["speaker_id"] = rewrite.speaker_id or None
            nodes[rewrite.node_id] = existing.model_copy(update=update)
            rewritten += 1

    valid_fields = {f.value for f in CharacterAttributeField}
    new_chars = dict(game.characters)
    char_updates = 0
    if char_result is not None:
        for upd in char_result.character_updates:
            char = new_chars.get(upd.character_id)
            if char is None or upd.field not in valid_fields:
                continue
            new_chars[upd.character_id] = char.model_copy(update={upd.field: upd.new_value})
            char_updates += 1

    new_world = game.world
    world_update_keys = {"summarizer_assessment", "overall_plot_direction"}
    world_patch: dict[str, str] = {}
    if world_result is not None:
        world_patch = {
            k: v
            for k, v in world_result.world_updates.items()
            if k in world_update_keys and isinstance(v, str) and v.strip()
        }
    if world_patch:
        new_world = new_world.model_copy(update=world_patch)

    new_tree = game.dialog_tree.model_copy(update={"nodes": nodes})
    new_game = game.model_copy(
        update={
            "dialog_tree": new_tree,
            "characters": new_chars,
            "world": new_world,
        }
    )
    ctx.session.install_game(new_game)
    if new_game.current_node_id:
        ctx.session.invalidate_speculation_from(new_game.current_node_id)
    await ctx.session.commit()

    _log.info(
        "global retcon applied: rewritten_beats=%d character_updates=%d",
        rewritten,
        char_updates,
    )
    return rewritten, char_updates


async def edit_history_retcon_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    """Apply a global player retcon. Thin wrapper over
    ``apply_global_retcon`` — kept for backward compatibility with
    the c2s/edit/history/retcon message; the safety path uses the
    helper directly."""
    assert isinstance(payload, C2SEditHistoryRetcon)
    await apply_global_retcon(ctx, payload.instructions, push_undo=True)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_state_full, _full_state_message(ctx.session))

    return gen()


async def edit_environment_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SEditEnvironment)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    envs = dict(ctx.session.game.environments)
    env = envs.get(payload.environment_id)
    if env is None:
        raise NotFoundError(f"unknown environment {payload.environment_id}")
    envs[env.id] = _validated_update(
        env,
        payload.field,
        payload.value,
        _EDITABLE_ENVIRONMENT_FIELDS,
        "edit_environment",
    )
    new_game = ctx.session.game.model_copy(update={"environments": envs})
    ctx.session.install_game(new_game)
    await ctx.session.commit()

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path=f"/environments/{env.id}/{payload.field}",
                        value=payload.value,
                    )
                ]
            ),
        )

    return gen()


async def edit_environment_rerender_handler(
    payload: BaseModel, ctx: HandlerContext
) -> HandlerResult:
    """Re-run the background pipeline for one environment, BUMPING
    the seed so the same prompt yields a different draw. Counterpart
    to ``edit_character_rerender_handler`` for scenes.

    Why bump the seed by default (unlike character rerender, which
    pins the seed): an environment doesn't have an "identity" the
    way a character does — players hit Rerender on a scene because
    the current draw doesn't match what they imagined, and they
    want a different sample. To pin a specific seed (e.g. they
    found one they like), they edit the seed field directly via
    ``c2s/edit/environment``.

    Idempotency note: bumping the seed changes
    ``_background_prompt_hash`` (which folds the seed in), so the
    cached PNG filename changes and the dedup check at
    ``ensure_backgrounds_for`` regenerates instead of reusing the
    stale file. The previous PNG stays on disk — content-addressed,
    so future renders that happen to land on that hash again
    short-circuit.
    """
    from ..orchestration.session import random_seed

    assert isinstance(payload, C2SEditEnvironmentRerender)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    envs = dict(ctx.session.game.environments)
    env = envs.get(payload.environment_id)
    if env is None:
        raise NotFoundError(f"unknown environment {payload.environment_id}")

    # Bump the seed (and clear the cached image) so the env reads as
    # stale to the redraw scheduler, which then renders the fresh draw.
    envs[env.id] = env.model_copy(update={"seed": random_seed(), "image_path": None})
    new_game = ctx.session.game.model_copy(update={"environments": envs})
    ctx.session.install_game(new_game)
    await ctx.session.commit()

    # Explicit player request: the +1000 boost jumps it ahead of turn
    # staleness, and an explicit background renders/announces even when
    # it isn't the active scene (the player is editing it directly). The
    # new draw arrives over the emit channel when it lands.
    ctx.session.render_scheduler.request_redraw(env.id)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_state_full, _full_state_message(ctx.session))

    return gen()


async def edit_environment_apply_handler(
    payload: BaseModel,
    ctx: HandlerContext,
) -> HandlerResult:
    """Swap the current scene's backdrop to a different environment.

    Mutates the CURRENT dialog node's ``location_id`` so the
    renderer's location-walk picks up the chosen environment's
    pre-rendered image. Doesn't touch any other node — past
    beats keep their original backdrops, future beats will pick
    a backdrop the same way they always did (LLM-narrated
    location_id on each new beat).

    Errors:
      * NotFoundError if no game loaded, the env id doesn't
        exist, or there's no current node to mutate.
      * No-op (still returns a state-patch echo so the renderer
        can sync) when the chosen env is already the current
        node's location_id — the player clicked Apply on the
        already-active backdrop, which is harmless.
    """
    assert isinstance(payload, C2SEditEnvironmentApply)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    game = ctx.session.game
    env = game.environments.get(payload.environment_id)
    if env is None:
        raise NotFoundError(f"unknown environment {payload.environment_id}")
    if not game.current_node_id:
        raise NotFoundError("no current node to apply backdrop against")
    nodes = dict(game.dialog_tree.nodes)
    current = nodes.get(game.current_node_id)
    if current is None:
        raise NotFoundError(f"current_node_id {game.current_node_id} missing from dialog tree")
    if current.location_id == env.id:
        # Already on this backdrop. Echo state for renderer
        # consistency and return — nothing to mutate.
        async def _noop() -> AsyncIterator[OutboundMessage]:
            yield (
                MessageType.s2c_state_patch,
                S2CStatePatch(
                    ops=[
                        PatchOp(
                            op="replace",
                            path=f"/dialog_tree/nodes/{current.id}/location_id",
                            value=env.id,
                        ),
                    ]
                ),
            )

        return _noop()

    nodes[current.id] = current.model_copy(update={"location_id": env.id})
    new_tree = game.dialog_tree.model_copy(update={"nodes": nodes})
    new_game = game.model_copy(update={"dialog_tree": new_tree})
    ctx.session.install_game(new_game)
    await ctx.session.commit()

    _log.info(
        "env apply: node=%s location_id %s -> %s",
        current.id,
        current.location_id,
        env.id,
    )

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace",
                        path=f"/dialog_tree/nodes/{current.id}/location_id",
                        value=env.id,
                    ),
                ]
            ),
        )

    return gen()


# ---------- settings ----------


async def settings_get_handler(_payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    settings = ctx.session.settings

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(op="replace", path="/settings", value=settings.model_dump(mode="json"))
                ]
            ),
        )

    return gen()


async def settings_update_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    assert isinstance(payload, C2SSettingsUpdate)
    from ..domain.settings import Settings
    from ..persistence import settings_store

    old_profile = ctx.session.settings.user_profile
    # Python-mode dump: ``llm.api_key`` stays a live ``SecretStr`` here,
    # so the read-modify-write below preserves the real key instead of
    # writing back the masked value the renderer was shown.
    merged = ctx.session.settings.model_dump()
    merged_recursive(merged, payload.patch)

    # Write-only secret. The renderer never receives the real key (the
    # JSON serializer masks it to ""), so every settings autosave echoes
    # that empty string back at us. Treating it as "clear the key" would
    # wipe the player's credentials the first time they nudged an
    # unrelated slider — so an empty/blank ``api_key`` means "leave it
    # alone". A genuinely new key is a non-blank string and lands
    # normally.
    llm_patch = payload.patch.get("llm")
    if isinstance(llm_patch, dict) and "api_key" in llm_patch:
        supplied = llm_patch["api_key"]
        if not isinstance(supplied, str) or not supplied.strip():
            merged.setdefault("llm", {})["api_key"] = ctx.session.settings.llm.api_key

    # If the player's update DROPPED entries from any of the
    # ``summarizer_*`` buckets (i.e., they hit the X next to a
    # tag in the "Inferred by the engine" section), record those
    # entries in the corresponding ``dismissed_*`` list. The
    # summarizer's merge logic then refuses to re-add anything
    # that's in dismissed, so deletions are sticky across future
    # consolidation passes. Without this, the next time the LLM
    # noticed the same pattern the deleted entry would silently
    # come back and the player's edits would feel ignored.
    new_profile_dict = merged.setdefault("user_profile", {})
    for bucket, dismissed_key in (
        ("summarizer_likes", "dismissed_likes"),
        ("summarizer_dislikes", "dismissed_dislikes"),
        ("summarizer_notes", "dismissed_notes"),
    ):
        new_bucket = new_profile_dict.get(bucket)
        if not isinstance(new_bucket, list):
            continue
        old_bucket = list(getattr(old_profile, bucket, []))
        new_set = {str(t).strip().casefold() for t in new_bucket if str(t).strip()}
        removed = [
            t for t in old_bucket if str(t).strip() and str(t).strip().casefold() not in new_set
        ]
        if not removed:
            continue
        existing_dismissed = new_profile_dict.get(dismissed_key)
        if not isinstance(existing_dismissed, list):
            existing_dismissed = list(getattr(old_profile, dismissed_key, []))
        seen = {str(t).strip().casefold() for t in existing_dismissed if str(t).strip()}
        for tag in removed:
            key = tag.strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            existing_dismissed.append(tag.strip())
        new_profile_dict[dismissed_key] = existing_dismissed

    new_settings = Settings.model_validate(merged)
    ctx.session.settings = new_settings
    settings_store.save_settings(new_settings)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_state_patch,
            S2CStatePatch(
                ops=[
                    PatchOp(
                        op="replace", path="/settings", value=new_settings.model_dump(mode="json")
                    )
                ]
            ),
        )

    return gen()


async def settings_validate_api_key_handler(
    payload: BaseModel,
    ctx: HandlerContext,
) -> HandlerResult:
    """Probe ``GET {base_url}/v1/models`` with the supplied key to tell
    the player immediately whether the key authenticates.

    The Settings + FirstTimeSetup inputs fire this on blur so a typo'd
    key surfaces with an inline red badge instead of waiting for the
    first real prompt round-trip to die mid-game with the generic
    "LLM provider failed" banner. ``/v1/models`` is the
    OpenAI-compatible "is this key alive" probe — cheap, no token
    cost on every endpoint the engine talks to, and uniformly
    authenticated via the Bearer header the engine already uses.

    No state is mutated here — validation is independent of the
    autosaving settings store. The caller passes the candidate
    base_url + key explicitly so they can probe a value BEFORE
    committing it.
    """
    import httpx

    from ..api.messages import C2SSettingsValidateApiKey, S2CSettingsApiKeyValidation

    assert isinstance(payload, C2SSettingsValidateApiKey)

    base_url = payload.base_url.strip().rstrip("/")
    api_key = payload.api_key.strip()

    def _reply(ok: bool, status: str, message: str) -> AsyncIterator[OutboundMessage]:
        async def gen() -> AsyncIterator[OutboundMessage]:
            yield (
                MessageType.s2c_settings_api_key_validation,
                S2CSettingsApiKeyValidation(ok=ok, status=status, message=message),
            )

        return gen()

    if not api_key:
        return _reply(False, "invalid_input", "API key is empty.")
    if not base_url:
        return _reply(False, "invalid_input", "Base URL is empty.")

    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    # Short timeout: the player is staring at the input waiting for the
    # badge to land. A slow provider just gets reported as "unreachable"
    # — we'd rather say so quickly than block the UI for 30 s on a
    # mistyped URL pointing at a non-server.
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return _reply(False, "unreachable", f"Could not reach {url}: {exc}")

    if 200 <= response.status_code < 300:
        return _reply(True, "valid", "API key is valid.")
    if response.status_code in (401, 403):
        return _reply(False, "unauthorized", "API key was rejected by the provider.")
    return _reply(
        False,
        "unreachable",
        f"Unexpected response from {url}: HTTP {response.status_code}.",
    )


def merged_recursive(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merged_recursive(target[key], value)
        else:
            target[key] = value


# ---------- embedded image backend ----------


def _embedded_models_dir(requested: str, ctx: HandlerContext) -> Path:
    """Resolve the models directory for an ``c2s/embedded/*`` request.

    The CONFIGURED directory (``ImageSettings.embedded_models_dir``, or
    the bundled default) is the root of trust. A ``models_dir`` on the
    request may narrow to a subdirectory of it — that's what lets the
    Settings screen probe a path the player is editing before committing
    it — but it can no longer REPLACE it. Before this, the wire value
    took precedence outright, and ``c2s/embedded/download_model`` would
    happily ``mkdir`` anywhere the process could write and stream
    gigabytes into it.
    """
    from ..providers.embedded_models import (
        ModelsDirOutsideRootError,
        resolve_models_dir,
    )

    configured = resolve_models_dir(ctx.session.settings.image.embedded_models_dir)
    if not (requested or "").strip():
        return configured
    try:
        return resolve_models_dir(requested, allowed_root=configured)
    except ModelsDirOutsideRootError as exc:
        raise SchemaError(str(exc)) from exc


async def embedded_list_models_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    """List safetensors / ckpt files in the embedded backend's
    models directory. Renderer uses the result to populate the
    Settings dropdown so the user can pick which checkpoint runs."""
    from ..api.messages import C2SEmbeddedListModels, S2CEmbeddedModels
    from ..providers.embedded_models import list_models

    assert isinstance(payload, C2SEmbeddedListModels)
    # Honour the request's ``models_dir`` so the Settings screen can
    # probe a path the user is editing live BEFORE committing it — but
    # only within the configured root (see ``_embedded_models_dir``).
    resolved = _embedded_models_dir(payload.models_dir, ctx)
    files = list_models(resolved)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_embedded_models,
            S2CEmbeddedModels(models_dir=str(resolved), models=files),
        )

    return gen()


async def embedded_recommend_model_handler(
    payload: BaseModel,
    ctx: HandlerContext,
) -> HandlerResult:
    """Report which base model the one-click download would fetch for
    this machine. Cheap (a GPU vendor probe + a best-effort VRAM read +
    a directory listing), so we compute it synchronously and emit one
    ``s2c/embedded/recommended_model``. The wizard uses ``display_name``
    for its ``<highlight>`` and ``key`` for the download button; it hides
    the offer entirely when ``has_models`` is already True."""
    from ..api.messages import C2SEmbeddedRecommendModel, S2CEmbeddedRecommendedModel
    from ..providers.embedded_models import (
        list_models,
        recommend_model,
    )

    assert isinstance(payload, C2SEmbeddedRecommendModel)
    resolved = _embedded_models_dir(payload.models_dir, ctx)
    has_models = len(list_models(resolved)) > 0
    spec = recommend_model()
    _log.info(
        "embedded recommend_model: %s (has_models=%s, dir=%s)",
        spec.key,
        has_models,
        resolved,
    )

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_embedded_recommended_model,
            S2CEmbeddedRecommendedModel(
                key=spec.key,
                display_name=spec.display_name,
                reason=(
                    f"Recommended for your hardware; downloads {spec.display_name} "
                    f"(~{spec.total_approx_bytes / 1e9:.1f} GB) from {spec.hf_repo}."
                ),
                models_dir=str(resolved),
                has_models=has_models,
                # The TOTAL, not just the checkpoint: the text encoder and
                # VAE this family fetches at first render are part of what
                # the player is agreeing to download.
                approx_bytes=spec.total_approx_bytes,
            ),
        )

    return gen()


async def embedded_download_model_handler(
    payload: BaseModel,
    ctx: HandlerContext,
) -> HandlerResult:
    """Download a base checkpoint into the embedded models directory,
    streaming byte progress over the websocket, then emit the refreshed
    model inventory so the dropdown updates in place.

    Mirrors ``torch_overlay_install_handler``: the blocking, network-heavy
    download runs on a worker thread and pushes ``(bytes_done, total)``
    onto an ``asyncio.Queue`` via a thread-safe callback; this coroutine
    drains the queue, yields ``s2c/embedded/download_progress`` frames as
    they arrive, then a terminal ``s2c/embedded/models`` on success or
    ``s2c/error`` on failure."""
    import asyncio as _asyncio

    from ..api.errors import LucidiumError, ProviderUnreachableError
    from ..api.messages import (
        C2SEmbeddedDownloadModel,
        S2CEmbeddedDownloadProgress,
        S2CEmbeddedModels,
    )
    from ..providers.embedded_models import (
        MODEL_CATALOG,
        download_model,
        list_models,
        recommend_model,
    )

    assert isinstance(payload, C2SEmbeddedDownloadModel)
    resolved = _embedded_models_dir(payload.models_dir, ctx)
    spec = MODEL_CATALOG.get(payload.key) if payload.key else None
    if spec is None:
        # Empty / unknown key falls back to the hardware recommendation so
        # a stale renderer key can't wedge the download.
        spec = recommend_model()
    _log.info("embedded download_model: key=%s -> %s into %s", payload.key, spec.key, resolved)

    loop = _asyncio.get_running_loop()
    queue: _asyncio.Queue[Any] = _asyncio.Queue(maxsize=256)

    def _offer(item: Any) -> None:
        try:
            queue.put_nowait(item)
        except _asyncio.QueueFull:
            pass  # Advisory progress frame; safe to drop under backpressure.

    def _on_progress(done: int, total: int | None) -> None:
        # Called FROM the worker thread; hop to the loop to touch the
        # (non-thread-safe) queue.
        loop.call_soon_threadsafe(_offer, ("progress", done, total))

    def _run_download() -> None:
        try:
            download_model(spec, resolved, on_progress=_on_progress)
            loop.call_soon_threadsafe(_offer, ("done", None))
        except BaseException as exc:
            loop.call_soon_threadsafe(_offer, ("error", exc))
            raise

    task = _asyncio.create_task(_asyncio.to_thread(_run_download))

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_embedded_download_progress,
            S2CEmbeddedDownloadProgress(
                key=spec.key, display_name=spec.display_name, stage="resolving"
            ),
        )
        error: BaseException | None = None
        while True:
            item = await queue.get()
            kind = item[0]
            if kind == "progress":
                _, done, total = item
                yield (
                    MessageType.s2c_embedded_download_progress,
                    S2CEmbeddedDownloadProgress(
                        key=spec.key,
                        display_name=spec.display_name,
                        stage="downloading",
                        bytes_done=done,
                        bytes_total=total,
                    ),
                )
            elif kind == "done":
                break
            elif kind == "error":
                error = item[1]
                break
        try:
            await task
        except BaseException:
            pass

        if error is not None:
            # Re-raise so the dispatcher emits an s2c/error. A LucidiumError
            # already carries its own code; anything else (ModelDownloadError,
            # a raw OSError) becomes a provider-unreachable so the wizard can
            # tell the player to retry or fall back to the manual route.
            if isinstance(error, LucidiumError):
                raise error
            raise ProviderUnreachableError(
                f"download of {spec.display_name} failed: {error}"
            ) from error

        # Success: re-list so the renderer's dropdown picks up the new
        # file (and the wizard can auto-select it).
        yield (
            MessageType.s2c_embedded_download_progress,
            S2CEmbeddedDownloadProgress(
                key=spec.key, display_name=spec.display_name, stage="saving"
            ),
        )
        yield (
            MessageType.s2c_embedded_models,
            S2CEmbeddedModels(models_dir=str(resolved), models=list_models(resolved)),
        )

    return gen()


# ---------- torch runtime overlay ----------


async def torch_overlay_status_handler(
    payload: BaseModel,
    ctx: HandlerContext,
) -> HandlerResult:
    """Report the recommended torch overlay flavor for this machine,
    which flavors are installed, and which is active.

    Network-free + cheap (it's a GPU probe + a couple of directory
    listings), so we just compute the snapshot synchronously and emit
    one ``s2c/torch_overlay/status``. The first-run / Settings screen
    calls this to decide whether to offer a torch download and which
    flavor button to highlight."""
    from ..api.messages import C2STorchOverlayStatus, S2CTorchOverlayStatus
    from ..providers import torch_overlay

    assert isinstance(payload, C2STorchOverlayStatus)
    snapshot = torch_overlay.status()
    _log.info("torch overlay status requested: %s", snapshot)

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_torch_overlay_status,
            S2CTorchOverlayStatus(
                recommended=str(snapshot["recommended"]),
                installed=list(cast("Sequence[str]", snapshot["installed"])),
                active=cast("str | None", snapshot["active"]),
                runtime_dir=str(snapshot["runtime_dir"]),
            ),
        )

    return gen()


async def torch_overlay_install_handler(
    payload: BaseModel,
    ctx: HandlerContext,
) -> HandlerResult:
    """Download + install (and by default activate) a torch overlay
    flavor, streaming progress over the websocket.

    Mirrors the SDXL "Download" handler's shape but adds incremental
    progress: the install runs on a worker thread (it's blocking,
    network + zip-heavy) and pushes ``(bytes_done, total)`` updates onto
    an ``asyncio.Queue`` from a thread-safe callback. This coroutine
    drains the queue and yields ``s2c/torch_overlay/progress`` events as
    they arrive, then a terminal ``s2c/torch_overlay/status`` on success
    or ``s2c/error`` on failure.

    The activation only flips the *next-launch* pointer — torch is
    already imported in this process (or excluded entirely in dev), so
    the new flavor can't load live; ``activated`` in the terminal status
    tells the UI to prompt a relaunch."""
    import asyncio as _asyncio

    from ..api.errors import LucidiumError, ProviderUnreachableError
    from ..api.messages import (
        C2STorchOverlayInstall,
        S2CTorchOverlayProgress,
        S2CTorchOverlayStatus,
    )
    from ..providers import torch_overlay

    assert isinstance(payload, C2STorchOverlayInstall)
    flavor = payload.flavor
    activate = payload.activate
    _log.info("torch overlay install requested: flavor=%s activate=%s", flavor, activate)

    loop = _asyncio.get_running_loop()
    # Bounded queue so a fast progress callback can't balloon memory if
    # the websocket drains slowly. ``None`` is the done sentinel; an
    # ``Exception`` instance signals failure to the consumer.
    queue: _asyncio.Queue[Any] = _asyncio.Queue(maxsize=256)

    def _on_progress(done: int, total: int | None) -> None:
        # Called FROM the worker thread. Hop back to the event loop to
        # touch the queue (asyncio.Queue is not thread-safe). Best-effort
        # via call_soon_threadsafe; dropping a frame on a full queue is
        # fine — progress is advisory.
        loop.call_soon_threadsafe(_offer, ("progress", done, total))

    def _offer(item: Any) -> None:
        try:
            queue.put_nowait(item)
        except _asyncio.QueueFull:
            pass  # Advisory progress frame; safe to drop under backpressure.

    def _run_install() -> torch_overlay.InstallResult:
        try:
            result = torch_overlay.install_flavor(
                flavor,
                on_progress=_on_progress,
                activate=activate,
            )
            loop.call_soon_threadsafe(_offer, ("done", result))
            return result
        except BaseException as exc:
            loop.call_soon_threadsafe(_offer, ("error", exc))
            raise

    task = _asyncio.create_task(_asyncio.to_thread(_run_install))

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_torch_overlay_progress,
            S2CTorchOverlayProgress(flavor=flavor, stage="resolving"),
        )
        result: torch_overlay.InstallResult | None = None
        error: BaseException | None = None
        while True:
            item = await queue.get()
            kind = item[0]
            if kind == "progress":
                _, done, total = item
                yield (
                    MessageType.s2c_torch_overlay_progress,
                    S2CTorchOverlayProgress(
                        flavor=flavor,
                        stage="downloading",
                        bytes_done=done,
                        bytes_total=total,
                    ),
                )
            elif kind == "done":
                result = item[1]
                break
            elif kind == "error":
                error = item[1]
                break
        # Let the worker task finish so its exception (if any) doesn't
        # surface as an un-retrieved-task warning.
        try:
            await task
        except BaseException:
            pass

        if error is not None:
            # Re-raise as a provider error the dispatcher turns into an
            # s2c/error, unless it's already a LucidiumError (carries its
            # own ErrorCode — e.g. ProviderValidationError for an unknown
            # flavor or no compatible wheel).
            if isinstance(error, LucidiumError):
                raise error
            raise ProviderUnreachableError(
                f"torch overlay install failed for flavor {flavor}: {error}"
            ) from error

        assert result is not None
        snapshot = torch_overlay.status()
        yield (
            MessageType.s2c_torch_overlay_progress,
            S2CTorchOverlayProgress(flavor=flavor, stage="installed"),
        )
        yield (
            MessageType.s2c_torch_overlay_status,
            S2CTorchOverlayStatus(
                recommended=str(snapshot["recommended"]),
                installed=list(cast("Sequence[str]", snapshot["installed"])),
                active=cast("str | None", snapshot["active"]),
                runtime_dir=str(snapshot["runtime_dir"]),
                activated=result.activated,
            ),
        )

    return gen()


# ---------- music inventory / connection probe ----------


async def music_inventory_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    """Probe an ACE-Step server for connectivity + available
    models. Powers the Settings screen's Test Connection button
    and music-model dropdown. Returns ``ok=false`` with a short
    error string instead of raising — the renderer surfaces the
    string verbatim and falls back to a free-text input so the
    player can still type a model name when the server isn't
    reachable yet."""
    from ..api.errors import ProviderUnreachableError, ProviderValidationError
    from ..api.messages import C2SMusicInventory, S2CMusicInventory
    from ..domain.settings import MusicSettings
    from ..providers.music_client import AceStepClient

    assert isinstance(payload, C2SMusicInventory)
    base_url = (payload.base_url or ctx.session.settings.music.base_url).strip()
    # Build an ad-hoc settings snapshot so the probe targets the
    # URL the user is editing live, without committing it yet.
    probe_settings = ctx.session.settings.music.model_copy(update={"base_url": base_url})
    if not isinstance(probe_settings, MusicSettings):  # pragma: no cover - defence
        probe_settings = MusicSettings(base_url=base_url)
    client = AceStepClient(probe_settings)
    error = ""
    models: list[str] = []
    try:
        body = await client.list_models()
    except ProviderUnreachableError as exc:
        error = f"server unreachable: {exc}"
    except ProviderValidationError as exc:
        error = f"server error: {exc}"
    except Exception as exc:
        error = f"unexpected error: {exc}"
    else:
        models = _flatten_music_inventory(body)
    finally:
        await client.aclose()

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (
            MessageType.s2c_music_inventory,
            S2CMusicInventory(
                base_url=base_url,
                ok=not error,
                models=models,
                error=error,
            ),
        )

    return gen()


async def music_regenerate_handler(payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    """Re-render the live game's background music. The Story panel's
    Music tab fires this when the player clicks Regenerate. The
    actual render runs in the background via ``_spawn_async_music``;
    a state/full echo lands when the new path is committed.

    A non-empty ``payload.prompt`` swaps the stored
    ``world.music_prompt`` BEFORE kicking the task — the new prompt
    feeds both the render and the prompt-hash cache key, so a
    duplicate regenerate request after a prompt change re-runs the
    pipeline instead of returning the stale cached track.
    """
    from ..api.messages import C2SMusicRegenerate

    assert isinstance(payload, C2SMusicRegenerate)
    if ctx.session.game is None:
        raise NotFoundError("no game loaded")
    if not ctx.session.settings.music.enabled:
        raise SchemaError("music generation is disabled in settings")
    new_prompt = payload.prompt.strip()
    extra_ops: list[PatchOp] = []
    if new_prompt and new_prompt != ctx.session.game.world.music_prompt:
        new_world = ctx.session.game.world.model_copy(
            update={"music_prompt": new_prompt},
        )
        ctx.session.install_game(
            ctx.session.game.model_copy(update={"world": new_world}),
        )
        await ctx.session.commit()
        extra_ops.append(
            PatchOp(
                op="replace",
                path="/world/music_prompt",
                value=new_prompt,
            )
        )
    prompt = (
        new_prompt
        or (ctx.session.game.world.music_prompt or "").strip()
        or _fallback_music_prompt(ctx.session.game.world)
    )
    if not prompt:
        raise SchemaError("no music prompt available; supply one in payload.prompt")
    _spawn_async_music(ctx.session, prompt)

    async def gen() -> AsyncIterator[OutboundMessage]:
        if extra_ops:
            yield (MessageType.s2c_state_patch, S2CStatePatch(ops=extra_ops))

    return gen()


def _flatten_music_inventory(body: object) -> list[str]:
    """Reduce ACE-Step's ``/v1/model_inventory`` payload to a flat
    list of DiT model IDs the renderer can drop straight into a
    dropdown.

    The upstream shape (server v1.0) is::

        {
          "data": {
            "models": [
              {"name": "...", "is_default": ..., "is_loaded": ...,
               "supported_task_types": [...]},
              ...
            ],
            "default_model": "...",
            "lm_models": [{"name": "...", "is_loaded": ...}, ...],
            "loaded_lm_model": "...",
            "llm_initialized": ...
          },
          "code": 200,
          ...
        }

    We pick out only the DiT entries — the LM list is irrelevant
    because the engine drives prompts through OpenRouter and
    asks the server for ``init_llm=False``. Order: default
    first, then the rest as the server returned them, deduped.
    """
    found: list[str] = []
    seen: set[str] = set()

    def _add(name: object, *, front: bool = False) -> None:
        if not isinstance(name, str) or not name or name in seen:
            return
        if front:
            found.insert(0, name)
        else:
            found.append(name)
        seen.add(name)

    if isinstance(body, dict):
        data = body.get("data", body)
        if isinstance(data, dict):
            for entry in data.get("models", []) or []:
                if isinstance(entry, dict):
                    _add(entry.get("name"))
                elif isinstance(entry, str):
                    _add(entry)
            default = data.get("default_model")
            if isinstance(default, str) and default:
                # Pull the default to the head of the list. We
                # don't add a NEW entry — only re-order if it was
                # already in models[].
                if default in seen:
                    found.remove(default)
                    found.insert(0, default)
                else:
                    _add(default, front=True)
            if found:
                return found

    # Fallback: tolerate unexpected shapes by walking for any
    # ``name`` field we can find. Skips the LM lists explicitly.
    def _walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in (
                    "lm_models",
                    "loaded_lm_model",
                    "lm",
                    "language_models",
                ):
                    continue
                if key == "name" and isinstance(value, str):
                    _add(value)
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(body)
    return found


# ---------- app exit ----------


async def app_exit_handler(_payload: BaseModel, ctx: HandlerContext) -> HandlerResult:
    await ctx.session.commit()

    async def gen() -> AsyncIterator[OutboundMessage]:
        yield (MessageType.s2c_state_patch, S2CStatePatch(ops=[]))
        return

    return gen()


# ---------- registry ----------


def build_default_registry() -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.register(MessageType.c2s_hello, hello_handler)
    registry.register(MessageType.c2s_saves_list, saves_list_handler)
    registry.register(MessageType.c2s_saves_continue, saves_continue_handler)
    registry.register(MessageType.c2s_saves_load, saves_load_handler)
    registry.register(MessageType.c2s_saves_rename, saves_rename_handler)
    registry.register(MessageType.c2s_saves_delete, saves_delete_handler)
    registry.register(MessageType.c2s_new_game_start, new_game_start_handler)
    registry.register(MessageType.c2s_new_game_answer, new_game_answer_handler)
    registry.register(
        MessageType.c2s_new_game_add_side_character, new_game_add_side_character_handler
    )
    registry.register(
        MessageType.c2s_new_game_edit_side_character,
        new_game_edit_side_character_handler,
    )
    registry.register(
        MessageType.c2s_new_game_delete_side_character,
        new_game_delete_side_character_handler,
    )
    registry.register(MessageType.c2s_new_game_edit_review, new_game_edit_review_handler)
    registry.register(MessageType.c2s_new_game_confirm, new_game_confirm_handler)
    registry.register(MessageType.c2s_new_game_go_back, new_game_go_back_handler)
    registry.register(MessageType.c2s_new_game_surprise_me, new_game_surprise_me_handler)
    registry.register(MessageType.c2s_play_advance, play_advance_handler)
    registry.register(MessageType.c2s_play_free_text, play_free_text_handler)
    registry.register(MessageType.c2s_play_undo, play_undo_handler)
    registry.register(MessageType.c2s_play_cancel, play_cancel_handler)
    registry.register(MessageType.c2s_edit_world, edit_world_handler)
    registry.register(MessageType.c2s_edit_character, edit_character_handler)
    registry.register(MessageType.c2s_edit_character_dismiss, edit_character_dismiss_handler)
    registry.register(MessageType.c2s_edit_character_show, edit_character_show_handler)
    registry.register(MessageType.c2s_edit_character_rerender, edit_character_rerender_handler)
    registry.register(MessageType.c2s_edit_history, edit_history_handler)
    registry.register(MessageType.c2s_edit_history_delete, edit_history_delete_handler)
    registry.register(MessageType.c2s_edit_history_retcon, edit_history_retcon_handler)
    registry.register(MessageType.c2s_edit_environment, edit_environment_handler)
    registry.register(
        MessageType.c2s_edit_environment_rerender,
        edit_environment_rerender_handler,
    )
    registry.register(
        MessageType.c2s_edit_environment_apply,
        edit_environment_apply_handler,
    )
    registry.register(MessageType.c2s_settings_get, settings_get_handler)
    registry.register(MessageType.c2s_settings_update, settings_update_handler)
    registry.register(
        MessageType.c2s_settings_validate_api_key,
        settings_validate_api_key_handler,
    )
    registry.register(MessageType.c2s_embedded_list_models, embedded_list_models_handler)
    registry.register(MessageType.c2s_embedded_recommend_model, embedded_recommend_model_handler)
    registry.register(MessageType.c2s_embedded_download_model, embedded_download_model_handler)
    registry.register(MessageType.c2s_torch_overlay_status, torch_overlay_status_handler)
    registry.register(MessageType.c2s_torch_overlay_install, torch_overlay_install_handler)
    registry.register(MessageType.c2s_music_inventory, music_inventory_handler)
    registry.register(MessageType.c2s_music_regenerate, music_regenerate_handler)
    registry.register(MessageType.c2s_app_exit, app_exit_handler)
    return registry


__all__ = [
    "Handler",
    "HandlerContext",
    "HandlerRegistry",
    "build_default_registry",
]

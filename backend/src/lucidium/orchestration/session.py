"""Session aggregate.

Owns the live ``Game`` (or absence of one), the render scheduler, and the
provider clients. Handlers reach into the session through a typed
``HandlerContext``; the ``WsServer`` constructs one session per
connection (and there is at most one connection in production).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..api.errors import ProviderValidationError
from ..config import (
    LLM_RETRY_BUDGET,
    saves_dir,
)
from ..domain.character import Character, CharacterImage
from ..domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
    premise_hash,
    world_snapshot_vector,
)
from ..domain.environment import Environment
from ..domain.game import Game, with_speaker_on_stage
from ..domain.settings import Settings
from ..domain.world import WorldState
from ..persistence import save_store, settings_store
from ..providers.image_client import (
    ImageClient,
    RecordedImageClient,
)
from ..providers.llm_client import LlmClient, RecordedLlmClient
from . import obsolescence
from .latency_audit import LatencyAudit

if TYPE_CHECKING:  # pragma: no cover -- imports for type checkers only
    from ..api.messages import MessageType
    from ..domain.settings import ImageSettings
    from ..providers.music_client import AceStepClient

_log = logging.getLogger(__name__)

OutboundEmit = Callable[["MessageType", BaseModel], None]


class InterviewState(BaseModel):
    """Transient state during the New Game interview."""

    setting_options: list[str] = []
    genre_options: list[str] = []
    visual_style_options: list[str] = []
    character_description_options: list[str] = []
    name_options: list[str] = []
    setting: str = ""
    genre: str = ""
    visual_style: str = ""
    character_description: str = ""
    character_name: str = ""
    # Free-text pronouns the player picked on the Identity step
    # (e.g. ``she/her``, ``they/them``, or a custom string).
    # Threaded into the player Character at confirm time so the
    # storyteller addresses the player correctly from beat 1.
    # Empty means "fall back to whatever pronouns the world_init
    # LLM emitted on its player_character descriptor."
    pronouns: str = ""
    side_character_descriptions: list[str] = []
    side_characters: list[Character] = []
    # FR-022: bundled "dream guide" + "white room" presented from the
    # first frame of the interview. Re-rendered in the chosen visual
    # style after the player answers the Visual Style step (FR-027a).
    dream_guide_image_path: str = ""
    white_room_image_path: str = ""
    # The menu pair the player saw at the moment they clicked New
    # Game. Sent by the renderer in c2s/new_game/start; the backend
    # re-renders the same character body+face in the player-chosen
    # visual style so the dream guide stays consistent across the
    # entire onboarding flow.
    preview_character_slug: str = ""
    # File paths populated by the post-visual-style preview render
    # task. Pushed back to the renderer via state/patch so the
    # MenuCarousel can swap from the cycling menu pairs to the
    # player's chosen-style imagery.
    preview_background_path: str = ""
    preview_guide_path: str = ""
    # Player-character portrait. Rendered after the player picks
    # their character description; the renderer replaces the dream
    # guide with this image once it arrives so the player sees
    # themselves by the time they reach Review.
    pc_portrait_path: str = ""


def random_seed() -> int:
    return secrets.randbits(64)


class Session:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm_client: LlmClient | None = None,
        image_client: ImageClient | None = None,
        saves_root: Path | None = None,
        emit: OutboundEmit | None = None,
        workflow_root: Path | None = None,
    ) -> None:
        self.settings = settings or settings_store.load_settings()
        self.saves_root = saves_root or saves_dir()
        self.emit = emit
        self.game: Game | None = None
        self.interview = InterviewState()
        # Shared lock between the local SDXL image client and the
        # ACE-Step music client. When BOTH backends run on the
        # same physical GPU, holding this lock during inference
        # serialises them so they don't both compete for VRAM at
        # the same time. Music acquires + evicts SDXL to CPU
        # before its HTTP call; image acquires + restores SDXL
        # to GPU before its diffusers call. Lazy-built so the
        # asyncio.Lock is bound to whichever event loop is live
        # when the first client asks for it (Sessions in the
        # tests get constructed without a loop).
        import asyncio as _asyncio

        self._gpu_lock: _asyncio.Lock | None = None
        # Play-handler serialization lock. ``c2s/play/advance``,
        # ``c2s/play/free_text``, and ``c2s/play/undo`` mutate the
        # dialog tree and current_node_id; running two of them
        # concurrently produced a "skip backwards" symptom where
        # play_advance's awaited speculation landed AFTER free_text
        # had already moved current_node_id forward. Serialising the
        # mutating play handlers is cheap (the player is only ever
        # in one of these states at a time on the renderer) and
        # eliminates the entire class of intra-handler races. Bound
        # lazily for the same reason as ``_gpu_lock`` — tests build
        # Session without a live event loop.
        self._play_lock: _asyncio.Lock | None = None
        # Serialises ``commit()``, whose save_store write now runs on a
        # worker thread and so can genuinely overlap itself. Lazily
        # bound for the same reason as the two locks above.
        self._commit_lock: _asyncio.Lock | None = None
        # Undo stack — every successful play_advance / free_text turn
        # pushes the pre-advance Game snapshot here. ``c2s/play/undo``
        # pops the top entry and reinstates that Game, undoing
        # everything the popped turn changed (including any
        # ``Character.removed`` flags that turn set). Bounded length
        # so a long session doesn't grow unbounded; old entries fall
        # off the bottom.
        self.undo_stack: list[Game] = []
        # Tracks the in-flight foreground LLM stream's drain task.
        # ``_generate_and_walk_chain`` emits beat 1 to the player,
        # spawns a background task to drain the rest of the chain
        # into the dialog tree, and stores that task here. Subsequent
        # play_advance handlers can then walk into already-installed
        # beats (or briefly wait on this task for the NEXT beat to
        # land) instead of blocking on the play_lock for the full
        # chain duration. Free-text invalidation cancels it.
        self._foreground_stream_task: _asyncio.Task[Any] | None = None
        # The dispatch task currently running a FOREGROUND play
        # generation (``c2s/play/advance`` / ``c2s/play/free_text``).
        # ``c2s/play/cancel`` cancels it; see ``cancel_foreground``.
        self._foreground_task: _asyncio.Task[Any] | None = None

        # ---- task surface ---------------------------------------------------
        # Everything below used to be grafted onto the instance at
        # runtime by handlers.py / render_scheduler.py via
        # ``getattr(session, "_x", None) or []``. Declaring them here
        # gives ``aclose`` a single, honest inventory of what the
        # session owns — an attribute that only exists once some
        # handler happens to run cannot be reliably torn down.
        #
        # Speculative branch generation, keyed by the node id being
        # speculated (``_ensure_speculative_branches``).
        self._speculative_tasks: dict[str, _asyncio.Task[Any]] = {}
        # Background summarizer passes (``_spawn_async_summarizer``).
        self._summarizer_tasks: list[_asyncio.Task[Any]] = []
        # Background music generation (``_spawn_async_music``).
        self._music_tasks: list[_asyncio.Task[Any]] = []
        # Render-scheduler pump tasks (``RenderScheduler._ensure_pump``).
        self._asset_tasks: list[_asyncio.Task[Any]] = []
        # Interview-time prefetch tasks. Each is cancelled + reset to
        # None by ``_cancel_pending_prefetch`` / new_game_go_back.
        self._world_init_task: _asyncio.Task[Any] | None = None
        self._char_desc_task: _asyncio.Task[Any] | None = None
        self._name_options_task: _asyncio.Task[Any] | None = None
        self._preview_bg_task: _asyncio.Task[Any] | None = None
        self._preview_guide_task: _asyncio.Task[Any] | None = None
        self._pc_portrait_task: _asyncio.Task[Any] | None = None

        # ---- provider client caches -----------------------------------------
        # Also previously grafted on demand; ``aclose`` needs to know
        # whether a client was ever built so it can await its aclose().
        self._llm_client_cache: LlmClient | None = None
        self._image_client_cache: ImageClient | None = None
        self._image_settings_fp: tuple[Any, ...] | None = None
        self._music_client_cache: AceStepClient | None = None
        self._music_client_fp: tuple[Any, ...] | None = None
        self._closed = False

        # Provider factories let settings updates rebind without restart.
        self._llm_override = llm_client
        self._image_override = image_client
        self._workflow_root = workflow_root or _default_workflow_root()

        # Latency-audit recorder. Disabled by default; the integration
        # test + ``scripts/audit_turn_latency.py`` flip it on so the
        # handler instrumentation actually appends events. In normal
        # operation every record() is a single-attribute is-enabled
        # branch and returns immediately.
        self.latency = LatencyAudit()

        # Per-entity redraw scheduler: scores every on-screen character +
        # the active background by how long they've been stale (plus a
        # big boost on an explicit Rerender) and renders the highest
        # scorer first, one at a time. Replaces the old fan-out of
        # per-turn asset tasks; see ``render_scheduler``.
        from .render_scheduler import RenderScheduler

        self.render_scheduler = RenderScheduler(self)

    # ---- provider factories -------------------------------------------------

    def _llm_factory(self) -> LlmClient:
        if self._llm_override is not None:
            return self._llm_override
        from ..config import is_offline

        if is_offline():
            return RecordedLlmClient(_default_fixture_root("llm"))
        from ..providers.llm_client import OpenAiCompatibleLlmClient

        client = OpenAiCompatibleLlmClient(self.settings.llm)
        # Held only so ``aclose`` can shut the httpx pool down on
        # teardown. Deliberately NOT a reuse cache — rebuilding per
        # call is what picks up a settings edit mid-session.
        self._llm_client_cache = client
        return client

    def _image_factory(self) -> ImageClient:
        if self._image_override is not None:
            return self._image_override
        from ..config import is_offline

        if is_offline():
            return RecordedImageClient(_default_fixture_root("image"))
        from ..domain.settings import ImageBackend

        fingerprint = _image_settings_fingerprint(self.settings.image)
        cached_fp: tuple[Any, ...] | None = getattr(self, "_image_settings_fp", None)
        cached_client: ImageClient | None = getattr(self, "_image_client_cache", None)

        if cached_client is not None and cached_fp == fingerprint:
            # Cheap mutable settings (face_detail toggle) flow through
            # without rebuilding the client — a rebuild would drop
            # every loaded pipeline and re-pay the multi-second
            # checkpoint reload cost just to flip a bool.
            self._sync_mutable_image_settings(cached_client)
            return cached_client

        # Settings changed (or no client cached). Tear down the
        # previous client BEFORE building the new one so a backend
        # swap or model change actually frees its VRAM-loaded
        # pipelines instead of leaving them pinned and blocking the
        # new render. Without this teardown the user's reported
        # symptom — "changing settings blocks renders until they
        # are reverted" — happens because the new EmbeddedImageClient
        # tries to load its checkpoint while the OLD one still has
        # its pipeline parked in CUDA memory.
        previous = cached_client
        if previous is not None:
            self._dispose_image_client(previous)

        if self.settings.image.backend == ImageBackend.embedded:
            from ..providers.embedded_image_client import EmbeddedImageClient

            client: ImageClient = EmbeddedImageClient(
                models_dir=self.settings.image.embedded_models_dir,
                model_name=self.settings.image.embedded_model_name,
                character_model_name=self.settings.image.embedded_character_model_name,
                environment_model_name=self.settings.image.embedded_environment_model_name,
                face_detail=self.settings.image.embedded_face_detail,
                gpu_lock=self._ensure_gpu_lock()
                if self.settings.music.local_gpu_coordination
                else None,
                # OOM-recovery hook: when an SDXL pipeline load
                # OOMs and we have nothing left to evict locally,
                # the image client calls this to ask the ACE-Step
                # server to unload its currently-resident model.
                # Best-effort — fails silently when the music
                # backend isn't on the same GPU or doesn't expose
                # an unload endpoint.
                unload_music_model=self._unload_music_model_hook,
            )
        else:
            from ..providers.image_client import ComfyUiImageClient

            client = ComfyUiImageClient(
                self.settings.image,
                workflow_root=self._workflow_root,
            )

        self._image_client_cache = client
        self._image_settings_fp = fingerprint
        return client

    def _ensure_gpu_lock(self) -> asyncio.Lock:
        """Lazily build the shared GPU lock. Sessions are constructed
        outside an event loop in tests; ``asyncio.Lock`` would bind
        to whichever loop is current at construction, then refuse
        to be acquired from a different loop. Building on first
        request defers the bind to the loop that's actually
        running the inference."""
        import asyncio as _asyncio

        if self._gpu_lock is None:
            self._gpu_lock = _asyncio.Lock()
        return self._gpu_lock

    def _ensure_play_lock(self) -> asyncio.Lock:
        """Lazy-build the play-handler serialization lock for the
        same reason ``_gpu_lock`` is lazy: Session can be constructed
        outside an event loop (tests, save migration tooling)."""
        import asyncio as _asyncio

        if self._play_lock is None:
            self._play_lock = _asyncio.Lock()
        return self._play_lock

    async def _unload_music_model_hook(self) -> bool:
        """Async hook the EmbeddedImageClient calls during OOM
        recovery. Routes through the cached music client if one
        exists; returns False (no-op) when music is disabled or
        not running on the same GPU as the image backend.

        Best-effort: HTTP failures are absorbed inside the music
        client and surfaced as ``False``; the image client treats
        a False return the same as a success and retries the load.
        Worst case: the retry OOMs again and we raise — which is
        the same outcome as if the hook didn't exist, so the hook
        can't make things worse than the no-coordination baseline.
        """
        if not self.settings.music.enabled:
            return False
        if not self.settings.music.local_gpu_coordination:
            # Music server is on a different GPU / box — its VRAM
            # isn't competing with our SDXL load.
            return False
        client = self.music_client()
        if client is None:
            return False
        unload = getattr(client, "unload_remote_model", None)
        if unload is None:
            return False
        try:
            return bool(await unload())
        except Exception:
            _log.warning(
                "music unload hook raised during OOM recovery",
                exc_info=True,
            )
            return False

    def _sync_mutable_image_settings(self, client: ImageClient) -> None:
        """Push the cheap-to-toggle image settings (currently just
        the face-detail flag) into a cached client. Avoids rebuild
        churn when the player flips a setting that doesn't require
        a different pipeline / different model on disk."""
        setter = getattr(client, "set_face_detail", None)
        if setter is not None:
            setter(self.settings.image.embedded_face_detail)

    def music_client(self) -> AceStepClient | None:
        """Return the cached ACE-Step client when music gen is
        enabled, else ``None``. Cached against a settings
        fingerprint (base_url, model name, all the request-shape
        knobs) so a settings flip rebuilds rather than reusing a
        stale ``httpx.AsyncClient``."""
        if not self.settings.music.enabled:
            cached: AceStepClient | None = getattr(self, "_music_client_cache", None)
            if cached is not None:
                self._dispose_music_client(cached)
                self._music_client_cache = None
                self._music_client_fp = None
            return None

        fingerprint = (
            self.settings.music.base_url,
            self.settings.music.model_name,
            self.settings.music.clip_seconds,
            self.settings.music.inference_steps,
            self.settings.music.guidance_scale,
        )
        prior: AceStepClient | None = getattr(self, "_music_client_cache", None)
        cached_fp: tuple[Any, ...] | None = getattr(self, "_music_client_fp", None)
        if prior is not None and cached_fp == fingerprint:
            return prior
        if prior is not None:
            self._dispose_music_client(prior)

        from ..providers.music_client import AceStepClient

        def _evict_image() -> None:
            image_client = getattr(self, "_image_client_cache", None)
            evict = getattr(image_client, "evict_to_cpu", None)
            if evict is not None:
                evict()

        client = AceStepClient(
            self.settings.music,
            gpu_lock=self._ensure_gpu_lock()
            if self.settings.music.local_gpu_coordination
            else None,
            evict_image_to_cpu=_evict_image,
        )
        self._music_client_cache = client
        self._music_client_fp = fingerprint
        return client

    def _dispose_music_client(self, client: AceStepClient) -> None:
        aclose = getattr(client, "aclose", None)
        if aclose is None:
            return
        import asyncio as _asyncio

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            try:
                _asyncio.run(aclose())
            except Exception:
                _log.warning(
                    "music client teardown raised; continuing",
                    exc_info=True,
                )
            return
        from ..background import spawn

        spawn(aclose(), label="music-client-aclose", loop=loop)

    def _dispose_image_client(self, client: ImageClient) -> None:
        """Best-effort teardown of a previous image client. Embedded
        clients hold pipelines in VRAM that must be released before
        the next client can load its own checkpoint; ComfyUI clients
        are stateless so the call is a no-op for them.

        The teardown is SYNCHRONOUS in spirit even when fired from
        async context — we drop the in-memory pipeline dict and
        empty the CUDA cache inline so the next pipeline load sees
        freed VRAM. Earlier this was a fire-and-forget
        ``loop.create_task(aclose())``, which scheduled the
        teardown for "soon" but didn't actually run it before the
        new client tried to load — the next render saw OLD
        pipelines still pinned in VRAM and either OOMed (low-VRAM
        cards) or rendered against the OLD checkpoint until a
        restart cleared the cache. The user's reported symptom:
        "characters not rendering after a model change in settings
        until restart." Inline drop fixes this.
        """
        # Drop the pipeline cache synchronously. EmbeddedImageClient
        # exposes a private dict + helper for this case so the
        # teardown doesn't depend on the asyncio loop.
        pipelines = getattr(client, "_pipelines", None)
        if pipelines is not None:
            try:
                from ..providers.embedded_image_client import _release_pipeline

                for pipe in list(pipelines.values()):
                    _release_pipeline(pipe)
                pipelines.clear()
                inference_locks = getattr(client, "_inference_locks", None)
                if inference_locks is not None:
                    inference_locks.clear()
                evicted = getattr(client, "_evicted", None)
                if evicted is not None:
                    evicted.clear()
            except Exception:
                _log.warning(
                    "image client pipeline teardown raised; continuing",
                    exc_info=True,
                )
        # ALSO call aclose() if available — for HTTP-based clients
        # (ComfyUI) this closes the connection pool. Fire as a task
        # so we don't block the loop on a transport-level shutdown;
        # the synchronous pipeline drop above is what actually frees
        # the VRAM the next render needs.
        aclose = getattr(client, "aclose", None)
        if aclose is None:
            return
        import asyncio as _asyncio

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            try:
                _asyncio.run(aclose())
            except Exception:
                _log.warning(
                    "image client aclose raised; continuing",
                    exc_info=True,
                )
            return
        from ..background import spawn

        spawn(aclose(), label="image-client-aclose", loop=loop)

    # ---- save commit --------------------------------------------------------

    async def commit(self) -> None:
        """Persist the current game. Awaitable because the write isn't
        cheap: ``commit_save`` does two mkstemp + fsync + ``os.replace``
        cycles, which measured 20-33 ms at a thousand-node tree — long
        enough that doing it inline on the loop visibly hitched the
        dialog stream, and it happens on every accepted turn plus every
        editor tweak. ``asyncio.to_thread`` is safe here because
        ``commit_save`` is pure with respect to session state: it takes
        an explicit ``root`` and touches nothing but the filesystem.

        Snapshots ``self.game`` before the hop so a concurrent
        ``install_game`` can't swap the model out mid-write.

        Serialised by ``_commit_lock``. On the loop thread the writes
        couldn't overlap; on worker threads they can, and two commits
        racing the same save directory lose on Windows — ``os.replace``
        of ``meta.json`` fails with ``WinError 5`` when the other
        commit's handle to the destination is still open. The lock is
        also what makes "last commit wins" true: without it the thread
        that started first could finish last and write a stale game.
        """
        game = self.game
        if game is None:
            return
        # Default save name = the world's game_name (set by world_init
        # from the LLM). ``commit_save`` only uses this on the FIRST
        # commit (when no meta exists yet); a player rename via
        # ``c2s/saves/rename`` writes a new meta that subsequent
        # commits preserve. Without this default every fresh save
        # landed as "Untitled" until the player renamed it manually.
        default_name = (game.world.game_name or "").strip()
        if self._commit_lock is None:
            self._commit_lock = asyncio.Lock()
        async with self._commit_lock:
            await asyncio.to_thread(
                save_store.commit_save,
                game,
                self.settings,
                name=default_name or None,
                root=self.saves_root,
            )

    def commit_blocking(self) -> None:
        """Synchronous ``commit`` for the handful of call sites that
        aren't coroutines (teardown paths, the safety-guard scan, and
        tests). Blocks the caller's thread — never call it from the
        event loop; use ``await commit()`` there."""
        game = self.game
        if game is None:
            return
        default_name = (game.world.game_name or "").strip()
        save_store.commit_save(
            game,
            self.settings,
            name=default_name or None,
            root=self.saves_root,
        )

    # ---- LLM critical-path helper ------------------------------------------

    async def llm_stream_chunks(
        self,
        prompt: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Run an LLM call in streaming mode, yielding each chunk as
        it arrives. Used by the foreground dialog generation flow so
        the streaming JSON parser can extract beats as soon as their
        closing brace lands — typewriter starts on beat 0 before the
        LLM has finished generating beats 1+.

        Cost telemetry is folded in once the stream ends (after the
        consumer drains the iterator). Retry budget mirrors
        ``llm_text`` so a transient validation/transport error on
        the first attempt re-establishes the stream.
        """
        import time

        from . import cost as cost_module

        client = self._llm_factory()
        last_exc: Exception | None = None
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.settings.llm.max_tokens
        )
        for _attempt in range(LLM_RETRY_BUDGET):
            started = time.monotonic()
            try:
                stream_iter: AsyncIterator[str] = await client.complete(
                    prompt,
                    model=self.settings.llm.model,
                    temperature=self.settings.llm.temperature,
                    max_tokens=effective_max_tokens,
                    stream=True,
                )
            except ProviderValidationError as exc:
                last_exc = exc
                continue
            total_chars = 0
            async for chunk in stream_iter:
                total_chars += len(chunk)
                yield chunk
            elapsed_ms = int((time.monotonic() - started) * 1000)
            prompt_chars = sum(len(m.get("content", "")) for m in prompt)
            delta = cost_module.estimate_llm_call(
                prompt_chars=prompt_chars,
                reply_chars=total_chars,
                latency_ms=elapsed_ms,
            )
            if self.game is not None:
                self.game = cost_module.fold_into_game(self.game, delta)
            return
        raise last_exc if last_exc else ProviderValidationError("LLM produced no output")

    async def llm_text(
        self,
        prompt: list[dict[str, str]],
        *,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> tuple[str, list[str]]:
        """Run an LLM call to completion, returning (full_text, chunks).

        Records a per-call cost-telemetry delta into the live ``Game``
        when one is installed (Constitution III, Efficiency).

        ``max_tokens`` overrides ``settings.llm.max_tokens`` for this
        single call. Use it for callers that emit unusually long
        structured output (notably the global retcon, which rewrites
        many beats + character updates in one response and otherwise
        truncates mid-JSON at the default 1024 cap).
        """
        import time

        from . import cost as cost_module

        client = self._llm_factory()
        chunks: list[str] = []
        last_exc: Exception | None = None
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.settings.llm.max_tokens
        )
        for _attempt in range(LLM_RETRY_BUDGET):
            started = time.monotonic()
            try:
                stream_iter: AsyncIterator[str] = await client.complete(
                    prompt,
                    model=self.settings.llm.model,
                    temperature=self.settings.llm.temperature,
                    max_tokens=effective_max_tokens,
                    stream=stream,
                )
            except ProviderValidationError as exc:
                last_exc = exc
                continue
            chunks = []
            async for chunk in stream_iter:
                chunks.append(chunk)
            full_text = "".join(chunks)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            prompt_chars = sum(len(m.get("content", "")) for m in prompt)
            delta = cost_module.estimate_llm_call(
                prompt_chars=prompt_chars, reply_chars=len(full_text), latency_ms=elapsed_ms
            )
            if self.game is not None:
                self.game = cost_module.fold_into_game(self.game, delta)
            return full_text, chunks
        raise last_exc if last_exc else ProviderValidationError("LLM produced no output")

    # ---- helpers exposed to handlers ---------------------------------------

    def world_fields_snapshot(self) -> dict[str, str]:
        if self.game is None:
            return {}
        world = self.game.world
        return {
            "game_name": world.game_name,
            "setting": world.setting,
            "genre": world.genre,
            "visual_style": world.visual_style,
            "overall_plot_direction": world.overall_plot_direction,
            "summarizer_assessment": world.summarizer_assessment,
            "active_threads": "|".join(t.id for t in world.active_plot_threads),
        }

    def character_attributes_snapshot(self) -> dict[str, dict[str, str]]:
        if self.game is None:
            return {}
        result: dict[str, dict[str, str]] = {}
        for cid, char in self.game.characters.items():
            result[cid] = {
                "outfit": char.outfit,
                "pose": char.pose,
                "expression": char.expression,
                "description": char.description,
            }
        return result

    def compute_premise_hash(
        self,
        *,
        parent_id: str | None,
        chosen_option_id: str | None,
    ) -> str:
        return premise_hash(
            parent_id=parent_id,
            chosen_option_id=chosen_option_id,
            snapshot_vector=world_snapshot_vector(
                on_stage_character_ids=self.game.on_stage if self.game else [],
                character_attributes=self.character_attributes_snapshot(),
                world_fields=self.world_fields_snapshot(),
            ),
        )

    def install_game(self, game: Game) -> None:
        # Every game mutation in the app funnels through here, which
        # makes this the one place that can guarantee a character is
        # never off-stage while they're talking. See
        # ``domain.game.with_speaker_on_stage``.
        self.game = with_speaker_on_stage(game)

    # ---- speculation invalidation (FR-016/17) ------------------------------

    def invalidate_speculation_from(self, root_node_id: str) -> list[str]:
        if self.game is None:
            return []
        invalidated = obsolescence.invalidate_descendants(self.game, root_node_id)
        # Drop pending redraws whose entity the invalidation just trimmed
        # (the speculative-and-not-picked discard). In-flight renders are
        # left to finish and self-discard at apply time.
        self.render_scheduler.on_invalidation()
        return invalidated

    # ---- cancellation ------------------------------------------------------

    def cancel_foreground(self) -> bool:
        """Cancel the in-flight FOREGROUND generation, if any.

        Two handles, because a play turn spans two tasks: the dispatch
        task running the handler (which is mid-``llm_stream_chunks``
        awaiting beat 1) and the drain task that keeps parsing beats 2+
        after beat 1 shipped. A player hitting Stop wants both gone.

        Returns ``True`` when something was actually cancelled — the
        caller reports that back so the renderer knows whether to
        expect the generation's messages to stop arriving.
        """
        import asyncio as _asyncio

        try:
            current = _asyncio.current_task()
        except RuntimeError:  # pragma: no cover -- no running loop
            current = None
        cancelled = False
        for attr in ("_foreground_task", "_foreground_stream_task"):
            task = getattr(self, attr, None)
            if task is None or task.done() or task is current:
                continue
            task.cancel()
            cancelled = True
        return cancelled

    # ---- teardown ----------------------------------------------------------

    def owned_tasks(self) -> list[Any]:
        """Every asyncio task this session started, flattened.

        The session's background work is spread across singleton
        handles (``_world_init_task``), dicts (``_speculative_tasks``),
        and lists (``_asset_tasks``), so teardown needs one place that
        knows the shape of each. Order is irrelevant — they all get
        cancelled together.
        """
        tasks: list[Any] = []
        for attr in (
            "_foreground_task",
            "_foreground_stream_task",
            "_world_init_task",
            "_char_desc_task",
            "_name_options_task",
            "_preview_bg_task",
            "_preview_guide_task",
            "_pc_portrait_task",
        ):
            # No getattr default: every name above is declared in
            # ``__init__``, so a typo here should be an AttributeError
            # rather than a silently skipped task group.
            task = getattr(self, attr)
            if task is not None:
                tasks.append(task)
        for attr in ("_summarizer_tasks", "_music_tasks", "_asset_tasks"):
            tasks.extend(getattr(self, attr))
        tasks.extend(self._speculative_tasks.values())
        pump = getattr(self.render_scheduler, "_pump", None)
        if pump is not None:
            tasks.append(pump)
        # De-duplicate by identity — the render pump registers itself on
        # ``_asset_tasks`` as well, and a drain task can be reachable
        # from two handles at once.
        unique: list[Any] = []
        seen: set[int] = set()
        for task in tasks:
            if id(task) not in seen:
                seen.add(id(task))
                unique.append(task)
        return unique

    async def aclose(self) -> None:
        """Cancel every task this session owns and close its provider
        clients.

        Called from the WebSocket connection's ``finally`` block. Before
        this existed, closing the window mid-turn left the LLM stream,
        the speculative branches, the summarizer, the music task and the
        render pump all running against an orphaned session — burning
        OpenRouter credits and GPU on output nobody would ever read, and
        (on reconnect) racing a second Session's image client for the
        same VRAM.

        We AWAIT the cancelled tasks rather than firing ``cancel()`` and
        walking away: ``cancel()`` only schedules the ``CancelledError``
        for the next time the task resumes, so returning immediately
        would let a task's ``finally`` block run after we'd already
        closed the image client out from under it. Idempotent.
        """
        import asyncio as _asyncio

        if self._closed:
            return
        self._closed = True

        try:
            current = _asyncio.current_task()
        except RuntimeError:  # pragma: no cover -- no running loop
            current = None

        tasks: list[Any] = [t for t in self.owned_tasks() if t is not current]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            # ``return_exceptions`` so one task's teardown blowing up
            # doesn't abandon the rest (or the client close below).
            await _asyncio.gather(*tasks, return_exceptions=True)

        # Drop the handles so a late caller can't resurrect a reference
        # to a dead task (and so the scheduler doesn't try to reuse a
        # cancelled pump).
        self._speculative_tasks = {}
        self._summarizer_tasks = []
        self._music_tasks = []
        self._asset_tasks = []
        for attr in (
            "_foreground_task",
            "_foreground_stream_task",
            "_world_init_task",
            "_char_desc_task",
            "_name_options_task",
            "_preview_bg_task",
            "_preview_guide_task",
            "_pc_portrait_task",
        ):
            setattr(self, attr, None)
        self.render_scheduler._pump = None

        await self._close_provider_clients()

    async def _close_provider_clients(self) -> None:
        """Await ``aclose()`` on every provider client the session
        built or was handed. Identity-deduped because an injected
        override is returned by the factory WITHOUT being cached, so
        the same object can appear under two attributes."""
        candidates = [
            self._llm_client_cache,
            self._llm_override,
            self._image_client_cache,
            self._image_override,
            self._music_client_cache,
        ]
        seen: set[int] = set()
        for client in candidates:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            aclose = getattr(client, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception:
                _log.warning(
                    "provider client aclose raised during session teardown; continuing",
                    exc_info=True,
                )
        self._llm_client_cache = None
        self._image_client_cache = None
        self._image_settings_fp = None
        self._music_client_cache = None
        self._music_client_fp = None


def _resolve_embedded_dir(image_settings: ImageSettings) -> Path:
    """Mirror ``embedded_models.resolve_models_dir`` without forcing
    the import at module load time (embedded_models is part of the
    optional backend; a comfyui-only install shouldn't pay its
    import cost)."""
    from ..providers.embedded_models import resolve_models_dir

    return resolve_models_dir(image_settings.embedded_models_dir)


def _image_settings_fingerprint(image_settings: ImageSettings) -> tuple[Any, ...]:
    """A hashable snapshot of every ImageSettings field that should
    invalidate the cached image client. When the player flips backend,
    swaps a model file, or retargets the embedded directory, the
    fingerprint changes and ``_image_factory`` rebuilds the client —
    a previous bug let stale clients linger so settings edits blocked
    renders until reverted."""
    return (
        image_settings.backend,
        image_settings.base_url,
        image_settings.portrait_workflow,
        image_settings.background_workflow,
        image_settings.embedded_models_dir,
        image_settings.embedded_model_name,
        image_settings.embedded_character_model_name,
        image_settings.embedded_environment_model_name,
    )


def _default_workflow_root() -> Path:
    """Resolve the default ComfyUI workflow directory.

    Production workflows live at ``backend/workflows/``. The toy
    placeholder graphs under ``backend/src/lucidium/orchestration/prompts/comfy/``
    are kept as a fallback for tests that don't have a real workflow
    set installed; they never get run against a real ComfyUI.
    """
    here = Path(__file__).resolve()
    backend_root = here.parents[3]  # backend/
    production = backend_root / "workflows"
    if production.exists():
        return production
    return here.parent / "prompts" / "comfy"


def _default_fixture_root(kind: str) -> Path:
    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / "tests" / "fixtures" / kind


def make_environment(*, label: str, prompt: str) -> Environment:
    return Environment(
        location_label=label,
        prompt=prompt,
        prompt_hash=premise_hash(parent_id=label, chosen_option_id=None, snapshot_vector=[prompt]),
    )


def make_dialog_node(
    *,
    parent_id: str | None,
    chosen_option_id: str | None,
    premise: str,
    text: str | None = None,
    state: DialogNodeState = DialogNodeState.speculative,
) -> DialogNode:
    return DialogNode(
        parent_id=parent_id,
        chosen_option_id=chosen_option_id,
        text=text,
        premise_hash=premise,
        state=state,
    )


def attach_image_to_character(
    character: Character,
    *,
    path: Path,
    prompt_hash: str,
    attributes: dict[str, str],
) -> Character:
    image = CharacterImage(
        path=str(path),
        prompt_hash=prompt_hash,
        attributes_snapshot=attributes,
    )
    return character.model_copy(update={"images": [image, *character.images]})


def empty_dialog_tree() -> DialogTree:
    return DialogTree()


def empty_world(*, setting: str, genre: str, visual_style: str) -> WorldState:
    return WorldState(
        game_name="(unnamed)",
        setting=setting,
        genre=genre,
        visual_style=visual_style,
    )


__all__ = [
    "InterviewState",
    "OutboundEmit",
    "Session",
    "attach_image_to_character",
    "empty_dialog_tree",
    "empty_world",
    "make_dialog_node",
    "make_environment",
    "random_seed",
]

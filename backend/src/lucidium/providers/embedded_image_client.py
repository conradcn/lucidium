"""In-process image generation that mirrors the ComfyUI workflows.

The ``ComfyUiImageClient`` POSTs the contents of
``backend/workflows/<workflow>.json`` to a ComfyUI server. The
embedded client implements the same ``ImageClient`` protocol but
runs the equivalent diffusers pipeline locally — no external server,
no HTTP. Same input contract (``workflow`` filename, ``params`` dict
of placeholder values, ``seed``); same output contract (PNG bytes
matching the workflow's stated dimensions). Tests assert this
contract holds so the rest of the engine doesn't need to know which
backend rendered any given asset.

Workflow parity, briefly:
  * ``character.json``  — SDXL @ 832x1216, then background removal
    via ``rembg``. The ComfyUI variant additionally runs a
    FaceDetailer pass; we skip that here because it depends on a
    ComfyUI custom-node pipeline that's not portable to diffusers.
    The embedded fallback renders the face in the base pass, which
    on SDXL-class checkpoints is generally clean at 1216 px tall.
  * ``background.json`` — SDXL @ 1536x1024, no post-processing.

The pipeline is loaded lazily on first ``generate`` call; subsequent
calls reuse the in-memory model. Switching the configured model file
forces a reload through ``aclose()`` + a fresh client.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, TypeVar

from ..api.errors import ProviderUnreachableError, ProviderValidationError
from . import clip_long_prompt
from .embedded_models import (
    pick_default_model,
    resolve_models_dir,
)

_log = logging.getLogger(__name__)
_call_log = logging.getLogger("lucidium.api_calls")

# Return type of the blocking helper handed to ``_run_cancellable``,
# so the awaited value keeps the helper's own type instead of ``Any``.
_RunResultT = TypeVar("_RunResultT")


# Dimension contract per workflow. Mirrors the EmptyLatentImage node
# in each ComfyUI workflow JSON so an embedded render lands at the
# same aspect ratio the engine elsewhere assumes (e.g., the renderer
# crops portraits at 832x1216 expectations).
#
# 832x1216 is the canonical SDXL / Pony portrait bucket (2:3) — both
# checkpoints were trained on it, so straying from the bucket
# (e.g. square 1024x1024) collapses anatomy quality. Backgrounds use
# the matching landscape bucket 1536x1024 (3:2). Same pixel count as
# 1024x1024 so VRAM budget is unchanged.
#
# Aliases below: the engine's settings default the workflow path to
# ``portrait.workflow.json`` / ``background.workflow.json``, but
# older saves and the bundled ComfyUI files use ``character.json`` /
# ``background.json``. Both names must resolve to portrait /
# landscape dims — without the aliases, ``.get()`` falls back to a
# square 1024x1024 default and Pony renders cropped, mis-anatomied
# figures.
_PORTRAIT_DIMS: tuple[int, int] = (832, 1216)
_BACKGROUND_DIMS: tuple[int, int] = (1536, 1024)
WORKFLOW_DIMENSIONS: dict[str, tuple[int, int]] = {
    "character.json": _PORTRAIT_DIMS,
    "portrait.workflow.json": _PORTRAIT_DIMS,
    "background.json": _BACKGROUND_DIMS,
    "background.workflow.json": _BACKGROUND_DIMS,
}


def _resolve_dimensions(workflow: str) -> tuple[int, int]:
    """Resolve dimensions for any character / portrait workflow name.
    Falls back to the canonical SDXL portrait bucket (832x1216) for
    unknown names rather than square 1024x1024 — the renderer pipes
    everything except backgrounds through the character pipeline, so
    a missing-key default of "portrait" is correct, not "square"."""
    if workflow in WORKFLOW_DIMENSIONS:
        return WORKFLOW_DIMENSIONS[workflow]
    lowered = workflow.lower()
    if "background" in lowered or "environment" in lowered or "scene" in lowered:
        return _BACKGROUND_DIMS
    return _PORTRAIT_DIMS


# The static negative prompts copied verbatim from the corresponding
# workflow JSON. Keeping them in code rather than re-parsing the
# workflow file lets the embedded backend run without the workflow
# directory present (e.g., in a packaged build that ships the
# workflows as data files).
_CHARACTER_NEGATIVE = (
    "blurry, low quality, watermark, text, extra limbs, deformed hands, "
    "deformed face, asymmetrical eyes, cropped, close-up, head shot, "
    "bust shot, portrait crop, cut off legs, cut off feet, out of frame, "
    "back view, side view, profile, facing away, turned away, "
    "multiple people"
)

_BACKGROUND_NEGATIVE = "blurry, low quality, watermark, text"


def _is_character_workflow(workflow: str) -> bool:
    """Workflows route by their filename — the engine passes
    ``character.json`` / ``portrait.workflow.json`` for portraits
    and ``background.json`` / ``background.workflow.json`` for
    backgrounds. The default in settings is ``portrait.workflow.json``
    so substring-matching ``character`` alone is NOT enough — a
    portrait render would otherwise route to the background
    pipeline (no FaceDetailer, landscape dims). Treat anything that
    isn't explicitly background-flavoured as a character workflow."""
    lowered = workflow.lower()
    if "background" in lowered or "environment" in lowered or "scene" in lowered:
        return False
    return True


class _NullAsyncCtx:
    """Async context manager that's a no-op. Used as a stand-in
    for the optional shared GPU lock so the call site can write
    ``async with gpu_ctx, inference_lock:`` unconditionally
    without an ``if gpu_lock is not None`` branch."""

    async def __aenter__(self) -> _NullAsyncCtx:
        return self

    async def __aexit__(self, *_a: object) -> bool:
        return False


_NULL_ASYNC_CTX = _NullAsyncCtx()


class RenderAborted(RuntimeError):
    """Raised inside the diffusion worker thread when its abort event
    fires. Never escapes to a caller: ``_run_cancellable`` is already
    unwinding on ``CancelledError`` by the time this surfaces."""


def _abort_callback(abort: threading.Event | None) -> Any:
    """Build a diffusers ``callback_on_step_end`` that aborts the
    denoise loop when ``abort`` is set, or ``None`` when there is
    nothing to watch.

    This is the ONLY way to stop a diffusers pipeline mid-run.
    ``asyncio.to_thread`` gives no handle on the worker thread, so
    cancelling the awaiting coroutine merely detaches the future —
    the thread keeps denoising every remaining step, holding
    ``gpu_lock`` and the per-pipeline inference lock the whole time,
    which queues the render the player is actually waiting for behind
    work whose result is already known to be garbage.
    """
    if abort is None:
        return None

    def _callback(
        _pipe: Any, _step: int, _timestep: Any, callback_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        if abort.is_set():
            raise RenderAborted("render aborted by caller")
        return callback_kwargs

    return _callback


def _install_abort_callback(
    pipeline: Any, kwargs: dict[str, Any], abort: threading.Event | None
) -> None:
    """Attach the abort callback to a pipeline call's kwargs, if this
    pipeline supports the hook. Older diffusers releases (and the fake
    pipelines the unit tests drive) don't take
    ``callback_on_step_end``; passing it there would be a TypeError, so
    the support check is a signature probe rather than a version test."""
    callback = _abort_callback(abort)
    if callback is None:
        return
    try:
        params = inspect.signature(pipeline.__call__).parameters
    except (TypeError, ValueError):  # pragma: no cover -- exotic callables
        return
    supported = "callback_on_step_end" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if supported:
        kwargs["callback_on_step_end"] = callback


async def _run_cancellable(
    fn: Callable[..., _RunResultT], /, *args: Any, **kwargs: Any
) -> _RunResultT:
    """Run a blocking render helper in a worker thread so that
    cancelling the awaiting coroutine actually stops the render.

    Two halves, both necessary:

      * a ``threading.Event`` is threaded into ``fn`` as ``abort`` (for
        helpers that accept it) and set on cancellation, so the
        diffusion loop raises out at its next step boundary;
      * the future is ``shield``ed and then awaited to completion on
        the cancellation path, so we don't return — and therefore
        don't release ``gpu_lock`` / the inference lock — until the
        worker thread has actually stopped touching the GPU. Returning
        early would hand the lock to the next render while the
        abandoned one still held VRAM, which is the OOM the lock
        exists to prevent.
    """
    abort = threading.Event()
    if _accepts_abort(fn):
        kwargs = {**kwargs, "abort": abort}
    future = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        abort.set()
        try:
            await future
        except BaseException:
            pass
        raise


def _accepts_abort(fn: Any) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover -- exotic callables
        return False
    return "abort" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


class EmbeddedImageClient:
    """Diffusers-backed implementation of the ``ImageClient`` protocol.

    Constructor never touches the model — load is deferred to the
    first ``generate`` call so a settings flip to ``embedded`` doesn't
    block on a multi-second pipeline init when the player isn't
    actually rendering yet. The ``_pipeline_factory`` knob lets tests
    inject a fake without monkey-patching diffusers.

    Per-workflow checkpoints: the player can pin different .safetensors
    files for character vs environment renders via
    ``ImageSettings.embedded_character_model_name`` /
    ``embedded_environment_model_name``. When the two paths differ,
    how many stay warm is bounded by ``max_resident_pipelines``
    (default 1, i.e. swap per render; override via the constructor or
    ``LUCIDIUM_MAX_RESIDENT_PIPELINES``). Loads that OOM additionally
    evict the least-recently-used pipeline and retry, so a low-VRAM
    setup adapts even under a generous cap. When the two paths resolve
    to the same file (or one is unset and the fallback chain points to
    the same file) only a single pipeline is loaded.
    """

    def __init__(
        self,
        *,
        models_dir: str = "",
        model_name: str = "",
        character_model_name: str = "",
        environment_model_name: str = "",
        device: str | None = None,
        pipeline_factory: Any = None,
        bg_remover: Any = None,
        face_detail: bool = False,
        face_inpaint_runner: Any = None,
        expression_inpaint_runner: Any = None,
        qwen_img2img_runner: Any = None,
        gpu_lock: asyncio.Lock | None = None,
        unload_music_model: Any = None,
        max_resident_pipelines: int | None = None,
    ) -> None:
        self._models_dir = resolve_models_dir(models_dir)
        # Fallback chain per workflow:
        #   character -> ``character_model_name`` or ``model_name``
        #   environment -> ``environment_model_name`` or ``model_name``
        # ``model_name`` ultimately falls back to "first .safetensors"
        # inside ``pick_default_model`` when itself empty.
        self._configured_name = model_name
        self._character_name = character_model_name
        self._environment_name = environment_model_name
        self._device = device
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self._bg_remover: Callable[[bytes], bytes] | None = bg_remover  # Lazy-loaded if None.
        # Face-detail toggle. Mutable across the client's lifetime —
        # ``set_face_detail`` flips it without rebuilding so the
        # player can A/B the option in Settings without paying a
        # multi-second pipeline reload per toggle.
        self._face_detail = bool(face_detail)
        # Hook so tests can swap the inpaint runner without standing
        # up a real diffusers img2img pipeline.
        self._face_inpaint_runner = face_inpaint_runner or _run_face_inpaint
        # Same seam for the expression-only refresh path
        # (``regenerate_expression``).
        self._expression_inpaint_runner = expression_inpaint_runner or _run_expression_inpaint
        # Seam for the Qwen img2img "character change" path
        # (``regenerate_from_image``). The default builds + caches a
        # ``QwenImageImg2ImgPipeline`` sharing the text2img components and
        # runs it; it returns ``None`` when the pipeline can't be built,
        # which the caller treats as a decline. Tests inject a fake to
        # exercise the path without a real diffusers pipeline.
        self._qwen_img2img_runner = qwen_img2img_runner or _default_qwen_img2img_runner
        # path -> loaded pipeline. Multiple entries when character
        # and environment workflows pin different checkpoints AND
        # there's enough VRAM to keep both warm.
        self._pipelines: dict[Path, Any] = {}
        # Insertion order = LRU order; entries at the front are the
        # least-recently-used and get evicted first.
        #
        # Hard cap on resident pipelines. Eviction used to happen ONLY
        # when an allocation failed and ``_is_oom_error`` recognised it
        # — which on DirectML / MPS / CPU it frequently didn't, so those
        # setups accumulated pipelines with no eviction path at all. The
        # cap makes eviction unconditional: OOM recovery is now a
        # fallback, not the only mechanism. Default 1 (swap per render)
        # because that's the setup that can't afford to be wrong; bump
        # it on a card with headroom to keep character + environment
        # checkpoints both warm.
        self._max_resident_pipelines = _resolve_pipeline_cap(max_resident_pipelines)
        self._lock = asyncio.Lock()
        # path -> per-pipeline inference lock. Diffusers pipelines
        # are NOT thread-safe — the scheduler is stateful (sigmas,
        # step_index, begin_index) and is mutated in-place during
        # ``set_timesteps`` and every ``step()`` call. The image
        # scheduler can fan multiple ``generate()`` calls out
        # concurrently (foreground ensure_assets + speculative
        # pre-render workers, both grabbing the same pipeline via
        # session._image_factory()), and without serialisation
        # those races trip ``IndexError: index N is out of bounds
        # for dimension 0 with size N`` mid-denoise. One lock per
        # loaded pipeline so character and environment renders on
        # SEPARATE pipelines can still run truly in parallel.
        self._inference_locks: dict[Path, asyncio.Lock] = {}
        # SHARED lock with the local ACE-Step / music client. When
        # set, the image client also holds it for every inference
        # — so an in-flight music render BLOCKS the next image
        # render and vice-versa. Coordinates VRAM contention when
        # both backends share the local GPU (a typical setup
        # where the player runs both an ACE-Step server and the
        # embedded SDXL pipeline on the same machine). ``None``
        # means no coordination (e.g. ACE-Step is on a remote
        # machine and doesn't compete for our GPU).
        self._gpu_lock = gpu_lock
        # Pipelines we shuttled to CPU on the last music render.
        # ``restore_to_gpu`` only acts on these — so when both
        # SDXL and ACE-Step fit on the GPU at once we never pay
        # the .to() cost. Empty after every full restore.
        self._evicted: set[Path] = set()
        # OOM-recovery hook. The Session wires this to the music
        # client's ``unload_remote_model`` so that when an SDXL
        # pipeline load hits CUDA OOM AND we have no more local
        # pipelines to evict, we can ask the ACE-Step server to
        # release its model from VRAM and retry. Async because
        # the unload is an HTTP round-trip.
        self._unload_music_model = unload_music_model

    def set_face_detail(self, enabled: bool) -> None:
        """Toggle the face-inpaint pass without rebuilding the
        client. Called by the Session when the corresponding
        setting flips, so the next character render reflects the
        new value with no pipeline reload."""
        self._face_detail = bool(enabled)

    # Free VRAM (GiB) above which we DO NOT shuttle SDXL pipelines
    # to CPU before a music render. Calibrated against an observed
    # 24 GB card OOMing by 2.5 GB on a co-resident
    # SDXL + ACE-Step + 5Hz-LM workload (combined ~26.5 GB). With
    # SDXL holding ~9 GB on that card, free VRAM at music time
    # is ~15 GB — comfortably below this threshold so shuttling
    # always engages on 24 GB. A 48 GB card with ~9 GB SDXL
    # resident has ~39 GB free, well above the threshold, so
    # both backends stay co-resident there. Bump if a user
    # reports OOMs at higher headroom.
    _SKIP_EVICT_FREE_GB = 19.0

    def evict_to_cpu(self) -> None:
        """Move loaded pipelines to CPU IF free VRAM is below the
        coexistence threshold. When free VRAM is high enough that
        ACE-Step fits alongside SDXL, this is a no-op (we still
        ``empty_cache`` to release reserved-but-unused fragments
        so the music server sees the freed bytes).

        Called by the music client (via the session's eviction
        hook) before each ACE-Step HTTP render fires on a shared
        GPU. Safe-by-construction: the music path holds
        ``gpu_lock`` for the duration of the call, so no image
        render can be in flight here, and new loads also acquire
        ``gpu_lock`` before starting — the dict is stable.
        """
        import torch

        accel = _accelerator_module(torch)
        if accel is None:
            return
        mem_info = getattr(accel, "mem_get_info", None)
        if callable(mem_info):
            try:
                free_bytes, _ = mem_info()
            except Exception:
                free_bytes = 0  # Fall through to shuttle on the safe side.
        else:
            # Backends without ``mem_get_info`` (notably MPS): we
            # can't measure free VRAM, so default to "evict" — safer
            # to over-evict than to leak across pipelines.
            free_bytes = 0
        free_gb = free_bytes / (1024**3)
        empty_cache = getattr(accel, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        if free_gb >= self._SKIP_EVICT_FREE_GB:
            _log.debug(
                "embedded image client: %.1f GiB free >= %.1f threshold; "
                "skipping CPU eviction for music render",
                free_gb,
                self._SKIP_EVICT_FREE_GB,
            )
            return
        _log.info(
            "embedded image client: %.1f GiB free < %.1f threshold; "
            "evicting %d pipeline(s) to CPU before music render",
            free_gb,
            self._SKIP_EVICT_FREE_GB,
            len(self._pipelines),
        )
        for path, pipeline in list(self._pipelines.items()):
            try:
                pipeline.to("cpu")
                self._evicted.add(path)
            except Exception:
                _log.warning(
                    "embedded image client: failed to evict pipeline %s to CPU; continuing",
                    path,
                    exc_info=True,
                )
        empty_cache = getattr(accel, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def restore_to_gpu(self) -> None:
        """Move pipelines previously evicted by ``evict_to_cpu``
        back to ``self._device``. No-op when nothing was evicted
        (the common path on hardware with enough VRAM to keep
        SDXL + ACE-Step resident together) or when the configured
        device is CPU.

        On failure (typically OOM during the .to() back to GPU —
        the music server may not have released its VRAM yet), we
        DROP the pipeline from cache entirely instead of leaving
        it stranded on CPU. Without this, a stranded pipeline
        would sit in ``_pipelines`` with components on CPU; the
        next ``generate()`` call would invoke it and trip
        "tensors on different devices" errors that don't tell the
        user what's actually wrong. Dropping forces the next
        call to take the cold-load path, which has its own OOM
        recovery (eviction + music-unload retry).
        """
        if self._device == "cpu" or not self._evicted:
            return
        for path in list(self._evicted):
            pipeline = self._pipelines.get(path)
            if pipeline is None:
                # Pipeline got evicted from the cache while on CPU
                # (e.g. via OOM eviction during a separate load).
                # Drop the bookkeeping entry; nothing to restore.
                self._evicted.discard(path)
                continue
            try:
                pipeline.to(self._device)
                self._evicted.discard(path)
            except Exception:
                _log.warning(
                    "embedded image client: failed to restore pipeline %s "
                    "to %s; dropping from cache so the next generate "
                    "triggers a fresh load",
                    path,
                    self._device,
                    exc_info=True,
                )
                # Drop from cache so the next generate() takes
                # the cold-load path. Don't leave a partially-
                # placed pipeline (some components on CPU, some
                # on GPU) in the cache — that's worse than no
                # cache entry at all.
                stale = self._evict(path)
                if stale is not None:
                    _release_pipeline(stale)

    def _evict(self, path: Path) -> Any:
        """Drop ``path`` from the cache and return the pipeline that was
        resident there (``None`` if there wasn't one).

        Single choke point for every eviction site so the companion
        bookkeeping — the per-pipeline inference lock and the
        moved-to-CPU marker — can never be left behind. A stale
        ``_inference_locks`` entry would be handed back out by
        ``_lock_for`` to a DIFFERENT pipeline object later loaded at the
        same path, which is exactly the aliasing the per-path lock
        exists to avoid.

        Releasing the returned pipeline is the caller's job: on the load
        path it's a synchronous ``_release_pipeline``, on the generate
        path it's batched through ``_release_evicted`` off the event
        loop.
        """
        pipeline = self._pipelines.pop(path, None)
        self._inference_locks.pop(path, None)
        self._evicted.discard(path)
        return pipeline

    def _evict_over_cap(self) -> list[Any]:
        """Evict least-recently-used pipelines until the cache is within
        ``_max_resident_pipelines``. Returns the evicted pipelines for
        the caller to release."""
        doomed: list[Any] = []
        # Never consider the most-recently-used entry (the last in
        # insertion order) — that's the pipeline the caller just loaded
        # or touched and is about to render with.
        candidates = list(self._pipelines)[:-1]
        for path in candidates:
            if len(self._pipelines) <= self._max_resident_pipelines:
                break
            lock = self._inference_locks.get(path)
            if lock is not None and lock.locked():
                # A render is mid-denoise on this pipeline. Freeing it
                # here would pull the weights out from under a running
                # ``__call__``. Skip it — the next load re-checks the
                # cap, and by then the render has finished and released
                # the lock. Being one over the cap for the length of a
                # render is strictly better than a use-after-free.
                _log.debug(
                    "embedded image client: skipping over-cap eviction of in-flight pipeline %s",
                    path,
                )
                continue
            pipeline = self._evict(path)
            _log.info(
                "embedded image client: pipeline cache over cap (%d); "
                "evicting least-recently-used pipeline %s",
                self._max_resident_pipelines,
                path,
            )
            if pipeline is not None:
                doomed.append(pipeline)
        return doomed

    def _lock_for(self, pipeline: Any) -> asyncio.Lock:
        """Return the per-pipeline inference lock, lazily creating
        one on first use. Looked up via the path the pipeline is
        cached under in ``_pipelines`` — that's a stable identity
        for the lifetime of the loaded checkpoint and survives the
        OOM-eviction path (a different pipeline at the same path
        would have already been evicted along with its lock)."""
        for path, cached in self._pipelines.items():
            if cached is pipeline:
                lock = self._inference_locks.get(path)
                if lock is None:
                    lock = asyncio.Lock()
                    self._inference_locks[path] = lock
                return lock
        # Fallback: pipeline not in cache (test stubs that bypass
        # _ensure_pipeline). One process-wide lock prevents concurrent
        # races on the stub too, at the cost of serialising tests.
        if not hasattr(self, "_fallback_inference_lock"):
            self._fallback_inference_lock = asyncio.Lock()
        return self._fallback_inference_lock

    async def generate(self, workflow: str, params: dict[str, Any], *, seed: int) -> bytes:
        # Refuse to render while a torch overlay is being downloaded /
        # unpacked. The seeded CPU overlay would otherwise carry the
        # render at minutes-per-image speed during the very window the
        # GPU wheel the player actually wants is installing. The product
        # call is: show the torch-overlay progress bar, not a CPU image.
        # The frontend interprets the error code and keeps the existing
        # GPU-provision progress UI visible — see GpuAccelStatus +
        # TorchOverlayPanel.
        from . import torch_overlay

        if torch_overlay.is_install_in_flight():
            raise ProviderUnreachableError(
                "torch_installing: GPU torch wheel is downloading; image "
                "generation is paused until the install completes. Watch "
                "the GPU acceleration progress bar."
            )
        positive = str(params.get("positive_prompt", "")).strip()
        face_prompt = str(params.get("face_prompt", "")).strip()
        negative_extras = str(params.get("negative_extras", "")).strip()
        # ``subject_kind`` is an optional override the asset
        # pipeline sets to ``"nonhuman"`` for monsters / robots /
        # spirits. Independent of the workflow filename: the
        # workflow stays ``character.workflow.json`` (so RMBG
        # still cuts the figure cleanly off its backdrop) but the
        # pipeline switches to the ENVIRONMENT checkpoint and the
        # face-detail inpaint pass is skipped (the human-face
        # detector smears nonhuman geometry into something
        # humanoid). Defaults to "human" for backwards compat
        # with callers that don't pass the field.
        subject_kind = str(params.get("subject_kind", "human")).strip().lower()
        nonhuman = subject_kind == "nonhuman"
        if not positive:
            raise ProviderValidationError("embedded backend requires a positive_prompt parameter")
        is_character = _is_character_workflow(workflow)
        # face_prompt handling depends on whether the inpaint pass
        # is enabled:
        #
        # * face_detail OFF — embedded has no second pass, so the
        #   body render is the only chance to land the expression.
        #   Append face_prompt at the END of positive_prompt where
        #   it tints without dominating composition.
        # * face_detail ON — DON'T append. The body pass renders a
        #   neutral face and the inpaint pass runs against the
        #   face_prompt as its only positive. This mirrors
        #   ComfyUI's setup (body uses main prompt; FaceDetailer
        #   wildcard alone drives expression) and makes the
        #   detailer's contribution actually visible — when both
        #   passes share the face_prompt, the body already matches
        #   it and the inpaint just refines, so the "before/after"
        #   diff looks too small to notice.
        # Skip the face_prompt append for nonhuman subjects too —
        # the face-detail inpaint pass is going to be skipped, and
        # the nonhuman subject's freeform ``physical_description``
        # already covers anatomy. Tacking the human-face-tagged
        # face_prompt onto the body prompt would push toward
        # humanoid geometry on a dragon / robot / spirit.
        # Pre-resolve the model path so we can spot Z-Image checkpoints
        # before the pipeline is loaded — the face-detail inpaint pass
        # can't run on Z-Image (no UNet / dual CLIP encoders), so if
        # we left face_detail=ON cause face_prompt to be deferred to
        # the inpaint pass, the cue would just be dropped on Z-Image
        # renders. Treat Z-Image like ``face_detail=OFF`` for the
        # append decision and let the body render carry the expression.
        target_name = self._resolve_target_name(workflow)
        if nonhuman:
            target_name = self._environment_name or self._configured_name
        resolved_path = pick_default_model(self._models_dir, target_name)
        target_is_z_image = resolved_path is not None and _is_z_image_model_path(resolved_path)
        # Krea 2 is in the same boat as Z-Image for both decisions below:
        # no UNet / dual CLIP encoders (so no SDXL inpaint pass) and a
        # bf16 + big-VRAM requirement DirectML can't meet.
        # Off-thread: unlike the Z-Image / Qwen sniffs (filename only)
        # this one opens the checkpoint and reads its safetensors
        # header, and it runs on every single render.
        target_is_krea = (
            resolved_path is not None
            and not target_is_z_image
            and await asyncio.to_thread(_is_krea_model_path, resolved_path)
        )
        target_needs_bf16_gpu = target_is_z_image or target_is_krea
        target_has_no_inpaint = target_is_z_image or target_is_krea
        # When the Z-Image guard below re-points us at an SDXL
        # checkpoint, this carries the override into ``_ensure_pipeline``
        # so the actual load matches the decision made here. ``None``
        # means "no override — resolve the model the usual way".
        z_image_fallback_name: str | None = None

        # Z-Image and Krea 2 cannot run on every GPU. Both need bf16 plus
        # ~18-24 GiB resident; DirectML doesn't expose a usable bf16 path
        # and the Windows-AMD cards it targets are typically 8-16 GiB.
        # Loading either there would crash deep inside diffusers or OOM
        # with a message that doesn't tell the user what's wrong. Catch
        # it HERE, before any load, and either re-point at an SDXL
        # checkpoint (preferred — keeps the player rendering) or raise
        # a clear, actionable error when there's no SDXL to fall back
        # to. Only DirectML is gated today; ROCm-on-Linux AMD cards
        # large enough flow through "cuda" and are left alone, so this
        # never penalises a capable AMD card.
        if target_needs_bf16_gpu and self._device_cannot_run_z_image():
            family = "Z-Image-Turbo" if target_is_z_image else "Krea 2"
            fallback_name = self._sdxl_fallback_name()
            if fallback_name is None:
                raise ProviderValidationError(
                    f"{family} isn't supported on this GPU "
                    "(DirectML / AMD-on-Windows): it needs bfloat16 "
                    "and 18+ GiB of VRAM, which the DirectML backend "
                    "can't provide. Drop an SDXL-family .safetensors "
                    "checkpoint into your models folder (e.g. SDXL "
                    "Turbo) and select it in Settings → Image "
                    "generator."
                )
            _log.warning(
                "embedded backend: resolved %s checkpoint %s but "
                "the active device cannot run it; falling back to SDXL "
                "checkpoint %s",
                family,
                resolved_path,
                fallback_name,
            )
            target_name = fallback_name
            z_image_fallback_name = fallback_name
            resolved_path = pick_default_model(self._models_dir, target_name)
            target_is_z_image = False
            target_is_krea = False
            target_has_no_inpaint = False

        inpaint_will_run = (
            is_character and self._face_detail and not nonhuman and not target_has_no_inpaint
        )
        if is_character and face_prompt and not nonhuman and not inpaint_will_run:
            positive = f"{positive}, {face_prompt}"
        width, height = _resolve_dimensions(workflow)
        # Match the ComfyUI workflows: each one composes its negative
        # by taking the static block from the JSON and appending the
        # caller's per-render extras.
        base_negative = _CHARACTER_NEGATIVE if is_character else _BACKGROUND_NEGATIVE
        negative = base_negative
        if negative_extras:
            negative = f"{base_negative}, {negative_extras}"

        pipeline = await self._ensure_pipeline(
            workflow,
            force_environment=nonhuman,
            override_name=z_image_fallback_name,
        )
        # Enable VAE tiling + slicing on every newly-loaded
        # pipeline. Idempotent (flag on the pipeline object), so
        # this is essentially free after the first call. Cuts
        # peak VRAM during the VAE decode step by ~3× — most
        # important for users who don't use face_detail (and so
        # never hit the inpaint path that also enables this).
        _apply_vae_memory_optimizations(pipeline)
        # Per-pipeline lock so concurrent callers (foreground
        # ensure_assets + render_scheduler workers, both holding the
        # same EmbeddedImageClient) don't race on the scheduler's
        # in-place state. Different pipelines have different locks
        # so character + environment renders on separate checkpoints
        # still parallelise. Looked up by ANY ``Path`` key in
        # _pipelines that maps to this pipeline; lazy-init on first
        # use so older paths still work.
        inference_lock = self._lock_for(pipeline)

        import time

        # Log the FULL composed prompts before generation so the
        # render itself isn't a black box. Tagged with backend so a
        # tail-friendly grep can pull just the embedded calls. The
        # positive line includes the face_prompt-prepended form so
        # it matches what the model actually saw.
        _call_log.info(
            "image-call workflow=%s subject_kind=%s seed=%d backend=embedded "
            "positive=%r negative=%r",
            workflow,
            subject_kind,
            seed,
            positive,
            negative,
        )

        started = time.monotonic()
        # SERIALISE all pipeline-touching work. Two layers:
        #
        #   1. ``gpu_lock`` (optional) — shared with the local
        #      ACE-Step / music client. When set, an in-flight
        #      music HTTP call holds it; this image render waits
        #      until the music finishes and the GPU is free.
        #      Distinct pipelines DO NOT parallelise across this
        #      lock because both compete for the same physical
        #      GPU. ``None`` skips this layer (music is remote /
        #      coordination disabled).
        #   2. ``inference_lock`` (per-pipeline) — diffusers
        #      pipelines are NOT thread-safe. Two concurrent
        #      ``generate()`` calls on the same pipeline race the
        #      stateful scheduler and trip an off-by-one
        #      ``IndexError``. Distinct pipelines still parallelise
        #      across this layer; only same-pipeline collisions
        #      serialise here.
        gpu_ctx: Any = self._gpu_lock if self._gpu_lock is not None else _NULL_ASYNC_CTX
        if self._gpu_lock is not None:
            # Restore SDXL pipelines to GPU. They may have been
            # evicted by the music client's last call; bring them
            # back before the inference layer below grabs the
            # per-pipeline lock and runs.
            _log.info(
                "image-call seed=%d step=restore_to_gpu (evicted=%d)",
                seed,
                len(self._evicted),
            )
            await asyncio.to_thread(self.restore_to_gpu)
            _log.info(
                "image-call seed=%d step=restore_to_gpu DONE (evicted=%d remaining)",
                seed,
                len(self._evicted),
            )
        _log.info("image-call seed=%d step=acquiring gpu_ctx + inference_lock", seed)
        async with gpu_ctx, inference_lock:
            _log.info("image-call seed=%d step=ENTERED critical section", seed)
            try:
                _log.info("image-call seed=%d step=spawning _run_pipeline thread", seed)
                image_bytes = await _run_cancellable(
                    _run_pipeline,
                    pipeline,
                    positive=positive,
                    negative=negative,
                    width=width,
                    height=height,
                    seed=seed,
                )
                _log.info("image-call seed=%d step=_run_pipeline returned", seed)
            except Exception as exc:
                # OOM during generation, not load: drop every OTHER
                # pipeline (keep the one we just used since reloading
                # it would defeat the point of staying warm) and retry
                # once. This is the same recovery path as the loader's
                # _load_with_oom_eviction, but it has to live here too
                # because activation tensors (latents, VAE buffers,
                # text encodes) can OOM on a render even when the
                # checkpoint loaded fine.
                if _is_oom_error(exc) and len(self._pipelines) > 1:
                    async with self._lock:
                        keep = pipeline
                        doomed = [
                            self._evict(path)
                            for path, p in list(self._pipelines.items())
                            if p is not keep
                        ]
                    # Freeing pipelines drops multi-GB of tensors and runs
                    # the CUDA allocator's collector — seconds, not
                    # milliseconds, on a card that just OOM'd.
                    await _release_evicted(doomed)
                    _call_log.warning(
                        "image-call workflow=%s seed=%d backend=embedded OOM during "
                        "generate; evicted other pipelines and retrying",
                        workflow,
                        seed,
                    )
                    try:
                        image_bytes = await _run_cancellable(
                            _run_pipeline,
                            pipeline,
                            positive=positive,
                            negative=negative,
                            width=width,
                            height=height,
                            seed=seed,
                        )
                    except Exception as retry_exc:
                        elapsed_ms = int((time.monotonic() - started) * 1000)
                        _call_log.warning(
                            "image-call workflow=%s seed=%d elapsed_ms=%d "
                            "backend=embedded FAILED after OOM-retry: %s",
                            workflow,
                            seed,
                            elapsed_ms,
                            retry_exc,
                        )
                        await _release_vram_async()
                        raise ProviderUnreachableError(
                            f"embedded image pipeline failed: {retry_exc}"
                        ) from retry_exc
                else:
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    _call_log.warning(
                        "image-call workflow=%s seed=%d elapsed_ms=%d backend=embedded FAILED: %s",
                        workflow,
                        seed,
                        elapsed_ms,
                        exc,
                    )
                    await _release_vram_async()
                    raise ProviderUnreachableError(
                        f"embedded image pipeline failed: {exc}"
                    ) from exc
            finally:
                # Release intermediate activations after every render —
                # SDXL holds onto latents and attention buffers in the
                # PyTorch allocator's cache, which is reusable across the
                # SAME pipeline but doesn't return memory to the OS for
                # OTHER allocations (rembg's onnxruntime, the second
                # checkpoint, etc.). This frees only intermediates; the
                # pipeline weights stay loaded so the next render of the
                # same workflow doesn't pay the multi-second checkpoint
                # cost. Inside ``finally`` so we still cleanup on OOM
                # paths above.
                await _release_vram_async()

            if is_character:
                # Mirror ``character.json``'s RMBG node FIRST — the
                # face-detail pass below positions its inpaint crop
                # using the alpha bounding box of the figure, which
                # only exists after rembg has alpha-cut the
                # background. Without this, the face crop was
                # geometry-guessed at ``y=height/8`` on the
                # assumption that the body prompt's "head fully
                # visible" rule consistently puts the head there —
                # but checkpoints and aspect ratios disagree on
                # where the head actually lands, so the inpaint
                # often targeted the chest instead of the face.
                # Running rembg in the inference lock costs nothing
                # (rembg uses a separate onnxruntime model on
                # different memory; the lock is held for ~200 ms
                # which is negligible against a 3-4 s SDXL pass).
                image_bytes = await _run_cancellable(self._remove_background, image_bytes)

            run_face_detail = (
                is_character
                and self._face_detail
                and face_prompt
                and not nonhuman
                # Z-Image lacks the UNet + dual CLIP encoders the SDXL
                # inpaint pipeline needs, so the inpaint runner would
                # build nothing and silently return the base PNG. We
                # already appended ``face_prompt`` to the body prompt
                # in this case (see the ``inpaint_will_run`` block
                # above); calling the runner here would be wasted work
                # AND would override that append with an identity
                # transform. Krea 2 is in the same position — its
                # transformer has neither a UNet nor CLIP encoders.
                and resolve_model_family(pipeline).supports_sdxl_face_inpaint(pipeline)
            )
            if run_face_detail:
                # Embedded face-detail pass — the equivalent of ComfyUI's
                # FaceDetailer node 12 in workflows/character.json. Runs
                # inpaint on a face-positioned crop with face_prompt as
                # its only positive prompt and a face-tight oval mask,
                # then composites the new face back over the body render
                # using the same mask so unmasked pixels stay byte-
                # identical to the body render. Stays inside the
                # per-pipeline lock so the inpaint pipeline (which
                # shares the UNet/VAE/encoders, even though it gets
                # its own scheduler instance) can't race against
                # another concurrent body render.
                face_seed = (seed * 2654435761) & 0xFFFFFFFF  # Knuth multiplicative
                try:
                    image_bytes = await _run_cancellable(
                        self._face_inpaint_runner,
                        pipeline,
                        image_bytes,
                        face_prompt=face_prompt,
                        negative=negative,
                        seed=face_seed,
                    )
                except Exception as exc:
                    _call_log.warning(
                        "image-call workflow=%s seed=%d backend=embedded face-detail "
                        "pass FAILED (continuing with body render): %s",
                        workflow,
                        seed,
                        exc,
                    )
                finally:
                    await _release_vram_async()

        elapsed_ms = int((time.monotonic() - started) * 1000)
        _call_log.info(
            "image-call workflow=%s seed=%d bytes=%d elapsed_ms=%d backend=embedded",
            workflow,
            seed,
            len(image_bytes),
            elapsed_ms,
        )
        _log_vram_diagnostics(f"workflow={workflow} seed={seed}")
        return image_bytes

    async def regenerate_expression(
        self, workflow: str, base_png: bytes, params: dict[str, Any], *, seed: int
    ) -> bytes | None:
        """Refresh ONLY the expression on an existing portrait.

        The orchestrator calls this (instead of :meth:`generate`) when a
        beat changed a character's expression but left identity, outfit,
        pose, lighting, and seed untouched. We img2img-inpaint just the
        face on ``base_png`` at 0.5 denoise with an OpenCV-detected,
        30 px soft-faded mask — far cheaper than a full render and it
        keeps the rest of the portrait byte-stable.

        Returns the new PNG bytes, or ``None`` when the fast path doesn't
        apply (non-human subject, a Z-Image checkpoint with no SDXL
        inpaint path, no face detected). On ``None`` the caller does a
        normal full re-render.
        """
        subject_kind = str(params.get("subject_kind", "human"))
        face_prompt = str(params.get("face_prompt", "")).strip()
        if subject_kind == "nonhuman" or not face_prompt:
            # Non-human portraits don't have a Haar-detectable face, and
            # with no expression cue there's nothing to refresh.
            return None

        negative_extras = str(params.get("negative_extras", ""))
        negative = _CHARACTER_NEGATIVE
        if negative_extras:
            negative = f"{_CHARACTER_NEGATIVE}, {negative_extras}"

        pipeline = await self._ensure_pipeline(
            workflow,
            force_environment=False,
            override_name=None,
        )
        if not resolve_model_family(pipeline).supports_sdxl_face_inpaint(pipeline):
            # Z-Image and Krea 2 lack the UNet + dual CLIP encoders the
            # SDXL inpaint pipeline needs — fall back to a full render.
            return None
        _apply_vae_memory_optimizations(pipeline)

        import time

        # Derive the inpaint seed the same way the face-detail pass does
        # so a given (portrait seed, expression) refresh is reproducible.
        inpaint_seed = (seed * 2654435761) & 0xFFFFFFFF
        inference_lock = self._lock_for(pipeline)
        gpu_ctx: Any = self._gpu_lock if self._gpu_lock is not None else _NULL_ASYNC_CTX
        if self._gpu_lock is not None:
            await asyncio.to_thread(self.restore_to_gpu)

        _call_log.info(
            "image-call workflow=%s seed=%d backend=embedded op=expression_inpaint face=%r",
            workflow,
            seed,
            face_prompt,
        )
        started = time.monotonic()
        async with gpu_ctx, inference_lock:
            try:
                result: bytes | None = await _run_cancellable(
                    self._expression_inpaint_runner,
                    pipeline,
                    base_png,
                    expression_prompt=face_prompt,
                    negative=negative,
                    seed=inpaint_seed,
                )
            except Exception as exc:
                _call_log.warning(
                    "image-call workflow=%s seed=%d backend=embedded "
                    "expression_inpaint FAILED (caller will full-render): %s",
                    workflow,
                    seed,
                    exc,
                )
                result = None
            finally:
                await _release_vram_async()

        elapsed_ms = int((time.monotonic() - started) * 1000)
        _call_log.info(
            "image-call workflow=%s seed=%d backend=embedded op=expression_inpaint "
            "ok=%s elapsed_ms=%d",
            workflow,
            seed,
            result is not None,
            elapsed_ms,
        )
        return result

    async def regenerate_from_image(
        self, workflow: str, base_png: bytes, params: dict[str, Any], *, seed: int
    ) -> bytes | None:
        """Render a NEW portrait by img2img-ing an existing one — the
        Qwen-Image "character change" path.

        When the configured checkpoint is a Qwen-Image model and a beat
        changes a character (new outfit, pose, expression, effects, …),
        the orchestrator calls this instead of :meth:`generate` so the
        change is applied as an img2img edit of the last portrait rather
        than a render from fresh noise. That keeps the character's
        identity, framing, and composition anchored to the previous
        image while the new prompt steers the requested change. Denoise
        runs at :data:`_QWEN_IMG2IMG_STRENGTH` (0.7) — enough to apply an
        outfit/pose swap, low enough to keep the same person.

        Returns ``None`` (declines the fast path) when:

          * the resolved checkpoint is NOT a Qwen-Image model — SDXL /
            Z-Image keep their existing full-render + face-inpaint
            behaviour, so this method is a no-op for them;
          * there's no positive prompt to steer toward;
          * diffusers can't build the img2img pipeline, or the render
            raises.

        On ``None`` the caller does a normal full :meth:`generate`, so a
        decline never drops a portrait.
        """
        from . import torch_overlay

        if torch_overlay.is_install_in_flight():
            raise ProviderUnreachableError(
                "torch_installing: GPU torch wheel is downloading; image "
                "generation is paused until the install completes. Watch "
                "the GPU acceleration progress bar."
            )
        positive = str(params.get("positive_prompt", "")).strip()
        if not positive:
            return None
        face_prompt = str(params.get("face_prompt", "")).strip()
        negative_extras = str(params.get("negative_extras", "")).strip()
        subject_kind = str(params.get("subject_kind", "human")).strip().lower()
        nonhuman = subject_kind == "nonhuman"
        is_character = _is_character_workflow(workflow)

        # Decline early (no pipeline load) when the resolved checkpoint
        # isn't Qwen — this method only owns the Qwen img2img path.
        target_name = self._resolve_target_name(workflow)
        if nonhuman:
            target_name = self._environment_name or self._configured_name
        resolved_path = pick_default_model(self._models_dir, target_name)
        if resolved_path is None or not _is_qwen_model_path(resolved_path):
            return None

        # Qwen has no separate face-detail inpaint pass, so fold the
        # face cue into the body prompt (mirrors the face_detail=OFF
        # branch in :meth:`generate`). Skip it for nonhuman subjects
        # whose freeform description already covers anatomy.
        if is_character and face_prompt and not nonhuman:
            positive = f"{positive}, {face_prompt}"
        width, height = _resolve_dimensions(workflow)
        base_negative = _CHARACTER_NEGATIVE if is_character else _BACKGROUND_NEGATIVE
        negative = base_negative
        if negative_extras:
            negative = f"{base_negative}, {negative_extras}"

        pipeline = await self._ensure_pipeline(
            workflow,
            force_environment=nonhuman,
            override_name=None,
        )
        if not resolve_model_family(pipeline).supports_qwen_img2img(pipeline):
            # The model changed under us (settings flip mid-render) and
            # is no longer Qwen — decline so the caller full-renders.
            return None
        _apply_vae_memory_optimizations(pipeline)

        import time

        inference_lock = self._lock_for(pipeline)
        gpu_ctx: Any = self._gpu_lock if self._gpu_lock is not None else _NULL_ASYNC_CTX
        if self._gpu_lock is not None:
            await asyncio.to_thread(self.restore_to_gpu)

        _call_log.info(
            "image-call workflow=%s seed=%d backend=embedded op=qwen_img2img "
            "strength=%.2f positive=%r",
            workflow,
            seed,
            _QWEN_IMG2IMG_STRENGTH,
            positive,
        )
        started = time.monotonic()
        async with gpu_ctx, inference_lock:
            try:
                image_bytes: bytes | None = await _run_cancellable(
                    self._qwen_img2img_runner,
                    pipeline,
                    base_png,
                    positive=positive,
                    negative=negative,
                    width=width,
                    height=height,
                    seed=seed,
                    strength=_QWEN_IMG2IMG_STRENGTH,
                )
            except Exception as exc:
                _call_log.warning(
                    "image-call workflow=%s seed=%d backend=embedded "
                    "qwen_img2img FAILED (caller will full-render): %s",
                    workflow,
                    seed,
                    exc,
                )
                await _release_vram_async()
                return None
            finally:
                await _release_vram_async()

            if image_bytes is None:
                # The img2img pipeline couldn't be built (diffusers
                # missing / unbuildable) — decline so the caller
                # full-renders rather than shipping nothing.
                return None

            if is_character:
                # Same RMBG cut the full character path runs — the
                # img2img output is a figure on a (gray) backdrop, so
                # re-cut it to a transparent portrait.
                image_bytes = await _run_cancellable(self._remove_background, image_bytes)

        elapsed_ms = int((time.monotonic() - started) * 1000)
        _call_log.info(
            "image-call workflow=%s seed=%d bytes=%d elapsed_ms=%d "
            "backend=embedded op=qwen_img2img",
            workflow,
            seed,
            len(image_bytes),
            elapsed_ms,
        )
        return image_bytes

    async def aclose(self) -> None:
        """Drop every loaded pipeline so the next generate reloads.
        Used by the Session when settings change (model name / dir /
        backend) — leaving stale pipelines pinned in VRAM blocks the
        next render's checkpoint load on a low-VRAM machine."""
        async with self._lock:
            for path in list(self._pipelines):
                pipeline = self._evict(path)
                if pipeline is not None:
                    _release_pipeline(pipeline)

    # ---- internals --------------------------------------------------------

    def _resolve_target_name(self, workflow: str) -> str:
        """Pick the configured filename for the workflow type, falling
        back through the legacy single-model field. Empty string means
        ``pick_default_model`` will choose the first .safetensors in
        the models directory."""
        if _is_character_workflow(workflow):
            return self._character_name or self._configured_name
        return self._environment_name or self._configured_name

    def _device_cannot_run_z_image(self) -> bool:
        """True when the active compute device can't run Z-Image-Turbo.

        Z-Image needs bf16 + ~18 GiB resident. DirectML ("dml") is the
        known-bad case today: no usable bf16 path and the Windows-AMD
        cards it targets are typically 8-16 GiB. ROCm-on-Linux AMD and
        every NVIDIA card present as "cuda" and are NOT gated here — a
        big AMD card on Linux runs Z-Image fine. Resolve the device the
        same way the factory does (configured override, else auto-
        detect) so the gate matches what the load would actually use.
        """
        device = self._device or detect_compute_device()[0]
        return device == "dml"

    def _sdxl_fallback_name(self) -> str | None:
        """Pick an SDXL-family checkpoint to fall back to when the
        configured model (Z-Image or Krea 2) can't run on this device.

        Preference order: the player's explicitly-configured names
        (character / environment / legacy single field) if any resolve
        to an SDXL-family file, then the first SDXL-family checkpoint in
        the models directory alphabetically. Returns ``None`` when the
        ONLY checkpoints on disk are Z-Image / Krea 2 variants — the
        caller raises a clear error in that case rather than loading
        something that will crash.
        """
        from .embedded_models import list_models

        # Honour explicit settings first so a player who pinned a
        # specific SDXL checkpoint for one workflow gets it.
        for configured in (
            self._character_name,
            self._environment_name,
            self._configured_name,
        ):
            name = (configured or "").strip()
            if not name:
                continue
            candidate = pick_default_model(self._models_dir, name)
            if candidate is not None and _is_sdxl_family_path(candidate):
                return candidate.name

        # Otherwise scan the directory for the first SDXL-family file.
        for name in list_models(self._models_dir):
            if _is_sdxl_family_path(self._models_dir / name):
                return name
        return None

    def _shared_model_short_circuit(self) -> Any:
        """When the player has pinned the SAME checkpoint for both
        character and environment workflows (the common Z-Image
        setup, where each pipeline is ~18 GiB resident and there's no
        room for a second one on a 24 GiB card), guarantee the cache
        returns ONE pipeline for every workflow + ``force_environment``
        combination — bypassing any path-resolution drift, lazy
        rename, or extension-case mismatch that could otherwise miss
        the cache and trigger a duplicate load.

        Returns the single cached pipeline when the short-circuit
        applies, else ``None`` so the regular resolution path runs.
        """
        # Need at least one loaded pipeline AND a unified configured
        # name across both workflow buckets. ``or self._configured_name``
        # mirrors the regular fallback chain so a single ``model_name``
        # also short-circuits (legacy single-field setting).
        if len(self._pipelines) != 1:
            return None
        char = (self._character_name or self._configured_name).strip()
        env = (self._environment_name or self._configured_name).strip()
        if not char or not env or char != env:
            return None
        sole_path, sole_pipeline = next(iter(self._pipelines.items()))
        if sole_path.name != char:
            return None
        return sole_pipeline

    async def _ensure_pipeline(
        self,
        workflow: str,
        *,
        force_environment: bool = False,
        override_name: str | None = None,
    ) -> Any:
        async with self._lock:
            # Same-model short-circuit. When the user has pinned the
            # SAME checkpoint for both character and environment, any
            # workflow — character OR background, with or without
            # ``force_environment`` — MUST return the single cached
            # pipeline. Without this guard the regular path still
            # routes correctly via the dict lookup below, but a future
            # change to the path-resolution helpers could re-introduce
            # a duplicate-load regression silently. On a low-VRAM
            # setup (Z-Image-Turbo at ~18 GiB resident) that means
            # OOM on the second workflow's first render.
            # Skip the same-model short-circuit when an override is in
            # play: the override exists precisely because the configured
            # (Z-Image) checkpoint can't run on this device, so the
            # short-circuit's "return the sole cached pipeline" would
            # hand back the wrong model.
            if override_name is None:
                shared = self._shared_model_short_circuit()
                if shared is not None:
                    return shared
            target_name = (
                override_name
                if override_name is not None
                else (self._environment_name or self._configured_name)
                if force_environment
                else self._resolve_target_name(workflow)
            )
            target_path = pick_default_model(self._models_dir, target_name)
            if target_path is None:
                # Empty directory. Lucidium does NOT auto-download
                # model weights — shipping a UI that fetches them
                # without explicit user action moves the project's
                # legal posture closer to "we distribute the model"
                # than is comfortable for an open-source build. The
                # FirstTimeSetup screen links the user to civitai
                # and asks them to drop a .safetensors file into
                # this folder themselves. SAFETY.md has the policy.
                raise ProviderUnreachableError(
                    "No image-generation weights found in "
                    f"{self._models_dir}. Download an SDXL / Pony / "
                    "Z-Image .safetensors checkpoint from "
                    "https://civitai.com/models?types=Checkpoint"
                    "&baseModels=SDXL+1.0&baseModels=Pony"
                    "&baseModels=Z-Image and drop the file into that "
                    "folder. See SAFETY.md for the policy on bundled "
                    "weights."
                )
            if target_path in self._pipelines:
                # Already-loaded pipeline. Move to back of LRU so it
                # doesn't get evicted by the next OOM-driven swap.
                pipeline = self._pipelines.pop(target_path)
                self._pipelines[target_path] = pipeline
                # Re-check the cap here too: an earlier enforcement may
                # have had to spare an in-flight pipeline, leaving the
                # cache one over. This is the next opportunity to
                # reclaim that slot.
                for stale in self._evict_over_cap():
                    _release_pipeline(stale)
                return pipeline
            return await self._load_with_oom_eviction(target_path)

    async def _load_with_oom_eviction(self, target_path: Path) -> Any:
        """Load ``target_path`` into the cache. If the load fails with
        a CUDA-out-of-memory error and we already have older pipelines
        loaded, drop the oldest one and retry — that's how the client
        adapts to low-VRAM setups without forcing the player to pin
        the same model on both workflows.

        Recovery escalation:
          1. OOM with another pipeline loaded → evict the oldest local
             SDXL pipeline, retry.
          2. OOM with no local pipelines left to evict → ask the music
             client to unload the ACE-Step model on the shared GPU,
             retry once.
          3. OOM persists → raise ``ProviderUnreachableError``.
        """
        # Tracks whether we've already burnt our one-shot music
        # unload retry on this load. The unload is best-effort and
        # may not actually free VRAM (older ACE-Step builds don't
        # ship the unload endpoint); a second retry would just
        # pile up the same OOM with no recourse.
        music_unload_attempted = False
        while True:
            _log.info(
                "embedded backend: loading checkpoint %s on device=%s "
                "(pipelines already loaded: %d)",
                target_path,
                self._device or "auto",
                len(self._pipelines),
            )
            try:
                pipeline = await asyncio.to_thread(
                    self._pipeline_factory,
                    target_path,
                    self._device,
                )
            except Exception as exc:
                if _is_oom_error(exc):
                    # First-line recovery: evict the oldest local
                    # SDXL pipeline if we have one warm.
                    if self._pipelines:
                        oldest_path = next(iter(self._pipelines))
                        _log.warning(
                            "embedded backend: OOM loading %s; evicting "
                            "local pipeline %s and retrying",
                            target_path,
                            oldest_path,
                        )
                        oldest_pipeline = self._evict(oldest_path)
                        if oldest_pipeline is not None:
                            _release_pipeline(oldest_pipeline)
                        continue
                    # Second-line recovery: ask ACE-Step to drop its
                    # loaded model. A common cause of "image gen OOMs
                    # with music enabled" is the ACE-Step server
                    # holding its model in VRAM from a previous
                    # generate_music call; nothing on the SDXL side
                    # can free that VRAM, but the music server can.
                    if not music_unload_attempted and self._unload_music_model is not None:
                        music_unload_attempted = True
                        _log.warning(
                            "embedded backend: OOM loading %s and no "
                            "local pipelines to evict; asking ACE-Step "
                            "to unload its model and retrying",
                            target_path,
                        )
                        try:
                            await self._unload_music_model()
                        except Exception:
                            _log.warning(
                                "music-model unload hook raised; continuing with retry anyway",
                                exc_info=True,
                            )
                        # Encourage CUDA to actually release the
                        # freed allocation back to the OS before we
                        # retry. Without this the retry sometimes
                        # OOMs again on the same nominal-free VRAM.
                        # Encourage the accelerator allocator to
                        # release freed allocations back to the OS
                        # before we retry. ``_release_vram`` handles
                        # CUDA / XPU / MPS uniformly.
                        await _release_vram_async()
                        continue
                raise ProviderUnreachableError(
                    f"failed to load embedded image pipeline: {exc}"
                ) from exc
            self._pipelines[target_path] = pipeline
            # Enforce the resident cap AFTER the successful load: the
            # newcomer is the most-recently-used entry, so the LRU walk
            # never targets it (a cap of 1 evicts everything else).
            for stale in self._evict_over_cap():
                _release_pipeline(stale)
            return pipeline

    def _remove_background(self, image_bytes: bytes) -> bytes:
        """Run the configured background-removal pipeline.

        Tests inject a stub via ``bg_remover``; production resolves
        ``rembg`` lazily because it pulls in onnxruntime.
        """
        if self._bg_remover is None:
            try:
                from rembg import remove as rembg_remove  # type: ignore
            except ImportError as exc:
                # Fall through with the original (un-cut) image: the
                # engine treats it as "rendered without post-
                # processing", which matches what the ComfyUI client
                # returns when RMBG is missing too. Log the actual
                # exception so a packaged build that subtly fails to
                # import rembg (transitive missing dep) tells us what
                # to fix instead of silently shipping with portrait
                # backgrounds intact.
                _log.warning(
                    "embedded backend: rembg import failed (%s); "
                    "skipping background removal. Install "
                    "lucidium[embedded] for full character-portrait "
                    "parity.",
                    exc,
                )
                return image_bytes
            self._bg_remover = rembg_remove
        return self._bg_remover(image_bytes)


def detect_compute_device() -> tuple[str, str]:
    """Return ``(device, diagnostic)`` describing what device image
    pipelines will land on.

    ``device`` is one of ``"cuda"`` (NVIDIA or AMD-via-ROCm — both
    show up on the CUDA backend), ``"xpu"`` (Intel Arc / oneAPI),
    ``"mps"`` (Apple Silicon), ``"dml"`` (DirectML — Windows-AMD and
    any DX12 GPU, via the separate ``torch-directml`` wheel), or
    ``"cpu"``. Note the ``"dml"`` sentinel is NOT a literal torch
    device string — it is mapped to ``torch_directml.device()`` at
    the point of use by :func:`_torch_device_arg`.

    ``diagnostic`` is a human-readable single-line summary of WHY
    that device was picked — empty on every accelerator path, and a
    clear "here's the fix" line when we land on CPU but a GPU is
    present. Used by the startup banner in
    ``app.py::_log_compute_device_diagnostics`` and by the one-shot
    UI notice emitted from ``_ensure_pipeline``.

    Precedence: CUDA → XPU → MPS → DirectML → CPU. CUDA is checked
    first because the upstream diffusers test matrix targets it and
    ROCm presents through the same API; XPU comes next so an Intel
    Arc on a system that ALSO has an iGPU doesn't accidentally
    fall through to CPU. DirectML is LAST before CPU: it is the
    Windows-AMD fallback (and works for any DX12 GPU), but it is
    slower and feature-poorer than the native backends, so a machine
    that has a real CUDA/XPU/MPS torch must never be diverted to it.
    """
    try:
        import torch
    except ImportError:
        return "cpu", (
            "PyTorch is not installed; image generation will fail. "
            "Install with: pip install --force-reinstall torch "
            "--index-url https://download.pytorch.org/whl/cu130 "
            "(NVIDIA) — see https://pytorch.org/get-started/locally/ "
            "for the AMD/Intel Arc/Apple Silicon variants."
        )

    torch_version = getattr(torch, "__version__", "?")
    has_cuda = torch.cuda.is_available()
    has_xpu = getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if has_cuda:
        return "cuda", ""
    if has_xpu:
        return "xpu", ""
    if has_mps:
        return "mps", ""

    # DirectML (Windows-AMD / any DX12 GPU). Checked AFTER the
    # first-class backends and BEFORE the CPU fallback so a machine
    # that has a real CUDA/XPU/MPS torch never accidentally lands on
    # the slower DML path. ``torch_directml`` is a SEPARATE wheel
    # (``torch-directml``) that registers a ``privateuseone`` device;
    # it is NOT part of torch itself, so the import is guarded — on
    # every install without it (including this test venv) we fall
    # straight through to the CPU diagnostics below.
    #
    # Defensive getattr on the availability predicate mirrors the MPS
    # idiom: older torch_directml builds expose only ``device_count()``
    # and never grew an ``is_available()``; treat "≥1 device" as
    # available so those builds still light up.
    try:
        import torch_directml  # type: ignore
    except ImportError:
        torch_directml = None
    else:
        is_available = getattr(torch_directml, "is_available", None)
        if callable(is_available):
            dml_available = bool(is_available())
        else:
            device_count = getattr(torch_directml, "device_count", None)
            dml_available = callable(device_count) and device_count() > 0
        if dml_available:
            return "dml", ""

    # Landed on CPU. Try to figure out WHY — CPU-only wheel vs
    # actual no-GPU machine. The ``+cpu`` build tag on torch's
    # version string is the most reliable signal: PyPI's default
    # torch wheel on Windows is the CPU build, and tons of users
    # accidentally end up with it because they ran ``pip install
    # torch`` instead of using the PyTorch index URL.
    is_cpu_wheel = "+cpu" in torch_version
    has_nvidia = _has_nvidia_gpu()
    has_amd = _has_amd_gpu()
    has_intel_arc = _has_intel_arc_gpu()

    cuda_install = (
        "pip install --force-reinstall torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu130"
    )
    xpu_install = (
        "pip install --force-reinstall torch torchvision --index-url "
        "https://download.pytorch.org/whl/xpu"
    )
    rocm_install = (
        "pip install --force-reinstall torch torchvision --index-url "
        "https://download.pytorch.org/whl/rocm7.2"
    )

    if is_cpu_wheel and has_nvidia:
        # Preserve the actionable cu130 command — that's the
        # copy-pasteable fix for the most common reported case
        # (Windows user accidentally on the +cpu wheel with a
        # CUDA card sitting idle).
        return "cpu", (
            f"torch {torch_version} is a CPU-ONLY build but an "
            "NVIDIA GPU was detected on this system. Image "
            "generation will run minutes-per-image on CPU vs "
            f"seconds on GPU. Fix: {cuda_install}"
        )
    if is_cpu_wheel and has_intel_arc:
        return "cpu", (
            f"torch {torch_version} is a CPU-ONLY build but an "
            "Intel Arc GPU was detected. Install the XPU wheel: "
            f"{xpu_install}"
        )
    if is_cpu_wheel and has_amd:
        return "cpu", (
            f"torch {torch_version} is a CPU-ONLY build but an "
            "AMD GPU was detected. On Linux: "
            + rocm_install
            + ". On Windows AMD inference requires WSL2 + ROCm "
            "or DirectML (not wired into Lucidium today)."
        )
    if is_cpu_wheel:
        # CPU-only wheel, no detectable GPU. Still surface the
        # primary CUDA install command — most users intending to
        # do image gen will move to an NVIDIA machine eventually.
        return "cpu", (
            f"torch {torch_version} is a CPU-ONLY build. Image "
            "generation will be unusably slow without a GPU. If "
            "you have an NVIDIA GPU, reinstall torch with the "
            f"CUDA index: {cuda_install}  Other accelerator paths "
            "(AMD ROCm on Linux, Intel Arc XPU, Apple Silicon MPS) "
            "are listed at https://pytorch.org/get-started/locally/"
        )
    if has_nvidia:
        return "cpu", (
            "An NVIDIA GPU is present but PyTorch reports "
            "torch.cuda.is_available() == False. Likely cause: "
            "the installed CUDA runtime version doesn't match "
            "the torch wheel. Update your NVIDIA driver and "
            "reinstall torch from "
            "https://download.pytorch.org/whl/cu130"
        )
    if has_intel_arc:
        return "cpu", (
            "An Intel Arc GPU is present but PyTorch reports "
            "torch.xpu.is_available() == False. Install the "
            f"XPU-built torch wheel: {xpu_install}"
        )
    if has_amd:
        return "cpu", (
            "An AMD GPU is present but PyTorch reports no "
            "accelerator. On Linux, install the ROCm-built "
            f"torch: {rocm_install}. On Windows, AMD inference "
            "requires WSL2 + ROCm or the DirectML backend (not "
            "currently wired into Lucidium)."
        )
    return "cpu", (
        "No CUDA / XPU / MPS device detected on this machine. "
        "Image generation will be extremely slow on CPU. "
        "Recommended: run on a machine with an NVIDIA GPU and the "
        "CUDA-enabled torch wheel; AMD-on-Linux (ROCm), Intel Arc "
        "(XPU), and Apple Silicon (MPS) are also supported."
    )


def _has_nvidia_gpu() -> bool:
    """Best-effort detection of an NVIDIA GPU on this machine,
    independent of whether torch sees it. Used to disambiguate
    "you have a GPU but the wrong torch" from "you have no GPU."
    """
    return _vendor_present(
        cli=("nvidia-smi", ["-L"]),
        wmi_match="NVIDIA",
        pci_match=("0x10de",),
    )


def _has_amd_gpu() -> bool:
    """Best-effort detection of an AMD discrete GPU. Reads the PCI
    vendor ID from sysfs on Linux (instant); falls back to
    ``rocminfo`` then WMI on Windows. The sysfs path is what
    rescued the "AppImage hangs on Linux/AMD" symptom — calling
    ``rocminfo`` on a broken HIP install used to spin the probe
    long enough that backend startup outran Electron's connection
    handshake."""
    return _vendor_present(
        cli=("rocminfo", []),
        wmi_match=("AMD", "Radeon", "ATI"),
        pci_match=("0x1002",),
    )


def _has_intel_arc_gpu() -> bool:
    """Best-effort detection of an Intel Arc discrete GPU. ``xpu-smi``
    ships with the Intel oneAPI runtime; WMI also surfaces Arc cards
    under "Intel(R) Arc". The sysfs PCI vendor ID 0x8086 matches
    ALL Intel GPUs (including the iGPU integrated graphics that
    can't run image-gen pipelines at usable speed); we trade the
    false-positive for the guaranteed-no-hang property and let the
    CPU diagnostic name "Intel Arc" — users with iGPUs ignore the
    Arc-specific URL, users with Arc follow it."""
    return _vendor_present(
        cli=("xpu-smi", ["discovery"]),
        wmi_match=("Intel(R) Arc", "Intel Arc"),
        pci_match=("0x8086",),
    )


def detect_total_vram_gb() -> float | None:
    """Best-effort total VRAM of the primary GPU, in GiB. ``None`` when
    it can't be determined.

    Used by ``embedded_models.recommend_model`` to size the first-run
    download: only a card with real headroom (~16 GiB+) is offered
    Z-Image-Turbo (which needs bf16 + ~18 GiB resident); smaller cards
    get full SDXL or the lighter, few-step SDXL-Turbo.

    Why not just ask torch? At first run the ACTIVE torch overlay is
    usually the seeded CPU build (the GPU flavor installs async), so
    ``torch.cuda.is_available()`` is ``False`` on a perfectly good
    NVIDIA box. We therefore probe in order of reliability:

      1. A live accelerator torch (CUDA / ROCm present as ``cuda``;
         Intel Arc as ``xpu``) — exact, when it happens to be active.
      2. ``nvidia-smi`` — reports total VRAM even when the running
         torch is CPU-only, so it covers the common first-run NVIDIA
         case the torch probe misses.
      3. ``rocm-smi`` (Linux AMD) — same idea for ROCm boxes.

    Everything is wrapped so a missing tool / parse failure just yields
    ``None`` (the recommender then plays safe and never offers Z-Image).
    """
    # 1. Live accelerator torch, if one is actually imported + visible.
    try:  # pragma: no cover - depends on the active overlay
        import torch

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return float(props.total_memory) / (1024**3)
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            props = xpu.get_device_properties(0)
            total = getattr(props, "total_memory", None)
            if total:
                return float(total) / (1024**3)
    except Exception:
        pass

    # 2. nvidia-smi: total VRAM regardless of which torch is active.
    mib = _query_nvidia_smi_vram_mib()
    if mib is not None:
        return mib / 1024.0

    # 3. rocm-smi on Linux AMD (best-effort).
    mib = _query_rocm_smi_vram_mib()
    if mib is not None:
        return mib / 1024.0

    return None


def _query_nvidia_smi_vram_mib() -> float | None:
    """Largest ``memory.total`` (MiB) nvidia-smi reports, or ``None``."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
    except Exception:
        return None
    best: float | None = None
    for line in out.stdout.splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        try:
            val = float(token)
        except ValueError:
            continue
        if best is None or val > best:
            best = val
    return best


def _query_rocm_smi_vram_mib() -> float | None:
    """Largest VRAM total (MiB) rocm-smi reports on Linux AMD, or
    ``None``. Parses the ``--showmeminfo vram --json`` output; any
    shape drift / missing tool yields ``None`` rather than raising."""
    import json
    import shutil
    import subprocess

    exe = shutil.which("rocm-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
        data = json.loads(out.stdout or "{}")
    except Exception:
        return None
    best: float | None = None
    for card in data.values() if isinstance(data, dict) else []:
        if not isinstance(card, dict):
            continue
        for key, raw in card.items():
            if "vram" not in key.lower() or "total" not in key.lower():
                continue
            try:
                # rocm-smi reports VRAM totals in bytes.
                val = float(raw) / (1024**2)
            except (TypeError, ValueError):
                continue
            if best is None or val > best:
                best = val
    return best


def _vendor_present(
    *,
    cli: tuple[str, list[str]] | None,
    wmi_match: str | tuple[str, ...],
    pci_match: tuple[str, ...] = (),
) -> bool:
    """Shared probe for GPU vendor detection.

    Order of preference per platform:

      * **Linux** — read ``/sys/class/drm/*/device/vendor`` (instant,
        kernel-side PCI vendor ID, no subprocess). Falls back to
        ``cli`` only if the sysfs probe yields nothing.
      * **Windows** — try the vendor CLI first (cheap on a healthy
        driver install), then WMI via PowerShell.
      * **macOS** — vendor CLIs only; WMI doesn't apply.

    Critical bug class this guards against: a broken ROCm install
    can make ``rocminfo`` hang during HIP runtime probe, and even
    with ``timeout=2.0`` the SIGTERM may not free the grandchild
    cleanly on every Linux distro. The sysfs path entirely avoids
    spawning a subprocess on Linux, which is where the
    "AppImage hangs on Linux/AMD" user report most likely lives.
    Vendor detection is best-effort; a wrong answer only softens
    the diagnostic message, never blocks a render.
    """
    import shutil
    import subprocess

    # Linux: PCI vendor IDs are exposed at /sys/class/drm/*/device/vendor
    # as 0x10de (NVIDIA), 0x1002 (AMD), 0x8086 (Intel). The sysfs
    # read takes microseconds and CANNOT hang.
    if sys.platform == "linux" and pci_match:
        try:
            from glob import glob

            for vendor_file in glob("/sys/class/drm/card?/device/vendor"):
                try:
                    with open(vendor_file, encoding="ascii") as f:
                        vid = f.read().strip().lower()
                    if vid in pci_match:
                        return True
                except OSError:
                    continue
        except Exception:
            pass

    if cli is not None:
        tool, args = cli
        path = shutil.which(tool)
        if path:
            try:
                # 1.0 s ceiling — vendor probes are diagnostic; never
                # delay backend startup past the LUCIDIUM_WS_PORT
                # announcement that Electron is waiting on.
                result = subprocess.run(
                    [path, *args],
                    capture_output=True,
                    timeout=1.0,
                )
                if result.returncode == 0 and result.stdout:
                    return True
            except Exception:
                pass

    if sys.platform != "win32":
        return False

    try:
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if ps is None:
            return False
        result = subprocess.run(
            [
                ps,
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            timeout=2.0,
        )
        out = result.stdout or b""
        needles = (wmi_match,) if isinstance(wmi_match, str) else wmi_match
        return any(needle.encode("utf-8") in out for needle in needles)
    except Exception:
        return False


def _is_z_image_model_path(model_path: Path) -> bool:
    """Pre-load filename sniff: does this checkpoint look like a
    Z-Image (Alibaba Tongyi) single-file model? Used by the factory
    to pick ``ZImagePipeline`` over ``StableDiffusionXLPipeline``
    before the safetensors header is read.

    Z-Image safetensors typically ship as ``Z-Image-Turbo.safetensors``
    or ``z-image-turbo-*.safetensors``. The substring match also
    accepts the upstream ``Tongyi-MAI`` repo prefix in case a user
    drops the file with the HF repo name preserved.
    """
    name = model_path.name.lower()
    return "z-image" in name or "zimage" in name or "tongyi-mai" in name


def _is_qwen_model_path(model_path: Path) -> bool:
    """Pre-load filename sniff: does this checkpoint look like a
    Qwen-Image single-file transformer? Used by the factory to pick
    ``QwenImagePipeline`` over ``StableDiffusionXLPipeline`` before the
    safetensors header is read.

    Qwen-Image ships ONLY a transformer (denoiser) in its single-file
    checkpoints — the Comfy-Org repackage names them
    ``qwen_image_*.safetensors``, and the one-click download saves as
    ``Qwen-Image.safetensors``. The text encoder (Qwen2.5-VL) + VAE are
    pulled from the upstream ``Qwen/Qwen-Image`` HF repo at load time
    (see :func:`_load_qwen_pipeline`). The substring match accepts any
    spelling a user might drop the file under.
    """
    name = model_path.name.lower()
    return "qwen-image" in name or "qwen_image" in name or "qwenimage" in name


def _is_sdxl_family_path(model_path: Path) -> bool:
    """True when the checkpoint would load through the SDXL branch of
    :func:`_default_pipeline_factory` — i.e. it is none of the
    special-cased families. Used by the DirectML fallback, which needs a
    checkpoint that actually runs on a small non-bf16 GPU.

    ``_is_krea_model_path`` reads the file header, so keep this ordered
    with the cheap filename sniffs first.
    """
    if _is_z_image_model_path(model_path) or _is_qwen_model_path(model_path):
        return False
    return not _is_krea_model_path(model_path)


def _load_z_image_pipeline(model_path: Path, *, device: str | None) -> Any:
    """Load a Z-Image pipeline that takes its transformer weights from
    the user's local single-file safetensors and pulls every other
    component (Qwen3 text encoder, VAE, tokenizer, scheduler config)
    from the upstream ``Tongyi-MAI/Z-Image-Turbo`` HF repo.

    ``ZImagePipeline.from_single_file`` alone fails on Z-Image
    safetensors with::

        SingleFileComponentError: Failed to load Qwen3Model. Weights
        for this component appear to be missing in the checkpoint.

    The single-file checkpoint carries the transformer (denoiser)
    weights only — the text encoder is a separate ~6 GB Qwen3 model
    and the VAE is its own ~300 MB file. Both have to be preloaded
    from the HF repo and handed in as kwargs. Caching them under the
    user's HF cache dir (the default behaviour of ``from_pretrained``)
    means subsequent loads are local-disk reads, not network fetches.

    Sequential CPU offload (``enable_model_cpu_offload``):

      Z-Image-Turbo all-resident totals ~18 GiB (transformer 11.5 GiB
      + Qwen3 text encoder 6 GiB + VAE 0.3 GiB). On a 24 GiB card the
      remaining headroom after one render is ~1.8 GiB — enough for
      a 1024² activation but too little for the 1536x1024 background
      bucket, which the user hit as CUDA-OOM.

      ``model_cpu_offload_seq = "text_encoder->transformer->vae"`` is
      already declared on ``ZImagePipeline``; calling
      ``enable_model_cpu_offload`` rotates components through the GPU
      in that order at inference time. Peak VRAM drops to
      ``max(component_size)`` ≈ the 11.5 GiB transformer instead of
      the sum, so a 1536x1024 background fits with comfortable
      headroom on a 24 GiB card. The per-render cost is one CPU↔GPU
      copy per component (~2 s extra) — a trade we have to take
      because the user's "no shuttling" preference can't coexist with
      a 24 GiB budget for this pipeline.

      IMPORTANT: ``enable_model_cpu_offload`` manages device
      placement itself. The caller MUST NOT also call
      ``pipeline.to(device)`` afterwards — doing so eagerly moves
      every component to GPU and silently undoes the offload. The
      factory honours this by skipping its usual ``.to`` step for the
      Z-Image branch.
    """
    from diffusers import AutoencoderKL, ZImagePipeline
    from transformers import AutoModel, AutoTokenizer

    hf_repo = "Tongyi-MAI/Z-Image-Turbo"
    dtype = _resolve_torch_dtype(prefer_bfloat16=True)

    text_encoder = AutoModel.from_pretrained(
        hf_repo,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo,
        subfolder="tokenizer",
    )
    vae = AutoencoderKL.from_pretrained(  # type: ignore[no-untyped-call]
        hf_repo,
        subfolder="vae",
        torch_dtype=dtype,
    )

    pipeline = ZImagePipeline.from_single_file(
        str(model_path),
        torch_dtype=dtype,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
    )

    # Activate sequential CPU offload on accelerator hardware. On
    # CPU-only installs there's nothing to offload TO. Across
    # accelerator families:
    #
    #   * CUDA / ROCm: fully supported; ideal trade for 24 GiB cards.
    #   * XPU (Intel Arc): supported in diffusers 0.27+.
    #   * MPS (Apple Silicon): historically NOT supported —
    #     ``enable_model_cpu_offload`` raises NotImplementedError /
    #     RuntimeError before 0.30, and even on supported builds
    #     the offload hooks interact poorly with Metal kernel
    #     launches. Fall back to plain ``.to(device)`` (all
    #     components resident) when offload errors out; on M-series
    #     hardware the unified-memory architecture means peak GPU
    #     pressure isn't the OOM trap it is on a discrete card.
    resolved_device = device or detect_compute_device()[0]
    if resolved_device == "cpu":
        return pipeline
    # ``_torch_device_arg`` turns the "dml" sentinel into the real
    # ``torch_directml.device()`` handle; pass-through for everything
    # else. (In practice Z-Image never reaches DirectML — the
    # model-resolution path in ``generate`` routes Z-Image away from
    # DML before we ever load it — but keep the placement correct in
    # case a future caller loads Z-Image on DML directly.)
    device_arg = _torch_device_arg(resolved_device)
    try:
        pipeline.enable_model_cpu_offload(device=device_arg)
    except TypeError:
        # Older diffusers releases (<0.26) don't accept the explicit
        # ``device`` kwarg. The no-arg form picks the first CUDA
        # device, which is correct on every supported pre-XPU setup.
        try:
            pipeline.enable_model_cpu_offload()
        except Exception:
            pipeline = pipeline.to(device_arg)
    except Exception:
        _log.warning(
            "embedded backend: enable_model_cpu_offload not supported on "
            "device=%s; falling back to all-resident placement. Peak "
            "VRAM will be the sum of (transformer + text_encoder + "
            "vae) instead of max.",
            resolved_device,
            exc_info=True,
        )
        pipeline = pipeline.to(device_arg)
    return pipeline


# Qwen-Image-Lightning few-step distill LoRA, fused into the base
# transformer at load (see :func:`_apply_qwen_lightning_lora`). The 8-step
# V2.0 LoRA at cfg 1.0 is a good speed/quality balance and is what makes
# Qwen render in seconds rather than minutes (8 forwards vs ~60).
_QWEN_LIGHTNING_LORA_REPO = "lightx2v/Qwen-Image-Lightning"
_QWEN_LIGHTNING_LORA_FILE = "Qwen-Image-Lightning-8steps-V2.0.safetensors"
# Per-variant sampling recipe: (num_inference_steps, true_cfg_scale).
#   * Lightning distill — few steps, CFG OFF (true_cfg_scale 1.0 → one
#     transformer forward per step).
#   * Undistilled base — more steps + real CFG (two forwards per step).
_QWEN_LIGHTNING_RECIPE: tuple[int, float] = (8, 1.0)
_QWEN_BASE_RECIPE: tuple[int, float] = (20, 4.0)
# Pipeline attribute carrying the resolved recipe so ``_run_pipeline`` /
# ``_run_qwen_img2img`` sample correctly. Set at load; defaults to the
# Lightning recipe (the shipped configuration) when absent.
_QWEN_RECIPE_ATTR = "_lucidium_qwen_recipe"


def _qwen_recipe(pipeline: Any) -> tuple[int, float]:
    """Return ``(num_inference_steps, true_cfg_scale)`` for this Qwen
    pipeline. Reads the flag stamped at load time; defaults to the
    Lightning recipe (the engine applies the Lightning LoRA by default)."""
    recipe = getattr(pipeline, _QWEN_RECIPE_ATTR, None)
    if isinstance(recipe, tuple) and len(recipe) == 2:
        return recipe
    return _QWEN_LIGHTNING_RECIPE


def _load_qwen_pipeline(model_path: Path, *, device: str | None) -> Any:
    """Load a Qwen-Image pipeline whose transformer (denoiser) weights
    come from the user's local single-file safetensors and whose every
    other component (Qwen2.5-VL text encoder, VAE, tokenizer, scheduler)
    is pulled from the upstream ``Qwen/Qwen-Image`` HF repo.

    Why assemble it by hand instead of ``from_pretrained`` /
    ``from_single_file``? ``QwenImagePipeline`` has no single-file loader
    (unlike Z-Image), and the single-file checkpoints in circulation
    (the Comfy-Org repackage, the one-click download) carry the
    transformer ONLY. So we load the transformer via the model-level
    ``from_single_file`` and hand-build the pipeline with the remaining
    components fetched from the diffusers repo — mirroring exactly what
    :func:`_load_z_image_pipeline` does for Z-Image. The HF components
    cache under the user's HF cache dir, so subsequent loads are local-
    disk reads, not network fetches.

    Residency strategy (see :func:`_apply_qwen_offload`): Qwen-Image's
    dense 60-layer transformer is ~40 GiB in bf16, but it does NOT need
    to be held resident at that size. ComfyUI runs the very same
    ``qwen_image_fp8_e4m3fn`` checkpoint on a 24 GiB 4090 by keeping the
    transformer in **fp8** (~20 GiB). We match that: torchao quantizes the
    transformer to native fp8 (GEMMs on the GPU's fp8 tensor cores via
    ``_scaled_mm``, ~5-7x faster per forward than upcasting), with
    diffusers' layerwise casting as the fallback where torchao / fp8
    hardware is absent. ``enable_model_cpu_offload`` then rotates whole
    components through the GPU so the fp8 transformer is resident during
    denoise on a 24 GiB card (the fast, default path on a 4090). Cards
    too tight even for the fp8 transformer fall back to block-level
    streaming, then sequential offload. Combined with the fused Lightning
    few-step distill LoRA this renders in seconds, not minutes. Either
    way the caller MUST NOT call ``pipeline.to(device)``
    afterwards — the offload mechanisms manage placement themselves and
    an eager ``.to`` undoes them — so the factory skips its usual ``.to``
    step for this branch.
    """
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        QwenImagePipeline,
        QwenImageTransformer2DModel,
    )

    hf_repo = "Qwen/Qwen-Image"
    dtype = _resolve_torch_dtype(prefer_bfloat16=True)

    # ``QwenImageTransformer2DModel`` has no default config repo in
    # diffusers' single-file map, so ``from_single_file`` would otherwise
    # fall back to the generic ``stable-diffusion-v1-5`` config and 404.
    # Point it at the real transformer config in the diffusers repo.
    transformer = QwenImageTransformer2DModel.from_single_file(
        str(model_path),
        torch_dtype=dtype,
        config=hf_repo,
        subfolder="transformer",
    )
    # The text encoder is Qwen2.5-VL; the pipeline calls it with
    # ``output_hidden_states=True`` and reads ``.hidden_states[-1]``.
    # Use the concrete class the pipeline type-hints, falling back to
    # ``AutoModel`` for transformers builds that expose it under a
    # different name (both shapes return ``.hidden_states``).
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_repo,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
    except Exception:
        from transformers import AutoModel

        text_encoder = AutoModel.from_pretrained(
            hf_repo,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_repo, subfolder="tokenizer")
    vae = AutoencoderKLQwenImage.from_pretrained(  # type: ignore[no-untyped-call]
        hf_repo,
        subfolder="vae",
        torch_dtype=dtype,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(  # type: ignore[no-untyped-call]
        hf_repo,
        subfolder="scheduler",
    )

    pipeline = QwenImagePipeline(  # type: ignore[no-untyped-call]
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )

    # Fuse the Lightning few-step distill LoRA BEFORE quantization (it
    # merges into the bf16 weights; quantizing first would freeze them).
    # Records the sampling recipe on the pipeline for _run_pipeline.
    lightning = _apply_qwen_lightning_lora(pipeline, model_path)
    setattr(
        pipeline,
        _QWEN_RECIPE_ATTR,
        _QWEN_LIGHTNING_RECIPE if lightning else _QWEN_BASE_RECIPE,
    )

    resolved_device = device or detect_compute_device()[0]
    if resolved_device == "cpu":
        return pipeline
    device_arg = _torch_device_arg(resolved_device)
    _apply_qwen_offload(pipeline, resolved_device, device_arg)
    return pipeline


def _apply_qwen_lightning_lora(pipeline: Any, model_path: Path) -> bool:
    """Fuse the Qwen-Image-Lightning few-step distill LoRA into the
    transformer so renders need ~8 steps at cfg 1.0 instead of ~20-30
    with CFG. Returns True when the Lightning recipe should be used.

    * If the checkpoint filename marks it as ALREADY few-step — a
      pre-merged Lightning checkpoint, a distilled checkpoint (the
      one-click download ships ``qwen_image_distill_full_fp8``), or a
      turbo / N-step build — we skip the LoRA fetch (the distillation is
      baked in; fusing on top would double-distill) and just use the
      few-step recipe.
    * Otherwise (a plain base Qwen checkpoint) we fetch the 8-step LoRA
      from HF, fuse it, and drop the adapter. Needs the PEFT backend; if
      PEFT is missing or the fuse fails we return False and the caller
      falls back to the slower base-model recipe rather than producing
      broken few-step renders.
    """
    name = model_path.name.lower()
    if any(tag in name for tag in ("lightning", "distill", "turbo", "step")):
        _log.info(
            "embedded backend: %s is already a few-step (distilled/Lightning) "
            "checkpoint; using the few-step recipe without fetching a LoRA",
            model_path.name,
        )
        return True
    try:
        pipeline.load_lora_weights(
            _QWEN_LIGHTNING_LORA_REPO,
            weight_name=_QWEN_LIGHTNING_LORA_FILE,
        )
        pipeline.fuse_lora()
        pipeline.unload_lora_weights()
    except Exception:
        _log.warning(
            "embedded backend: could not fuse the Qwen-Image-Lightning LoRA "
            "(install the 'peft' backend for few-step renders); falling back "
            "to the slower base-model recipe",
            exc_info=True,
        )
        return False
    _log.info(
        "embedded backend: fused Qwen-Image-Lightning (%s) — few-step recipe",
        _QWEN_LIGHTNING_LORA_FILE,
    )
    return True


# Resident VRAM needed once the transformer is stored in fp8 (the ComfyUI
# scheme — see ``_try_qwen_fp8_casting``): the ~40 GiB bf16 transformer
# becomes ~20 GiB, so component-level ``enable_model_cpu_offload`` (peak =
# largest single component) fits a 24 GiB card. This mirrors ComfyUI, which
# runs ``qwen_image_fp8_e4m3fn`` at ~86% of a 24 GiB 4090.
_QWEN_FP8_RESIDENT_MIN_VRAM_GB: float = 22.0
# Resident VRAM needed WITHOUT fp8 casting (transformer stays ~40 GiB bf16).
# Only a big card clears this; smaller ones fall back to block streaming.
_QWEN_RESIDENT_MIN_VRAM_GB: float = 44.0


def _apply_qwen_offload(pipeline: Any, resolved_device: str, device_arg: Any) -> None:
    """Place a Qwen-Image pipeline for inference using the residency
    strategy that fits the active GPU (see :func:`_load_qwen_pipeline`).

    Steps:
      0. Shrink the transformer to fp8 — torchao native fp8 (fast) or
         diffusers layerwise casting — so the ~40 GiB bf16 transformer
         becomes ~20 GiB and fits a 24 GiB card.
      1. Enough VRAM for the fp8 transformer → CPU-ENCODE RESIDENT mode
         (see :func:`_setup_qwen_cpu_encode`): transformer + VAE resident
         on the GPU, text encoder on the CPU. The render encodes the
         prompt on the CPU and passes the embeds in, so nothing has to be
         offloaded mid-pipeline. This is the 24 GiB fast path (~10 s/
         render) and, having NO diffusers offload, works identically for
         text2img and img2img (``enable_model_cpu_offload``'s fixed
         eviction chain breaks on img2img's vae-encode-first order, and
         torchao's fp8 tensors can't be moved across devices at all).
      2. Too tight / non-CUDA → block-level group offload, then
         sequential offload, then ``enable_model_cpu_offload`` — the
         streaming fallbacks for cards that can't hold the transformer.
    """
    # Prefer torchao native fp8 (fast — fp8 tensor cores via _scaled_mm);
    # fall back to diffusers layerwise casting (fp8 storage, bf16 compute)
    # where torchao is absent or the GPU lacks fp8 (pre-Ada).
    fp8 = _try_qwen_torchao_fp8(pipeline, resolved_device, device_arg) or _try_qwen_fp8_casting(
        pipeline, resolved_device
    )
    vram = detect_total_vram_gb()
    # Preferred: fp8 transformer resident + CPU text-encode. Only on CUDA
    # (the CPU-encode dance assumes a single GPU execution device).
    if (
        fp8
        and resolved_device == "cuda"
        and (vram is None or vram >= _QWEN_FP8_RESIDENT_MIN_VRAM_GB)
    ):
        if _setup_qwen_cpu_encode(pipeline, device_arg, resolved_device):
            return
    # The un-cast (bf16) path needs a ~44 GiB card to hold a component
    # resident; on such a card component-level offload is fine.
    resident_floor = _QWEN_FP8_RESIDENT_MIN_VRAM_GB if fp8 else _QWEN_RESIDENT_MIN_VRAM_GB
    if vram is None or vram >= resident_floor:
        if _try_qwen_model_offload(pipeline, device_arg, resolved_device):
            return
    # Tighter than the resident floor (or model-offload failed): stream.
    if _try_qwen_block_offload(pipeline, device_arg, resolved_device):
        return
    try:
        pipeline.enable_sequential_cpu_offload(device=device_arg)
        _log.info(
            "embedded backend: Qwen-Image using sequential CPU offload on "
            "device=%s (fits low VRAM; slow per step)",
            resolved_device,
        )
        return
    except Exception:
        _log.warning(
            "embedded backend: sequential CPU offload failed for Qwen-Image; "
            "falling back to model_cpu_offload (may OOM on a small card)",
            exc_info=True,
        )
    _try_qwen_model_offload(pipeline, device_arg, resolved_device)


# Pipeline attribute marking CPU-encode-resident mode (see
# _setup_qwen_cpu_encode). When set, _run_pipeline / _run_qwen_img2img
# encode the prompt on the CPU text encoder and pass embeds in.
_QWEN_CPU_ENCODE_ATTR = "_lucidium_qwen_cpu_encode"


def _setup_qwen_cpu_encode(
    pipeline: Any,
    device_arg: Any,
    resolved_device: str,
) -> bool:
    """Qwen-Image's CPU-encode-resident placement. Thin wrapper over the
    family-neutral :func:`_setup_cpu_encode_resident`."""
    return _setup_cpu_encode_resident(
        pipeline,
        device_arg,
        resolved_device,
        attr=_QWEN_CPU_ENCODE_ATTR,
        family="Qwen-Image",
    )


def _setup_cpu_encode_resident(
    pipeline: Any,
    device_arg: Any,
    resolved_device: str,
    *,
    attr: str,
    family: str,
) -> bool:
    """Place a big-transformer pipeline for the CPU-encode-resident fast
    path: transformer + VAE resident on the GPU, text encoder parked on
    the CPU, VAE tiling on. No diffusers offload is installed — the
    render encodes the prompt on the CPU encoder and hands the embeds to
    the pipeline (see :func:`_cpu_encode_prompts`).

    Shared by Qwen-Image and Krea 2: both pair a torchao-fp8 transformer
    with a multi-GB Qwen-family text encoder and a 3D-conv VAE, so the
    placement decision is identical. ``attr`` is the pipeline flag
    ``_run_pipeline`` reads to know it must CPU-encode; ``family`` only
    labels the log line.

    Why this shape on a 24 GiB card:
      * The fp8 transformer is ~12-20 GiB and torchao's fp8 tensors CANNOT
        be moved across devices (``.to`` strands the weight scale → a
        ``_scaled_mm`` device mismatch), so the transformer must stay
        resident.
      * The multi-GB text encoder therefore can't be co-resident, and
        ``enable_model_cpu_offload`` (which would swap it) both breaks on
        img2img's call order and can't move the torchao transformer. So
        we never put the text encoder on the GPU at all — it encodes on
        the CPU (~5 s) and we move the small embeds to the GPU.
      * The Qwen-Image VAE (which Krea 2 also uses) is 3D-conv and OOMs
        at 832×1216 next to the resident transformer without tiling, so
        tiling + slicing are enabled.

    Returns False (caller falls back to streaming offload) if placement
    errors.
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return False
    try:
        transformer.to(device_arg)  # resident (torchao already on-device)
        vae = getattr(pipeline, "vae", None)
        if vae is not None:
            vae.to(device_arg)
            # Tame the 3D-conv VAE's peak so it fits beside the resident
            # transformer. Prefer the module-level API (the pipeline-level
            # wrappers are deprecated for QwenImage).
            for meth in ("enable_tiling", "enable_slicing"):
                fn = getattr(vae, meth, None)
                if callable(fn):
                    fn()
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None:
            text_encoder.to("cpu")  # encode on CPU; never resident on GPU
    except Exception:
        _log.warning(
            "embedded backend: %s CPU-encode-resident placement failed; "
            "falling back to streaming offload",
            family,
            exc_info=True,
        )
        return False
    try:
        setattr(pipeline, attr, True)
    except (AttributeError, TypeError):
        pass
    _log.info(
        "embedded backend: %s transformer + VAE resident on %s, "
        "text encoder CPU-encoding (order-independent; ~10 s/render, "
        "fits 24 GiB)",
        family,
        resolved_device,
    )
    return True


def _qwen_cpu_encode(
    pipeline: Any,
    positive: str,
    negative: str,
    true_cfg: float,
) -> dict[str, Any]:
    """Qwen-Image's CPU prompt encode. The unconditional branch only runs
    when real CFG is on (``true_cfg > 1``); a Lightning render at cfg 1.0
    needs the positive only."""
    return _cpu_encode_prompts(
        pipeline,
        positive,
        negative,
        encode_negative=bool(true_cfg and true_cfg > 1),
    )


def _cpu_encode_prompts(
    pipeline: Any,
    positive: str,
    negative: str,
    *,
    encode_negative: bool,
) -> dict[str, Any]:
    """Encode the prompt(s) on the CPU text encoder and return the
    ``prompt_embeds`` kwargs (moved to the GPU) for a CPU-encode-resident
    pipeline. ``encode_prompt`` returns a ``None`` mask when it's all-ones
    — propagated as-is (the pipeline treats ``None`` as no mask).

    Shared by Qwen-Image and Krea 2: both expose the same
    ``encode_prompt(prompt, device, num_images_per_prompt) ->
    (embeds, mask)`` shape and the same ``prompt_embeds`` /
    ``prompt_embeds_mask`` call kwargs."""
    import torch

    device = torch.device("cpu")

    def _to_gpu(t: Any) -> Any:
        return t.to("cuda") if t is not None else None

    with torch.no_grad():
        pe, pem = pipeline.encode_prompt(
            prompt=positive,
            device=device,
            num_images_per_prompt=1,
        )
        out: dict[str, Any] = {
            "prompt_embeds": _to_gpu(pe),
            "prompt_embeds_mask": _to_gpu(pem),
        }
        if encode_negative:
            ne, nem = pipeline.encode_prompt(
                prompt=negative,
                device=device,
                num_images_per_prompt=1,
            )
            out["negative_prompt_embeds"] = _to_gpu(ne)
            out["negative_prompt_embeds_mask"] = _to_gpu(nem)
    return out


def _run_qwen_call(pipeline: Any, kwargs: dict[str, Any]) -> Any:
    """Invoke a CPU-encode-resident pipeline (Qwen-Image or Krea 2) with
    its text encoder temporarily detached. The text encoder is parked on
    the CPU; left attached it can become the pipeline's device anchor
    (``self.device`` iterates components in an arbitrary set order) and
    make the pipeline build latents on the CPU, mismatching the resident
    CUDA transformer. ``__call__`` never references ``text_encoder`` once
    embeds are passed, so detaching it for the call is safe; restored in
    ``finally``."""
    try:
        saved = pipeline.text_encoder
        pipeline.text_encoder = None
    except (AttributeError, TypeError):
        return pipeline(**kwargs)
    try:
        return pipeline(**kwargs)
    finally:
        pipeline.text_encoder = saved


def _try_qwen_torchao_fp8(
    pipeline: Any,
    resolved_device: str,
    device_arg: Any,
    *,
    family: str = "Qwen-Image",
) -> bool:
    """Quantize the transformer to fp8 with torchao's dynamic
    activation + weight float8 config — the GEMMs then run on the GPU's
    fp8 tensor cores via ``torch._scaled_mm`` (ComfyUI's fp8 speed),
    while the weights drop to ~20 GiB so the 20B model fits a 24 GiB
    card. ~5-7x faster per forward than layerwise casting (which upcasts
    to bf16 for the matmul).

    Returns False — so the caller falls back to layerwise casting — when
    torchao isn't installed, the GPU has no fp8 path (pre-Ada / non-CUDA),
    or quantization errors / OOMs. Only the transformer is quantized; the
    text encoder runs bf16 and is rotated by the offload step.
    """
    if resolved_device not in ("cuda",):
        # fp8 _scaled_mm is a CUDA (Ada/Hopper) path; skip elsewhere.
        return False
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return False
    try:
        from torchao.quantization import (  # type: ignore
            Float8DynamicActivationFloat8WeightConfig,
            quantize_,
        )
    except Exception:
        return False
    try:
        quantize_(
            transformer,
            Float8DynamicActivationFloat8WeightConfig(),
            device=device_arg,
        )
    except Exception:
        _log.warning(
            "embedded backend: torchao fp8 quantization of the %s "
            "transformer failed; falling back to layerwise casting",
            family,
            exc_info=True,
        )
        return False
    _log.info(
        "embedded backend: %s transformer quantized to native fp8 "
        "via torchao (fp8 tensor cores; roughly halves resident VRAM)",
        family,
    )
    return True


def _try_qwen_fp8_casting(pipeline: Any, resolved_device: str) -> bool:
    """Store the transformer (+ text encoder) weights in fp8 with bf16
    compute via diffusers' layerwise casting — the same "fp8 weights,
    upcast per layer at compute" scheme ComfyUI uses to run
    ``qwen_image_fp8_e4m3fn`` on a 24 GiB card. Roughly halves resident
    VRAM (transformer ~40 GiB bf16 → ~20 GiB fp8) at a small quality /
    speed cost. Returns True if it was applied to at least one module.

    No-op (returns False) on CPU, on torch builds without
    ``float8_e4m3fn``, or diffusers builds without ``enable_layerwise_
    casting`` — the caller then sizes its offload strategy for the larger
    bf16 footprint.
    """
    import torch

    if resolved_device == "cpu":
        return False
    storage = getattr(torch, "float8_e4m3fn", None)
    if storage is None:
        return False
    compute = _resolve_torch_dtype(prefer_bfloat16=True)
    applied = 0
    for name in ("transformer", "text_encoder"):
        module = getattr(pipeline, name, None)
        if module is None or not hasattr(module, "enable_layerwise_casting"):
            continue
        try:
            module.enable_layerwise_casting(
                storage_dtype=storage,
                compute_dtype=compute,
            )
            applied += 1
        except Exception:
            _log.warning(
                "embedded backend: fp8 layerwise casting failed for Qwen-Image "
                "%s; keeping bf16 (needs more VRAM)",
                name,
                exc_info=True,
            )
    if applied:
        _log.info(
            "embedded backend: Qwen-Image stored in fp8 (bf16 compute) on "
            "%d module(s) — ~half the resident VRAM, fits a 24 GiB card",
            applied,
        )
    return applied > 0


def _try_qwen_model_offload(
    pipeline: Any,
    device_arg: Any,
    resolved_device: str,
) -> bool:
    """Component-level ``enable_model_cpu_offload``. Returns True on
    success. Tolerates older diffusers that lack the ``device`` kwarg."""
    try:
        pipeline.enable_model_cpu_offload(device=device_arg)
    except TypeError:
        try:
            pipeline.enable_model_cpu_offload()
        except Exception:
            return False
    except Exception:
        return False
    _log.info(
        "embedded backend: Qwen-Image using model CPU offload on device=%s",
        resolved_device,
    )
    return True


def _try_qwen_block_offload(
    pipeline: Any,
    device_arg: Any,
    resolved_device: str,
    *,
    family: str = "Qwen-Image",
) -> bool:
    """Block-level group offload on the transformer + text encoder (the
    two big modules), with a CUDA prefetch stream to hide the extra
    copies; the small VAE is kept resident for a fast decode. Returns
    True on success, False if the diffusers build doesn't support it."""
    import torch

    offload_device = torch.device("cpu")
    # ``use_stream`` overlaps the next block's host→device copy with the
    # current block's compute; only valid on CUDA. record_stream avoids a
    # use-after-free of the streamed buffers.
    use_stream = str(resolved_device) == "cuda"
    streamed = 0
    for name in ("transformer", "text_encoder"):
        module = getattr(pipeline, name, None)
        if module is None or not hasattr(module, "enable_group_offload"):
            continue
        try:
            module.enable_group_offload(
                onload_device=device_arg,
                offload_device=offload_device,
                offload_type="block_level",
                num_blocks_per_group=1,
                use_stream=use_stream,
                record_stream=use_stream,
            )
            streamed += 1
        except Exception:
            _log.warning(
                "embedded backend: group offload failed for %s %s; trying the next strategy",
                family,
                name,
                exc_info=True,
            )
            return False
    if streamed == 0:
        return False
    # Keep the small VAE resident so the decode isn't streamed too.
    vae = getattr(pipeline, "vae", None)
    if vae is not None:
        try:
            vae.to(device_arg)
        except Exception:
            pass
    _log.info(
        "embedded backend: %s using block-level group offload on "
        "device=%s (streamed %d module(s); fits a 24 GiB-class card)",
        family,
        resolved_device,
        streamed,
    )
    return True


# HF repos the non-transformer Krea 2 components come from. Krea's own
# ``krea/Krea-2-*`` repos are GATED (a 401 without an accepted licence),
# so we deliberately DON'T depend on them: Krea 2 pairs the stock
# Qwen-Image VAE with a stock Qwen3-VL-4B text encoder, both ungated and
# both already cached for players who use the Qwen-Image checkpoint.
# This is the same pairing Comfy-Org ships in ``Comfy-Org/Krea-2``.
_KREA_TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-4B-Instruct"
_KREA_VAE_REPO = "Qwen/Qwen-Image"

# Per-variant sampling recipe: (num_inference_steps, guidance_scale).
# Krea 2's CFG convention is ``cond + scale * (cond - uncond)``, so a
# scale of 0.0 means guidance OFF (one transformer forward per step).
#   * Turbo / TDM distill — 8 steps, guidance OFF.
#   * Raw (midtrain) base — 28 steps with real CFG at 4.5.
_KREA_DISTILLED_RECIPE: tuple[int, float] = (8, 0.0)
_KREA_RAW_RECIPE: tuple[int, float] = (28, 4.5)
# Pipeline attributes carrying the resolved recipe / placement mode so
# ``_run_pipeline`` samples correctly.
_KREA_RECIPE_ATTR = "_lucidium_krea_recipe"
_KREA_CPU_ENCODE_ATTR = "_lucidium_krea_cpu_encode"

# Scheduler config for Krea 2's resolution-aware exponential time shift.
# The upstream scheduler config lives only in the gated repo, so it's
# reproduced here from the values the Krea 2 pipeline documents.
_KREA_SCHEDULER_CONFIG: dict[str, Any] = {
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 6400,
}

# Resident VRAM needed for the fp8 Krea 2 transformer (~12 GiB) plus the
# VAE and one render's activations, with the text encoder CPU-parked.
# Lower than Qwen's floor because Krea 2's transformer is 12B, not 20B.
_KREA_FP8_RESIDENT_MIN_VRAM_GB: float = 16.0
# Resident floor WITHOUT fp8 — the transformer stays ~24 GiB in bf16.
_KREA_RESIDENT_MIN_VRAM_GB: float = 32.0


def _krea_recipe(pipeline: Any) -> tuple[int, float]:
    """Return ``(num_inference_steps, guidance_scale)`` for this Krea 2
    pipeline. Reads the flag stamped at load time; defaults to the
    distilled recipe, matching :func:`krea_checkpoint.krea_is_distilled`."""
    recipe = getattr(pipeline, _KREA_RECIPE_ATTR, None)
    if isinstance(recipe, tuple) and len(recipe) == 2:
        return recipe
    return _KREA_DISTILLED_RECIPE


def _is_krea_model_path(model_path: Path) -> bool:
    """Does this checkpoint hold a Krea 2 transformer?

    Unlike the Z-Image / Qwen sniffs this reads the safetensors HEADER
    rather than trusting the filename: Krea 2 finetunes on Civitai carry
    arbitrary names (``krea2GPTGrandPTruth_gptINT4INT8Convrot`` is a real
    one) and the header check is both cheap — 8 bytes plus a JSON blob,
    no tensor data mapped — and exact. The filename is used only as a
    fast pre-filter so we don't open every SDXL checkpoint in the folder.
    """
    from .krea_checkpoint import is_krea_state_dict, read_safetensors_header

    if "krea" not in model_path.name.lower():
        return False
    if model_path.suffix.lower() != ".safetensors":
        return False
    try:
        return is_krea_state_dict(list(read_safetensors_header(model_path).keys()))
    except Exception:
        _log.debug(
            "embedded backend: could not read the safetensors header of %s "
            "while sniffing for Krea 2",
            model_path,
            exc_info=True,
        )
        return False


def _load_krea_pipeline(model_path: Path, *, device: str | None) -> Any:
    """Load a Krea 2 pipeline whose transformer comes from the user's
    local single-file safetensors and whose remaining components
    (Qwen3-VL-4B text encoder, Qwen-Image VAE, tokenizer, scheduler) are
    pulled from ungated upstream HF repos.

    Krea 2 needs more assembly than Z-Image or Qwen-Image because
    diffusers ships ``Krea2Pipeline`` (0.39+) with NO single-file loader
    at all — not even at the model level. Every Krea 2 checkpoint in
    circulation is a ComfyUI-format transformer under the reference
    implementation's key names, in one of three quantizations. So we
    derive the config from the tensor shapes, rename + dequantize the
    weights (see :mod:`krea_checkpoint`), and load them into an
    empty-initialised ``Krea2Transformer2DModel``.

    Residency (see :func:`_apply_krea_offload`): the 12B transformer is
    ~24 GiB in bf16 — too big to hold resident on a 24 GiB card next to
    the text encoder. torchao quantizes it to native fp8 (~12 GiB, GEMMs
    on the fp8 tensor cores) and the text encoder stays on the CPU, which
    is comfortably the fast path on a 4090. The caller MUST NOT
    ``pipeline.to(device)`` afterwards — the placement is managed here.
    """
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        Krea2Pipeline,
        Krea2Transformer2DModel,
    )
    from transformers import (
        AutoTokenizer,
        Qwen3VLForConditionalGeneration,
    )

    from .krea_checkpoint import (
        KreaCheckpointError,
        infer_krea_config,
        krea_is_distilled,
        krea_rotation_is_folded,
        load_krea_transformer_state_dict,
        read_safetensors_header,
    )

    dtype = _resolve_torch_dtype(prefer_bfloat16=True)
    header = read_safetensors_header(model_path)
    # Checked before the 3-minute weight load, because the failure is
    # otherwise silent: a rotated checkpoint loads and samples happily
    # and decodes to noise.
    if krea_rotation_is_folded(model_path, header):
        raise KreaCheckpointError(
            f"{model_path.name} is a rotated ('convrot') int8 Krea 2 checkpoint, "
            "which Lucidium can't run: the quantization folds a Hadamard rotation "
            "into the weights and expects the inference engine to rotate "
            "activations to match, but the rotation isn't stored in the file. Use "
            "an fp8_scaled or bf16 build of the same model instead (e.g. "
            "krea2_turbo_fp8_scaled.safetensors)."
        )
    config = infer_krea_config(header, path_name=model_path.name)
    hidden_size = config["attention_head_dim"] * config["num_attention_heads"]
    _log.info(
        "embedded backend: loading Krea 2 transformer %s (%d blocks, hidden %d, %d text layers)",
        model_path.name,
        config["num_layers"],
        hidden_size,
        config["num_text_layers"],
    )

    transformer = _build_krea_transformer(
        Krea2Transformer2DModel,
        config,
        model_path,
        dtype=dtype,
        hidden_size=hidden_size,
        load_state_dict=load_krea_transformer_state_dict,
    )

    # Qwen3-VL-4B, tapped at 12 intermediate layers by the pipeline. The
    # multimodal wrapper is never used — only the text decoder stack —
    # but the pipeline indexes ``outputs.hidden_states``, which requires
    # the ``Qwen3VLModel`` class the pipeline type-hints.
    #
    # It MUST be reached through the generation wrapper. The upstream
    # checkpoint is saved from ``Qwen3VLForConditionalGeneration``, so
    # every tensor is stored under a ``model.`` prefix; loading it with
    # ``Qwen3VLModel.from_pretrained`` matches NOTHING and silently
    # returns a randomly initialised encoder (transformers only warns).
    # The symptom is not a crash — it's a render that decodes to
    # saturated blocky noise, because the transformer is conditioned on
    # garbage. ``.model`` is the same ``Qwen3VLModel``, correctly loaded.
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        _KREA_TEXT_ENCODER_REPO,
        dtype=dtype,
    ).model
    tokenizer = AutoTokenizer.from_pretrained(_KREA_TEXT_ENCODER_REPO)
    vae = AutoencoderKLQwenImage.from_pretrained(  # type: ignore[no-untyped-call]
        _KREA_VAE_REPO,
        subfolder="vae",
        torch_dtype=dtype,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(**_KREA_SCHEDULER_CONFIG)  # type: ignore[no-untyped-call]

    distilled = krea_is_distilled(model_path)
    pipeline = Krea2Pipeline(  # type: ignore[no-untyped-call]
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=distilled,
    )
    setattr(
        pipeline,
        _KREA_RECIPE_ATTR,
        _KREA_DISTILLED_RECIPE if distilled else _KREA_RAW_RECIPE,
    )
    _log.info(
        "embedded backend: Krea 2 %s recipe for %s (%d steps, guidance %.1f)",
        "distilled" if distilled else "raw/base",
        model_path.name,
        *(_KREA_DISTILLED_RECIPE if distilled else _KREA_RAW_RECIPE),
    )

    resolved_device = device or detect_compute_device()[0]
    if resolved_device == "cpu":
        return pipeline
    _apply_krea_offload(pipeline, resolved_device, _torch_device_arg(resolved_device))
    return pipeline


def _build_krea_transformer(
    model_cls: Any,
    config: dict[str, Any],
    model_path: Path,
    *,
    dtype: Any,
    hidden_size: int,
    load_state_dict: Any,
) -> Any:
    """Instantiate ``Krea2Transformer2DModel`` without allocating a
    throwaway 24 GiB of random weights, then fill it from the converted
    checkpoint.

    ``init_empty_weights`` puts the parameters on the meta device, so
    ``load_state_dict(..., assign=True)`` adopts the loaded tensors
    directly instead of copying into pre-allocated storage — one copy of
    the weights in host RAM rather than two.

    The RMSNorm scales are then restored to float32: diffusers declares
    them in ``_keep_in_fp32_modules`` (the norm runs in float32 and the
    ``weight + 1.0`` centring is precision-sensitive), and that
    convention is applied by ``from_pretrained``, which we bypass.
    """
    import torch
    from accelerate import init_empty_weights  # type: ignore

    with init_empty_weights():
        transformer = model_cls(**config)

    state_dict = load_krea_state_dict_checked(
        load_state_dict,
        model_path,
        dtype=dtype,
        hidden_size=hidden_size,
    )
    missing, unexpected = transformer.load_state_dict(
        state_dict,
        strict=False,
        assign=True,
    )
    if missing or unexpected:
        from .krea_checkpoint import KreaCheckpointError

        raise KreaCheckpointError(
            f"{model_path.name} does not match Krea2Transformer2DModel: "
            f"{len(missing)} missing tensor(s) {sorted(missing)[:4]}, "
            f"{len(unexpected)} unexpected {sorted(unexpected)[:4]}"
        )
    del state_dict

    for name, module in transformer.named_modules():
        if name.rsplit(".", 1)[-1] in ("norm", "norm1", "norm2", "norm_q", "norm_k"):
            module.to(torch.float32)
    return transformer


def load_krea_state_dict_checked(
    loader: Callable[..., dict[str, Any]],
    model_path: Path,
    *,
    dtype: Any,
    hidden_size: int,
) -> dict[str, Any]:
    """Call the checkpoint loader, translating any failure into a
    ``ProviderUnreachableError`` that names the file — a raw
    ``KreaCheckpointError`` would otherwise surface to the player as an
    unexplained load failure."""
    try:
        return loader(model_path, dtype=dtype, hidden_size=hidden_size)
    except Exception as exc:
        raise ProviderUnreachableError(
            f"failed to read the Krea 2 checkpoint {model_path.name}: {exc}"
        ) from exc


def _apply_krea_offload(
    pipeline: Any,
    resolved_device: str,
    device_arg: Any,
) -> None:
    """Place a Krea 2 pipeline for inference, mirroring the Qwen-Image
    strategy (see :func:`_apply_qwen_offload`) at Krea 2's smaller sizes.

      0. Shrink the transformer to fp8 — torchao native fp8 (fast) or
         diffusers layerwise casting — so the ~24 GiB bf16 transformer
         becomes ~12 GiB.
      1. Enough VRAM → CPU-encode-resident: transformer + VAE on the GPU,
         Qwen3-VL text encoder on the CPU.
      2. Too tight / non-CUDA → block-level group offload, then
         sequential offload, then component-level offload.
    """
    fp8 = _try_qwen_torchao_fp8(
        pipeline, resolved_device, device_arg, family="Krea 2"
    ) or _try_qwen_fp8_casting(pipeline, resolved_device)
    vram = detect_total_vram_gb()
    if (
        fp8
        and resolved_device == "cuda"
        and (vram is None or vram >= _KREA_FP8_RESIDENT_MIN_VRAM_GB)
    ):
        if _setup_cpu_encode_resident(
            pipeline,
            device_arg,
            resolved_device,
            attr=_KREA_CPU_ENCODE_ATTR,
            family="Krea 2",
        ):
            return
    resident_floor = _KREA_FP8_RESIDENT_MIN_VRAM_GB if fp8 else _KREA_RESIDENT_MIN_VRAM_GB
    if vram is None or vram >= resident_floor:
        if _try_qwen_model_offload(pipeline, device_arg, resolved_device):
            return
    if _try_qwen_block_offload(
        pipeline,
        device_arg,
        resolved_device,
        family="Krea 2",
    ):
        return
    try:
        pipeline.enable_sequential_cpu_offload(device=device_arg)
        _log.info(
            "embedded backend: Krea 2 using sequential CPU offload on "
            "device=%s (fits low VRAM; slow per step)",
            resolved_device,
        )
        return
    except Exception:
        _log.warning(
            "embedded backend: sequential CPU offload failed for Krea 2; "
            "falling back to model_cpu_offload (may OOM on a small card)",
            exc_info=True,
        )
    _try_qwen_model_offload(pipeline, device_arg, resolved_device)


def _default_pipeline_factory(model_path: Path, device: str | None) -> Any:
    """Production pipeline factory: pick a diffusers pipeline class by
    filename sniff, then load via the single-file loader. Kept
    module-level (and out of the class) so tests can swap it without
    subclassing.

    Three supported families:

      * ``StableDiffusionXLPipeline`` (default). Pins
        ``EulerAncestralDiscreteScheduler`` because (1) it matches
        the ``sampler_name: euler_ancestral`` setting in
        ``backend/workflows/character.json``, so embedded renders
        track what ComfyUI users get; (2) the default
        ``EulerDiscreteScheduler`` has a known ``step_index + 1``
        off-by-one that surfaces as ``IndexError`` on the final
        step of certain ``num_inference_steps`` settings. The
        ancestral variant runs the same algorithm without the
        buggy guard.
      * ``ZImagePipeline`` — Alibaba's Z-Image-Turbo and family.
        Selected when the filename matches
        :func:`_is_z_image_model_path`. Z-Image ships its own
        ``FlowMatchEulerDiscreteScheduler`` already configured to
        match the upstream training recipe, so we DON'T swap it.
        Loaded in bfloat16 on CUDA per the upstream model card; the
        SDXL fp16 default is not safe for Z-Image's transformer.
      * ``QwenImagePipeline`` — Alibaba's Qwen-Image. Selected when the
        filename matches :func:`_is_qwen_model_path`. Like Z-Image it
        ships its own ``FlowMatchEulerDiscreteScheduler`` and loads in
        bfloat16 with sequential CPU offload (see
        :func:`_load_qwen_pipeline`). Character-change renders route
        through ``QwenImageImg2ImgPipeline`` instead of a fresh text2img
        pass (see :meth:`EmbeddedImageClient.regenerate_from_image`).
      * ``Krea2Pipeline`` — Krea AI's Krea 2 (Turbo and Raw, plus the
        Civitai finetunes built on them). Selected when the checkpoint's
        safetensors header matches :func:`_is_krea_model_path`. diffusers
        has no single-file loader for this family at all, so the
        transformer is converted from the ComfyUI key layout and the
        remaining components are fetched from ungated upstream repos
        (see :func:`_load_krea_pipeline`).
    """
    from diffusers import (
        EulerAncestralDiscreteScheduler,
        StableDiffusionXLPipeline,
    )

    resolved_device = device or detect_compute_device()[0]

    if _is_z_image_model_path(model_path):
        # Z-Image manages its own device placement via
        # ``enable_model_cpu_offload`` (see _load_z_image_pipeline).
        # An eager ``.to(device)`` here would pin every component on
        # GPU and silently undo the offload, putting peak VRAM back
        # over 24 GiB.
        return _load_z_image_pipeline(model_path, device=resolved_device)

    if _is_qwen_model_path(model_path):
        # Qwen-Image likewise manages its own placement via
        # ``enable_model_cpu_offload`` (see _load_qwen_pipeline) — don't
        # eager-``.to`` it here for the same reason as Z-Image.
        return _load_qwen_pipeline(model_path, device=resolved_device)

    if _is_krea_model_path(model_path):
        # Krea 2 also manages its own placement (torchao fp8 + CPU-encode
        # residency, see _load_krea_pipeline) — an eager ``.to`` here
        # would strand the fp8 weight scales and break ``_scaled_mm``.
        return _load_krea_pipeline(model_path, device=resolved_device)

    pipeline = StableDiffusionXLPipeline.from_single_file(
        str(model_path),
        torch_dtype=_resolve_torch_dtype(),
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(  # type: ignore[no-untyped-call]
        pipeline.scheduler.config
    )
    # Route placement through ``_torch_device_arg`` so the "dml"
    # sentinel becomes a real ``torch_directml.device()`` handle.
    # For cuda/xpu/mps/cpu this is a pass-through — byte-identical
    # to ``pipeline.to(resolved_device)``.
    return pipeline.to(_torch_device_arg(resolved_device))


def _make_seeded_generator(torch_mod: Any, device: str, seed: int) -> Any:
    """Build a seeded ``torch.Generator`` pinned to ``device`` — except
    on DirectML, where the generator is built on CPU.

    On CUDA/XPU/MPS the generator MUST live on the pipeline's device
    or diffusers silently ignores the CPU generator and swaps in a
    fresh random GPU generator, breaking seed reproducibility (the
    reason this was pinned in the first place). DirectML is the
    exception: many ``torch_directml`` builds DON'T implement a
    device-side Philox generator, and ``torch.Generator(device=dml)``
    raises ``RuntimeError: ... privateuseone``. A CPU generator is
    accepted by the DML pipeline and still seeds the initial latent
    deterministically, so for "dml" / the registered ``privateuseone``
    device we deliberately build the generator with no device arg.
    Byte-identical to the prior inline call for every other backend.
    """
    if device == "dml" or device.startswith("privateuseone"):
        return torch_mod.Generator().manual_seed(int(seed))
    return torch_mod.Generator(device=device).manual_seed(int(seed))


def _torch_device_arg(device: str | None) -> Any:
    """Map a Lucidium device string to the argument torch actually
    wants for ``pipeline.to(...)`` / ``torch.Generator(device=...)``.

    Every supported backend except DirectML uses a literal device
    string ("cuda" / "xpu" / "mps" / "cpu") that torch accepts as-is,
    so this is a pass-through for them — placement on CUDA/XPU/MPS/CPU
    stays byte-identical to before this helper existed.

    DirectML is the exception: ``torch_directml`` registers a
    ``privateuseone`` device and the real handle is
    ``torch_directml.device()`` — the literal string "dml" is NOT a
    valid torch device and ``pipeline.to("dml")`` raises. So for the
    "dml" sentinel we resolve the live handle here. If
    ``torch_directml`` somehow isn't importable at the call site
    (shouldn't happen — we only return "dml" when it imported during
    detection) we fall back to the raw string so the caller gets a
    clear torch-level error rather than a None.
    """
    if device == "dml":
        try:
            import torch_directml
        except ImportError:
            return device
        return torch_directml.device()
    return device


def _accelerator_module(torch_mod: Any) -> Any:
    """Return the torch submodule (``torch.cuda``, ``torch.xpu``,
    ``torch.mps``) that owns the currently-active accelerator, or
    ``None`` on CPU-only installs. Lets the rest of the codebase
    call ``empty_cache`` / ``memory_allocated`` / ``mem_get_info``
    without an ``if torch.cuda.is_available() else if torch.xpu …``
    ladder at every call site.

    Note on MPS: ``torch.mps`` exists in recent PyTorch but its
    introspection surface is much thinner than CUDA / XPU
    (``empty_cache`` is there; ``memory_allocated`` and
    ``mem_get_info`` are not). Callers must defensively
    ``getattr(module, ...)`` rather than assume parity.

    Note on DirectML: there is NO accelerator module to return.
    ``torch_directml`` exposes a device handle but no
    ``empty_cache`` / ``memory_allocated`` / ``mem_get_info`` parity
    with ``torch.cuda``, so the right answer is ``None``. Every
    caller of this helper (``_release_vram``, ``evict_to_cpu``,
    ``restore_to_gpu``, ``_log_vram_diagnostics``) already early-
    returns on ``None`` — so returning ``None`` makes the VRAM-
    juggling paths cleanly no-op on DirectML rather than crashing on
    a missing ``empty_cache``. dtype selection does NOT go through
    this None (see :func:`_resolve_torch_dtype`, which detects DML
    independently so it doesn't fall to the CPU fp32 default).
    """
    if torch_mod.cuda.is_available():
        return torch_mod.cuda
    xpu = getattr(torch_mod, "xpu", None)
    if xpu is not None and xpu.is_available():
        return xpu
    backends_mps = getattr(torch_mod.backends, "mps", None)
    if backends_mps is not None and backends_mps.is_available():
        return getattr(torch_mod, "mps", None)
    # DirectML deliberately returns None — no cuda-parity memory API
    # exists, and the callers all no-op safely on None.
    return None


def _resolve_torch_dtype(*, prefer_bfloat16: bool = False) -> Any:
    """fp16 (or bfloat16) on accelerators; fp32 on CPU. Kept defensive
    so a CPU-only install doesn't trip on torch.float16 underflow.

    ``prefer_bfloat16`` is set by callers loading transformer-based
    image models (Z-Image, Flux) that are trained in bf16 and
    underflow in fp16. Falls back to fp16 if the host accelerator
    does not advertise bf16 support (pre-Ampere CUDA, Apple MPS,
    older Intel Arc drivers).
    """
    try:
        import torch
    except ImportError:
        return None
    accel = _accelerator_module(torch)
    if accel is None:
        # ``_accelerator_module`` returns None for BOTH CPU-only and
        # DirectML. They want OPPOSITE dtypes: CPU needs fp32 (fp16
        # underflows and many CPU kernels lack fp16 support), but a
        # DirectML GPU needs fp16 — running SDXL in fp32 on DML both
        # doubles VRAM (DML cards are usually 8-16 GB) and is far
        # slower. ``torch_directml`` is a separate optional wheel, so
        # probe it defensively; if it isn't installed we're genuinely
        # CPU-only and fp32 is correct. bf16 is intentionally NOT
        # offered on DML — DirectML's bf16 support is spotty across
        # driver/runtime versions, so we pin fp16 as the safe choice.
        try:
            import torch_directml
        except ImportError:
            return torch.float32
        is_available = getattr(torch_directml, "is_available", None)
        if callable(is_available):
            dml_available = bool(is_available())
        else:
            device_count = getattr(torch_directml, "device_count", None)
            dml_available = callable(device_count) and device_count() > 0
        if dml_available:
            return torch.float16
        return torch.float32
    if prefer_bfloat16:
        # NVIDIA / AMD-ROCm expose ``is_bf16_supported`` on
        # ``torch.cuda``; Intel XPU exposes the same name; Apple MPS
        # silently coerces unsupported dtypes so we treat the lack
        # of the predicate as "not safe to ask for bf16".
        supported = getattr(accel, "is_bf16_supported", None)
        if callable(supported) and supported():
            return torch.bfloat16
    return torch.float16


# ComfyUI Impact-Pack ``FaceDetailer`` defaults (modules/impact/core.py
# enhance_detail). Defining them as constants keeps the parity contract
# explicit — anyone updating these should also revisit
# ``test_face_detail_parity_with_comfy.py``.
_GUIDE_SIZE: int = 512  # target FACE size in inpaint resolution
_MAX_CROP_SIZE: int = 1024  # ceiling for the upscaled crop size
_CROP_FACTOR: float = 3.0  # padding around the bbox (Impact-Pack default)


def _run_face_inpaint(
    pipeline: Any,
    base_png: bytes,
    *,
    face_prompt: str,
    negative: str,
    seed: int,
    strength: float = 0.5,
    inference_steps: int = 18,
    guidance_scale: float = 6.0,
    abort: threading.Event | None = None,
) -> bytes:
    """Run a SDXL inpaint pass on every detected face in the rendered
    image and composite the inpaint output back. Embedded equivalent
    of ComfyUI Impact-Pack's ``FaceDetailer`` node — operates on any
    SDXL checkpoint without inpaint-finetuned weights.

    Maps onto Impact-Pack's ``enhance_detail`` (modules/impact/core.py)
    one step at a time:

      1. Detect ALL faces (Impact-Pack uses a YOLO bbox detector;
         we use opencv's Haar cascade with stricter filtering).
      2. For each face, build a padded crop region (``crop_factor``
         around the bbox).
      3. Resize the crop UP to ``guide_size`` so the face occupies
         a meaningful share of the inpaint latent. WITHOUT THIS, a
         small detected face on an 832×1216 canvas runs through SDXL
         at ~12 latent rows — far below the resolution SDXL was
         trained on, producing the "nightmarish" detail the user
         reported. After upscaling, the face has ~64 latent rows,
         which matches Impact-Pack's behaviour.
      4. Run inpaint at the upscaled resolution with a face-tight
         mask. Use ``StableDiffusionXLInpaintPipeline``'s composite-
         at-each-step path (preserves init_latents under the mask)
         so unmasked regions stay anchored — no pose / shoulder
         drift even with a face-only prompt.
      5. Resize the inpaint output BACK DOWN to the original crop
         size and composite into the base image through the same
         mask × original-alpha so unmasked / transparent pixels
         stay byte-identical to the body render.
    """
    from PIL import Image

    base_image: Image.Image = Image.open(io.BytesIO(base_png))
    has_alpha = base_image.mode in ("RGBA", "LA") or "transparency" in base_image.info
    if has_alpha:
        base_image = base_image.convert("RGBA")
        alpha_full = base_image.getchannel("A")
        figure_bbox = alpha_full.getbbox()
    else:
        base_image = base_image.convert("RGB")
        alpha_full = None
        figure_bbox = None
    width, height = base_image.size

    detected = _detect_face_bboxes(
        base_image,
        figure_bbox=figure_bbox,
        alpha=alpha_full,
    )
    fallback_used = False
    if not detected:
        # No face survived detection — fall back to a single-face
        # geometric guess so the pass still does *something* on
        # profile / occluded renders the cascade misses.
        face_h_est, face_w_est = _estimate_face_dimensions(
            figure_bbox,
            canvas_width=width,
            canvas_height=height,
        )
        if face_h_est <= 0 or face_w_est <= 0:
            return base_png
        if figure_bbox is not None:
            fig_left, fig_top, fig_right, fig_bottom = figure_bbox
            fig_height = fig_bottom - fig_top
            head_strip_height = max(20, fig_height // 8)
            head_strip = (
                alpha_full.crop(
                    (
                        fig_left,
                        fig_top,
                        fig_right,
                        fig_top + head_strip_height,
                    )
                )
                if alpha_full is not None
                else None
            )
            head_bbox = head_strip.getbbox() if head_strip is not None else None
            if head_bbox is not None:
                head_cx = fig_left + (head_bbox[0] + head_bbox[2]) // 2
            else:
                head_cx = (fig_left + fig_right) // 2
            face_top_y = fig_top
        else:
            head_cx = width // 2
            face_top_y = max(0, height // 8)
        detected = [
            (
                head_cx - face_w_est // 2,
                face_top_y,
                head_cx + face_w_est // 2,
                face_top_y + face_h_est,
            )
        ]
        fallback_used = True

    inpaint = _get_or_build_inpaint_pipeline(pipeline)
    if inpaint is None:
        return base_png

    _log.info(
        "face-detail: %d face(s) to inpaint (source=%s)",
        len(detected),
        "fallback" if fallback_used else "haar",
    )

    # Inpaint each face in turn. Updates accumulate on ``base_image``
    # so a later face's inpaint sees the earlier ones' results in
    # case they overlap.
    for idx, face_bbox in enumerate(detected):
        try:
            base_image = _inpaint_one_face(
                base_image=base_image,
                alpha_full=alpha_full,
                face_bbox=face_bbox,
                face_prompt=face_prompt,
                negative=negative,
                seed=(seed ^ (idx * 0x9E3779B1)) & 0xFFFFFFFF,
                strength=strength,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                inpaint_pipeline=inpaint,
                abort=abort,
            )
        except RenderAborted:
            # Caller cancelled — stop, don't fall through to the next
            # face's inpaint as though this one merely failed.
            raise
        except Exception as exc:
            _log.warning(
                "face-detail: inpaint of face %d (bbox=%s) failed; skipping that face: %s",
                idx,
                face_bbox,
                exc,
            )
            continue

    out = io.BytesIO()
    base_image.save(out, format="PNG")
    return out.getvalue()


# Expression-only refresh tuning. ``_EXPRESSION_BOUNDARY_PX`` is the
# margin (in canvas pixels) grown around the detected face bbox before
# the soft Gaussian fade; ``_EXPRESSION_DENOISE`` is the img2img strength
# — 0.5 changes the expression while keeping identity, framing, and the
# surrounding pixels intact.
_EXPRESSION_BOUNDARY_PX: int = 30
_EXPRESSION_DENOISE: float = 0.5


def _run_expression_inpaint(
    pipeline: Any,
    base_png: bytes,
    *,
    expression_prompt: str,
    negative: str,
    seed: int,
    strength: float = _EXPRESSION_DENOISE,
    boundary_radius: int = _EXPRESSION_BOUNDARY_PX,
    inference_steps: int = 18,
    guidance_scale: float = 6.0,
    abort: threading.Event | None = None,
) -> bytes | None:
    """Refresh ONLY the expression on an existing portrait, in place.

    Used when a beat changed a character's expression but nothing else
    (same identity, outfit, pose, lighting, seed). Rather than
    re-rendering the whole portrait from noise — which would also drift
    the pose, hair, and framing — we img2img-inpaint just the face at
    ``strength`` (0.5) denoise:

      1. Locate the face with OpenCV (the shared Haar-cascade detector).
      2. Mask the face bbox grown by ``boundary_radius`` (30 px) with a
         Gaussian soft fade, so the new pixels blend into the unchanged
         surroundings.
      3. Inpaint with the new expression prompt at 0.5 denoise and
         composite back through the mask × figure-alpha (so transparent /
         unmasked pixels stay byte-identical to the original portrait).

    Returns the new PNG bytes, or ``None`` when the fast path can't run
    (no face found, no SDXL inpaint pipeline available) — the caller then
    falls back to a full re-render.
    """
    from PIL import Image

    base_image: Image.Image = Image.open(io.BytesIO(base_png))
    has_alpha = base_image.mode in ("RGBA", "LA") or "transparency" in base_image.info
    if has_alpha:
        base_image = base_image.convert("RGBA")
        alpha_full = base_image.getchannel("A")
        figure_bbox = alpha_full.getbbox()
    else:
        base_image = base_image.convert("RGB")
        alpha_full = None
        figure_bbox = None

    detected = _detect_face_bboxes(
        base_image,
        figure_bbox=figure_bbox,
        alpha=alpha_full,
    )
    if not detected:
        # No confident face — don't risk inpainting the wrong region of
        # a good portrait; let the caller re-render fully instead.
        _log.info("expression-inpaint: no face detected; falling back to full render")
        return None

    inpaint = _get_or_build_inpaint_pipeline(pipeline)
    if inpaint is None:
        return None

    # A portrait has one subject — target the largest detected face.
    detected.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    face_bbox = detected[0]
    _log.info(
        "expression-inpaint: refreshing face bbox=%s at denoise=%.2f boundary=%dpx",
        face_bbox,
        strength,
        boundary_radius,
    )
    try:
        base_image = _inpaint_one_face(
            base_image=base_image,
            alpha_full=alpha_full,
            face_bbox=face_bbox,
            face_prompt=expression_prompt,
            negative=negative,
            seed=seed,
            strength=strength,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            inpaint_pipeline=inpaint,
            boundary_px=boundary_radius,
            abort=abort,
        )
    except RenderAborted:
        raise
    except Exception as exc:
        _log.warning("expression-inpaint: inpaint failed (%s); falling back", exc)
        return None
    finally:
        _release_vram()

    out = io.BytesIO()
    base_image.save(out, format="PNG")
    return out.getvalue()


def _inpaint_one_face(
    *,
    base_image: Any,
    alpha_full: Any,
    face_bbox: tuple[int, int, int, int],
    face_prompt: str,
    negative: str,
    seed: int,
    strength: float,
    inference_steps: int,
    guidance_scale: float,
    inpaint_pipeline: Any,
    boundary_px: int = 0,
    abort: threading.Event | None = None,
) -> Any:
    """Inpaint a single face. Returns the updated base image (with
    the inpainted face composited in). All steps mirror Impact-Pack
    ``enhance_detail`` — see the parent's docstring for the mapping."""
    from PIL import Image, ImageChops

    width, height = base_image.size
    fl, ft, fr, fb = face_bbox
    face_w = fr - fl
    face_h = fb - ft

    # ----- Crop region with crop_factor padding ----------------------------
    # Impact-Pack: padded width = bbox_width × crop_factor (each side
    # gets (crop_factor - 1)/2 of bbox_width as padding).
    # crop_factor=3 → 1×bbox padding on each side → 3× total.
    pad_w = int(face_w * (_CROP_FACTOR - 1) / 2)
    pad_h = int(face_h * (_CROP_FACTOR - 1) / 2)
    crop_left = max(0, fl - pad_w)
    crop_top = max(0, ft - pad_h)
    crop_right = min(width, fr + pad_w)
    crop_bottom = min(height, fb + pad_h)
    # Make the crop SQUARE — the inpaint pipeline expects equal h/w
    # and works best at square latent grids. Take the larger side
    # and centre on the face.
    crop_w = crop_right - crop_left
    crop_h = crop_bottom - crop_top
    crop_side = max(crop_w, crop_h)
    cx = (crop_left + crop_right) // 2
    cy = (crop_top + crop_bottom) // 2
    half = crop_side // 2
    crop_left = max(0, cx - half)
    crop_top = max(0, cy - half)
    crop_right = min(width, crop_left + crop_side)
    crop_bottom = min(height, crop_top + crop_side)
    # If the canvas was smaller than the desired square, pull the
    # box back into the canvas by adjusting the top-left.
    crop_left = max(0, crop_right - crop_side)
    crop_top = max(0, crop_bottom - crop_side)
    crop_box = (crop_left, crop_top, crop_right, crop_bottom)
    crop_w = crop_right - crop_left
    crop_h = crop_bottom - crop_top

    # ----- guide_size upscaling --------------------------------------------
    # Impact-Pack: upscale so the FACE bbox reaches guide_size. Cap
    # the upscale so the resulting CROP doesn't exceed max_size. This
    # is the missing step that made small detected faces inpaint at
    # ~12 latent rows; with this step they hit ~64 latent rows.
    upscale = max(1.0, _GUIDE_SIZE / max(face_h, face_w))
    if upscale * crop_w > _MAX_CROP_SIZE:
        upscale = _MAX_CROP_SIZE / crop_w
    upscale = max(1.0, upscale)
    inpaint_w = round(crop_w * upscale)
    inpaint_h = round(crop_h * upscale)
    # Round to multiple of 8 (SDXL latent stride).
    inpaint_w -= inpaint_w % 8
    inpaint_h -= inpaint_h % 8
    if inpaint_w < 8 or inpaint_h < 8:
        return base_image

    face_crop = base_image.crop(crop_box)
    if face_crop.mode != "RGB":
        face_crop_rgb = face_crop.convert("RGB")
    else:
        face_crop_rgb = face_crop
    if (inpaint_w, inpaint_h) != face_crop_rgb.size:
        face_crop_rgb = face_crop_rgb.resize(
            (inpaint_w, inpaint_h),
            Image.LANCZOS,  # type: ignore[attr-defined]  # runtime alias missing from Pillow stubs
        )

    # ----- Mask geometry (in upscaled coordinates) ------------------------
    # Convert face bbox from canvas coords → upscaled crop coords.
    face_top_in_crop = (ft - crop_top) * upscale
    face_left_in_crop = (fl - crop_left) * upscale
    face_h_in_crop = face_h * upscale
    face_w_in_crop = face_w * upscale
    if boundary_px > 0:
        # Expression-only refresh: a rectangular mask = the face bbox
        # grown by ``boundary_px`` (canvas px, scaled into crop coords)
        # with a Gaussian soft fade, so the change blends into the
        # surrounding pixels instead of leaving a seam.
        inpaint_mask = _expression_inpaint_mask(
            crop_size_h=inpaint_h,
            crop_size_w=inpaint_w,
            face_top=int(face_top_in_crop),
            face_left=int(face_left_in_crop),
            face_height=int(face_h_in_crop),
            face_width=int(face_w_in_crop),
            boundary=round(boundary_px * upscale),
        )
    else:
        inpaint_mask = _face_inpaint_mask(
            crop_size_h=inpaint_h,
            crop_size_w=inpaint_w,
            face_top=int(face_top_in_crop),
            face_left=int(face_left_in_crop),
            face_height=int(face_h_in_crop),
            face_width=int(face_w_in_crop),
        )

    kwargs: dict[str, Any] = {
        "image": face_crop_rgb,
        "mask_image": inpaint_mask,
        "strength": strength,
        "num_inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "height": inpaint_h,
        "width": inpaint_w,
    }
    try:
        import torch

        device = _resolve_pipeline_device(inpaint_pipeline)
        kwargs["generator"] = _make_seeded_generator(torch, device, seed)
    except ImportError:
        pass

    embeds = _encode_long_prompt(inpaint_pipeline, face_prompt, negative)
    if embeds is None:
        kwargs["prompt"] = face_prompt
        kwargs["negative_prompt"] = negative
    else:
        kwargs.update(embeds)

    try:
        import torch

        ctx: AbstractContextManager[Any] = torch.inference_mode()
    except ImportError:
        from contextlib import nullcontext

        ctx = nullcontext()
    _install_abort_callback(inpaint_pipeline, kwargs, abort)
    with ctx:
        result = inpaint_pipeline(**kwargs)
    new_face = result.images[0]
    if new_face.mode != "RGB":
        new_face = new_face.convert("RGB")

    # Resize the inpainted face BACK DOWN to the source crop's
    # native dimensions before compositing. The composite mask is
    # built at upscaled resolution and resized too, so mask ×
    # alpha multiplication operates at the same native scale.
    if new_face.size != (crop_w, crop_h):
        # ``Image.LANCZOS`` exists at runtime but not in Pillow's stubs.
        new_face = new_face.resize((crop_w, crop_h), Image.LANCZOS)  # type: ignore[attr-defined]
    composite_mask = inpaint_mask
    if composite_mask.size != (crop_w, crop_h):
        composite_mask = composite_mask.resize(
            (crop_w, crop_h),
            Image.LANCZOS,  # type: ignore[attr-defined]  # runtime alias missing from Pillow stubs
        )
    if alpha_full is not None:
        crop_alpha = alpha_full.crop(crop_box)
        composite_mask = ImageChops.multiply(composite_mask, crop_alpha)
    base_image.paste(new_face, crop_box, composite_mask)

    # CUDA leak mitigation. Each inpaint pass holds:
    #   * ``result`` — diffusers Output dataclass with image tensor
    #     refs the GC won't collect until this scope exits;
    #   * ``embeds`` — compel-encoded prompt_embeds /
    #     pooled_prompt_embeds, all CUDA tensors;
    #   * ``kwargs`` — closes over both, plus the torch.Generator;
    #   * ``new_face`` / ``face_crop_rgb`` — PIL images whose
    #     in-memory buffers came from CUDA via VAE decode.
    # On a multi-face render those refs accumulate per face — the
    # CUDA allocator's cache balloons and the next render OOMs.
    # Explicit deletion + ``empty_cache`` here forces the
    # collector to release the CUDA blocks back to the pool
    # BEFORE the next face's inpaint allocates fresh tensors.
    del result
    del new_face
    del face_crop_rgb
    del face_crop
    del kwargs
    if embeds is not None:
        del embeds
    _release_vram()
    return base_image


_HAAR_CASCADE = None  # lazy-loaded on first detection call


def _bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Intersection-over-union for two ``(l, t, r, b)`` boxes."""
    inter_l = max(a[0], b[0])
    inter_t = max(a[1], b[1])
    inter_r = min(a[2], b[2])
    inter_b = min(a[3], b[3])
    if inter_r <= inter_l or inter_b <= inter_t:
        return 0.0
    inter = (inter_r - inter_l) * (inter_b - inter_t)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, area_a + area_b - inter)


def _bbox_overlap_fraction(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> float:
    """Fraction of ``inner``'s area that falls inside ``outer``.

    Used to filter face detections that landed outside the figure's
    alpha silhouette — a face on a wall, in a reflection, on the
    fabric of a dress, etc., that the Haar cascade flagged as a
    false positive."""
    inter_l = max(inner[0], outer[0])
    inter_t = max(inner[1], outer[1])
    inter_r = min(inner[2], outer[2])
    inter_b = min(inner[3], outer[3])
    if inter_r <= inter_l or inter_b <= inter_t:
        return 0.0
    inter = (inter_r - inter_l) * (inter_b - inter_t)
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / inner_area


def _detect_face_bbox(image: Any) -> tuple[int, int, int, int] | None:
    """Single-face detection — kept as a thin wrapper around the
    multi-face path for the existing tests. Returns the LARGEST
    detected face bbox or ``None`` if nothing survived filtering."""
    bboxes = _detect_face_bboxes(image)
    if not bboxes:
        return None
    return max(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def _detect_face_bboxes(
    image: Any,
    *,
    figure_bbox: tuple[int, int, int, int] | None = None,
    alpha: Any = None,
) -> list[tuple[int, int, int, int]]:
    """Run an opencv Haar-cascade face detector on ``image`` and
    return ALL surviving face bboxes as ``(left, top, right, bottom)``.

    Filtering pipeline (applied in this order):

      1. **Cascade detection** with stricter ``minNeighbors=8`` and
         ``scaleFactor=1.08``. Higher than opencv's defaults, lower
         false-positive rate. minSize is 5% of the shorter side.
      2. **Aspect ratio** — face_h / face_w must be in [0.7, 1.6].
         Anything wider or taller is almost certainly a non-face
         pattern the cascade hallucinated (a dress fold, a column,
         a wall texture). The user reported a false positive that
         "put an eye in the middle of someone's dress" — that was
         a low-aspect detection on cloth folds.
      3. **Figure containment** — when the alpha figure bbox is
         known, every detection must overlap the figure by ≥70%
         of its area. Faces on the BACKGROUND (a poster on the
         wall, a reflection in a window) get filtered.
      4. **Alpha sampling** — when a per-pixel alpha mask is
         available, the detection's centre pixel must be opaque
         (alpha > 200). Catches false positives on the see-through
         halo that survives rembg's threshold.
      5. **NMS-lite** — drop any detection that overlaps another
         (kept) detection by IoU > 0.4. The cascade often flags
         the same face two or three times at different scales.

    Returns the filtered list in descending order of area so the
    primary subject's face comes first.
    """
    global _HAAR_CASCADE
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    if _HAAR_CASCADE is None:
        try:
            # cv2's bundled stubs omit ``data`` and ``CascadeClassifier``.
            cascade_path = (
                cv2.data.haarcascades  # type: ignore[attr-defined]
                + "haarcascade_frontalface_default.xml"
            )
            cascade = cv2.CascadeClassifier(cascade_path)  # type: ignore[attr-defined]
            if cascade.empty():
                _log.warning(
                    "face-detail: opencv haar cascade XML loaded empty",
                )
                return []
            _HAAR_CASCADE = cascade
        except Exception:
            _log.warning(
                "face-detail: failed to load opencv haar cascade",
                exc_info=True,
            )
            return []

    try:
        if image.mode != "RGB":
            rgb = image.convert("RGB")
        else:
            rgb = image
        arr = np.array(rgb)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        raw = _HAAR_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=8,
            minSize=(max(48, min(arr.shape[:2]) // 20),) * 2,
        )
    except Exception:
        _log.warning(
            "face-detail: opencv detection raised",
            exc_info=True,
        )
        return []
    if len(raw) == 0:
        return []

    # Convert raw (x,y,w,h) → (l,t,r,b) for the rest of the pipeline.
    candidates: list[tuple[int, int, int, int]] = [
        (int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in raw
    ]

    alpha_arr = None
    if alpha is not None:
        try:
            alpha_arr = np.array(alpha)
        except Exception:
            alpha_arr = None

    surviving: list[tuple[int, int, int, int]] = []
    for bbox in sorted(
        candidates,
        key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])),
    ):
        bl, bt, br, bb = bbox
        bw = br - bl
        bh = bb - bt
        if bw <= 0 or bh <= 0:
            continue
        # Aspect ratio guard.
        aspect = bh / bw
        if aspect < 0.7 or aspect > 1.6:
            _log.debug(
                "face-detail: dropped detection %s — aspect %.2f outside [0.7, 1.6]",
                bbox,
                aspect,
            )
            continue
        # Figure containment guard.
        if figure_bbox is not None:
            overlap = _bbox_overlap_fraction(bbox, figure_bbox)
            if overlap < 0.7:
                _log.debug(
                    "face-detail: dropped detection %s — only %.0f%% inside figure",
                    bbox,
                    overlap * 100,
                )
                continue
        # Alpha-centre guard.
        if alpha_arr is not None:
            cx = (bl + br) // 2
            cy = (bt + bb) // 2
            try:
                if alpha_arr[cy, cx] < 200:
                    _log.debug(
                        "face-detail: dropped detection %s — centre alpha=%d",
                        bbox,
                        int(alpha_arr[cy, cx]),
                    )
                    continue
            except IndexError:
                continue
        # NMS-lite against already-kept detections.
        suppress = False
        for kept in surviving:
            if _bbox_iou(bbox, kept) > 0.4:
                suppress = True
                break
        if suppress:
            continue
        surviving.append(bbox)

    return surviving


def _estimate_face_dimensions(
    figure_bbox: tuple[int, int, int, int] | None,
    *,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int]:
    """Best-effort guess at how tall + wide the face is in pixels.

    Without face detection we approximate from FIGURE PROPORTIONS
    + how much of the canvas the figure fills:

      * Full body (figure spans most of the canvas) — face is
        ~14% of figure height (classic "7-and-a-half heads tall"
        proportion);
      * Head-and-shoulders (figure fills less of the canvas) —
        face is roughly half the figure height;
      * Mid framings interpolate.

    The earlier impl ignored framing and sized the inpaint mask as
    a fixed fraction of the crop. On a full-body render that put
    the mask oval over the chest instead of the face — the
    user-reported "inpaint area is beneath the face". Returning a
    height + width tuple lets the caller size both the crop AND
    the mask off the actual face proportions.

    Faces are taller than wide; width ≈ 0.7 × height matches typical
    SDXL outputs. Returns ``(0, 0)`` when no figure was found.
    """
    if figure_bbox is None:
        # No alpha — fall back to a generic small face roughly
        # appropriate for a centered portrait at any resolution.
        face_h = max(64, canvas_height // 5)
        return face_h, int(face_h * 0.7)

    _fig_left, fig_top, _fig_right, fig_bottom = figure_bbox
    fig_height = fig_bottom - fig_top
    if fig_height <= 0:
        return 0, 0
    extent = fig_height / max(canvas_height, 1)
    if extent >= 0.85:
        face_fraction = 1.0 / 7  # full body — head is ~14% of figure
    elif extent <= 0.55:
        face_fraction = 0.5  # tight portrait — face fills half
    else:
        # Linear interp between those anchors.
        t = (extent - 0.55) / (0.85 - 0.55)
        face_fraction = 0.5 + t * ((1.0 / 7) - 0.5)
    face_h = max(48, int(fig_height * face_fraction))
    face_w = max(40, int(face_h * 0.7))
    return face_h, face_w


def _face_inpaint_mask(
    *,
    crop_size_h: int,
    crop_size_w: int,
    face_top: int,
    face_left: int,
    face_height: int,
    face_width: int,
) -> Any:
    """Build the SAMPLER inpaint mask — white where the FACE is
    inside the crop, black everywhere else. The white region is an
    oval at ``(face_left, face_top)`` with the supplied dimensions,
    plus a small upward extension to catch hair and a small
    downward extension to catch the chin.

    Crop dimensions are passed separately so non-square crops (the
    edge case when the bbox sits near a canvas edge and gets
    clamped) still get a correctly-shaped mask.

    Soft-edged via Gaussian blur so the mask transition isn't sharp
    — SDXL produces visible halos at hard mask boundaries. Blur
    radius scales with face size so the soft zone stays
    proportional regardless of crop / face dimensions.
    """
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new("L", (crop_size_w, crop_size_h), 0)
    draw = ImageDraw.Draw(mask)
    hair_pad = max(4, face_height // 10)
    oval_top = max(0, face_top - hair_pad)
    oval_bottom = min(crop_size_h, face_top + face_height + hair_pad // 2)
    oval_left = max(0, face_left)
    oval_right = min(crop_size_w, face_left + face_width)
    draw.ellipse((oval_left, oval_top, oval_right, oval_bottom), fill=255)
    blur_radius = max(6, face_height // 12)
    return mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def _expression_inpaint_mask(
    *,
    crop_size_h: int,
    crop_size_w: int,
    face_top: int,
    face_left: int,
    face_height: int,
    face_width: int,
    boundary: int,
) -> Any:
    """Mask for the expression-only refresh: a solid white rectangle
    over the face bbox grown by ``boundary`` pixels on every side, then
    Gaussian-blurred so the edge is a soft fade rather than a hard seam.

    Unlike :func:`_face_inpaint_mask` (a face-tight oval tuned for the
    full FaceDetailer pass) this keeps the masked region a touch larger
    and rectangular: at 0.5 denoise the model only nudges the expression,
    and the ``boundary`` halo + blur let the new pixels blend into the
    unchanged hair / neck / background around the face.

    ``boundary`` is already in crop (upscaled) coordinates. The blur
    radius is half the boundary so the fade spans roughly the grown
    margin — wider boundary, softer fade.
    """
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new("L", (crop_size_w, crop_size_h), 0)
    draw = ImageDraw.Draw(mask)
    left = max(0, face_left - boundary)
    top = max(0, face_top - boundary)
    right = min(crop_size_w, face_left + face_width + boundary)
    bottom = min(crop_size_h, face_top + face_height + boundary)
    draw.rectangle((left, top, right, bottom), fill=255)
    # Half the boundary gives a fade that lives mostly inside the grown
    # margin, so the fully-opaque core still covers the whole face.
    blur_radius = max(4, boundary // 2)
    return mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))


_INPAINT_CACHE_ATTR = "_lucidium_inpaint_cached"
_VAE_MEMORY_OPT_ATTR = "_lucidium_vae_memory_opt_applied"
_QWEN_IMG2IMG_CACHE_ATTR = "_lucidium_qwen_img2img_cached"

# img2img denoise strength for the Qwen "character change" path. 0.7
# applies an outfit / pose / expression change clearly while keeping the
# character's identity, framing, and composition anchored to the prior
# portrait (strength=1.0 would start from pure noise and ignore the init;
# lower values barely change it). Tunable here.
_QWEN_IMG2IMG_STRENGTH: float = 0.7


def _apply_vae_memory_optimizations(pipeline: Any) -> None:
    """Enable diffusers' VAE memory optimizations on the shared
    VAE. Idempotent — flagged on first call so repeated invocations
    are no-ops.

    *** WHY ***

    The face-inpaint pass is what users actually feel as a "VRAM
    spike": the body render's VAE decode peaks at 2-3 GiB on top
    of resident weights, the inpaint then runs ANOTHER VAE
    encode + decode for the face crop, and the combined
    activation peak can push a 24 GiB card into OOM territory
    when paired with a music model on the same GPU.

    Diffusers exposes two cheap, lossless toggles:

      * ``enable_vae_slicing`` — decode batched latents one-by-
        one instead of all at once. Free for batch_size=1 (our
        case), but harmless to enable.
      * ``enable_vae_tiling`` — split VAE decode into overlapping
        tiles. Cuts peak VRAM during decode by ~3× at the cost
        of barely-perceptible seam artefacts that the rembg cut
        + face composite step blur out anyway.

    Both flags ride on the VAE module itself, so flipping them
    on the shared VAE benefits BOTH text2img and inpaint passes.
    The text2img pass at 832×1216 sees a smaller benefit (already
    fits comfortably) but the inpaint at 512×512 face crops sees
    the full peak reduction.
    """
    if getattr(pipeline, _VAE_MEMORY_OPT_ATTR, False):
        return
    vae = getattr(pipeline, "vae", None)
    if vae is None:
        return
    try:
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
    except Exception:
        _log.warning(
            "embedded backend: failed to enable VAE memory optimizations; "
            "continuing (renders may spike VRAM during VAE decode)",
            exc_info=True,
        )
        return
    try:
        setattr(pipeline, _VAE_MEMORY_OPT_ATTR, True)
    except (AttributeError, TypeError):
        # Slotted / frozen pipeline classes — flag would be
        # rejected. The flags on the VAE itself still stick;
        # we just lose the idempotency guard and toggle them
        # again next call (which is also idempotent inside
        # diffusers, so this is safe).
        pass


def _get_or_build_inpaint_pipeline(text2img_pipeline: Any) -> Any:
    """Return a cached inpaint pipeline wrapping ``text2img_pipeline``,
    constructing one on first use.

    Mirrors the Compel cache — same lifecycle, dies with the
    text2img pipeline on eviction. The inpaint wrapper shares
    components with text2img (no extra weights in VRAM) but the
    wrapper class registers config metadata on construction;
    rebuilding it per render churned that metadata and added a
    fresh module dict every time. Cache → one wrapper per
    text2img pipeline, no per-render churn.

    On first construction we ALSO enable VAE tiling + slicing on
    the shared VAE. The inpaint pass's encode+decode is the
    heaviest VRAM event in the render pipeline; tiling cuts its
    peak by ~3× without changing the rendered output (the rembg
    cut + face composite step blurs over any tile seams).
    """
    cached = getattr(text2img_pipeline, _INPAINT_CACHE_ATTR, None)
    if cached is not None:
        return cached
    inpaint = _build_inpaint_pipeline(text2img_pipeline)
    if inpaint is None:
        return None
    # Apply memory optimizations to the shared VAE — benefits both
    # text2img and inpaint passes since they share the module.
    _apply_vae_memory_optimizations(text2img_pipeline)
    _apply_vae_memory_optimizations(inpaint)
    try:
        setattr(text2img_pipeline, _INPAINT_CACHE_ATTR, inpaint)
    except (AttributeError, TypeError):
        return inpaint
    return inpaint


def _build_inpaint_pipeline(text2img_pipeline: Any) -> Any:
    """Construct an SDXL inpaint pipeline that SHARES weights with
    the supplied text2img pipeline (no extra VRAM, no extra disk
    load).

    Diffusers' ``StableDiffusionXLInpaintPipeline`` works with both
    inpaint-finetuned UNets (9 input channels) and regular SDXL
    UNets (4 input channels). When the UNet is regular it uses the
    "composite at each step" code path — at every denoise step the
    unmasked latent regions are restored from the appropriately-
    noised init_latents, exactly the mechanism ComfyUI's
    Impact-Pack FaceDetailer relies on. We share the loaded
    components so the inpaint pass costs no extra VRAM beyond
    activation tensors.

    Returns ``None`` when diffusers isn't available or the pipeline
    can't be promoted (test stubs without UNet / VAE / encoders).
    """
    try:
        from diffusers import StableDiffusionXLInpaintPipeline
    except ImportError:
        return None
    required = (
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "unet",
        "scheduler",
    )
    if not all(hasattr(text2img_pipeline, attr) for attr in required):
        return None
    try:
        # Each pipeline MUST have its own scheduler instance.
        # EulerAncestralDiscreteScheduler is stateful (tracks
        # step_index, sigmas, begin_index across calls), and the
        # inpaint pipeline's ``get_timesteps`` truncates
        # ``self.scheduler.timesteps`` directly + calls
        # ``set_begin_index`` for the strength<1 path. Sharing one
        # scheduler instance between text2img and inpaint leaks
        # that state — the next text2img call sees a truncated
        # sigmas array but the full step_index counter and trips
        # ``IndexError: index 19 is out of bounds for dimension 0
        # with size 19`` on the final denoise step. Cloning via
        # ``from_config`` produces an independent instance with
        # identical settings.
        inpaint_scheduler = type(text2img_pipeline.scheduler).from_config(
            text2img_pipeline.scheduler.config
        )
        return StableDiffusionXLInpaintPipeline(  # type: ignore[no-untyped-call]
            vae=text2img_pipeline.vae,
            text_encoder=text2img_pipeline.text_encoder,
            text_encoder_2=text2img_pipeline.text_encoder_2,
            tokenizer=text2img_pipeline.tokenizer,
            tokenizer_2=text2img_pipeline.tokenizer_2,
            unet=text2img_pipeline.unet,
            scheduler=inpaint_scheduler,
        )
    except Exception:
        _log.warning(
            "embedded backend: failed to build inpaint pipeline for face-detail pass; skipping",
            exc_info=True,
        )
        return None


def _restore_qwen_transformer_param_dtype(transformer: Any, compute_dtype: Any) -> None:
    """Cast a Qwen transformer's plain float32 params back to ``compute_dtype``
    (bf16) IN PLACE, leaving the torchao ``Float8Tensor`` weights untouched.

    ``from_pipe`` upcasts every non-fp8 parameter (biases + norm/embed
    weights) to float32; torchao's fp8 ``addmm`` then rejects the float32
    bias. We can't ``.to`` the whole module (that would re-cast — and break —
    the fp8 weights), so we walk the parameters and restore only the genuine
    float32 ones. The fp8 weights are tensor subclasses whose ``.dtype``
    reports the bf16 compute dtype (not float32), so the dtype filter skips
    them; the explicit subclass-name guard is belt-and-suspenders. No-ops on
    test stubs / a missing transformer."""
    params = getattr(transformer, "parameters", None)
    if not callable(params):
        return
    try:
        import torch
    except ImportError:
        return
    _QUANT_TENSOR_NAMES = ("Float8Tensor", "AffineQuantizedTensor")
    try:
        with torch.no_grad():
            for p in params(recurse=True):
                data = getattr(p, "data", p)
                if type(data).__name__ in _QUANT_TENSOR_NAMES:
                    continue
                if getattr(data, "dtype", None) == torch.float32:
                    p.data = data.to(compute_dtype)  # keeps device
    except Exception:
        _log.warning(
            "embedded backend: could not restore the Qwen transformer's "
            "param dtype after from_pipe; character-change renders may fail "
            "or be slow",
            exc_info=True,
        )


def _get_or_build_qwen_img2img(text2img_pipeline: Any) -> Any:
    """Return a cached ``QwenImageImg2ImgPipeline`` that SHARES every
    component (transformer, text encoder, VAE, tokenizer) with the
    supplied Qwen text2img pipeline — no extra weights in VRAM, no
    second checkpoint load. Constructed on first use and cached on the
    text2img pipeline object so it dies with it on eviction (mirrors
    the SDXL inpaint-pipeline cache).

    Built via diffusers' ``from_pipe``, which reuses the loaded module
    objects by reference. Because the modules are shared, any CPU-offload
    hooks the text2img pipeline installed (see :func:`_load_qwen_pipeline`)
    ride along on those same modules — so the img2img wrapper inherits
    the parent's device placement without a second
    ``enable_model_cpu_offload`` call (which would double-hook the shared
    modules). Returns ``None`` when diffusers isn't importable or the
    wrapper can't be built (test stubs); the caller then declines the
    fast path and full-renders.
    """
    cached = getattr(text2img_pipeline, _QWEN_IMG2IMG_CACHE_ATTR, None)
    if cached is not None:
        return cached
    try:
        from diffusers import QwenImageImg2ImgPipeline
    except ImportError:
        return None
    try:
        img2img = QwenImageImg2ImgPipeline.from_pipe(text2img_pipeline)
    except Exception:
        _log.warning(
            "embedded backend: failed to build Qwen img2img pipeline for "
            "the character-change path; will full-render instead",
            exc_info=True,
        )
        return None
    # Share the VAE memory optimizations too (same module, so this is
    # idempotent, but keeps the flag consistent on the wrapper).
    _apply_vae_memory_optimizations(img2img)
    # ``from_pipe`` UPCASTS every non-fp8 weight to float32 (the new
    # pipeline's default dtype) on the SHARED modules, even though the
    # loader built them in bf16. Two distinct breakages result, both fixed
    # by restoring the loader's bf16 compute dtype here:
    #
    #   1. VAE + text encoder -> float32. The img2img init latents come
    #      from the VAE, so a float32 VAE makes the denoise loop run
    #      float32 hidden states through the torchao-fp8 transformer —
    #      bypassing its fp8 tensor-core fast path (catastrophic without
    #      Triton on Windows): ~20x slower (47 s vs 2.2 s per step).
    #   2. The transformer's 846 biases (+ norm/embed weights) -> float32,
    #      while its quantized weights stay ``Float8Tensor``. torchao's fp8
    #      ``addmm`` then rejects the float32 bias ("Bias must be BFloat16
    #      or Half, but got Float"). Because the module is shared, this also
    #      corrupts the next text2img render off the same pipeline.
    #
    # The transformer's fp8 weights must NOT be ``.to``'d (that strands the
    # weight scale -> a ``_scaled_mm`` error), so we cast only its plain
    # float32 params in place and skip the ``Float8Tensor`` weights.
    compute_dtype = _resolve_torch_dtype(prefer_bfloat16=True)
    if compute_dtype is not None:
        # Non-quantized components: a whole-module dtype cast is safe.
        for comp_name in ("vae", "text_encoder"):
            comp = getattr(img2img, comp_name, None)
            to_fn = getattr(comp, "to", None)
            if comp is None or not callable(to_fn):
                continue
            try:
                to_fn(dtype=compute_dtype)  # keeps each component's device
            except Exception:
                _log.warning(
                    "embedded backend: could not restore %s to %s on the "
                    "Qwen img2img pipeline; character-change renders may be "
                    "slow",
                    comp_name,
                    compute_dtype,
                    exc_info=True,
                )
        _restore_qwen_transformer_param_dtype(
            getattr(img2img, "transformer", None),
            compute_dtype,
        )
    # Carry our private flags (sampling recipe + CPU-encode mode) onto the
    # wrapper so _run_qwen_img2img behaves like text2img — from_pipe
    # doesn't copy them, and the img2img shares the same resident
    # transformer + CPU text encoder.
    try:
        setattr(img2img, _QWEN_RECIPE_ATTR, _qwen_recipe(text2img_pipeline))
        setattr(
            img2img,
            _QWEN_CPU_ENCODE_ATTR,
            getattr(text2img_pipeline, _QWEN_CPU_ENCODE_ATTR, False),
        )
    except (AttributeError, TypeError):
        pass
    try:
        setattr(text2img_pipeline, _QWEN_IMG2IMG_CACHE_ATTR, img2img)
    except (AttributeError, TypeError):
        return img2img
    return img2img


def _default_qwen_img2img_runner(
    text2img_pipeline: Any,
    base_png: bytes,
    *,
    positive: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    strength: float,
    abort: threading.Event | None = None,
) -> bytes | None:
    """Default ``qwen_img2img_runner``: build (or reuse) the shared-
    component ``QwenImageImg2ImgPipeline`` for ``text2img_pipeline`` and
    run one img2img pass. Returns ``None`` when the img2img pipeline
    can't be built (diffusers absent / a test stub) so the caller
    declines the fast path and full-renders. Split from the client so
    ``asyncio.to_thread`` doesn't bind ``self`` and tests can inject a
    fake runner."""
    img2img = _get_or_build_qwen_img2img(text2img_pipeline)
    if img2img is None:
        return None
    return _run_qwen_img2img(
        img2img,
        base_png,
        positive=positive,
        negative=negative,
        width=width,
        height=height,
        seed=seed,
        strength=strength,
        abort=abort,
    )


def _run_qwen_img2img(
    pipeline: Any,
    base_png: bytes,
    *,
    positive: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    strength: float,
    abort: threading.Event | None = None,
) -> bytes:
    """Run a single Qwen-Image img2img pass and return PNG bytes.

    Lifted out of the client (like :func:`_run_pipeline`) so
    ``asyncio.to_thread`` doesn't bind ``self`` and tests can drive it
    with a fake pipeline. The init image is the previous portrait —
    typically a transparent cut-out — so we composite it onto neutral
    mid-gray before handing it to the VAE: dropping alpha to pure black
    makes the model paint a black field behind the figure, whereas a
    plain gray backdrop reads as "figure on a blank wall" and rembg
    re-cuts it cleanly afterwards.
    """
    from PIL import Image

    init: Image.Image = Image.open(io.BytesIO(base_png))
    if init.mode in ("RGBA", "LA") or "transparency" in init.info:
        init = init.convert("RGBA")
        backdrop = Image.new("RGBA", init.size, (128, 128, 128, 255))
        backdrop.alpha_composite(init)
        init = backdrop.convert("RGB")
    else:
        init = init.convert("RGB")
    if init.size != (width, height):
        # ``Image.LANCZOS`` exists at runtime but not in Pillow's stubs.
        init = init.resize((width, height), Image.LANCZOS)  # type: ignore[attr-defined]

    steps, true_cfg = _qwen_recipe(pipeline)
    cpu_encode = getattr(pipeline, _QWEN_CPU_ENCODE_ATTR, False)
    kwargs: dict[str, Any] = {
        "image": init,
        "width": width,
        "height": height,
        "strength": strength,
        # Same variant-aware recipe as _run_pipeline (Lightning few-step /
        # cfg-off, or base). img2img runs ``steps × strength`` effective
        # denoise steps, which keeps a Lightning edit in the few-step
        # regime.
        "num_inference_steps": steps,
        "true_cfg_scale": true_cfg,
        "guidance_scale": 1.0,
    }
    if cpu_encode:
        # CPU-encode-resident mode: encode on the CPU text encoder, pass
        # embeds, detach the encoder for the call (see _run_qwen_call).
        kwargs.update(_qwen_cpu_encode(pipeline, positive, negative, true_cfg))
    else:
        kwargs["prompt"] = positive
        kwargs["negative_prompt"] = negative
    try:
        import torch

        device = _resolve_pipeline_device(pipeline)
        kwargs["generator"] = _make_seeded_generator(torch, device, seed)
        # ``no_grad`` not ``inference_mode`` — torchao fp8 tensors trip
        # inference_mode's strict functionalization (see _run_pipeline).
        ctx: AbstractContextManager[Any] = torch.no_grad()
    except ImportError:
        from contextlib import nullcontext

        ctx = nullcontext()
    _install_abort_callback(pipeline, kwargs, abort)
    with ctx:
        result = _run_qwen_call(pipeline, kwargs) if cpu_encode else pipeline(**kwargs)
    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    del image
    del result
    del kwargs
    return png_bytes


def _feather_mask(width: int, height: int, *, feather: int) -> Any:
    """Build an L-mode (8-bit grayscale) PIL mask: solid white in
    the inner region, smooth gradient to black ``feather`` pixels
    from each edge. Used as the alpha channel when pasting the
    inpainted face back over the body render so the seam blurs out
    instead of cutting hard at the crop rectangle."""
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    # Inner solid box leaves a ``feather``-px border of black around
    # the edge; the box-blur below softens that border into a
    # gradient. Empirically a feather of ~64 px on a 512 face hides
    # the seam fully without smearing the face out of the crop.
    draw.rectangle(
        (feather, feather, width - feather, height - feather),
        fill=255,
    )
    return mask.filter(ImageFilter.GaussianBlur(radius=feather / 2))


def _run_pipeline(
    pipeline: Any,
    *,
    positive: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    abort: threading.Event | None = None,
) -> bytes:
    """Run a single inference pass. Lifted out of the client so
    ``asyncio.to_thread`` doesn't have to bind ``self`` (and tests
    can drive it standalone with a fake pipeline). SDXL Turbo wants
    1-step / cfg=0; the if-branch keeps multi-step checkpoints
    working at the same call site.

    Long-prompt handling: when the composed prompt would exceed
    CLIP's 77-token window (which a portrait prompt routinely
    does once expression / pose / outfit / style are all
    folded in), we encode it via ``providers.clip_long_prompt``
    instead. That module chunks the prompt into 75-token slices,
    encodes each slice against both SDXL CLIP encoders, and
    concatenates the resulting embeddings — the model gets to see
    every token instead of silently dropping the tail. Falls back
    to plain string prompts when the pipeline doesn't expose the
    tokenizer/text-encoder pair the chunker needs (the usual case
    in test venvs that mock the pipeline).
    """
    family = resolve_model_family(pipeline)
    kwargs: dict[str, Any] = {"width": width, "height": height}
    try:
        import torch

        # The generator MUST live on the same device as the pipeline.
        # ``torch.Generator()`` defaults to CPU; if the pipeline is on
        # CUDA, diffusers silently ignores the CPU generator and uses
        # a fresh per-call random GPU generator instead — every render
        # came back different even with the same seed. Pin to the
        # pipeline's device so manual_seed actually takes effect.
        device = _resolve_pipeline_device(pipeline)
        # Route through ``_make_seeded_generator`` so DirectML gets a
        # CPU generator (DML often lacks a device-side generator);
        # identical to the prior inline call on CUDA/XPU/MPS/CPU.
        kwargs["generator"] = _make_seeded_generator(torch, device, seed)
    except ImportError:
        pass
    # Sampling recipe, owned by the family (see the registry above).
    kwargs.update(family.sampling_kwargs(pipeline))

    cpu_encode_attr = family.cpu_encode_attr(pipeline)
    cpu_encode = bool(cpu_encode_attr is not None and getattr(pipeline, cpu_encode_attr, False))
    if cpu_encode:
        # CPU-encode-resident mode (see _setup_cpu_encode_resident): the
        # text encoder lives on the CPU, so encode there and pass the
        # embeds in (the GPU holds only the resident transformer + VAE).
        # Skip the unconditional branch when guidance is off — Krea 2
        # gates it on ``guidance_scale > 0``, Qwen on ``true_cfg > 1``.
        encode_negative = family.cpu_encode_negative(kwargs)
        kwargs.update(
            _cpu_encode_prompts(
                pipeline,
                positive,
                negative,
                encode_negative=encode_negative,
            )
        )
        embeds = None
    elif family.prompt_strategy(pipeline) == "plain":
        # Z-Image, Qwen-Image and Krea 2 all use a single, long-context
        # text encoder (Qwen3 / Qwen2.5-VL / Qwen3-VL) rather than SDXL's
        # dual CLIP, so the compel chunking dance the SDXL path uses is both
        # unnecessary and incompatible — ``Compel`` insists on a dual-
        # tokenizer SDXL shape it can't build against these pipelines.
        # Pass the prompts as plain strings and let the pipeline's own
        # ``encode_prompt`` handle the (much larger) token window.
        kwargs["prompt"] = positive
        kwargs["negative_prompt"] = negative
        embeds = None
    else:
        embeds = _encode_long_prompt(pipeline, positive, negative)
        if embeds is None:
            kwargs["prompt"] = positive
            kwargs["negative_prompt"] = negative
        else:
            kwargs.update(embeds)
    # Run inside ``inference_mode`` so PyTorch doesn't keep autograd
    # tensors around for the entire denoising loop. Diffusers does
    # this internally in newer versions but pinning it here is cheap
    # insurance — autograd buffers were a measurable VRAM tax on
    # older diffusers releases. Falls back to a no-op context when
    # torch isn't installed (test stubs land here).
    try:
        import torch

        # Qwen and Krea 2 run under ``no_grad`` rather than
        # ``inference_mode``: the torchao fp8 tensor subclass trips
        # ``inference_mode``'s strict functionalization. ``no_grad`` is
        # equivalent for inference and doesn't. SDXL / Z-Image keep
        # ``inference_mode``. Which one is the family's call.
        ctx = family.inference_context(torch)
    except ImportError:
        from contextlib import nullcontext

        ctx = nullcontext()
    _install_abort_callback(pipeline, kwargs, abort)
    with ctx:
        result = _run_qwen_call(pipeline, kwargs) if cpu_encode else pipeline(**kwargs)
    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    # Drop refs to the heaviest objects BEFORE returning so the
    # caller's ``_release_vram`` empty_cache has something to free.
    # Without these explicit deletes the closure keeps prompt embeds
    # and the result tensor alive until the to_thread worker exits,
    # which on a tight VRAM budget is enough to OOM the next call.
    del image
    del result
    del kwargs
    if embeds is not None:
        del embeds
    return png_bytes


def _encode_long_prompt(
    pipeline: Any,
    positive: str,
    negative: str,
) -> dict[str, Any] | None:
    """Encode prompts past CLIP's 77-token window.

    Returns the keyword-argument dict that should replace ``prompt`` /
    ``negative_prompt`` on the pipeline call, or ``None`` when the
    pipeline isn't SDXL-shaped (single-encoder models like Qwen-Image /
    Krea 2 / Z-Image have no 77-token cliff) or encoding failed — in
    which case the caller falls back to plain-string prompts, with
    diffusers' silent truncation behaviour.

    The chunking itself lives in ``providers.clip_long_prompt``; see
    that module for why it replaced the ``compel`` dependency and how
    the chunk/pool split works. Nothing here is cached on the pipeline:
    unlike compel, the encoder registers no forward hooks, so there is
    no per-render hook (and no pinned VRAM) to accumulate.
    """
    return clip_long_prompt.encode_long_prompt(pipeline, positive, negative)


def _resolve_pipeline_device(pipeline: Any) -> str:
    """Return the device string the pipeline is currently on
    (``"cuda"`` / ``"mps"`` / ``"cpu"``). Generator must match this
    so ``manual_seed`` actually takes effect — diffusers silently
    swaps in a fresh random generator otherwise."""
    dev = getattr(pipeline, "device", None)
    if dev is not None:
        return str(dev)
    # SDXL pipelines don't always expose ``.device`` directly; fall
    # back to whichever device the unet's parameters live on.
    unet = getattr(pipeline, "unet", None)
    if unet is not None:
        try:
            return str(next(unet.parameters()).device)
        except (StopIteration, AttributeError):
            pass
    try:
        import torch
    except ImportError:
        return "cpu"
    accel = _accelerator_module(torch)
    if accel is torch.cuda:
        return "cuda"
    if accel is getattr(torch, "xpu", None) and accel is not None:
        return "xpu"
    if accel is getattr(torch, "mps", None) and accel is not None:
        return "mps"
    return "cpu"


def _is_oom_error(exc: BaseException) -> bool:
    """Best-effort detection of an out-of-memory failure on ANY backend.

    Diffusers / PyTorch raise ``torch.cuda.OutOfMemoryError`` on CUDA and
    ``RuntimeError`` with a "CUDA out of memory" /
    "CUBLAS_STATUS_NOT_INITIALIZED" message on older builds. Non-CUDA
    backends word it differently and used to fall through as "not an
    OOM", so an AMD (DirectML / ROCm) or Apple (MPS) user never got the
    eviction-and-retry recovery at all:

      * DirectML surfaces the D3D12 allocator failure —
        "Could not allocate tensor with N bytes",
        "DML allocator out of memory", ``E_OUTOFMEMORY``;
      * MPS reports "MPS backend out of memory" / "Insufficient
        memory on MPS device";
      * plain CPU allocation failures raise ``MemoryError``, or a
        RuntimeError phrased "DefaultCPUAllocator: can't allocate
        memory" / "Unable to allocate".

    Tests inject plain Exception objects with the magic substrings to
    drive eviction without a real accelerator runtime.
    """
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "out of memory",
            "cuda_error_out_of_memory",
            "e_outofmemory",
            "could not allocate",
            "can't allocate memory",
            "cannot allocate memory",
            "unable to allocate",
            "insufficient memory",
            "not enough memory",
        )
    )


# Env override for the resident-pipeline cap, so a player with a big
# card can keep both workflow checkpoints warm without a rebuild.
_PIPELINE_CAP_ENV = "LUCIDIUM_MAX_RESIDENT_PIPELINES"


def _resolve_pipeline_cap(explicit: int | None) -> int:
    """Resolve the resident-pipeline cap: explicit constructor argument,
    else the ``LUCIDIUM_MAX_RESIDENT_PIPELINES`` env var, else 1.
    Always at least 1 — a cap of 0 would evict the pipeline we just
    loaded and livelock the render."""
    if explicit is not None:
        return max(1, int(explicit))
    raw = os.environ.get(_PIPELINE_CAP_ENV, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            _log.warning(
                "embedded image client: ignoring non-integer %s=%r",
                _PIPELINE_CAP_ENV,
                raw,
            )
    return 1


def _release_vram() -> None:
    """Free the active accelerator's allocator cache without
    unloading any pipelines. ``empty_cache`` returns blocks the
    allocator has cached internally (latents, attention buffers,
    intermediate activations) back to the OS pool — model weights
    stay pinned in the loaded modules. Without this call, a
    sequence of distinct renders accumulates allocator fragmentation
    until subsequent renders OOM even though no pipeline is
    actually that big.

    Generic across CUDA / XPU / MPS; CPU-only installs are a no-op.
    Cheap when there's nothing to free; safe to call from any
    success / failure path.
    """
    try:
        import gc

        import torch

        gc.collect()
        accel = _accelerator_module(torch)
        if accel is None:
            return
        empty_cache = getattr(accel, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        # ipc_collect releases memory held by inter-process accelerator
        # contexts. CUDA-only attribute; mostly a no-op outside multi-
        # process scenarios but free to call when available.
        ipc_collect = getattr(accel, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except Exception:
                pass
    except ImportError:
        pass


# Detached VRAM-release tasks, kept alive against the GC. ``asyncio``
# only holds a weak reference to a task, so a release that outlives the
# coroutine that started it (the cancellation path in
# ``_release_vram_async``) would otherwise be collected mid-flight.
_PENDING_RELEASES: set[Any] = set()


async def _release_vram_async() -> None:
    """``_release_vram`` on a worker thread.

    It looks cheap but isn't: ``gc.collect()`` plus the allocator's
    ``empty_cache`` walks and unmaps every cached block, which on a card
    holding an SDXL working set is tens to hundreds of milliseconds of
    fully synchronous work. Every render passes through here at least
    once, so on the loop thread it lands as a hitch in whatever dialog
    is streaming.

    Runs as a detached task behind a shield because most call sites are
    ``finally`` blocks: if a cancellation arrives mid-render, awaiting
    directly would abandon the release and strand the allocator blocks
    the next render needs. The shield lets the cancellation propagate
    while the release still runs to completion.
    """
    task = asyncio.create_task(asyncio.to_thread(_release_vram))
    _PENDING_RELEASES.add(task)
    task.add_done_callback(_PENDING_RELEASES.discard)
    await asyncio.shield(task)


async def _release_evicted(pipelines: list[Any]) -> None:
    """Free a batch of dropped pipelines off the loop thread. Called on
    the OOM-recovery path, where each entry is multiple GB."""

    def _run() -> None:
        for pipeline in pipelines:
            _release_pipeline(pipeline)
        _release_vram()

    await asyncio.to_thread(_run)


def _log_vram_diagnostics(label: str) -> None:
    """Log per-render VRAM diagnostics: current allocated, peak
    allocated since last reset, and free / total on the device.
    Resets the peak tracker after logging so the next render's
    peak is measured against THIS render's baseline rather than
    an ever-growing session-wide max.

    Costs ~one driver query per render — negligible. Output goes
    to the call log alongside the existing ``image-call`` lines so
    a tail-friendly grep can pull both together when debugging an
    OOM. Disabled silently when no accelerator is reachable, and
    backend-aware: CUDA / ROCm / XPU expose the full surface
    (allocated, peak, free, total); MPS exposes only ``empty_cache``
    so the diagnostic logs what it can and leaves the rest at
    "0.0" / "?".
    """
    try:
        import torch

        accel = _accelerator_module(torch)
        if accel is None:
            return
        backend = (
            "cuda"
            if accel is torch.cuda
            else "xpu"
            if accel is getattr(torch, "xpu", None)
            else "mps"
            if accel is getattr(torch, "mps", None)
            else "?"
        )
        mem_alloc = getattr(accel, "memory_allocated", None)
        max_alloc = getattr(accel, "max_memory_allocated", None)
        mem_info = getattr(accel, "mem_get_info", None)
        reset_peak = getattr(accel, "reset_peak_memory_stats", None)

        allocated = (mem_alloc() / (1024**3)) if callable(mem_alloc) else 0.0
        peak = (max_alloc() / (1024**3)) if callable(max_alloc) else 0.0
        try:
            if callable(mem_info):
                free_bytes, total_bytes = mem_info()
                free_gb = free_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
            else:
                free_gb = total_gb = 0.0
        except Exception:
            free_gb = total_gb = 0.0
        _call_log.info(
            "vram-diag backend=%s %s allocated=%.2f GiB peak=%.2f GiB free=%.2f / %.2f GiB",
            backend,
            label,
            allocated,
            peak,
            free_gb,
            total_gb,
        )
        if callable(reset_peak):
            try:
                reset_peak()
            except Exception:
                pass
    except ImportError:
        return
    except Exception:
        return


def _release_pipeline(pipeline: Any) -> None:
    """Drop refs + nudge the accelerator allocator so the freed VRAM
    is actually returned to the pool. Without ``empty_cache`` the
    backend allocator holds onto the freed blocks for reuse, which
    defeats the point of evicting a pipeline to make room for a
    different one. Backend-aware via ``_accelerator_module``."""
    try:
        if hasattr(pipeline, "to"):
            pipeline.to("cpu")
    except Exception:
        pass
    try:
        import gc

        import torch

        del pipeline
        gc.collect()
        accel = _accelerator_module(torch)
        if accel is not None:
            empty_cache = getattr(accel, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
    except ImportError:
        pass


def _pipeline_name_or_path(pipeline: Any) -> str:
    """Lower-cased ``config._name_or_path`` for the pipeline, or ``""``
    when diffusers' config metadata is missing (e.g. test stubs)."""
    cfg = getattr(pipeline, "config", None)
    if cfg is None:
        return ""
    return str(getattr(cfg, "_name_or_path", "") or "").lower()


# ---------------------------------------------------------------------------
# Model-family registry
# ---------------------------------------------------------------------------
#
# Every checkpoint family the embedded backend supports differs at exactly
# four call sites:
#
#   1. :meth:`EmbeddedImageClient.generate` — whether the SDXL face-detail
#      inpaint pass can run at all (``supports_sdxl_face_inpaint``).
#   2. :meth:`EmbeddedImageClient.regenerate_expression` — same question for
#      the cheap expression refresh (same hook).
#   3. :meth:`EmbeddedImageClient.regenerate_img2img` — whether the Qwen
#      img2img fast path owns this pipeline (``supports_qwen_img2img``).
#   4. :func:`_run_pipeline` — sampling recipe (``sampling_kwargs``), how
#      prompts reach the pipeline (``prompt_strategy`` / ``cpu_encode_attr``
#      / ``cpu_encode_negative``), and the autograd context
#      (``inference_context``).
#
# Previously each of those sites open-coded an if/elif chain over four
# ``_is_*_pipeline`` predicates, and because the sniffers overlap (Krea 2's
# components are Qwen-derived, Z-Image ships as "Z-Image-Turbo") the
# *evaluation order* was load-bearing and duplicated per site. Adding a
# family meant finding every chain and getting the order right, with no
# interface to check against.
#
# Now each family is one class owning its own sniffer plus all four call-site
# hooks, and precedence lives in exactly one place: ``_FAMILY_PRECEDENCE``.
# ``ModelFamily`` is an ABC with no defaults, so a new family that forgets a
# hook fails to instantiate rather than silently inheriting SDXL behaviour.


class ModelFamily(ABC):
    """One checkpoint family and everything the render paths do
    differently for it. Subclasses MUST implement every hook — no
    defaults, so adding a family cannot silently miss a branch point.
    """

    #: Stable identifier, used in logs and tests.
    name: str = ""

    @abstractmethod
    def matches(self, pipeline: Any) -> bool:
        """True when ``pipeline`` belongs to this family.

        Sniffers may overlap; :func:`resolve_model_family` resolves the
        ambiguity by ``_FAMILY_PRECEDENCE`` order.
        """

    @abstractmethod
    def supports_sdxl_face_inpaint(self, pipeline: Any) -> bool:
        """True when the SDXL face-detail / expression inpaint pipeline
        can be built from this pipeline's components (needs a UNet plus
        the dual CLIP text encoders)."""

    @abstractmethod
    def supports_qwen_img2img(self, pipeline: Any) -> bool:
        """True when :meth:`EmbeddedImageClient.regenerate_img2img` may
        drive this pipeline through the Qwen img2img runner."""

    @abstractmethod
    def sampling_kwargs(self, pipeline: Any) -> dict[str, Any]:
        """Step count / guidance keyword arguments for one inference
        pass. May read per-pipeline recipe flags stamped at load."""

    @abstractmethod
    def prompt_strategy(self, pipeline: Any) -> str:
        """``"clip_long"`` to route prompts through the SDXL 77-token
        chunker, or ``"plain"`` to hand the pipeline raw strings (single
        long-context text encoders have no 77-token cliff)."""

    @abstractmethod
    def cpu_encode_attr(self, pipeline: Any) -> str | None:
        """Name of the pipeline attribute marking CPU-encode-resident
        mode, or ``None`` when the family has no such mode."""

    @abstractmethod
    def cpu_encode_negative(self, kwargs: dict[str, Any]) -> bool:
        """Whether to encode the negative prompt in CPU-encode mode,
        given the family's own guidance kwargs."""

    @abstractmethod
    def inference_context(self, torch: Any) -> Any:
        """Torch context manager for the denoising loop."""


class _SdxlFamily(ModelFamily):
    """Plain multi-step SDXL — the fallback family.

    ``matches`` returns ``False`` by design: this family is never
    sniffed for, it is what :func:`resolve_model_family` falls back to
    when no other family claims the pipeline. That keeps "exactly one
    family matches a given pipeline" true for every real checkpoint.
    """

    name = "sdxl"

    def matches(self, pipeline: Any) -> bool:
        return False

    def supports_sdxl_face_inpaint(self, pipeline: Any) -> bool:
        return True

    def supports_qwen_img2img(self, pipeline: Any) -> bool:
        return False

    def sampling_kwargs(self, pipeline: Any) -> dict[str, Any]:
        return {"num_inference_steps": 25, "guidance_scale": 7.0}

    def prompt_strategy(self, pipeline: Any) -> str:
        return "clip_long"

    def cpu_encode_attr(self, pipeline: Any) -> str | None:
        return None

    def cpu_encode_negative(self, kwargs: dict[str, Any]) -> bool:
        return True

    def inference_context(self, torch: Any) -> Any:
        return torch.inference_mode()


class _TurboFamily(_SdxlFamily):
    """SDXL Turbo: 1 step, CFG off. Structurally SDXL (same UNet + dual
    CLIP), so it inherits the SDXL hooks and overrides only the recipe.
    Its ``turbo`` name marker also appears in "Z-Image-Turbo", which is
    why it sits AFTER Z-Image in ``_FAMILY_PRECEDENCE``."""

    name = "turbo"

    def matches(self, pipeline: Any) -> bool:
        return "turbo" in _pipeline_name_or_path(pipeline)

    def sampling_kwargs(self, pipeline: Any) -> dict[str, Any]:
        return {"num_inference_steps": 1, "guidance_scale": 0.0}


class _ZImageFamily(ModelFamily):
    """Z-Image, any variant (text2img, img2img, inpaint).

    Class-name match is the primary signal because ``ZImagePipeline``
    is imported on demand inside :func:`_default_pipeline_factory` and
    the sniffer must work even when diffusers isn't installed (the SDXL
    fake-pipeline tests would otherwise fail to import). Falls back to
    the shared ``_name_or_path`` substring so test stubs can opt in by
    setting ``config._name_or_path = "Z-Image-Turbo"``.
    """

    name = "z_image"

    def matches(self, pipeline: Any) -> bool:
        if type(pipeline).__name__.startswith("ZImage"):
            return True
        lowered = _pipeline_name_or_path(pipeline)
        return "z-image" in lowered or "zimage" in lowered

    def supports_sdxl_face_inpaint(self, pipeline: Any) -> bool:
        # No UNet and no dual CLIP encoders, so the inpaint runner would
        # build nothing and silently return the base PNG.
        return False

    def supports_qwen_img2img(self, pipeline: Any) -> bool:
        return False

    def sampling_kwargs(self, pipeline: Any) -> dict[str, Any]:
        # Z-Image-Turbo recipe: 9 steps, guidance 0.0 (CFG OFF). With
        # ``guidance_scale > 0`` Z-Image's pipeline runs a second
        # transformer pass for the unconditional branch, which both
        # doubles VRAM and contradicts the model's Turbo training.
        # Negative prompts are still passed through — they're ignored
        # when ``do_classifier_free_guidance == False``.
        return {"num_inference_steps": 9, "guidance_scale": 0.0}

    def prompt_strategy(self, pipeline: Any) -> str:
        return "plain"

    def cpu_encode_attr(self, pipeline: Any) -> str | None:
        return None

    def cpu_encode_negative(self, kwargs: dict[str, Any]) -> bool:
        return True

    def inference_context(self, torch: Any) -> Any:
        return torch.inference_mode()


class _KreaFamily(ModelFamily):
    """Krea 2.

    Class-name match is the primary signal — ``Krea2Pipeline`` is
    imported on demand inside the loader, so keying off the name keeps
    the sniffer working when diffusers is absent or predates 0.39.
    Falls back to the shared ``_name_or_path`` substring so a test stub
    can opt in with ``config._name_or_path = "Krea-2"``.

    Ordering hazard: Krea 2's components are Qwen-derived, so a Krea
    checkpoint whose ``_name_or_path`` mentions Qwen also satisfies the
    Qwen sniffer — hence Krea precedes Qwen in ``_FAMILY_PRECEDENCE``.
    """

    name = "krea"

    def matches(self, pipeline: Any) -> bool:
        if type(pipeline).__name__.startswith("Krea2"):
            return True
        return "krea" in _pipeline_name_or_path(pipeline)

    def supports_sdxl_face_inpaint(self, pipeline: Any) -> bool:
        # Transformer-only: neither a UNet nor CLIP encoders.
        return False

    def supports_qwen_img2img(self, pipeline: Any) -> bool:
        return False

    def sampling_kwargs(self, pipeline: Any) -> dict[str, Any]:
        # Krea 2's recipe depends on the variant (see _krea_recipe): the
        # Turbo/TDM distill renders in 8 flow-match steps with guidance
        # OFF, the raw midtrain base needs 28 steps at guidance 4.5.
        # Krea 2's CFG convention is ``cond + scale * (cond - uncond)``,
        # so 0.0 (not 1.0) is the "guidance off" value; the negative
        # prompt is only consulted above 0.
        steps, guidance = _krea_recipe(pipeline)
        return {
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            # The transformer consumes a fixed-length text block; 512 is
            # the upstream default and comfortably fits a composed
            # portrait prompt (no CLIP-style truncation to work around).
            "max_sequence_length": 512,
        }

    def prompt_strategy(self, pipeline: Any) -> str:
        return "plain"

    def cpu_encode_attr(self, pipeline: Any) -> str | None:
        return _KREA_CPU_ENCODE_ATTR

    def cpu_encode_negative(self, kwargs: dict[str, Any]) -> bool:
        return bool(kwargs.get("guidance_scale", 0.0) > 0)

    def inference_context(self, torch: Any) -> Any:
        # ``no_grad`` rather than ``inference_mode``: the torchao fp8
        # tensor subclass trips ``inference_mode``'s strict
        # functionalization. ``no_grad`` is equivalent for inference.
        return torch.no_grad()


class _QwenFamily(ModelFamily):
    """Qwen-Image, any variant (text2img, img2img, inpaint, edit).

    Class-name match is the primary signal — ``QwenImagePipeline`` /
    ``QwenImageImg2ImgPipeline`` are imported on demand, so keying off
    the name keeps the sniffer working when diffusers isn't installed.
    Falls back to the shared ``_name_or_path`` substring so a test stub
    can opt in with ``config._name_or_path = "Qwen-Image"``.
    """

    name = "qwen"

    def matches(self, pipeline: Any) -> bool:
        if type(pipeline).__name__.startswith("QwenImage"):
            return True
        return "qwen" in _pipeline_name_or_path(pipeline)

    def supports_sdxl_face_inpaint(self, pipeline: Any) -> bool:
        # Historically the face-detail site only excluded Z-Image and
        # Krea; Qwen renders reach it and the runner declines internally
        # (there is no Qwen inpaint pipeline to build), and the Qwen
        # img2img fast path is chosen upstream by model path anyway.
        return True

    def supports_qwen_img2img(self, pipeline: Any) -> bool:
        return True

    def sampling_kwargs(self, pipeline: Any) -> dict[str, Any]:
        # Qwen-Image recipe depends on the variant (see _qwen_recipe):
        # the Lightning distill renders in ~8 flow-match steps with CFG
        # OFF (true_cfg_scale 1.0 → one transformer forward per step),
        # while the undistilled base needs more steps + real CFG. Real
        # negative-prompt steering rides on ``true_cfg_scale``;
        # ``guidance_scale`` is the distilled embedded-guidance knob.
        steps, true_cfg = _qwen_recipe(pipeline)
        return {
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg,
            "guidance_scale": 1.0,
        }

    def prompt_strategy(self, pipeline: Any) -> str:
        return "plain"

    def cpu_encode_attr(self, pipeline: Any) -> str | None:
        return _QWEN_CPU_ENCODE_ATTR

    def cpu_encode_negative(self, kwargs: dict[str, Any]) -> bool:
        return bool(kwargs.get("true_cfg_scale", 1.0) > 1)

    def inference_context(self, torch: Any) -> Any:
        # See _KreaFamily.inference_context — torchao fp8 needs no_grad.
        return torch.no_grad()


Z_IMAGE_FAMILY = _ZImageFamily()
KREA_FAMILY = _KreaFamily()
QWEN_FAMILY = _QwenFamily()
TURBO_FAMILY = _TurboFamily()
SDXL_FAMILY = _SdxlFamily()

#: Sniffed families in resolution order. The order is load-bearing and
#: documented per family:
#:
#:   1. ``z_image`` — ships as "Z-Image-Turbo", so it must outrank turbo.
#:   2. ``krea``    — Qwen-derived components, so it must outrank qwen.
#:   3. ``qwen``
#:   4. ``turbo``   — weakest signal (a bare "turbo" name substring).
#:
#: ``sdxl`` is not listed: it is the fallback when nothing matches.
_FAMILY_PRECEDENCE: tuple[ModelFamily, ...] = (
    Z_IMAGE_FAMILY,
    KREA_FAMILY,
    QWEN_FAMILY,
    TURBO_FAMILY,
)

#: Every family, including the fallback — for exhaustiveness checks.
ALL_MODEL_FAMILIES: tuple[ModelFamily, ...] = (*_FAMILY_PRECEDENCE, SDXL_FAMILY)

#: The call-site hooks every family must implement.
MODEL_FAMILY_HOOKS: tuple[str, ...] = (
    "matches",
    "supports_sdxl_face_inpaint",
    "supports_qwen_img2img",
    "sampling_kwargs",
    "prompt_strategy",
    "cpu_encode_attr",
    "cpu_encode_negative",
    "inference_context",
)


def resolve_model_family(pipeline: Any) -> ModelFamily:
    """Return the single :class:`ModelFamily` owning ``pipeline``.

    Walks ``_FAMILY_PRECEDENCE`` in order and returns the first match,
    falling back to :data:`SDXL_FAMILY`. This is the ONLY place family
    precedence is encoded — call sites ask the returned family for the
    behaviour they need instead of re-deriving it from the sniffers.
    """
    for family in _FAMILY_PRECEDENCE:
        if family.matches(pipeline):
            return family
    return SDXL_FAMILY


# Thin back-compat aliases over the family sniffers. Kept because they
# read well at a glance and several tests pin these names; the family
# classes above are the single source of truth for the detection logic.
def _is_turbo_pipeline(pipeline: Any) -> bool:
    """Detect SDXL Turbo by config marker."""
    return TURBO_FAMILY.matches(pipeline)


def _is_z_image_pipeline(pipeline: Any) -> bool:
    """Detect Z-Image (any variant: text2img, img2img, inpaint)."""
    return Z_IMAGE_FAMILY.matches(pipeline)


def _is_qwen_pipeline(pipeline: Any) -> bool:
    """Detect Qwen-Image (any variant: text2img, img2img, inpaint, edit)."""
    return QWEN_FAMILY.matches(pipeline)


def _is_krea_pipeline(pipeline: Any) -> bool:
    """Detect Krea 2."""
    return KREA_FAMILY.matches(pipeline)

"""Synchronous asset generation helper.

``render_scheduler.py`` owns the per-entity redraw pipeline; this
module is the simpler on-demand path used by the new-game and play
handlers. It generates
whatever images are missing for the current node + on-stage roster and
attaches them to the live ``Game``.

Image bytes are stored content-addressed under
``<save-id>/images/<prompt-hash>.png`` so identical prompts dedupe
across runs (Constitution III, Efficiency).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..config import saves_dir
from ..domain.character import Character, CharacterImage, CharacterKind
from ..domain.environment import Environment
from ..persistence.atomic import atomic_write_bytes
from ..providers.image_client import ImageClient
from .portrait_post import crop_to_figure
from .prompts import image_prompts

if TYPE_CHECKING:
    from ..domain.game import Game
    from ..domain.world import WorldState
    from ..providers.music_client import AceStepClient
    from .session import Session


class _ImageRegenerator(Protocol):
    """Shape of the optional ``regenerate_expression`` /
    ``regenerate_from_image`` fast paths. Only the embedded client
    implements them, so they are looked up with ``getattr`` and are
    ``None`` on backends that decline."""

    async def __call__(
        self, workflow: str, base_png: bytes, params: dict[str, object], *, seed: int
    ) -> bytes | None: ...


_log = logging.getLogger(__name__)

# Process-wide map of in-flight image renders, keyed by absolute
# target path. Used by ``_render_or_await`` to collapse the
# ensure_assets / ensure_portraits_for race where multiple
# concurrent commits each pass the ``target.exists()`` check before
# any of them writes the file — without this, the same prompt-hash
# was observed regenerating 4× back-to-back during fast playthrough
# (same seed, same prompt, identical output bytes). Entries are
# removed when the original generator's task finishes (success or
# failure). Bounded implicitly by the lifespan of those tasks.
_INFLIGHT_RENDERS: dict[str, asyncio.Event] = {}


async def _render_or_await(
    target: Path,
    *,
    render: Callable[[], Awaitable[bytes]],
    post_process: Callable[[bytes], bytes] = lambda b: b,
    safety_check: Callable[[bytes], bool] | None = None,
) -> bool:
    """Generate ``target`` exactly once across concurrent callers.

    If another coroutine is already rendering the same path, await
    its completion and return ``True`` if the file landed (the
    other render succeeded). Otherwise call ``render`` ourselves,
    optionally post-process the bytes (e.g. ``crop_to_figure``),
    optionally run a safety check that can veto the write, and
    write atomically. Returns ``True`` iff ``target`` exists at
    the end.

    Doesn't take a session — the dedup is purely path-based, which
    is correct because identical prompt-hashes resolve to identical
    paths.

    Each decision (cache hit, dedup wait, fresh render) logs at
    INFO level so a session log can be cross-referenced against
    the embedded_image_client's ``image-call`` lines without
    guessing whether a missing ``image-call`` means "it ran but
    didn't log" or "it never ran because dedup served the file
    from disk".
    """
    target_label = target.name
    # ASYNC240 is suppressed here (and at the other local stat() calls in
    # this module): a stat() on a local save directory is microseconds, and
    # awaiting it would insert a suspension point between the existence
    # check and the in-flight registration below, opening the dedup race
    # this function exists to close.
    if target.exists():  # noqa: ASYNC240
        _log.info(
            "image dedup HIT (already on disk): %s — skipping render",
            target_label,
        )
        return True
    key = str(target)
    existing = _INFLIGHT_RENDERS.get(key)
    if existing is not None:
        # Another caller is already rendering this exact prompt
        # hash. Wait for them, then trust the file (or fall through
        # to a fresh render if they failed).
        _log.info(
            "image dedup WAIT: %s — another coroutine is rendering this hash",
            target_label,
        )
        await existing.wait()
        if target.exists():  # noqa: ASYNC240  -- local stat(), see above
            _log.info(
                "image dedup WAIT-HIT: %s landed via the other coroutine",
                target_label,
            )
            return True
        # Original render failed silently — fall through and try
        # again ourselves. Don't recurse — just generate inline
        # without re-registering, since the original entry is gone.
        # (Race where two callers both fall through is bounded by
        # the atomic write below.)
        _log.info(
            "image dedup WAIT-MISS: %s did not land (other coroutine failed); rendering ourselves",
            target_label,
        )
    event = asyncio.Event()
    _INFLIGHT_RENDERS[key] = event
    try:
        _log.info("image render START: %s", target_label)
        payload = await render()
        # ``safety_check`` and ``post_process`` are both CPU-bound and
        # measured in the tens-to-hundreds of milliseconds — the safety
        # filter constructs NudeDetector + insightface on first use and
        # runs two ONNX inferences plus a tempfile write, and
        # ``crop_to_figure`` decodes and re-encodes a 832x1216 RGBA PNG.
        # Run on the loop thread they stall every in-flight dialog
        # stream for the duration, so each hops to a worker thread. The
        # atomic write joins them: it's an fsync, and there's no reason
        # to pay for it here either.
        if safety_check is not None and not await asyncio.to_thread(safety_check, payload):
            _log.info(
                "image render SAFETY-DROP: %s — safety_check vetoed payload",
                target_label,
            )
            return False
        processed = await asyncio.to_thread(post_process, payload)
        await asyncio.to_thread(atomic_write_bytes, target, processed)
        _log.info("image render WROTE: %s", target_label)
        return True
    except BaseException as exc:
        _log.warning(
            "image render FAILED: %s — %s: %s",
            target_label,
            type(exc).__name__,
            exc,
        )
        raise
    finally:
        # Always release waiters and drop the slot — even on
        # failure or cancellation. A waiter that woke to find
        # ``target.exists() is False`` will fall through to its
        # own render attempt.
        event.set()
        _INFLIGHT_RENDERS.pop(key, None)


@dataclass
class GeneratedAsset:
    kind: str  # "portrait" | "background" | "music"
    target_id: str
    image_path: Path
    prompt_hash: str


def _safety_retcon_age_correction(
    session: Session, character_id: str, name: str, old_age: int
) -> None:
    """Spawn a fire-and-forget retcon task that rewrites past
    history references to the character's prior age.

    Runs after the immediate age mutation + notice. Failures are
    logged and swallowed: the storage age is already corrected,
    the SDXL prompt is already safe, and the modal has already
    informed the player. The history rewrite is a polish pass.
    """
    import asyncio as _asyncio

    instructions = (
        f"Retcon: the character {name} (id: {character_id}) is now 18 years old. "
        f"Walk the committed history and rewrite any narration or dialogue that "
        f"described {name} as younger than 18 (e.g. explicit ages below 18 like "
        f"'{old_age}-year-old' or 'fifteen', references to them as 'a child', "
        f"'a kid', 'a minor', or 'a teenager of {old_age}'). Keep their name, "
        f"personality, role in the plot, and every other established trait "
        f"unchanged. Update character.description ONLY if the existing "
        f"description literally states an age younger than 18."
    )

    async def _run() -> None:
        from ..api.handlers import HandlerContext, _full_state_message, apply_global_retcon
        from ..api.messages import MessageType

        try:
            ctx = HandlerContext(session=session)
            await apply_global_retcon(ctx, instructions, push_undo=True)
        except Exception:
            _log.exception("safety retcon failed for char=%s", character_id)
            return
        # Push the rewritten history out — the modal already
        # explained why; this is what makes the retcon actually
        # visible in the rendered scrollback.
        if session.emit is not None and session.game is not None:
            session.emit(
                MessageType.s2c_state_full,
                _full_state_message(session),
            )

    try:
        loop = _asyncio.get_running_loop()
    except RuntimeError:
        # No running loop (e.g. called from a test that mutated
        # state directly). Skip the retcon — the in-memory age is
        # already corrected; that's what the SDXL prompt cares about.
        _log.info(
            "no running loop; skipping safety retcon for char=%s "
            "(age mutation + notice already applied)",
            character_id,
        )
        return
    from ..background import spawn

    spawn(_run(), label=f"safety-retcon:{character_id}", loop=loop)


def check_minor_nudity_and_correct(session: Session) -> list[str]:
    """Scan ``session.game.characters`` for any human character who is
    described with an outfit that would trigger the SDXL nudity tag
    (``"nude"`` / ``"mostly nude"``) AND whose stored ``age`` is
    below 18.

    For every match: bump their age to 18 in storage, install the
    revised game, emit ``s2c/notice`` so the renderer pops a modal,
    and fire-and-forget a global retcon so any past narration
    that referenced the prior age gets rewritten.

    Returns the list of corrected character ids (empty when no
    correction was needed). Idempotent: the next call sees age=18
    on already-corrected characters and short-circuits.

    The age floor inside ``character.age_band`` ALSO catches this
    (it raises any sub-18 age to 18 inside the prompt builder), so
    even if this scan misses for any reason the SDXL pipeline
    never sees a minor. This function exists so the player gets a
    visible signal AND the stored age gets corrected to match the
    rendered intent.
    """
    game = session.game
    if game is None:
        return []
    new_chars = dict(game.characters)
    corrections: list[tuple[str, str, int]] = []
    for cid, char in game.characters.items():
        if char.kind != CharacterKind.human:
            continue
        if char.age >= 18:
            continue
        if not image_prompts.is_nudity_outfit(char.outfit):
            continue
        corrections.append((cid, char.name, char.age))
        new_chars[cid] = char.model_copy(update={"age": 18})

    if not corrections:
        return []

    new_game = game.model_copy(update={"characters": new_chars})
    session.install_game(new_game)
    try:
        # ``commit_blocking``, not ``await commit()``: this scan stays a
        # plain function (it's called from sync test code and from three
        # async sites) and it only reaches this line when a correction
        # actually fired, which is rare enough that a one-off blocking
        # write isn't worth making the whole call chain async.
        session.commit_blocking()
    except Exception:
        _log.warning("commit failed after age correction", exc_info=True)

    for cid, name, old_age in corrections:
        _log.warning(
            "minor-nudity guard tripped: bumped %s (%s) from age=%d → 18",
            name,
            cid,
            old_age,
        )
        if session.emit is not None:
            from ..api.messages import MessageType, NoticeKind, S2CNotice

            session.emit(
                MessageType.s2c_notice,
                S2CNotice(
                    title="Age safety correction",
                    body=(
                        f"{name} was described in a way that combined an under-18 "
                        f"age with nudity, which the engine never sends to the "
                        f"image renderer. Their age has been corrected to 18, and "
                        f"a quick rewrite is updating any narration that "
                        f"referenced their previous age."
                    ),
                    kind=NoticeKind.warning,
                ),
            )
        _safety_retcon_age_correction(session, cid, name, old_age)

    return [cid for cid, _name, _old in corrections]


def _save_root_for(game_id: str, root: Path | None) -> Path:
    return (root or saves_dir()) / game_id


def _images_dir(game_id: str, root: Path | None) -> Path:
    return _save_root_for(game_id, root) / "images"


def _hash_prompt(*parts: str) -> str:
    payload = "\x1e".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# Filename substrings that mark a checkpoint as a long-prompt model
# (Qwen-Image, Flux, Z-Image). These read natural-language prose rather
# than SDXL/Pony weighted-tag soup, so their prompts are built by the
# ``long_context`` branch of the image_prompts builders — see
# :func:`_long_context_image_model`. Kept aligned with the pipeline
# sniffs in ``providers/embedded_image_client.py``.
_LONG_CONTEXT_MODEL_MARKERS: tuple[str, ...] = (
    "qwen-image",
    "qwen_image",
    "qwenimage",
    "z-image",
    "zimage",
    "tongyi-mai",
    "flux",
)


def _long_context_image_model(session: Session, *, environment: bool = False) -> bool:
    """True when the active image checkpoint handles long natural-language
    prompts (Qwen-Image / Flux / Z-Image) as opposed to SDXL/Pony, which
    truncate at CLIP's 77-token window and rely on ``(tag:weight)`` syntax.

    Detection keys off the checkpoint FILENAME using the same markers the
    embedded factory uses to pick its pipeline. Crucially we resolve the
    file that will ACTUALLY load — running ``pick_default_model`` against
    the configured name — rather than trusting the settings string alone:
    when the player downloaded a model via the first-run wizard and left
    the model-name field empty (the common "auto-pick the only checkpoint"
    case), the settings name is blank but the on-disk file is e.g.
    ``Z-Image-Turbo.safetensors``. Sniffing only the settings string
    missed that and silently fell back to the terse SDXL prompt.

    ``environment`` selects the background checkpoint over the portrait
    one. The ComfyUI backend keeps its model inside the workflow JSON
    rather than a settings filename, so we sniff the workflow filename as
    a best-effort opt-in for ComfyUI users running a Qwen/Flux workflow.
    """
    from ..domain.settings import ImageBackend

    try:
        img = session.settings.image
    except Exception:
        return False

    if img.backend == ImageBackend.comfyui:
        workflow = img.background_workflow if environment else img.portrait_workflow
        name = (workflow or "").lower()
        return any(marker in name for marker in _LONG_CONTEXT_MODEL_MARKERS)

    # Embedded: mirror EmbeddedImageClient._resolve_target_name — the
    # per-workflow override falls back to the legacy single model name.
    configured = (
        (img.embedded_environment_model_name if environment else img.embedded_character_model_name)
        or img.embedded_model_name
        or ""
    )
    resolved_name = configured
    try:
        from ..providers.embedded_models import pick_default_model, resolve_models_dir

        models_dir = resolve_models_dir(img.embedded_models_dir)
        path = pick_default_model(models_dir, configured)
        if path is not None:
            resolved_name = path.name
    except Exception:
        pass
    name = resolved_name.lower()
    return any(marker in name for marker in _LONG_CONTEXT_MODEL_MARKERS)


def _portrait_prompt_hash(
    world: WorldState, character: Character, lighting: str = "", *, long_context: bool = False
) -> str:
    from ..domain.character import CharacterKind

    return _hash_prompt(
        image_prompts.portrait_prompt(
            world=world,
            character=character,
            lighting=lighting,
            long_context=long_context,
        ),
        image_prompts.portrait_face_prompt(character=character),
        str(character.seed),
        # Fold ``kind`` into the hash so a save that flipped a
        # character's kind gets a fresh re-render rather than
        # serving the stale cached file from the previous
        # pipeline (different model, different output).
        f"kind:{character.kind.value}" if character.kind == CharacterKind.nonhuman else "",
    )


def _active_lighting(game: Game) -> str:
    """Lighting descriptor of the most recently-walked node's
    environment, or empty string if none. Threaded into portrait
    prompts so character renders match the room they're standing in."""
    env_id = _resolve_active_environment_id(game)
    if env_id is None:
        return ""
    env = game.environments.get(env_id)
    return env.lighting if env else ""


def _background_prompt_hash(
    world: WorldState, environment: Environment, *, long_context: bool = False
) -> str:
    # Seed mixes in so a "rerender" (which bumps the seed) produces
    # a different content hash and therefore a different cached
    # filename — without the seed in the hash, the same prompt
    # always maps to the same PNG on disk and the dedup short-
    # circuits regeneration. Mirrors how _portrait_prompt_hash
    # folds in character.seed.
    return _hash_prompt(
        image_prompts.background_prompt(
            world=world, environment=environment, long_context=long_context
        ),
        str(environment.seed),
    )


def _resolve_active_environment_id(game: Game) -> str | None:
    """Walk back along the committed path until we find a node with a
    location (FR-035 fallback)."""
    if not game.dialog_tree.committed_path:
        return None
    nodes = game.dialog_tree.nodes
    for nid in reversed(game.dialog_tree.committed_path):
        node = nodes.get(nid)
        if node and node.location_id:
            return node.location_id
    return None


def _portrait_already_current(character: Character, expected_hash: str) -> bool:
    return any(image.prompt_hash == expected_hash for image in character.images)


def _portrait_identity_hash(
    world: WorldState,
    character: Character,
    lighting: str = "",
    *,
    long_context: bool = False,
) -> str:
    """Hash of every portrait input EXCEPT the expression.

    Built by blanking ``expression`` and reusing the exact same hashing
    as :func:`_portrait_prompt_hash`, so two renders that differ ONLY in
    expression collide here while any other change (outfit, pose,
    lighting, seed, kind, appearance) produces a different identity.

    When a fresh render's ``prompt_hash`` is new but its identity matches
    an existing portrait, the expression is the lone difference — and the
    embedded backend can refresh just the face via inpaint instead of
    re-rendering the whole figure.
    """
    neutral = character.model_copy(update={"expression": ""})
    return _portrait_prompt_hash(world, neutral, lighting, long_context=long_context)


def _find_expression_base(character: Character, identity_hash: str) -> CharacterImage | None:
    """An existing on-disk portrait whose identity matches ``identity_hash``
    — i.e. a previous render of this exact character that differs only by
    expression. Returns ``None`` when there's no such image (nothing to
    inpaint from; the caller renders fully). Most-recent images come
    first in ``character.images``, so the freshest match wins."""
    if not identity_hash:
        return None
    for image in character.images:
        if (
            image.identity_hash
            and image.identity_hash == identity_hash
            and Path(image.path).exists()
        ):
            return image
    return None


def _find_latest_portrait(character: Character) -> CharacterImage | None:
    """The most recent on-disk portrait for this character, regardless of
    identity — the freshest base for an img2img "character change". Unlike
    :func:`_find_expression_base` it does NOT require an identity match: a
    Qwen character-change edit works from whatever the last rendered
    portrait was. ``character.images`` is most-recent-first, so the first
    entry whose file still exists wins. Returns ``None`` for a character
    with no prior portrait (the first render must be a full text2img)."""
    for image in character.images:
        if Path(image.path).exists():
            return image
    return None


def _build_portrait_render(
    *,
    session: Session,
    image_client: ImageClient,
    character: Character,
    positive: str,
    face: str,
    negative_extras: str,
    identity_hash: str,
) -> Callable[[], Awaitable[bytes]]:
    """Build the ``render()`` closure for ``_render_or_await``.

    Render-strategy precedence, each falling back to the next when the
    backend declines or no suitable base exists on disk:

      1. **Qwen img2img character change** — when the backend exposes
         ``regenerate_from_image`` (the embedded Qwen path) AND any prior
         portrait of this character is on disk, edit that portrait toward
         the new prompt instead of rendering from fresh noise. Keeps the
         character's identity / framing while applying the requested
         outfit / pose / expression change. ``regenerate_from_image``
         returns ``None`` on non-Qwen backends, so this is a no-op for
         SDXL / Z-Image.
      2. **SDXL expression-only refresh** — when ``regenerate_expression``
         is available AND a prior portrait with the SAME identity (only
         the expression changed) is on disk, inpaint just the face.
      3. **Full ``generate``** — a fresh text2img render.

    The closure always yields a valid portrait: every fast path falls
    back to the full render on decline (non-human subject, a Z-Image
    checkpoint, no detectable face, no prior image, or a read/render
    error).
    """
    workflow = session.settings.image.portrait_workflow
    seed = character.seed
    kind = character.kind.value
    full_params: dict[str, object] = {
        "positive_prompt": positive,
        "face_prompt": face,
        "negative_extras": negative_extras,
        # ``subject_kind`` switches the embedded backend to the
        # environment SDXL checkpoint for nonhuman subjects AND skips the
        # face-detail inpaint pass. Background removal still runs (gated
        # on the workflow, not the kind) so a nonhuman portrait still
        # arrives as a transparent cut-out figure.
        "subject_kind": kind,
    }

    async def _full() -> bytes:
        return await image_client.generate(workflow, full_params, seed=seed)

    # ---- (2) SDXL expression-only refresh, else full render -------------
    regen: _ImageRegenerator | None = getattr(image_client, "regenerate_expression", None)
    base = _find_expression_base(character, identity_hash) if callable(regen) else None
    # ``regen is None`` is implied by ``base is None`` (the lookup above
    # only searches for a base when the backend exposes the fast path);
    # it is spelled out so the closure below sees a narrowed callable.
    if base is None or regen is None:
        sdxl_path = _full
    else:
        base_path = base.path

        async def _expression_or_full() -> bytes:
            try:
                base_png = await asyncio.to_thread(Path(base_path).read_bytes)
            except OSError:
                return await _full()
            result = await regen(
                workflow,
                base_png,
                {
                    "face_prompt": face,
                    "negative_extras": negative_extras,
                    "subject_kind": kind,
                },
                seed=seed,
            )
            if result is not None:
                _log.info(
                    "portrait expression-only refresh for %s via face inpaint",
                    character.id,
                )
                return result
            # Backend declined the fast path — render the portrait normally.
            return await _full()

        sdxl_path = _expression_or_full

    # ---- (1) Qwen img2img character change, else the SDXL path ----------
    img2img: _ImageRegenerator | None = getattr(image_client, "regenerate_from_image", None)
    latest = _find_latest_portrait(character) if callable(img2img) else None
    # A POSE change is structural: img2img edits the prior portrait in
    # place and preserves its composition, so it cannot reliably restage
    # a new pose — the user reported "changing pose and clicking rerender
    # doesn't rerender" precisely because the img2img path returned a
    # near-identical frame. When the pose differs from the base portrait,
    # skip img2img and fall to a full text2img (the stable per-character
    # seed still holds identity). Outfit / expression edits stay on the
    # identity-preserving img2img path.
    if latest is not None:
        base_pose = str((getattr(latest, "attributes_snapshot", None) or {}).get("pose", ""))
        if base_pose != str(getattr(character, "pose", "") or ""):
            _log.info(
                "portrait pose change for %s (%r -> %r): full text2img, "
                "skipping identity-preserving img2img",
                character.id,
                base_pose,
                character.pose,
            )
            latest = None
    # As above, ``img2img is None`` already implies ``latest is None``;
    # restated for the closure's benefit.
    if latest is None or img2img is None:
        return sdxl_path

    latest_path = latest.path

    async def _img2img_or_sdxl() -> bytes:
        try:
            base_png = await asyncio.to_thread(Path(latest_path).read_bytes)
        except OSError:
            return await sdxl_path()
        # Pass the full new prompt — img2img edits the prior portrait
        # toward the changed appearance, not just the face.
        result = await img2img(workflow, base_png, full_params, seed=seed)
        if result is not None:
            _log.info(
                "portrait character-change for %s via Qwen img2img (base=%s)",
                character.id,
                latest_path,
            )
            return result
        # Backend declined (not a Qwen checkpoint, or the render failed)
        # — fall back to the SDXL expression/full path.
        return await sdxl_path()

    return _img2img_or_sdxl


def _portrait_safety_check(session: Session, character: Character) -> Callable[[bytes], bool]:
    """Bind ``character`` into the output-filter closure.

    A factory rather than an inline lambda so the loop that builds these
    per-character checks captures THIS iteration's character instead of
    sharing the loop variable (which would make every queued check see
    the last character in the loop).
    """
    return lambda payload: _output_filter_passes(session, character, payload)


def _output_filter_passes(
    session: Session,
    character: Character,
    payload: bytes,
) -> bool:
    """Run the output-side ML content filter on a freshly-rendered
    portrait. Returns True if the caller should save the image,
    False if the image should be discarded.

    On a ``blocked`` verdict (nudity + estimated minor face): emit
    an ``s2c/notice`` so the renderer pops a modal, log the
    diagnostic detail, and return False. The caller skips
    ``atomic_write_bytes`` and the image never reaches disk.

    On ``unavailable`` (filter not installed): log once, allow the
    save. The prompt-side and storage-side guards still apply;
    the player can install the safety extras later.
    """
    from . import content_filter

    flt = content_filter.default_filter()
    try:
        result = flt.classify(payload)
    except Exception:
        _log.exception(
            "output filter raised on portrait for %s; allowing image through",
            character.id,
        )
        return True

    if result.verdict == content_filter.Verdict.blocked:
        _log.warning(
            "output filter BLOCKED portrait for %s (%s): %s",
            character.id,
            character.name,
            result.detail,
        )
        if session.emit is not None:
            from ..api.messages import MessageType, NoticeKind, S2CNotice

            session.emit(
                MessageType.s2c_notice,
                S2CNotice(
                    title="Portrait blocked by content filter",
                    body=(
                        f"The portrait the engine just rendered for "
                        f"{character.name} was rejected by the output "
                        f"safety filter — the image combined nudity with "
                        f"a face the age estimator scored as under 18. "
                        f"No image was saved. You can edit the "
                        f"character's outfit or description and rerender "
                        f"from the Cast panel."
                    ),
                    kind=NoticeKind.warning,
                ),
            )
        return False

    if result.verdict == content_filter.Verdict.unavailable:
        # SAFETY.md §3 promises this filter always runs. It isn't
        # running, so say so — ERROR in the log and a one-time modal
        # in the renderer — rather than degrading silently. The image
        # is still allowed through: the prompt-side age floor and the
        # storage-side correction remain active, and refusing every
        # portrait would be a worse failure mode than a loud warning.
        _log.debug(
            "output filter unavailable for %s; allowing through "
            "(prompt+storage guards still active)",
            character.id,
        )
        content_filter.report_unavailable(
            getattr(session, "emit", None),
            detail=result.detail,
        )
    return True


async def render_portrait_to_disk(
    *,
    session: Session,
    image_client: ImageClient,
    character: Character,
    lighting: str,
    images_dir: Path,
) -> CharacterImage | None:
    """Render ONE character's CURRENT state to a content-addressed PNG
    and return the ``CharacterImage`` describing it — WITHOUT touching
    ``session.game``.

    The single render primitive shared by ``ensure_portraits_for`` and
    the per-entity :class:`RenderScheduler`. Idempotent on disk via
    ``_render_or_await``. Returns ``None`` when nothing landed (the
    output safety filter vetoed the payload, or the render failed) so
    the caller leaves the previous portrait in place.
    """
    # Callers only reach the render primitives with a loaded game.
    assert session.game is not None
    world = session.game.world
    long_context = _long_context_image_model(session)
    prompt_hash = _portrait_prompt_hash(world, character, lighting, long_context=long_context)
    identity_hash = _portrait_identity_hash(world, character, lighting, long_context=long_context)
    target = images_dir / f"{prompt_hash}.png"
    if not target.exists():
        positive = image_prompts.portrait_prompt(
            world=world,
            character=character,
            lighting=lighting,
            long_context=long_context,
        )
        face = image_prompts.portrait_face_prompt(character=character)
        negative_extras = image_prompts.portrait_negative_extras(character=character)
        _log.info(
            "render_portrait_to_disk: generating portrait for %s (%s) kind=%s",
            character.id,
            character.name,
            character.kind.value,
        )
        character_capture = character
        render = _build_portrait_render(
            session=session,
            image_client=image_client,
            character=character,
            positive=positive,
            face=face,
            negative_extras=negative_extras,
            identity_hash=identity_hash,
        )
        wrote = await _render_or_await(
            target,
            render=render,
            post_process=crop_to_figure,
            safety_check=lambda payload: _output_filter_passes(
                session,
                character_capture,
                payload,
            ),
        )
        if not wrote:
            return None
    return CharacterImage(
        path=str(target),
        prompt_hash=prompt_hash,
        identity_hash=identity_hash,
        attributes_snapshot={
            "outfit": character.outfit,
            "pose": character.pose,
            "expression": character.expression,
            "effects": character.effects,
            "decals": character.decals,
            "hair_color": character.hair_color,
            "eye_color": character.eye_color,
        },
    )


async def render_background_to_disk(
    *,
    session: Session,
    image_client: ImageClient,
    env: Environment,
    images_dir: Path,
) -> tuple[Path, str] | None:
    """Render ONE environment's CURRENT state to a content-addressed
    PNG and return ``(path, prompt_hash)`` — WITHOUT touching
    ``session.game``. Mirror of :func:`render_portrait_to_disk` for
    backgrounds. Returns ``None`` only when the render failed."""
    # Callers only reach the render primitives with a loaded game.
    assert session.game is not None
    world = session.game.world
    long_context = _long_context_image_model(session, environment=True)
    prompt_hash = _background_prompt_hash(world, env, long_context=long_context)
    target = images_dir / f"{prompt_hash}.png"
    if not target.exists():
        positive = image_prompts.background_prompt(
            world=world, environment=env, long_context=long_context
        )
        _log.info("render_background_to_disk: generating background for %s", env.id)

        async def _bg_render() -> bytes:
            return await image_client.generate(
                session.settings.image.background_workflow,
                {"positive_prompt": positive},
                seed=env.seed,
            )

        wrote = await _render_or_await(target, render=_bg_render)
        if not wrote:
            return None
    return target, prompt_hash


async def ensure_portraits_for(
    *,
    session: Session,
    image_client: ImageClient,
    character_ids: list[str],
    saves_root: Path | None = None,
    allow_player: bool = False,
) -> list[GeneratedAsset]:
    """Generate portraits for a specific set of characters, even when
    they aren't in ``game.on_stage`` yet.

    Used by speculative branch pre-rendering: the LLM may have queued
    a future beat that will introduce a character; we pre-render the
    portrait so that, when the player actually walks the speculative
    beat, the image is already on disk and ``ensure_assets`` skips
    regeneration via its prompt-hash check.

    Mirrors the on-stage portrait loop in ``ensure_assets`` but
    iterates a caller-supplied id list and tolerates ids that don't
    exist in ``game.characters`` yet (returns no work for them).

    ``allow_player`` defaults to ``False``: the PC is the camera lens,
    not an on-stage figure, so every automatic caller is gated out.
    The one legitimate exception is the explicit, user-triggered
    ``edit_character_rerender_handler`` (the player edits their own
    attributes and asks to see a refreshed render); that caller sets
    the flag to ``True``.
    """
    game = session.game
    if game is None or not character_ids:
        return []

    # Safety guard FIRST: if any candidate character is under 18 and
    # described with a nudity outfit, bump their age to 18 (and
    # retcon past references) BEFORE we build any prompt. The
    # ``age_band`` floor at the prompt level catches this too, but
    # this pass also corrects the stored age and tells the player.
    check_minor_nudity_and_correct(session)
    game = session.game  # may have been replaced by the safety pass.
    # The safety pass only ever installs a revised game, never clears it.
    assert game is not None

    images_dir = _images_dir(game.id, saves_root)
    images_dir.mkdir(parents=True, exist_ok=True)

    lighting = _active_lighting(game)
    long_context = _long_context_image_model(session)
    generated: list[GeneratedAsset] = []
    for cid in character_ids:
        character = game.characters.get(cid)
        if character is None:
            continue
        # The PC is rendered through the player's perspective, never
        # as an on-stage figure. Defense-in-depth: even if the caller
        # accidentally passes the PC's id (e.g. a future speculation
        # path that forgets to filter), we refuse to spawn the work
        # unless the caller has explicitly opted in via
        # ``allow_player`` (the user-triggered rerender handler).
        if character.is_player and not allow_player:
            _log.warning(
                "ensure_portraits_for: refusing to render PC %s "
                "(is_player=True, allow_player=False); caller "
                "should either filter this id or pass allow_player=True",
                character.id,
            )
            continue
        prompt_hash = _portrait_prompt_hash(
            game.world, character, lighting, long_context=long_context
        )
        if _portrait_already_current(character, prompt_hash):
            continue
        new_image = await render_portrait_to_disk(
            session=session,
            image_client=image_client,
            character=character,
            lighting=lighting,
            images_dir=images_dir,
        )
        if new_image is None:
            continue
        updated = character.model_copy(update={"images": [new_image, *character.images]})
        game.characters[cid] = updated
        generated.append(
            GeneratedAsset(
                kind="portrait",
                target_id=character.id,
                image_path=Path(new_image.path),
                prompt_hash=new_image.prompt_hash,
            )
        )
    return generated


async def ensure_backgrounds_for(
    *,
    session: Session,
    image_client: ImageClient,
    environment_ids: list[str],
    saves_root: Path | None = None,
    force: bool = False,
) -> list[GeneratedAsset]:
    """Generate / regenerate backgrounds for a specific set of
    environments, even when they aren't currently the active scene.

    Used by the player-facing "Rerender" button on the
    EnvironmentsTab. ``force=True`` clears the env's cached
    ``image_path`` so the regeneration runs against a fresh seed
    even if the previous file is still on disk; ``force=False``
    short-circuits when a current PNG exists (mirrors the standard
    ``ensure_assets`` behaviour for new envs the player hasn't
    rendered yet).

    Mirrors ``ensure_portraits_for``'s shape so the calling handler
    can chain identical s2c/image/ready emission.
    """
    game = session.game
    if game is None or not environment_ids:
        return []

    images_dir = _images_dir(game.id, saves_root)
    images_dir.mkdir(parents=True, exist_ok=True)

    long_context = _long_context_image_model(session, environment=True)
    generated: list[GeneratedAsset] = []
    for eid in environment_ids:
        env = game.environments.get(eid)
        if env is None:
            continue
        if force:
            # Drop the cached image_path so the gate below regenerates
            # against the (now fresh) seed.
            env = env.model_copy(update={"image_path": None})
            game.environments[eid] = env
        prompt_hash = _background_prompt_hash(game.world, env, long_context=long_context)
        # Local stat() — see the ASYNC240 note in ``_render_or_await``.
        if not force and env.image_path and Path(env.image_path).exists():  # noqa: ASYNC240
            # Already current and not forced — no work.
            continue
        target = images_dir / f"{prompt_hash}.png"
        if not target.exists():
            positive = image_prompts.background_prompt(
                world=game.world, environment=env, long_context=long_context
            )
            _log.info(
                "ensure_backgrounds_for: generating background for %s (force=%s)",
                env.id,
                force,
            )
            payload = await image_client.generate(
                session.settings.image.background_workflow,
                {"positive_prompt": positive},
                seed=env.seed,
            )
            atomic_write_bytes(target, payload)
        updated = env.model_copy(update={"image_path": str(target), "prompt_hash": prompt_hash})
        game.environments[eid] = updated
        generated.append(
            GeneratedAsset(
                kind="background",
                target_id=env.id,
                image_path=target,
                prompt_hash=prompt_hash,
            )
        )
    return generated


def music_prompt_hash(prompt: str, model_name: str) -> str:
    """Content-addressed hash for the world's background music. The
    model name folds in so a player who switches ACE-Step models
    (3.5B → fine-tuned variant) gets a fresh render against the
    new model rather than re-using the cached MP3 from the prior
    one. Mirrors the seed-folding the image hashes do."""
    return _hash_prompt(prompt, model_name)


async def ensure_music_for_world(
    *,
    session: Session,
    music_client: AceStepClient,
    prompt: str,
    saves_root: Path | None = None,
) -> GeneratedAsset | None:
    """Render a single background music track for the current
    ``WorldState`` from ``prompt`` and update the world's
    ``music_path`` / ``music_prompt`` / ``music_prompt_hash`` to
    point at the freshly-rendered file.

    Idempotent: if the world already has a music track whose
    ``music_prompt_hash`` matches the (prompt, model) tuple AND
    the file is on disk, no work happens. ``music_client`` is
    expected to carry the model name in the settings it was
    constructed with (see ``Session.music_client``).

    Returns the generated asset (or ``None`` when no work was
    done / the prompt was empty / the music client was missing).
    """
    if music_client is None:
        return None
    text = (prompt or "").strip()
    if not text:
        return None
    game = session.game
    if game is None:
        return None

    images_dir = _images_dir(game.id, saves_root)
    images_dir.mkdir(parents=True, exist_ok=True)

    model_name = session.settings.music.model_name
    new_hash = music_prompt_hash(text, model_name)
    world = game.world
    if world.music_path and world.music_prompt_hash == new_hash:
        # Local stat() — see the ASYNC240 note in ``_render_or_await``.
        if Path(world.music_path).exists():  # noqa: ASYNC240
            return None

    from ..providers.music_client import music_audio_extension

    extension = music_audio_extension(session.settings.music)
    target = images_dir / f"music-{new_hash}.{extension}"
    if not target.exists():
        # Seed the music render off the prompt hash so the same
        # (prompt, model) combo always reproduces. Players who
        # want variety bump the prompt; the engine doesn't pin a
        # per-world music seed because mid-game music_change
        # swaps shouldn't preserve a stale seed unrelated to the
        # new prompt.
        seed = abs(hash(new_hash)) % (2**31)
        try:
            audio_bytes = await music_client.generate_music(text, seed=seed)
        except Exception:
            _log.warning(
                "ensure_music_for_world: music generation failed for prompt=%r",
                text,
                exc_info=True,
            )
            return None
        atomic_write_bytes(target, audio_bytes)

    new_world = world.model_copy(
        update={
            "music_path": str(target),
            "music_prompt": text,
            "music_prompt_hash": new_hash,
        }
    )
    new_game = game.model_copy(update={"world": new_world})
    session.install_game(new_game)
    return GeneratedAsset(
        kind="music",
        target_id="world",
        image_path=target,
        prompt_hash=new_hash,
    )


async def ensure_assets(
    *,
    session: Session,
    image_client: ImageClient,
    saves_root: Path | None = None,
) -> list[GeneratedAsset]:
    """Generate any missing portrait/background for the current state.

    Returns the freshly-generated assets so the caller can emit
    ``s2c/image/ready`` events for the renderer.
    """
    game = session.game
    if game is None:
        return []

    # Safety guard FIRST — see ``check_minor_nudity_and_correct``.
    # Runs before any portrait prompt is built so an under-18
    # character described with nudity gets their age bumped (and a
    # notice + retcon fired) before SDXL sees the prompt.
    check_minor_nudity_and_correct(session)
    game = session.game
    # The safety pass only ever installs a revised game, never clears it.
    assert game is not None

    images_dir = _images_dir(game.id, saves_root)
    images_dir.mkdir(parents=True, exist_ok=True)

    generated: list[GeneratedAsset] = []

    # 1. Portraits for every on-stage character — ALWAYS rendered
    #    before the background. On a single GPU the portrait and the
    #    backdrop render sequentially through the same image client; a
    #    player registers a missing or stale FACE far more than a
    #    missing backdrop, so character work is dispatched FIRST and the
    #    background (step 2) only after every on-stage portrait has been
    #    handled. ``_resolve_active_environment_id`` (used for the
    #    lighting descriptor here and the background below) reads only
    #    the committed path, so neither pass can target a future node.
    lighting = _active_lighting(game)
    long_context = _long_context_image_model(session)
    long_context_bg = _long_context_image_model(session, environment=True)
    for cid in list(game.on_stage):
        character = game.characters.get(cid)
        if character is None:
            continue
        # The PC is the camera lens, not a portrait — they should
        # never reach on_stage (``_apply_node`` filters them out
        # before the list is updated). Defensive belt-and-braces:
        # if they DO slip through, refuse to render rather than
        # producing a silent FPV portrait on disk.
        if character.is_player:
            _log.warning(
                "ensure_assets: PC %s found in on_stage; refusing "
                "to render — _apply_node should have filtered them",
                character.id,
            )
            continue
        prompt_hash = _portrait_prompt_hash(
            game.world, character, lighting, long_context=long_context
        )
        if _portrait_already_current(character, prompt_hash):
            continue
        identity_hash = _portrait_identity_hash(
            game.world, character, lighting, long_context=long_context
        )
        target = images_dir / f"{prompt_hash}.png"
        if not target.exists():
            positive = image_prompts.portrait_prompt(
                world=game.world,
                character=character,
                lighting=lighting,
                long_context=long_context,
            )
            face = image_prompts.portrait_face_prompt(character=character)
            negative_extras = image_prompts.portrait_negative_extras(character=character)
            _log.info(
                "ensure_assets: generating portrait for %s (%s) kind=%s",
                character.id,
                character.name,
                character.kind.value,
            )

            portrait_render = _build_portrait_render(
                session=session,
                image_client=image_client,
                character=character,
                positive=positive,
                face=face,
                negative_extras=negative_extras,
                identity_hash=identity_hash,
            )

            wrote = await _render_or_await(
                target,
                render=portrait_render,
                post_process=crop_to_figure,
                # Output-side ML safety filter — drops the image if
                # it combined nudity with a minor-face estimate.
                safety_check=_portrait_safety_check(session, character),
            )
            if not wrote:
                continue
        attribute_snapshot = {
            "outfit": character.outfit,
            "pose": character.pose,
            "expression": character.expression,
            "effects": character.effects,
            "hair_color": character.hair_color,
            "eye_color": character.eye_color,
        }
        new_image = CharacterImage(
            path=str(target),
            prompt_hash=prompt_hash,
            identity_hash=identity_hash,
            attributes_snapshot=attribute_snapshot,
        )
        # Most-recent first so the renderer's `images[0]` is current.
        updated = character.model_copy(update={"images": [new_image, *character.images]})
        game.characters[cid] = updated
        generated.append(
            GeneratedAsset(
                kind="portrait",
                target_id=character.id,
                image_path=target,
                prompt_hash=prompt_hash,
            )
        )

    # 2. Background for the active environment (lowest priority — every
    #    on-stage portrait above has already been dispatched). The env
    #    is resolved from the committed path, so this is always the
    #    CURRENT scene, never a speculative / future node.
    env_id = _resolve_active_environment_id(game)
    if env_id and env_id in game.environments:
        env = game.environments[env_id]
        prompt_hash = _background_prompt_hash(game.world, env, long_context=long_context_bg)
        # Local stat() — see the ASYNC240 note in ``_render_or_await``.
        if env.image_path is None or not Path(env.image_path).exists():  # noqa: ASYNC240
            target = images_dir / f"{prompt_hash}.png"
            if not target.exists():
                positive = image_prompts.background_prompt(
                    world=game.world, environment=env, long_context=long_context_bg
                )
                _log.info("ensure_assets: generating background for %s", env.id)

                async def _bg_render() -> bytes:
                    return await image_client.generate(
                        session.settings.image.background_workflow,
                        {"positive_prompt": positive},
                        seed=env.seed,
                    )

                await _render_or_await(target, render=_bg_render)
            updated_env = env.model_copy(
                update={"image_path": str(target), "prompt_hash": prompt_hash}
            )
            game.environments[env_id] = updated_env
            generated.append(
                GeneratedAsset(
                    kind="background",
                    target_id=env.id,
                    image_path=target,
                    prompt_hash=prompt_hash,
                )
            )

    return generated

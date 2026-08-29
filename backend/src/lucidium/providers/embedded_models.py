"""Helpers for the embedded image-generation backend's model directory.

The embedded backend reads checkpoints from a user-configurable
directory. The directory listing is exposed to the renderer via
``c2s/embedded/list_models`` so the Settings UI can populate a
dropdown with whatever files the user has on disk.

OPTIONAL one-click bootstrap: when the directory is empty the first-
run wizard offers to download a sensible base model for the player's
hardware (see :data:`MODEL_CATALOG` / :func:`recommend_model` /
:func:`download_model`). This is always a deliberate, user-clicked
action — nothing is fetched silently, and the wizard still links out
to Civitai for players who'd rather pick their own fine-tune. The
weights come straight from the upstream HuggingFace repo; Lucidium
neither hosts nor relicenses them.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import embedded_models_dir as default_models_dir

_log = logging.getLogger(__name__)

# Extensions we list for the user. The diffusers loader supports
# safetensors directly; .ckpt is older but tolerated.
_MODEL_SUFFIXES: tuple[str, ...] = (".safetensors", ".ckpt")


class ModelsDirOutsideRootError(ValueError):
    """A requested models-dir resolved outside the allowed root.

    Raised by :func:`resolve_models_dir` when the caller supplied an
    ``allowed_root``. The download path creates directories and streams
    multi-GB files into whatever it's handed, so a value that came off
    the wire must be pinned to the directory the player configured
    rather than being able to name (say) the Startup folder."""


def resolve_models_dir(
    configured: str,
    *,
    allowed_root: Path | None = None,
) -> Path:
    """Return the absolute models-dir to use.

    ``configured`` is the user-supplied path from
    ``ImageSettings.embedded_models_dir`` (empty string means
    "use the bundled default"). This is the only place where the
    "empty string falls back" rule lives — every other module reads
    the resolved Path.

    ``allowed_root`` confines the result: the resolved directory must be
    ``allowed_root`` itself or live underneath it, otherwise
    :class:`ModelsDirOutsideRootError` is raised. Callers handling a
    value that originated on the WebSocket pass the *configured* dir as
    the root, so a request can at most narrow to a subdirectory of the
    place the player already chose.
    """
    if configured.strip():
        resolved = Path(configured).expanduser().resolve()
    else:
        resolved = default_models_dir()
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve()
        if resolved != root and not resolved.is_relative_to(root):
            raise ModelsDirOutsideRootError(
                f"models directory {resolved} is outside the configured root {root}"
            )
    return resolved


def list_models(models_dir: Path) -> list[str]:
    """Return the model filenames inside ``models_dir`` (NOT recursive,
    just the top level — diffusers loads single-file checkpoints by
    path). Sorted alphabetically so the UI dropdown is stable across
    renders. Returns an empty list when the directory doesn't exist
    yet so the caller can decide whether to surface the manual-download
    instructions."""
    if not models_dir.exists():
        return []
    out: list[str] = []
    for entry in models_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in _MODEL_SUFFIXES:
            out.append(entry.name)
    return sorted(out)


def pick_default_model(models_dir: Path, configured_name: str) -> Path | None:
    """Resolve the checkpoint path the embedded client should load.

    If ``configured_name`` matches a file in ``models_dir`` — use
    that. Otherwise fall back to the first file in alphabetical
    order. Returns ``None`` when the directory is empty (the caller
    raises an actionable error pointing the user at the manual-
    download instructions in that case).
    """
    available = list_models(models_dir)
    if configured_name and configured_name in available:
        return models_dir / configured_name
    if not available:
        return None
    return models_dir / available[0]


# ---------------------------------------------------------------------------
# One-click base-model download (first-run bootstrap)
# ---------------------------------------------------------------------------


class ModelDownloadError(RuntimeError):
    """Raised when a one-click base-model download can't complete (network
    failure, unexpected HTTP status, disk error). Surfaced to the renderer
    via ``s2c/error`` so the wizard can tell the player to retry or fall
    back to the manual Civitai route."""


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable base model.

    ``hf_repo`` / ``hf_filename`` locate the single-file checkpoint in a
    public HuggingFace repo; we fetch it from the stable ``resolve/main``
    URL (no extra dependency — plain ``urllib``). ``local_filename`` is
    what we save it as on disk, chosen to be human-friendly AND, for
    Z-Image, to satisfy ``_is_z_image_model_path`` (which keys off a
    ``z-image`` / ``zimage`` substring) so the loader picks the right
    pipeline.

    ``approx_bytes`` is the size of THAT ONE FILE. ``aux_approx_bytes``
    is everything else the family needs before it can render a single
    image — for the transformer-only entries (Z-Image, Qwen-Image,
    Krea 2) the text encoder + VAE + tokenizer are fetched from their
    own upstream repos at first render, and they are not small (8-17 GB).
    The UI must quote :attr:`total_approx_bytes`, not ``approx_bytes``:
    a player on a metered connection deciding from the checkpoint size
    alone would be off by more than a factor of two.

    ``sha256`` is the upstream LFS object digest (``git lfs`` ``oid``,
    which for HuggingFace is the sha256 of the file contents). It is the
    integrity check for a download: the URL pins ``resolve/main``, a
    moving ref, and the target-exists fast path means a bad file is
    never re-fetched — so a corrupt or truncated checkpoint published
    into place would be permanent. ``None`` means no published digest,
    in which case :func:`download_model` falls back to the (weaker)
    Content-Length comparison.
    """

    key: str
    display_name: str
    hf_repo: str
    hf_filename: str
    local_filename: str
    approx_bytes: int
    # Text encoder + VAE + tokenizer fetched lazily at first render.
    # Zero for the single-file SDXL entries, which carry everything.
    aux_approx_bytes: int = 0
    sha256: str | None = None

    @property
    def total_approx_bytes(self) -> int:
        """Every byte the player will end up downloading for this model —
        the checkpoint plus the lazily-fetched components. This is the
        number to show before asking someone to click Download."""
        return self.approx_bytes + self.aux_approx_bytes


# The families the embedded pipeline can load: SDXL, the distilled
# SDXL-Turbo, Alibaba's Z-Image-Turbo and Qwen-Image, and Krea AI's
# Krea 2. Z-Image has no single-file checkpoint in the official
# ``Tongyi-MAI`` diffusers repo, so we pull the ComfyUI repackage's
# consolidated bf16 transformer; the text encoder + VAE are fetched from
# ``Tongyi-MAI/Z-Image-Turbo`` lazily at first render (see
# ``embedded_image_client._load_z_image_pipeline``). Qwen-Image and
# Krea 2 follow the same transformer-only pattern.
#
# NOTE ``recommend_model`` below deliberately never returns Krea 2: it's
# downloadable by explicit key, but the first-run hardware default is
# left as-is so an existing install's wizard behaviour doesn't change.
MODEL_CATALOG: dict[str, ModelSpec] = {
    "sdxl": ModelSpec(
        key="sdxl",
        display_name="SDXL",
        hf_repo="stabilityai/stable-diffusion-xl-base-1.0",
        hf_filename="sd_xl_base_1.0.safetensors",
        local_filename="sdxl-base-1.0.safetensors",
        # Exact upstream LFS sizes + digests (``<repo>/raw/main/<file>``
        # serves the git-lfs pointer, whose ``oid sha256:`` IS the
        # content hash). Single-file: no lazily-fetched components.
        approx_bytes=6_938_078_334,
        sha256="31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
    ),
    "sdxl-turbo": ModelSpec(
        key="sdxl-turbo",
        display_name="SDXL Turbo",
        hf_repo="stabilityai/sdxl-turbo",
        hf_filename="sd_xl_turbo_1.0_fp16.safetensors",
        local_filename="sdxl-turbo.safetensors",
        # NOT the same file as SDXL base despite the near-identical size
        # (these two carried byte-identical ``approx_bytes`` until the
        # figures were taken from the upstream pointers).
        approx_bytes=6_938_081_905,
        sha256="e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26",
    ),
    "z-image-turbo": ModelSpec(
        key="z-image-turbo",
        display_name="Z-Image Turbo",
        hf_repo="Comfy-Org/z_image_turbo",
        # The repo keeps the consolidated checkpoint under
        # ``split_files/diffusion_models/`` — a bare
        # ``z_image_turbo_bf16.safetensors`` at the repo root is a 404.
        hf_filename="split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        # Hyphenated so ``_is_z_image_model_path`` matches; the Comfy-Org
        # filename uses underscores and would NOT be detected as Z-Image.
        local_filename="Z-Image-Turbo.safetensors",
        approx_bytes=12_309_866_400,
        # ``Tongyi-MAI/Z-Image-Turbo`` text_encoder (3 shards, ~8.0 GB)
        # + vae (~168 MB) + tokenizer, at first render.
        aux_approx_bytes=8_228_529_974,
        sha256="2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6",
    ),
    "qwen-image": ModelSpec(
        key="qwen-image",
        display_name="Qwen-Image",
        # Pre-distilled, few-step Qwen-Image transformer (fp8) from the
        # Comfy-Org repackage. This is the official base transformer with
        # the step/CFG distillation already baked in — same tensor keys as
        # the plain base, so it loads through the same single-file path AND
        # avoids fetching + fusing a Lightning LoRA at load (which cost
        # ~3 min every load). The text encoder (Qwen2.5-VL) + VAE come from
        # ``Qwen/Qwen-Image`` at first load. With torchao native-fp8 it
        # renders in seconds on a 24 GiB card (transformer ~20 GiB
        # resident, text encoder CPU-encoded — see
        # ``embedded_image_client._load_qwen_pipeline``).
        hf_repo="Comfy-Org/Qwen-Image_ComfyUI",
        hf_filename="non_official/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors",
        # ``Qwen-Image`` keeps ``_is_qwen_model_path`` matching; ``Distill``
        # flags it to the loader as a pre-distilled few-step checkpoint so
        # it skips the Lightning LoRA fuse.
        local_filename="Qwen-Image-Distill.safetensors",
        approx_bytes=20_430_635_136,
        # ``Qwen/Qwen-Image`` text_encoder (4 shards, ~16.6 GB) + vae
        # (~254 MB) + tokenizer, at first render.
        aux_approx_bytes=16_843_285_101,
        sha256="a1d50aa60140f156fc2fae6b4b370c95fde65d5d3485dbbd7a1b387fd7ae8612",
    ),
    "krea-2-turbo": ModelSpec(
        key="krea-2-turbo",
        display_name="Krea 2 Turbo",
        # Krea AI's own ``krea/Krea-2-Turbo`` repo is GATED (licence
        # acceptance required), so we pull the Comfy-Org repackage of the
        # same weights — the fp8-scaled few-step Turbo transformer. The
        # text encoder (Qwen3-VL-4B) and VAE (Qwen-Image) are fetched from
        # their own ungated upstream repos at first load; see
        # ``embedded_image_client._load_krea_pipeline``.
        hf_repo="Comfy-Org/Krea-2",
        hf_filename="diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        # ``krea`` in the name keeps the loader's cheap filename
        # pre-filter matching before it reads the safetensors header;
        # ``Turbo`` marks it few-step so the 8-step / no-guidance recipe
        # is selected instead of the 28-step raw one.
        local_filename="Krea-2-Turbo.safetensors",
        approx_bytes=13_141_730_784,
        # ``Qwen/Qwen3-VL-4B-Instruct`` text encoder (~8.9 GB) + the
        # ``Qwen/Qwen-Image`` vae (~254 MB) + tokenizer, at first render.
        aux_approx_bytes=9_141_018_253,
        sha256="eb4dd8c612cfd10f64f25b057e6e6bbcb5737c94a7372177e456dbf7579502f1",
    ),
}

# Below this VRAM we don't offer Z-Image-Turbo — it needs bf16 + ~18 GiB
# resident, so a smaller card would just OOM at render time.
_Z_IMAGE_MIN_VRAM_GB = 16.0
# VRAM at/above which we OFFER Qwen-Image as the first-run DEFAULT. This
# is a "recommend at resident-class speed" threshold, NOT a hard floor:
# Qwen-Image also runs on much smaller cards (e.g. a 24 GiB 4090) because
# the loader streams the dense transformer at block level instead of
# holding it resident (see ``embedded_image_client._apply_qwen_offload``).
# We just don't auto-recommend it below this — block streaming is slower
# per step — but it stays explicitly downloadable by key for anyone who
# wants its quality and accepts the speed.
_QWEN_MIN_VRAM_GB = 40.0
# Below this VRAM a GPU gets the lighter, few-step SDXL-Turbo instead of
# full SDXL base (which wants ~8 GiB of comfortable headroom).
_FULL_SDXL_MIN_VRAM_GB = 8.0


def recommend_model(
    *,
    flavor: str | None = None,
    vram_gb: float | None = None,
) -> ModelSpec:
    """Pick the base model to offer for THIS machine. Pure given its two
    signals, so the decision table is unit-testable without hardware.

    Signals default to the live probes (``torch_overlay.recommend_flavor``
    for the GPU vendor/runtime, ``embedded_image_client.detect_total_vram_gb``
    for headroom); tests inject explicit values.

    Decision table:

      * CUDA / ROCm **and** VRAM ≥ 40 GiB → Qwen-Image. The highest-
        fidelity family, but its transformer alone needs ~40 GiB resident
        even with CPU offload, so only a very large card gets it offered
        as the default (it stays downloadable by key on smaller cards).
      * CUDA / ROCm **and** VRAM ≥ 16 GiB → Z-Image-Turbo. Newest fast
        family, but it only fits (and only has a bf16 path) on a real
        NVIDIA / Linux-AMD card with headroom.
      * Any other GPU (incl. DirectML / Intel Arc), or CUDA/ROCm with
        unknown / 8-16 GiB VRAM → SDXL base. Full quality, runs
        comfortably on a typical 8-12 GiB modern dGPU.
      * GPU with KNOWN < 8 GiB VRAM → SDXL-Turbo. Few-step + lighter so
        a small card still renders quickly.
      * No GPU (CPU) → SDXL-Turbo. CPU renders are slow regardless;
        Turbo's 1-4 steps minimise the wait.
    """
    if flavor is None:
        from . import torch_overlay

        flavor = torch_overlay.recommend_flavor()
    if vram_gb is None:
        from .embedded_image_client import detect_total_vram_gb

        vram_gb = detect_total_vram_gb()

    if flavor in ("cuda", "rocm") and vram_gb is not None:
        if vram_gb >= _QWEN_MIN_VRAM_GB:
            return MODEL_CATALOG["qwen-image"]
        if vram_gb >= _Z_IMAGE_MIN_VRAM_GB:
            return MODEL_CATALOG["z-image-turbo"]

    has_gpu = flavor != "cpu"
    if has_gpu:
        if vram_gb is not None and vram_gb < _FULL_SDXL_MIN_VRAM_GB:
            return MODEL_CATALOG["sdxl-turbo"]
        return MODEL_CATALOG["sdxl"]

    return MODEL_CATALOG["sdxl-turbo"]


# A thread-safe byte-progress callback: ``(bytes_done, bytes_total|None)``.
ProgressCallback = Callable[[int, int | None], None]

_HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB

# ---------------------------------------------------------------------------
# Per-target download locks. Downloads run on worker threads (the handler
# hands the blocking work to ``run_in_executor``), and a dropped WebSocket
# does NOT cancel the in-flight one — so a reconnect-and-retry lands a
# SECOND thread on the same target while the first is still streaming
# gigabytes. Serialising on the target path means the retry waits, sees the
# now-published file, and returns it instead of racing the writer.
# ``_target_locks`` itself is guarded by ``_locks_guard``; entries are never
# evicted (one small lock per checkpoint path, bounded by the catalog).
# ---------------------------------------------------------------------------
_locks_guard = threading.Lock()
_target_locks: dict[str, threading.Lock] = {}


def _lock_for(target: Path) -> threading.Lock:
    key = str(target)
    with _locks_guard:
        lock = _target_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _target_locks[key] = lock
        return lock


def _partial_path(target: Path) -> Path:
    """A ``.part`` path unique to THIS download attempt.

    A single fixed ``<name>.part`` was the sharp edge: a retry opened it
    ``"wb"``, truncating the still-running download's temp file, and
    whichever thread finished first ``replace()``d a multi-GB file full of
    zero holes into place — where the ``target.exists()`` fast path made it
    permanently sticky (an opaque safetensors error at every render, with
    no way to force a re-fetch). A uuid means the two attempts can never
    write to the same bytes.
    """
    return target.with_name(f"{target.name}.{uuid.uuid4().hex[:12]}.part")


def _verify_download(
    tmp: Path,
    spec: ModelSpec,
    *,
    done: int,
    total: int | None,
) -> None:
    """Gate the ``tmp -> target`` publish. Raises :class:`ModelDownloadError`
    (leaving the caller to clean up) unless the bytes we hold are the bytes
    upstream published.

    Two checks, strongest first:

      * ``spec.sha256`` — the upstream LFS digest. Reuses
        ``torch_overlay._verify_hash`` so checkpoints and torch wheels
        verify through exactly one implementation.
      * otherwise Content-Length — far weaker (it can't catch corruption,
        only truncation) but it *does* catch the common failure: a
        connection cut at 80%, whose partial file would otherwise be
        renamed into place and cached forever.
    """
    if spec.sha256:
        from ..api.errors import ProviderValidationError
        from .torch_overlay import _verify_hash

        try:
            _verify_hash(tmp, ("sha256", spec.sha256))
        except ProviderValidationError as exc:
            raise ModelDownloadError(
                f"integrity check failed for {spec.display_name}: {exc}"
            ) from exc
        return

    if total is not None and done != total:
        raise ModelDownloadError(
            f"download of {spec.display_name} is incomplete: got {done} bytes "
            f"but the server advertised {total}; refusing to save a truncated "
            "checkpoint"
        )


def download_model(
    spec: ModelSpec,
    models_dir: Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download ``spec``'s checkpoint into ``models_dir`` and return the
    saved path. Idempotent: if the target file already exists it's
    returned untouched (no re-download).

    Streams straight from the upstream HuggingFace ``resolve/main`` URL
    with stdlib ``urllib`` (no ``huggingface_hub`` dependency, so it works
    in the frozen bundle), writing to a per-attempt ``.part`` temp that's
    atomically renamed on success — a half-finished download never looks
    like a valid checkpoint to ``list_models``. ``on_progress`` is invoked
    with ``(bytes_done, bytes_total)`` as chunks land (``bytes_total`` is
    ``None`` when the server omits Content-Length).

    SAFE TO RETRY, by construction:

      * calls for the same target serialise on a per-path lock, so a
        reconnect-driven retry never races the download still in flight;
      * each attempt gets its own uuid-suffixed ``.part``, so even an
        un-serialised caller can't truncate another's temp file;
      * the rename is gated on :func:`_verify_download` — the published
        sha256 when there is one, else the Content-Length — so a
        truncated or corrupt stream is deleted rather than cached
        forever behind the exists-check above.

    Raises :class:`ModelDownloadError` on any network / HTTP / disk /
    integrity failure, leaving ``models_dir`` exactly as it was (the
    partial file is removed).
    """
    with _lock_for(models_dir / spec.local_filename):
        return _download_model_locked(spec, models_dir, on_progress=on_progress)


def _download_model_locked(
    spec: ModelSpec,
    models_dir: Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Real download body. Held under the per-target lock, so the
    ``target.exists()`` fast path below also covers "another thread just
    finished this exact download while we were queued"."""
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / spec.local_filename
    if target.exists():
        _log.info("embedded backend: %s already present at %s", spec.key, target)
        return target

    url = _HF_RESOLVE.format(repo=spec.hf_repo, filename=spec.hf_filename)
    tmp = _partial_path(target)
    _log.info(
        "embedded backend: downloading %s (%s) -> %s",
        spec.display_name,
        url,
        target,
    )
    # User-Agent: some CDNs 403 the stdlib default; a real-looking one is
    # also good manners for an upstream we don't host.
    request = urllib.request.Request(url, headers={"User-Agent": "lucidium/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            if status and status >= 400:
                raise ModelDownloadError(f"download of {spec.display_name} failed: HTTP {status}")
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            done = 0
            with open(tmp, "wb") as handle:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(done, total)
            _verify_download(tmp, spec, done=done, total=total)
    except ModelDownloadError:
        _cleanup_partial(tmp)
        raise
    except Exception as exc:
        _cleanup_partial(tmp)
        raise ModelDownloadError(
            f"failed to download {spec.display_name} from {spec.hf_repo}: {exc}"
        ) from exc

    tmp.replace(target)
    _log.info("embedded backend: %s downloaded (%d bytes)", spec.key, target.stat().st_size)
    return target


def _cleanup_partial(tmp: Path) -> None:
    """Remove a half-written ``.part`` file, ignoring errors — the caller
    is already raising/handling the real failure."""
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        _log.debug("could not remove partial download %s", tmp, exc_info=True)

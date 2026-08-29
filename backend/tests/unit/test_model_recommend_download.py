"""Hardware-based model recommendation + one-click download bootstrap.

Covers ``embedded_models.recommend_model`` (the GPU-vendor × VRAM
decision table the first-run wizard's ``<highlight>`` is built from),
``embedded_models.download_model`` (the stdlib-urllib streaming download
behind the wizard's one-click button), and — at the end —
``embedded_download_model_handler``, the frame that actually reaches
both of them off an unauthenticated local socket. The library-level
tests monkeypatch ``urlopen`` on the module, so on their own they say
nothing about where the handler puts the file or what the player sees
when the download dies.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace
from pathlib import Path

import pytest

from lucidium.api.errors import ProviderUnreachableError, SchemaError
from lucidium.api.handlers import HandlerContext, embedded_download_model_handler
from lucidium.api.messages import C2SEmbeddedDownloadModel, MessageType
from lucidium.domain.settings import ImageSettings, Settings
from lucidium.providers import embedded_models as em

# --------------------------------------------------------------------------
# recommend_model — the (flavor, vram) decision table
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flavor", "vram_gb", "expected_key"),
    [
        # Huge CUDA / ROCm cards (≥40 GiB) get Qwen-Image — its
        # transformer alone needs ~40 GiB resident even with offload.
        ("cuda", 48.0, "qwen-image"),
        ("rocm", 40.0, "qwen-image"),
        # Big-but-not-huge CUDA / ROCm cards get Z-Image-Turbo
        # (needs bf16 + ~18 GiB, but Qwen wouldn't fit).
        ("cuda", 24.0, "z-image-turbo"),
        ("rocm", 16.0, "z-image-turbo"),
        # A huge NON-cuda/rocm GPU is still gated off Qwen (no bf16 path).
        ("directml", 48.0, "sdxl"),
        # Mid CUDA/ROCm and any non-cuda/rocm GPU → full SDXL base.
        ("cuda", 12.0, "sdxl"),
        ("directml", 16.0, "sdxl"),  # Windows-AMD: gated off Z-Image.
        ("xpu", 16.0, "sdxl"),  # Intel Arc: gated off Z-Image.
        # Small GPU → the lighter, few-step SDXL-Turbo.
        ("cuda", 6.0, "sdxl-turbo"),
        ("directml", 4.0, "sdxl-turbo"),
        # No GPU → SDXL-Turbo (fewest steps minimises the CPU wait).
        ("cpu", None, "sdxl-turbo"),
        ("cpu", 0.0, "sdxl-turbo"),
    ],
)
def test_recommend_model_decision_table(flavor, vram_gb, expected_key):
    spec = em.recommend_model(flavor=flavor, vram_gb=vram_gb)
    assert spec.key == expected_key
    # Every recommendation must be a real, downloadable catalog entry.
    assert spec is em.MODEL_CATALOG[expected_key]


def test_recommend_model_gpu_unknown_vram_is_safe(monkeypatch):
    """A GPU with undetectable VRAM falls back to full SDXL — we can't
    promise the headroom Z-Image needs, but a dGPU handles SDXL base.
    (Passing ``vram_gb=None`` means "probe live", so force the probe to
    return ``None`` to exercise the genuine unknown branch.)"""
    monkeypatch.setattr(
        "lucidium.providers.embedded_image_client.detect_total_vram_gb",
        lambda: None,
    )
    assert em.recommend_model(flavor="cuda").key == "sdxl"


def test_recommend_model_probes_when_signals_omitted(monkeypatch):
    """With neither signal supplied, the function pulls the live flavor
    + VRAM probes (the codebase's None-means-probe idiom)."""
    monkeypatch.setattr("lucidium.providers.torch_overlay.recommend_flavor", lambda: "cuda")
    monkeypatch.setattr(
        "lucidium.providers.embedded_image_client.detect_total_vram_gb",
        lambda: 24.0,
    )
    assert em.recommend_model().key == "z-image-turbo"


def test_z_image_local_name_is_detectable():
    """The Z-Image download must save under a name the loader's
    ``_is_z_image_model_path`` sniff recognises, or the pipeline factory
    would try to load it as SDXL and crash."""
    from lucidium.providers.embedded_image_client import _is_z_image_model_path

    spec = em.MODEL_CATALOG["z-image-turbo"]
    assert _is_z_image_model_path(Path(spec.local_filename))


def test_qwen_local_name_is_detectable():
    """The Qwen download must save under a name the loader's
    ``_is_qwen_model_path`` sniff recognises, or the pipeline factory
    would try to load it as SDXL and crash."""
    from lucidium.providers.embedded_image_client import _is_qwen_model_path

    spec = em.MODEL_CATALOG["qwen-image"]
    assert _is_qwen_model_path(Path(spec.local_filename))


# --------------------------------------------------------------------------
# download_model — streaming, progress, idempotency, failure cleanup
# --------------------------------------------------------------------------


class _FakeResponse(io.BytesIO):
    """Minimal context-manager stand-in for an ``urlopen`` response."""

    def __init__(self, data: bytes, *, status: int = 200, length: bool = True):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))} if length else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_download_model_streams_and_reports_progress(tmp_path, monkeypatch):
    payload = b"x" * (3 * (1 << 20) + 17)  # >3 chunks so progress fires repeatedly
    # The catalog digest belongs to the real 6.9 GB checkpoint, so point
    # the spec at the stand-in payload's hash — the download still has to
    # pass the same integrity gate the shipped entry does.
    spec = replace(em.MODEL_CATALOG["sdxl-turbo"], sha256=hashlib.sha256(payload).hexdigest())

    def fake_urlopen(request, timeout=None):
        assert spec.hf_repo in request.full_url
        assert spec.hf_filename in request.full_url
        return _FakeResponse(payload)

    monkeypatch.setattr(em.urllib.request, "urlopen", fake_urlopen)

    seen: list[tuple[int, int | None]] = []
    out = em.download_model(spec, tmp_path, on_progress=lambda d, t: seen.append((d, t)))

    assert out == tmp_path / spec.local_filename
    assert out.read_bytes() == payload
    # Progress is monotonic, ends at the full size, and carries the total.
    assert seen[-1] == (len(payload), len(payload))
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
    # No leftover .part temp (the name carries a per-attempt uuid now, so
    # glob rather than naming it).
    assert list(tmp_path.glob("*.part")) == []


def test_download_model_is_idempotent(tmp_path, monkeypatch):
    spec = em.MODEL_CATALOG["sdxl"]
    target = tmp_path / spec.local_filename
    target.write_bytes(b"already here")

    def boom(*a, **k):
        raise AssertionError("must not download when the file already exists")

    monkeypatch.setattr(em.urllib.request, "urlopen", boom)
    out = em.download_model(spec, tmp_path)
    assert out == target
    assert target.read_bytes() == b"already here"


def test_download_model_cleans_up_on_failure(tmp_path, monkeypatch):
    spec = em.MODEL_CATALOG["sdxl"]

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(b"", status=404)

    monkeypatch.setattr(em.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(em.ModelDownloadError):
        em.download_model(spec, tmp_path)
    # The failed download leaves the directory clean — no real checkpoint
    # and no dangling .part file.
    assert not (tmp_path / spec.local_filename).exists()
    assert list(tmp_path.glob("*.part")) == []


# --------------------------------------------------------------------------
# embedded_download_model_handler — the frame that reaches all of the above
# --------------------------------------------------------------------------


class _StubSession:
    """The slice of ``Session`` the embedded handlers touch."""

    def __init__(self, models_root: Path) -> None:
        self.settings = Settings(
            image=ImageSettings(embedded_models_dir=str(models_root)),
        )
        self.game = None
        self.emit = None


def _ctx(models_root: Path) -> HandlerContext:
    return HandlerContext(session=_StubSession(models_root))  # type: ignore[arg-type]


async def _drain(result) -> list:
    return [message async for message in result]


# Filesystem probes live in sync helpers so the async tests don't touch
# pathlib on the event loop (ASYNC240).
def _exists(path: Path | str) -> bool:
    return Path(path).exists()


def _entries(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


@pytest.fixture
def offline_download(monkeypatch: pytest.MonkeyPatch):
    """Record every ``(spec, destination)`` the handler asks for and
    write a stand-in file there, without touching the network."""
    calls: list[tuple[str, Path]] = []

    def fake_download(spec, models_dir, on_progress=None):
        calls.append((spec.key, models_dir))
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / spec.local_filename
        target.write_bytes(b"checkpoint")
        if on_progress is not None:
            on_progress(10, 10)
        return target

    monkeypatch.setattr(em, "download_model", fake_download)
    return calls


@pytest.mark.asyncio
async def test_handler_downloads_into_the_configured_directory(
    tmp_path: Path,
    offline_download,
) -> None:
    """With no ``models_dir`` on the wire, the file lands in the
    directory the player configured — and the terminal frame re-lists
    it so the Settings dropdown can select it without a reload."""
    root = tmp_path / "models"
    root.mkdir()

    messages = await _drain(
        await embedded_download_model_handler(
            C2SEmbeddedDownloadModel(key="sdxl-turbo", models_dir=""),
            _ctx(root),
        )
    )

    spec = em.MODEL_CATALOG["sdxl-turbo"]
    assert offline_download == [("sdxl-turbo", root.resolve())]
    assert (root / spec.local_filename).exists()

    stages = [
        payload.stage
        for kind, payload in messages
        if kind is MessageType.s2c_embedded_download_progress
    ]
    assert stages[0] == "resolving"
    assert "downloading" in stages and stages[-1] == "saving"

    kind, payload = messages[-1]
    assert kind is MessageType.s2c_embedded_models
    assert Path(payload.models_dir) == root.resolve()
    assert spec.local_filename in [
        Path(entry).name if isinstance(entry, str) else entry.name for entry in payload.models
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hostile",
    [
        "C:/Windows/System32",
        "/etc",
        "..",
        "../elsewhere",
        "sibling",
    ],
)
async def test_handler_refuses_a_destination_outside_the_configured_root(
    tmp_path: Path,
    hostile: str,
    offline_download,
) -> None:
    """``models_dir`` is attacker-controlled — it arrives on an
    unauthenticated loopback socket. It may narrow within the
    configured root, never replace it, or one frame could point a
    multi-GB stream at (say) the Startup folder."""
    root = tmp_path / "models"
    root.mkdir()
    target = hostile if hostile.startswith(("C:", "/")) else str(tmp_path / hostile)
    existed_before = _exists(target)

    with pytest.raises(SchemaError):
        await embedded_download_model_handler(
            C2SEmbeddedDownloadModel(key="sdxl", models_dir=target),
            _ctx(root),
        )

    # Rejected before anything ran, and nothing was created on the way:
    # no download, no mkdir of a path that wasn't already there, and
    # not a single byte inside the legitimate root either.
    assert offline_download == []
    assert _exists(target) is existed_before
    assert _entries(root) == []


@pytest.mark.asyncio
async def test_handler_allows_narrowing_within_the_root(
    tmp_path: Path,
    offline_download,
) -> None:
    """The Settings screen probes sub-paths of the configured root
    while the player edits the field; that has to keep working."""
    root = tmp_path / "models"
    root.mkdir()
    nested = root / "sdxl"

    await _drain(
        await embedded_download_model_handler(
            C2SEmbeddedDownloadModel(key="sdxl", models_dir=str(nested)),
            _ctx(root),
        )
    )

    assert offline_download == [("sdxl", nested.resolve())]
    assert (nested / em.MODEL_CATALOG["sdxl"].local_filename).exists()


@pytest.mark.asyncio
async def test_handler_falls_back_to_the_recommendation_on_an_unknown_key(
    tmp_path: Path,
    offline_download,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale renderer key must not wedge the wizard's one button —
    the handler substitutes the hardware recommendation."""
    monkeypatch.setattr(em, "recommend_model", lambda: em.MODEL_CATALOG["sdxl-turbo"])
    root = tmp_path / "models"
    root.mkdir()

    await _drain(
        await embedded_download_model_handler(
            C2SEmbeddedDownloadModel(key="model-from-a-future-version", models_dir=""),
            _ctx(root),
        )
    )

    assert offline_download == [("sdxl-turbo", root.resolve())]


@pytest.mark.asyncio
async def test_handler_reports_a_failed_download_as_provider_unreachable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead mirror surfaces as an ``s2c/error`` the wizard can act on
    (retry, or fall back to the manual route) rather than a silent hang
    or a bare traceback — and the directory is left clean.

    The failure happens on a worker thread and is relayed to the
    generator through an asyncio queue, so this is also the test that
    the relay path doesn't swallow it.
    """
    root = tmp_path / "models"
    root.mkdir()

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(b"", status=503)

    monkeypatch.setattr(em.urllib.request, "urlopen", fake_urlopen)

    result = await embedded_download_model_handler(
        C2SEmbeddedDownloadModel(key="sdxl-turbo", models_dir=""),
        _ctx(root),
    )
    with pytest.raises(ProviderUnreachableError) as excinfo:
        await _drain(result)

    spec = em.MODEL_CATALOG["sdxl-turbo"]
    assert spec.display_name in str(excinfo.value)
    # No checkpoint, and no dangling .part for the next run to mistake
    # for a finished download.
    assert not (root / spec.local_filename).exists()
    assert list(root.glob("*.part")) == []


@pytest.mark.asyncio
async def test_handler_preserves_a_lucidium_error_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure that already carries a protocol error code is re-raised
    as-is; only unclassified failures get wrapped."""
    root = tmp_path / "models"
    root.mkdir()

    def boom(spec, models_dir, on_progress=None):
        raise SchemaError("models directory vanished mid-download")

    monkeypatch.setattr(em, "download_model", boom)

    result = await embedded_download_model_handler(
        C2SEmbeddedDownloadModel(key="sdxl", models_dir=""),
        _ctx(root),
    )
    with pytest.raises(SchemaError):
        await _drain(result)

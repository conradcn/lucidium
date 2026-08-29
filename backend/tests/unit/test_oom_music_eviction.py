"""OOM-recovery escalation in the embedded image client.

Pin: when ``_load_with_oom_eviction`` hits a CUDA OOM and there
are no local SDXL pipelines left to evict, the client calls the
``unload_music_model`` hook (wired by the Session to ACE-Step's
``unload_remote_model`` HTTP call) and retries the load once.

Also pins the silent-skip → ``s2c/notice`` plumbing in
``_emit_render_failure_notice``: a swallowed render exception
must produce a player-visible modal with an actionable message
(special-cased for OOM, generic otherwise).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lucidium.api.errors import ProviderUnreachableError
from lucidium.api.handlers import _emit_render_failure_notice
from lucidium.api.messages import MessageType, NoticeKind, S2CNotice
from lucidium.providers.embedded_image_client import EmbeddedImageClient


class _SimulatedOomError(RuntimeError):
    """Stand-in for the CUDA out-of-memory exception. ``_is_oom_error``
    matches on the message substring, which is what diffusers'
    actual error reads as."""

    def __init__(self) -> None:
        super().__init__("CUDA out of memory. Tried to allocate ...")


@pytest.mark.asyncio
async def test_oom_with_pipelines_loaded_evicts_oldest_first() -> None:
    """Baseline: when local pipelines exist, OOM recovery evicts
    the oldest before falling through to the music-unload hook."""
    factory_calls: list[Path] = []

    def factory(path: Path, _device: Any) -> Any:
        factory_calls.append(path)
        # First two loads succeed (warm both pipelines into cache),
        # third raises OOM, fourth (after eviction) succeeds.
        if len(factory_calls) == 3:
            raise _SimulatedOomError()
        return object()

    music_unload_calls: list[None] = []

    async def unload_music() -> bool:
        music_unload_calls.append(None)
        return True

    client = EmbeddedImageClient(
        models_dir=".",
        pipeline_factory=factory,
        bg_remover=lambda b: b,
        unload_music_model=unload_music,
        # Headroom for three resident pipelines so the LRU cap stays
        # out of the way — this test is about the OOM path specifically.
        max_resident_pipelines=3,
    )
    p1 = Path("a.safetensors")
    p2 = Path("b.safetensors")
    p3 = Path("c.safetensors")
    await client._load_with_oom_eviction(p1)
    await client._load_with_oom_eviction(p2)
    await client._load_with_oom_eviction(p3)

    # Music unload hook NOT called — local eviction was sufficient.
    assert music_unload_calls == []
    # p1 (oldest) got evicted; p2 and p3 remain.
    paths_in_cache = list(client._pipelines.keys())
    assert p1 not in paths_in_cache
    assert p2 in paths_in_cache
    assert p3 in paths_in_cache


@pytest.mark.asyncio
async def test_oom_with_no_local_pipelines_calls_music_unload_then_retries() -> None:
    """The escalation path: OOM with empty local cache fires the
    music-unload hook exactly once, then retries the load. On
    retry success, the load completes normally."""
    factory_calls: list[Path] = []

    def factory(path: Path, _device: Any) -> Any:
        factory_calls.append(path)
        # First call raises OOM (cold cache), second succeeds.
        if len(factory_calls) == 1:
            raise _SimulatedOomError()
        return object()

    music_unload_calls: list[None] = []

    async def unload_music() -> bool:
        music_unload_calls.append(None)
        return True

    client = EmbeddedImageClient(
        models_dir=".",
        pipeline_factory=factory,
        bg_remover=lambda b: b,
        unload_music_model=unload_music,
    )
    pipeline = await client._load_with_oom_eviction(Path("first.safetensors"))
    assert pipeline is not None
    # Hook called exactly once.
    assert len(music_unload_calls) == 1
    # Two factory calls — initial OOM + retry after music unload.
    assert len(factory_calls) == 2


@pytest.mark.asyncio
async def test_oom_persists_after_music_unload_raises_provider_error() -> None:
    """Defence-in-depth: if the OOM persists EVEN AFTER the music
    unload, we don't loop forever. A second OOM (third would be) on
    the same load gets converted to ProviderUnreachableError."""

    def factory(_path: Path, _device: Any) -> Any:
        raise _SimulatedOomError()

    async def unload_music() -> bool:
        return True

    client = EmbeddedImageClient(
        models_dir=".",
        pipeline_factory=factory,
        bg_remover=lambda b: b,
        unload_music_model=unload_music,
    )
    with pytest.raises(ProviderUnreachableError) as excinfo:
        await client._load_with_oom_eviction(Path("loop.safetensors"))
    assert "out of memory" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_oom_without_music_hook_raises_directly() -> None:
    """When music isn't running on the same GPU (no hook wired),
    the second OOM converts to ProviderUnreachableError without
    looping."""

    def factory(_path: Path, _device: Any) -> Any:
        raise _SimulatedOomError()

    client = EmbeddedImageClient(
        models_dir=".",
        pipeline_factory=factory,
        bg_remover=lambda b: b,
        unload_music_model=None,
    )
    with pytest.raises(ProviderUnreachableError):
        await client._load_with_oom_eviction(Path("nope.safetensors"))


@pytest.mark.asyncio
async def test_music_unload_hook_failure_is_swallowed_and_retry_still_runs() -> None:
    """If the unload HTTP call itself raises, we log + retry the
    load anyway. A flaky music server should not block image
    rendering completely; worst case the retry OOMs again and we
    raise via the same path as ``test_oom_persists_after_music_unload``.
    """
    factory_calls: list[Path] = []

    def factory(path: Path, _device: Any) -> Any:
        factory_calls.append(path)
        if len(factory_calls) == 1:
            raise _SimulatedOomError()
        return object()  # second attempt succeeds

    async def unload_music() -> bool:
        raise RuntimeError("music server unreachable")

    client = EmbeddedImageClient(
        models_dir=".",
        pipeline_factory=factory,
        bg_remover=lambda b: b,
        unload_music_model=unload_music,
    )
    pipeline = await client._load_with_oom_eviction(Path("retry.safetensors"))
    assert pipeline is not None
    assert len(factory_calls) == 2


@pytest.mark.asyncio
async def test_oom_does_not_call_unload_when_local_pipelines_exist() -> None:
    """Strict ordering: even when an OOM hits, if there's a local
    pipeline available to evict, that goes FIRST. The music-unload
    hook is the second-line recovery, never used while we still
    have unused VRAM headroom on our own side."""
    factory_calls: list[Path] = []

    def factory(path: Path, _device: Any) -> Any:
        factory_calls.append(path)
        # 1st: load a.safetensors (warms cache)
        # 2nd: load b.safetensors → OOM (a is in cache)
        # 3rd: load b.safetensors after evicting a → succeeds
        if len(factory_calls) == 2:
            raise _SimulatedOomError()
        return object()

    music_unload_calls: list[None] = []

    async def unload_music() -> bool:
        music_unload_calls.append(None)
        return True

    client = EmbeddedImageClient(
        models_dir=".",
        pipeline_factory=factory,
        bg_remover=lambda b: b,
        unload_music_model=unload_music,
    )
    await client._load_with_oom_eviction(Path("a.safetensors"))
    await client._load_with_oom_eviction(Path("b.safetensors"))

    # Music hook never touched — local eviction sufficed.
    assert music_unload_calls == []


# ---------- Silent-skip notice -----------------------------------------------


class _StubSession:
    def __init__(self) -> None:
        self.emitted: list[tuple[MessageType, S2CNotice]] = []
        self.emit = self._emit

    def _emit(self, mt: MessageType, payload: object) -> None:
        self.emitted.append((mt, payload))  # type: ignore[arg-type]


def test_emit_render_failure_notice_special_cases_oom() -> None:
    """OOM exception → notice body must mention "GPU memory" and
    suggest disabling music. Generic exceptions don't get the
    OOM-specific hint."""
    session = _StubSession()
    _emit_render_failure_notice(
        session,
        RuntimeError("CUDA out of memory. Tried to allocate ..."),
        kind="portrait/background",
    )
    assert len(session.emitted) == 1
    msg_type, payload = session.emitted[0]
    assert msg_type == MessageType.s2c_notice
    assert isinstance(payload, S2CNotice)
    assert payload.kind == NoticeKind.warning
    assert "GPU memory" in payload.body
    # Actionable: tells the player what to try.
    assert "music" in payload.body.lower()


def test_emit_render_failure_notice_provider_unreachable_blames_backend() -> None:
    """ProviderUnreachableError → notice points at Settings → Image,
    not at GPU memory."""
    session = _StubSession()
    _emit_render_failure_notice(
        session,
        ProviderUnreachableError("ComfyUI server returned 503"),
        kind="portrait/background",
    )
    assert len(session.emitted) == 1
    body = session.emitted[0][1].body
    assert "Settings" in body
    assert "Image" in body
    assert "GPU memory" not in body


def test_emit_render_failure_notice_generic_exception_falls_back() -> None:
    """Any other exception type still produces a notice — the
    player must always know when a render was silently dropped."""
    session = _StubSession()
    _emit_render_failure_notice(
        session,
        KeyError("missing field"),
        kind="portrait/background",
    )
    assert len(session.emitted) == 1
    body = session.emitted[0][1].body
    # Falls back to the generic copy.
    assert "skipped" in body.lower()


def test_emit_render_failure_notice_handles_missing_emit_gracefully() -> None:
    """A session with ``emit=None`` (e.g. no live WS connection)
    must NOT raise from inside the catch path — that would just
    pile errors on top of errors."""

    class _NoEmitSession:
        emit = None

    # No exception expected.
    _emit_render_failure_notice(
        _NoEmitSession(),
        RuntimeError("boom"),
        kind="portrait/background",
    )


def test_emit_render_failure_notice_truncates_runaway_exception_strings() -> None:
    """Some libraries embed full tracebacks in str(exc). The
    notice modal is a one-paragraph body; trim hard so the modal
    stays readable."""
    session = _StubSession()
    long_msg = "x" * 5000
    _emit_render_failure_notice(
        session,
        RuntimeError(long_msg),
        kind="portrait/background",
    )
    body = session.emitted[0][1].body
    assert len(body) < 1000
    # Truncation marker is present so the player knows there's more
    # in the log.
    assert "…" in body

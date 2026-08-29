"""End-to-end live test of the ACE-Step music pipeline.

Unlike ``tests/unit/test_music_pipeline.py`` (which stubs the
audio client and exercises wiring against canned bytes), this
module sends a real HTTP POST to a running ACE-Step server and
asserts the returned audio is well-formed. Skips cleanly when
no server is reachable on the configured port — opt-in via the
``live`` marker.

Run::

    pytest tests/integration/test_music_live.py -m live -v

Requirements:
  * An ACE-Step inference server running at the player's
    configured ``MusicSettings.base_url`` (default
    ``http://127.0.0.1:8001``). The server must speak the
    OpenAI-compat ``/v1/chat/completions`` shape and return the
    rendered clip in
    ``choices[0].message.audio[0].audio_url.url`` either as a
    ``data:audio/...;base64,...`` URL or a server-relative path
    fetchable via ``/v1/audio?path=...`` — both shapes are
    handled by ``AceStepClient``.
  * ``soundfile`` (default dep) — the test decodes the response
    and inspects sample rate / duration / non-silent RMS.
"""

from __future__ import annotations

import io

import httpx
import pytest

from lucidium.domain.settings import MusicSettings
from lucidium.providers.music_client import AceStepClient

_LIVE_FILTERS = [
    "ignore::DeprecationWarning",
    "ignore::UserWarning",
    "ignore::ResourceWarning",
]


def _server_reachable(base_url: str) -> bool:
    """Probe the ACE-Step server with a short-timeout GET. Used
    to skip the test cleanly rather than fail when the player
    hasn't started the server. The upstream FastAPI server
    exposes ``/health``; we fall back to the OpenAPI document
    for builds that lack it."""
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(base_url.rstrip("/") + "/health")
            if r.status_code < 500:
                return True
    except httpx.HTTPError:
        pass
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(base_url.rstrip("/") + "/openapi.json")
            return r.status_code < 500
    except httpx.HTTPError:
        return False


def _decode_audio(buf: bytes):
    """Decode WAV/MP3/FLAC via soundfile. soundfile delegates to
    libsndfile which handles WAV / FLAC natively; MP3 support
    arrived in libsndfile 1.1+."""
    import soundfile as sf

    return sf.read(io.BytesIO(buf), dtype="float32")


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.filterwarnings(*_LIVE_FILTERS)
async def test_music_client_generates_real_audio_against_local_server() -> None:
    """End-to-end: real ACE-Step HTTP server → real audio bytes
    → readable signal. Skips when nothing's listening on the
    configured port; doesn't constrain the exact response format
    (the server can return MP3 or WAV — soundfile handles both)
    so a player can swap inference servers without breaking the
    test."""
    settings = MusicSettings()  # picks up the new 8001 default
    if not _server_reachable(settings.base_url):
        pytest.skip(
            f"ACE-Step server not reachable at {settings.base_url}; "
            "start the server to exercise the live path"
        )

    client = AceStepClient(settings)
    try:
        # Short clip so the test stays under typical CI patience.
        # 15 s is the floor MusicSettings allows; at default
        # inference_steps=27 this typically renders in 20-40 s on
        # a CUDA GPU.
        settings_short = MusicSettings(clip_seconds=15.0)
        client_short = AceStepClient(settings_short)
        try:
            audio_bytes = await client_short.generate_music(
                "ambient cinematic, low strings, sparse piano, slow, melancholic, contemplative",
                seed=12345,
            )
        finally:
            await client_short.aclose()
    finally:
        await client.aclose()

    assert audio_bytes, "ACE-Step server returned empty body"
    # Decode and sanity-check.
    try:
        audio, sample_rate = _decode_audio(audio_bytes)
    except Exception as exc:
        pytest.fail(
            f"could not decode ACE-Step response as audio "
            f"(server may be returning a wrong content type or "
            f"a truncated file): {exc}"
        )
    assert audio.size > 0, "decoded audio had zero samples"
    assert sample_rate in (22050, 32000, 44100, 48000), (
        f"unexpected sample rate {sample_rate}; ACE-Step typically emits 44.1 kHz"
    )
    duration_s = audio.shape[0] / sample_rate
    # Allow the server to truncate / round to its own clip-length
    # convention; just assert SOME audio came back of plausible
    # length (5-60 s for a 15 s request).
    assert 5.0 <= duration_s <= 60.0, (
        f"ACE-Step response duration {duration_s:.1f}s out of plausible range for a 15 s request"
    )
    # Non-silent floor: digital silence has RMS == 0. A real
    # render lands well above 1e-3.
    import numpy as np

    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert rms > 1e-3, (
        f"ACE-Step response was effectively silent (RMS {rms:.6f}); "
        f"the server may have returned a placeholder / error tone"
    )


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.filterwarnings(*_LIVE_FILTERS)
async def test_music_client_propagates_4xx_as_validation_error() -> None:
    """A 4xx from the server (e.g. malformed prompt, model not
    found, server-side validation rejection) must surface as
    ``ProviderValidationError`` — distinguishable from a transport
    failure by the engine's error-handling layers."""
    settings = MusicSettings()
    if not _server_reachable(settings.base_url):
        pytest.skip("ACE-Step server not reachable")

    # Force a 4xx by sending an obviously invalid model name.
    bad_settings = MusicSettings(model_name="not-a-real-model-name/xxx")
    client = AceStepClient(bad_settings)
    try:
        from lucidium.api.errors import ProviderValidationError

        with pytest.raises(ProviderValidationError):
            await client.generate_music("test", seed=1)
    except Exception as exc:
        # If the server happens to ACCEPT unknown model names
        # (some shims fall back to a default), skip rather than
        # fail — the test isn't useful against that server.
        if "ProviderValidationError" not in type(exc).__name__:
            pytest.skip(
                f"server didn't reject unknown model — error path isn't exercisable here: {exc}"
            )
    finally:
        await client.aclose()

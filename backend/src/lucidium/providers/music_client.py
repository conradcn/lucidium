"""ACE-Step background-music generation client.

The engine talks to a configurable ACE-Step inference server
over HTTP — typically the player's own local instance at
``http://127.0.0.1:8001``. We don't ship the model in-process
the way the SDXL client does because ACE-Step's pipeline is
large (~10 GB) and serving it locally is well-trod by upstream
tooling.

Wire shape — matches the real ACE-Step server's
OpenAI-compatible endpoint:

  1. ``POST {base_url}/v1/init`` ONCE per process to load the
     selected model into the server's slot 1. We retry-tolerate
     this — already-initialized servers respond 200 with the
     existing slot info, and a missing endpoint downshifts the
     init step to a no-op.
  2. ``POST {base_url}/v1/chat/completions`` with the prompt
     wrapped in an OpenAI ``messages: [{role: user, content: ...}]``
     payload. The server emits a chat-completion JSON whose
     ``choices[0].message.audio[0].audio_url.url`` is a base64
     ``data:audio/mpeg;base64,...`` URL containing the rendered
     clip. The engine decodes the data URL and writes the bytes
     directly to disk.

This contract is the one the upstream ACE-Step server (FastAPI,
title ``ACE-Step API``, version 1.0) actually exposes. Earlier
versions of this client targeted a hypothetical
``/api/v1/generate`` endpoint that no real server speaks; the
end-to-end live test verified the OpenAI shape against a running
server before this rewrite landed.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import httpx

from ..api.errors import ProviderUnreachableError, ProviderValidationError
from ..domain.settings import MusicSettings

_log = logging.getLogger(__name__)
_call_log = logging.getLogger("lucidium.api_calls")


class AceStepClient:
    """Thin HTTP client that POSTs a music-generation request to an
    ACE-Step inference server and returns the audio bytes.

    Stateless — no model is held in this process. Every call goes
    over the wire. Players who don't run an ACE-Step server keep
    ``MusicSettings.enabled = False`` and the engine never reaches
    this client.

    The client lazily initialises the server's DiT slot the first
    time ``generate_music`` is called for a given model name and
    base URL — subsequent calls skip the init handshake.

    LM bypass: ACE-Step's chat-completions handler will load the
    bundled 5 Hz LM whenever any of ``thinking``, ``sample_mode``,
    ``use_format``, ``use_cot_caption``, or ``use_cot_language``
    is true (see ``acestep/api/llm_generation_inputs.py``). All
    five default to true on the server's request model, but we
    set them ALL to false in the request payload so the LM is
    never invoked and ``init_llm`` can stay false during
    initialisation. OpenRouter is already producing richer
    tag-style audio captions on our side; we pipe those captions
    straight into the DiT without ACE-Step's LM rewriting,
    expanding, or chain-of-thought-ing them. This saves ~3.5 GB
    of VRAM (the 5 Hz LM weights), removes one LM round trip's
    latency from every render, and keeps the music language
    grounded in our scene-aware prompts rather than ACE-Step's
    generic caption training distribution.
    """

    def __init__(
        self,
        settings: MusicSettings,
        *,
        http: httpx.AsyncClient | None = None,
        gpu_lock: asyncio.Lock | None = None,
        evict_image_to_cpu: Callable[[], None] | None = None,
    ) -> None:
        self._settings = settings
        # Music inference is slow (10-90 s typical). Longer than
        # the default httpx timeout — wire the read budget to
        # match. We hold the connection open over a generous
        # window rather than chunked-stream the audio because the
        # server's contract returns one complete response.
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
        self._http = http or httpx.AsyncClient(timeout=timeout)
        # Shared lock with the local SDXL image client. When set
        # AND ``settings.local_gpu_coordination`` is true, the
        # engine assumes ACE-Step is on the same GPU and uses
        # the lock to serialise music+image inference.
        self._gpu_lock = gpu_lock
        self._evict_image_to_cpu = evict_image_to_cpu
        # ``(base_url, model_name)`` pairs we have already
        # /v1/init'd against. Keeps init out of the hot path
        # without wiring a separate "warm" call from the
        # orchestration layer.
        self._initialised: set[tuple[str, str]] = set()
        self._init_lock = asyncio.Lock()
        # Sticky "server is down" flag. Set on the first transport-
        # level failure to /v1/init; subsequent ``generate_music``
        # calls then short-circuit BEFORE the GPU-coordination
        # eviction step. Without this, a player whose ACE-Step
        # server is offline pays the eviction cost on every beat
        # change — SDXL flips to CPU, the music call fails 3-4 s
        # later, the next image render has to flip SDXL back to
        # GPU. Persists for the lifetime of the client; re-enable
        # by clearing settings.json music.enabled (rebuilds the
        # client) or restarting the backend after starting the
        # ACE-Step server.
        self._unreachable = False

    async def _ensure_initialised(self, base_url: str, model_name: str) -> None:
        """POST ``/v1/init`` once per (base_url, model) pair with
        ``init_llm=False`` — we route around ACE-Step's bundled
        5 Hz LM by setting every LM-trigger flag false in the
        chat-completions payload, so loading those weights would
        only cost VRAM with no benefit.

        Tolerates already-initialised servers (they respond 200
        with the existing slot info)."""
        key = (base_url, model_name)
        if key in self._initialised:
            return
        async with self._init_lock:
            if key in self._initialised:
                return
            url = base_url.rstrip("/") + "/v1/init"
            try:
                response = await self._http.post(
                    url,
                    json={"model": model_name, "slot": 1, "init_llm": False},
                )
            except httpx.HTTPError as exc:
                # Mark the client unreachable so subsequent
                # generate_music calls can fail-fast before the
                # GPU-eviction step — see the _unreachable comment
                # in __init__ for the full rationale.
                self._unreachable = True
                raise ProviderUnreachableError(
                    f"ACE-Step server unreachable at {url}: {exc}"
                ) from exc
            if response.status_code >= 500:
                raise ProviderUnreachableError(
                    f"ACE-Step /v1/init returned {response.status_code}: {response.text[:200]}"
                )
            if response.status_code >= 400:
                raise ProviderValidationError(
                    f"ACE-Step /v1/init rejected request "
                    f"({response.status_code}): {response.text[:200]}"
                )
            self._initialised.add(key)

    async def unload_remote_model(self) -> bool:
        """Ask the ACE-Step server to drop its loaded model from
        VRAM. Best-effort: on success the in-memory ``_initialised``
        map is cleared so the next ``generate_music`` call goes
        through the normal warm-up path again.

        Returns True iff the server confirmed an unload (any 2xx
        response), False otherwise. Failures are LOGGED but never
        raised — the image client calls this from its OOM recovery
        path and a stuck music server should not break image
        rendering further.

        ACE-Step exposes this as ``DELETE /v1/load_slot/{slot}``
        in newer builds. Older builds return 404; we try the
        DELETE first, then fall back to a ``POST /v1/cleanup``
        call that some forks ship. Tune the endpoint here if
        your ACE-Step build uses different routes — the image
        client's OOM-recovery call doesn't depend on a specific
        URL, just on the side effect of freeing VRAM.
        """
        if not self._initialised:
            # Nothing to unload from our perspective. We still
            # send the request, because the server may have a
            # model loaded from a previous engine session — but
            # we treat a 404 as "fine, it was already empty."
            pass
        base_url = self._settings.base_url.rstrip("/")
        candidate_calls: list[tuple[str, str, dict[str, Any] | None]] = [
            ("DELETE", f"{base_url}/v1/load_slot/1", None),
            ("POST", f"{base_url}/v1/cleanup", {"slot": 1}),
        ]
        for method, url, body in candidate_calls:
            try:
                if method == "DELETE":
                    response = await self._http.delete(url)
                else:
                    response = await self._http.post(url, json=body or {})
            except httpx.HTTPError as exc:
                _log.warning(
                    "ace-step unload via %s %s: transport failure (%s); trying next candidate",
                    method,
                    url,
                    exc,
                )
                continue
            if 200 <= response.status_code < 300:
                _log.info(
                    "ace-step unload via %s %s: success (%d)",
                    method,
                    url,
                    response.status_code,
                )
                self._initialised.clear()
                return True
            if response.status_code == 404:
                _log.debug(
                    "ace-step unload via %s %s: 404 (endpoint not "
                    "supported on this build); trying next candidate",
                    method,
                    url,
                )
                continue
            _log.warning(
                "ace-step unload via %s %s: %d %s",
                method,
                url,
                response.status_code,
                response.text[:200],
            )
            continue
        _log.warning(
            "ace-step unload: no supported endpoint responded; the music model may still hold VRAM"
        )
        return False

    async def list_models(self) -> dict[str, Any]:
        """Probe ``GET /v1/model_inventory``. Used by the settings
        UI to populate the model dropdown and verify the server is
        reachable. Returns the parsed JSON body verbatim — the UI
        is expected to be tolerant of shape variation across
        ACE-Step versions."""
        url = self._settings.base_url.rstrip("/") + "/v1/model_inventory"
        try:
            response = await self._http.get(url)
        except httpx.HTTPError as exc:
            raise ProviderUnreachableError(f"ACE-Step server unreachable at {url}: {exc}") from exc
        if response.status_code >= 500:
            raise ProviderUnreachableError(
                f"ACE-Step model_inventory returned {response.status_code}: {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise ProviderValidationError(
                f"ACE-Step model_inventory rejected request "
                f"({response.status_code}): {response.text[:200]}"
            )
        try:
            inventory: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ProviderValidationError(
                f"ACE-Step model_inventory returned non-JSON body: {response.text[:200]}"
            ) from exc
        return inventory

    async def generate_music(
        self,
        prompt: str,
        *,
        seed: int,
    ) -> bytes:
        """Render an instrumental looping clip and return its
        bytes. Raises ``ProviderUnreachableError`` on transport
        failure / 5xx, ``ProviderValidationError`` on 4xx /
        unparseable response."""
        text = (prompt or "").strip()
        if not text:
            raise ProviderValidationError("ACE-Step backend requires a non-empty music prompt")

        # Sticky-down short-circuit: if a prior call already
        # discovered the server is offline, skip the GPU
        # eviction + connect attempt and fail immediately. Saves
        # one full eviction-and-reload cycle per attempted music
        # render for the rest of the session. See the
        # ``_unreachable`` comment in __init__.
        if self._unreachable:
            raise ProviderUnreachableError(
                f"ACE-Step server at {self._settings.base_url} marked "
                "unreachable from a prior failed init; not retrying "
                "until the client is rebuilt"
            )

        base_url = self._settings.base_url
        model_name = self._settings.model_name
        chat_url = base_url.rstrip("/") + "/v1/chat/completions"
        # Wrap the player-driven prompt as the user message and
        # ask the server NOT to stream — we want one response
        # body to parse. ``audio_config.duration`` is THE field
        # the server reads for clip length; placing duration at
        # the top level (as we did initially) silently falls
        # through to the model's default of ~138 s. ``inference_steps``
        # and ``guidance_scale`` live at the top level alongside
        # ``seed``. ``modalities`` is set explicitly so the
        # server emits both the audio and the metadata caption.
        payload = {
            "model": model_name,
            "stream": False,
            "messages": [{"role": "user", "content": text}],
            "modalities": ["audio", "text"],
            "seed": int(seed),
            "audio_config": {
                "duration": float(self._settings.clip_seconds),
                "format": "mp3",
                "instrumental": True,
            },
            "inference_steps": int(self._settings.inference_steps),
            "guidance_scale": float(self._settings.guidance_scale),
            # LM bypass — see class docstring. Every flag below
            # is one that the server's llm_generation_inputs gate
            # checks before lazily loading the 5 Hz LM. With all
            # five false, the LM stays unloaded and the DiT
            # consumes our caption directly.
            "thinking": False,
            "sample_mode": False,
            "use_format": False,
            "use_cot_caption": False,
            "use_cot_language": False,
        }
        _call_log.info(
            "music-call backend=ace-step seed=%d duration=%.1f model=%s prompt=%r",
            seed,
            self._settings.clip_seconds,
            model_name,
            text,
        )
        # GPU coordination: when ACE-Step runs on the same GPU
        # as our embedded SDXL pipeline, hold the shared lock for
        # the duration of the HTTP call AND evict the SDXL
        # pipelines to CPU before posting. The HTTP server has
        # full GPU access for the inference window; the next
        # image render reacquires the lock, brings SDXL back to
        # GPU, and proceeds. Without coordination both clients
        # race for VRAM and OOM-trip the smaller side.
        if self._settings.local_gpu_coordination and self._gpu_lock is not None:
            gpu_ctx: Any = self._gpu_lock
        else:
            gpu_ctx = nullcontext()
        started = time.monotonic()
        async with gpu_ctx:
            # Init the music server FIRST. Eviction is a destructive
            # operation on the SDXL pipeline (CPU↔GPU round trip)
            # and the next image render has to undo it via
            # ``restore_to_gpu``. Observed in the wild: when
            # ACE-Step is offline, the original order
            # (evict → init → fail) left SDXL on CPU, the restore
            # path completed cleanly per its own logs, but the
            # next ``_run_pipeline`` call hung indefinitely inside
            # torch — apparently the round trip leaves some
            # diffusers internal state mid-device. Initialising
            # first means a failed init aborts BEFORE any pipeline
            # gets touched, and SDXL stays cleanly resident on GPU.
            await self._ensure_initialised(base_url, model_name)
            if self._settings.local_gpu_coordination and self._evict_image_to_cpu is not None:
                try:
                    self._evict_image_to_cpu()
                except Exception:
                    _log.warning(
                        "ace-step: image-pipeline eviction hook raised; continuing",
                        exc_info=True,
                    )
            try:
                response = await self._http.post(chat_url, json=payload)
            except httpx.HTTPError as exc:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                _call_log.warning(
                    "music-call seed=%d elapsed_ms=%d FAILED transport: %s",
                    seed,
                    elapsed_ms,
                    exc,
                )
                raise ProviderUnreachableError(
                    f"ACE-Step server unreachable at {chat_url}: {exc}"
                ) from exc
            if response.status_code >= 500:
                raise ProviderUnreachableError(
                    f"ACE-Step server returned {response.status_code}: {response.text[:200]}"
                )
            if response.status_code >= 400:
                raise ProviderValidationError(
                    f"ACE-Step server rejected request ({response.status_code}): "
                    f"{response.text[:200]}"
                )
            audio_bytes = await self._extract_audio(response, base_url)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            _call_log.info(
                "music-call seed=%d bytes=%d elapsed_ms=%d backend=ace-step",
                seed,
                len(audio_bytes),
                elapsed_ms,
            )
            return audio_bytes

    async def _extract_audio(self, response: httpx.Response, base_url: str) -> bytes:
        """Pull the audio bytes out of an ACE-Step chat-completion
        response. The server's documented shape puts the rendered
        clip at ``choices[0].message.audio[0].audio_url.url`` as a
        ``data:audio/...;base64,...`` URL. We also handle the
        path-pointer fallback (``...url`` is a relative server
        path that we fetch via ``/v1/audio?path=...``) for
        forward-compat with builds that decline to inline the
        audio."""
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderValidationError(
                f"ACE-Step server returned non-JSON body: {response.text[:200]}"
            ) from exc
        try:
            audio_url = body["choices"][0]["message"]["audio"][0]["audio_url"]["url"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderValidationError(
                "ACE-Step server response missing choices[0].message.audio[0].audio_url.url"
            ) from exc
        if not isinstance(audio_url, str) or not audio_url:
            raise ProviderValidationError("ACE-Step server returned empty audio_url")
        if audio_url.startswith("data:"):
            try:
                _, _, b64 = audio_url.partition(",")
                return base64.b64decode(b64, validate=False)
            except Exception as exc:
                raise ProviderValidationError(
                    f"ACE-Step server returned malformed data URL: {exc}"
                ) from exc
        # Path-pointer fallback: fetch the audio file separately.
        fetch_url = base_url.rstrip("/") + "/v1/audio"
        try:
            audio_response = await self._http.get(fetch_url, params={"path": audio_url})
        except httpx.HTTPError as exc:
            raise ProviderUnreachableError(
                f"ACE-Step audio fetch unreachable at {fetch_url}: {exc}"
            ) from exc
        if audio_response.status_code >= 400:
            raise ProviderValidationError(
                f"ACE-Step audio fetch failed "
                f"({audio_response.status_code}): "
                f"{audio_response.text[:200]}"
            )
        if not audio_response.content:
            raise ProviderValidationError("ACE-Step audio fetch returned empty body")
        return audio_response.content

    async def aclose(self) -> None:
        try:
            await self._http.aclose()
        except Exception:
            _log.debug("ACE-Step client close raised", exc_info=True)


def music_audio_extension(settings: MusicSettings) -> str:
    """Pick the on-disk extension for ACE-Step output. The official
    server returns MP3 by default. Stays a function (not a constant)
    so a future shim could vary by ``settings.model_name`` without
    changing call sites."""
    return "mp3"


__all__ = ["AceStepClient", "music_audio_extension"]

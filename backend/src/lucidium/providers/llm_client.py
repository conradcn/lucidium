"""LLM client protocol + OpenAI-compatible HTTP impl + recorded-fixture fake.

Constitution IV (Testability): the recorded fake is the default in tests.
``conftest.py`` injects it via the offline gate; production code receives
the real client.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import httpx

from ..api.errors import (
    ProviderCreditsError,
    ProviderUnreachableError,
    ProviderValidationError,
)
from ..config import LLM_RETRY_BUDGET
from ..domain.settings import LlmSettings

# Dedicated logger for the provider call timing line. Routed through
# the standard logging tree so the renderer-side console (Electron)
# and the systemd / launchd journal pick it up without extra wiring.
_call_log = logging.getLogger("lucidium.api_calls")


def _raise_for_credits_error(response: httpx.Response) -> None:
    """Detect OpenRouter / OpenAI-compat "out of credits" responses
    and raise ``ProviderCreditsError`` so the renderer can show an
    actionable banner. Three signals to look for:

      * ``402 Payment Required`` — the canonical HTTP code.
      * ``429 Too Many Requests`` whose body matches the credits
        phrasing — OpenRouter sends 429 with code ``insufficient_credits``
        for accounts that have funded > 0 but burned through the
        balance.
      * Any 4xx whose body contains the structured error code
        ``insufficient_credits`` — covers provider tweaks to the
        status code without our regex chasing them.

    Returns silently when the response is fine; raises on credits
    exhaustion. Other 4xx errors fall through to
    ``raise_for_status`` so they keep their existing taxonomy.
    """
    status = response.status_code
    if status < 400:
        return
    if status not in (402, 429) and status >= 500:
        return
    # Best-effort body parse — if it's not JSON, fall through and
    # the caller's existing error path runs.
    try:
        body = response.json()
    except Exception:
        body = None
    code = ""
    message = ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            code = str(err.get("code") or err.get("type") or "").lower()
            message = str(err.get("message") or "")
        elif isinstance(err, str):
            message = err
        if not message:
            message = str(body.get("message") or "")
    raw = (message + " " + code).lower()
    looks_like_credits = (
        status == 402
        or "insufficient_credits" in raw
        or "insufficient credits" in raw
        or "out of credits" in raw
        or "credits remaining" in raw
        or "add credits" in raw
        or ("credit" in raw and "balance" in raw)
    )
    if looks_like_credits:
        # Surface the provider's own message verbatim when it has
        # one — usually the most actionable text — and fall back
        # to a sensible default.
        text = message.strip() or (
            "Your OpenRouter account has run out of credits. "
            "Top up at openrouter.ai/credits to continue."
        )
        raise ProviderCreditsError(text)


class LlmClient(Protocol):
    async def complete(
        self,
        prompt: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> AsyncIterator[str]: ...


def fixture_key(prompt: list[dict[str, str]], *, model: str, temperature: float) -> str:
    """Deterministic fixture key from prompt content."""
    canonical = json.dumps(
        {"prompt": prompt, "model": model, "temperature": round(temperature, 4)},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


class OpenAiCompatibleLlmClient:
    """OpenAI Chat Completions client. Used with OpenRouter by default."""

    def __init__(self, settings: LlmSettings, *, http: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        # Long timeouts on the read leg specifically — large reasoning
        # models can take 60–120 s to start streaming the first token,
        # and OpenRouter occasionally adds a queue wait on top. The
        # earlier 60 s blanket timeout fired during legitimate slow
        # responses and the handler emitted ``s2c/error`` —
        # ``useOptimisticAction`` released its guard, the player saw
        # their choice "reset to the last choice", and the underlying
        # cause was invisible. 300 s is generous but bounded so a
        # truly hung provider still surfaces eventually.
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
        self._http = http or httpx.AsyncClient(timeout=timeout)

    async def complete(
        self,
        prompt: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> AsyncIterator[str]:
        body = {
            "model": model,
            "messages": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        headers = {"Content-Type": "application/json"}
        # ``api_key`` is a ``SecretStr``: always truthy, so test the
        # unwrapped value. This is the only place the real key is read.
        api_key = self._settings.api_key.get_secret_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{self._settings.base_url.rstrip('/')}/chat/completions"

        prompt_chars = sum(len(m.get("content", "")) for m in prompt)
        last_exc: Exception | None = None
        for _attempt in range(LLM_RETRY_BUDGET):
            started = time.monotonic()
            try:
                if not stream:
                    response = await self._http.post(url, json=body, headers=headers)
                    # Check for credits-exhausted before
                    # raise_for_status — that path raises a generic
                    # HTTPError that loses the response body. We
                    # want the body's structured message so the
                    # renderer can surface it.
                    _raise_for_credits_error(response)
                    response.raise_for_status()
                    # OpenRouter occasionally returns a 200 with a
                    # non-JSON body (a fragment of HTML, an empty
                    # body, or a half-written SSE chunk that landed
                    # on the non-stream path) — without this catch,
                    # the bare ``json.JSONDecodeError`` bypasses the
                    # ``httpx.HTTPError`` retry block below and
                    # crashes the handler. Treat it as a transient
                    # provider failure: retry, and on final failure
                    # surface a typed ``ProviderUnreachableError``.
                    try:
                        data = response.json()
                    except json.JSONDecodeError as exc:
                        body_preview = response.text[:200]
                        last_exc = ProviderUnreachableError(
                            f"LLM returned non-JSON 200 body "
                            f"({len(response.content)} bytes): {body_preview!r}"
                        )
                        elapsed_ms = int((time.monotonic() - started) * 1000)
                        _call_log.warning(
                            "llm-call model=%s stream=False prompt_chars=%d "
                            "elapsed_ms=%d attempt=%d FAILED non-JSON body: %s",
                            model,
                            prompt_chars,
                            elapsed_ms,
                            _attempt,
                            exc,
                        )
                        await asyncio.sleep(0.25 * (1 + _attempt))
                        continue
                    text = self._extract_text(data)
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    _call_log.info(
                        "llm-call model=%s stream=False prompt_chars=%d reply_chars=%d "
                        "elapsed_ms=%d attempt=%d",
                        model,
                        prompt_chars,
                        len(text),
                        elapsed_ms,
                        _attempt,
                    )
                    return _single_chunk(text)
                # Streaming path can't measure reply size up front; wrap
                # the iterator so the timing line lands when the stream
                # completes (or errors out).
                return _logged_stream(
                    self._http,
                    url,
                    body,
                    headers,
                    model=model,
                    prompt_chars=prompt_chars,
                    started=started,
                    attempt=_attempt,
                )
            except httpx.HTTPError as exc:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                _call_log.warning(
                    "llm-call model=%s stream=%s prompt_chars=%d elapsed_ms=%d "
                    "attempt=%d FAILED: %s",
                    model,
                    stream,
                    prompt_chars,
                    elapsed_ms,
                    _attempt,
                    exc,
                )
                last_exc = exc
                await asyncio.sleep(0.25 * (1 + _attempt))
        raise ProviderUnreachableError(f"LLM provider failed: {last_exc}")

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderValidationError(f"unexpected LLM response shape: {data}") from exc

    async def aclose(self) -> None:
        await self._http.aclose()


async def _single_chunk(text: str) -> AsyncIterator[str]:
    yield text


async def _stream(
    http: httpx.AsyncClient,
    url: str,
    body: dict[str, object],
    headers: dict[str, str],
) -> AsyncIterator[str]:
    async with http.stream("POST", url, json=body, headers=headers) as response:
        # On a credits-exhausted reply the body is a one-shot JSON
        # error envelope, not an SSE stream. Read it eagerly so
        # ``_raise_for_credits_error`` can match the structured
        # message before raise_for_status throws away the body.
        if response.status_code >= 400:
            await response.aread()
            _raise_for_credits_error(response)
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                return
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            try:
                delta = event["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, TypeError):
                continue
            if delta:
                yield str(delta)


async def _logged_stream(
    http: httpx.AsyncClient,
    url: str,
    body: dict[str, object],
    headers: dict[str, str],
    *,
    model: str,
    prompt_chars: int,
    started: float,
    attempt: int,
) -> AsyncIterator[str]:
    """Wrap ``_stream`` with a timing log line that fires once the
    stream has fully drained (or errored out). The log keeps the
    same shape as the non-streaming path so a later parser doesn't
    need a second case.

    Wraps httpx errors that happen inside the stream into
    ``ProviderUnreachableError`` so the WS server's ``LucidiumError``
    catch-clause emits a structured ``s2c/error`` (the renderer's
    optimistic-action guard listens for ``s2c/error`` to release
    the pending button — without the wrap the bare httpx exception
    would fall into the generic catch and the player would see no
    actionable signal).
    """
    reply_chars = 0
    try:
        async for chunk in _stream(http, url, body, headers):
            reply_chars += len(chunk)
            yield chunk
    except httpx.HTTPError as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _call_log.warning(
            "llm-call model=%s stream=True prompt_chars=%d elapsed_ms=%d attempt=%d FAILED: %s",
            model,
            prompt_chars,
            elapsed_ms,
            attempt,
            exc,
        )
        raise ProviderUnreachableError(f"LLM stream failed: {exc}") from exc
    finally:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _call_log.info(
            "llm-call model=%s stream=True prompt_chars=%d reply_chars=%d elapsed_ms=%d attempt=%d",
            model,
            prompt_chars,
            reply_chars,
            elapsed_ms,
            attempt,
        )


class RecordedLlmClient:
    """Replays fixture files keyed by ``fixture_key`` of the request.

    Fixture format (JSON):
        {"text": "..."}    # full reply
    Or:
        {"chunks": ["..", ".."]}    # streamed chunks

    Missing fixtures raise ``ProviderUnreachableError`` so a missing
    recording fails the test loudly rather than silently fabricating
    output.
    """

    def __init__(self, fixture_root: Path) -> None:
        self._root = fixture_root

    async def complete(
        self,
        prompt: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> AsyncIterator[str]:
        key = fixture_key(prompt, model=model, temperature=temperature)
        fixture_file = self._root / f"{key}.json"
        if not fixture_file.exists():
            raise ProviderUnreachableError(
                f"missing LLM fixture {fixture_file.name} for model={model!r}"
            )
        data = json.loads(fixture_file.read_text(encoding="utf-8"))
        if "text" in data:
            return _single_chunk(str(data["text"]))
        if "chunks" in data:
            return _replay_chunks([str(c) for c in data["chunks"]])
        raise ProviderValidationError(
            f"fixture {fixture_file.name} must contain 'text' or 'chunks'"
        )


async def _replay_chunks(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk

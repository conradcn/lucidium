"""Incremental JSON parser for streaming ``LlmDialogPayload`` responses.

The dialog generation prompt produces a single JSON object shaped
like:

    {
      "beats": [ {...}, {...}, ... ],
      "options": [ {...}, {...} ]
    }

Without streaming, the whole object had to land before ``parse_json_object``
could run, which meant the typewriter started typing only after the
entire LLM response (often 5–15 seconds) finished. With streaming, the
chunks arrive as the model emits them; this parser extracts each
top-level beat object as soon as its closing ``}`` arrives so the
foreground play-advance flow can emit text + state patches per-beat
incrementally.

State machine:

    * ``BEFORE_BEATS`` — accumulate chunks until we find ``"beats" : [``.
    * ``IN_BEATS_ARRAY`` — scan for top-level ``{ ... }`` objects;
      string + escape-aware bracket counting handles JSON strings
      that contain unbalanced curlies.
    * ``AFTER_BEATS_ARRAY`` — accumulate the tail; on stream end, the
      buffer is wrapped back into a single object and parsed for
      ``options``.

The parser is RESILIENT to LLM quirks:

    * Markdown ``` ```json``` fences at the start are stripped (same as
      ``parse_json_object``).
    * Whitespace and commas between array elements are skipped.
    * A truncated final beat is dropped on stream-end if the parser
      can't repair it; everything that DID parse is preserved.

Pydantic-side: each extracted dict is validated through ``LlmBeat``
with the existing null-coercion path, so the same recovery rules
apply per-beat that ``parse_json_object`` provides for the whole
response.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from pydantic import ValidationError

from ..api.errors import ProviderValidationError
from .responses import (
    LlmBeat,
    LlmDialogPayload,
    _coerce_legacy_dialog_shape,
    _coerce_null_typed_fields,
    _strip_json_preamble,
)

_log = logging.getLogger(__name__)


@dataclass
class StreamingDialogParser:
    """Stateful incremental parser. Feed string chunks via ``feed``;
    each call returns any newly-complete ``LlmBeat`` instances.
    On stream end, call ``finalize`` to parse the trailing
    ``options`` and produce the full ``LlmDialogPayload``.

    Beats already returned from ``feed`` are NOT re-emitted by
    ``finalize``. The caller can stitch the final payload together
    using the streamed beats + the finalized payload's options.
    """

    _buffer: str = ""
    _state: str = "BEFORE_BEATS"
    _emitted_beats: list[LlmBeat] = field(default_factory=list)
    # Index into ``_buffer`` where the next scan should resume.
    # Lets ``feed`` skip re-scanning prefix bytes that have already
    # been classified as preamble or as part of an emitted beat.
    _scan_pos: int = 0

    def feed(self, chunk: str) -> list[LlmBeat]:
        """Append a chunk to the running buffer and return any
        newly-complete beats. Empty list when more data is needed."""
        if not chunk:
            return []
        self._buffer += chunk
        out: list[LlmBeat] = []
        # State transitions are linear; loop until no further
        # progress is possible on this buffer.
        while True:
            advanced = False
            if self._state == "BEFORE_BEATS":
                advanced = self._advance_before_beats()
            elif self._state == "IN_BEATS_ARRAY":
                beat = self._try_extract_next_beat()
                if beat is not None:
                    out.append(beat)
                    self._emitted_beats.append(beat)
                    advanced = True
                elif self._array_closed():
                    self._state = "AFTER_BEATS_ARRAY"
                    advanced = True
            else:  # AFTER_BEATS_ARRAY
                # No more incremental work possible until finalize().
                break
            if not advanced:
                break
        return out

    def finalize(self) -> LlmDialogPayload:
        """Parse the full payload after the stream ends. Combines
        already-emitted beats with the trailing structure (which
        carries ``options``). Falls back to attempting a whole-
        buffer parse when the streaming path didn't make it past
        BEFORE_BEATS — that's the case where the LLM never even
        opened the array (truncation, malformed response).
        """
        # Apply the same preamble strip ``parse_json_object`` uses so
        # the legacy whole-buffer fallback works on responses that
        # start with conversational text or a Markdown fence.
        candidate = _strip_json_preamble(self._buffer)

        try:
            # ``raw_decode`` ignores trailing prose / fence-close
            # after the JSON value.
            data, _end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError as exc:
            # Best-effort: if we got SOME beats out via streaming,
            # return a payload built from those beats + empty
            # options. The caller treats no-options as "click
            # Continue" — better than losing the whole response.
            if self._emitted_beats:
                _log.warning(
                    "streaming dialog parse: buffer didn't fully parse "
                    "(%s); returning %d streamed beats with empty "
                    "options",
                    exc,
                    len(self._emitted_beats),
                )
                return LlmDialogPayload(
                    beats=list(self._emitted_beats),
                    options=[],
                )
            raise ProviderValidationError(f"streaming dialog parse failed: {exc}") from exc
        # Coerce legacy + null-typed shapes the same way
        # ``parse_json_object`` does.
        data = _coerce_legacy_dialog_shape(data)
        try:
            payload = LlmDialogPayload.model_validate(data)
        except ValidationError as exc:
            coerced = _coerce_null_typed_fields(data, exc)
            if coerced is not None:
                try:
                    payload = LlmDialogPayload.model_validate(coerced)
                except ValidationError:
                    if self._emitted_beats:
                        _log.warning(
                            "streaming dialog parse: tail validation "
                            "failed; returning %d streamed beats with "
                            "empty options",
                            len(self._emitted_beats),
                        )
                        return LlmDialogPayload(
                            beats=list(self._emitted_beats),
                            options=[],
                        )
                    raise ProviderValidationError(f"streaming dialog parse failed: {exc}") from exc
            else:
                if self._emitted_beats:
                    return LlmDialogPayload(
                        beats=list(self._emitted_beats),
                        options=[],
                    )
                raise ProviderValidationError(f"streaming dialog parse failed: {exc}") from exc
        return payload

    @property
    def emitted_beats(self) -> list[LlmBeat]:
        return list(self._emitted_beats)

    # ---- internals -------------------------------------------------

    def _advance_before_beats(self) -> bool:
        """Scan for ``"beats" : [`` and transition into
        ``IN_BEATS_ARRAY`` once found."""
        # Find the literal ``"beats"`` key. The LLM may emit it
        # with or without surrounding whitespace, but it's always
        # quoted — we don't have to deal with bare-word identifiers.
        idx = self._buffer.find('"beats"', self._scan_pos)
        if idx == -1:
            # Nothing yet. Don't advance scan_pos so partial
            # matches across chunk boundaries get reconsidered.
            return False
        # Past the key, find the colon then the opening bracket.
        post_key = idx + len('"beats"')
        bracket = self._buffer.find("[", post_key)
        if bracket == -1:
            return False
        # Position the scanner just inside the array.
        self._scan_pos = bracket + 1
        self._state = "IN_BEATS_ARRAY"
        return True

    def _try_extract_next_beat(self) -> LlmBeat | None:
        """Scan from ``_scan_pos`` looking for the next complete
        top-level object ``{...}``. Returns the parsed ``LlmBeat``
        if one is available, else ``None`` (need more data) or
        signals array close via the separate ``_array_closed``."""
        # Skip whitespace + commas between elements.
        i = self._scan_pos
        n = len(self._buffer)
        while i < n and self._buffer[i] in ", \t\r\n":
            i += 1
        if i >= n:
            self._scan_pos = i
            return None
        ch = self._buffer[i]
        if ch == "]":
            # Array closed; let _array_closed pick this up.
            return None
        if ch != "{":
            # Either malformed or we're not actually at a beat.
            # Skip one char and retry on the next call. Defensive.
            self._scan_pos = i + 1
            return None
        # Scan from ``i`` looking for the matching ``}``.
        end = self._find_matching_close(i)
        if end is None:
            # Need more data.
            self._scan_pos = i
            return None
        obj_text = self._buffer[i : end + 1]
        try:
            obj = json.loads(obj_text)
        except json.JSONDecodeError as exc:
            _log.warning(
                "streaming dialog parse: object spanning %d–%d failed json.loads (%s); skipping",
                i,
                end,
                exc,
            )
            self._scan_pos = end + 1
            return None
        try:
            beat = LlmBeat.model_validate(obj)
        except ValidationError as exc:
            coerced = _coerce_null_typed_fields(obj, exc)
            if coerced is not None:
                try:
                    beat = LlmBeat.model_validate(coerced)
                except ValidationError:
                    _log.warning(
                        "streaming dialog parse: beat object failed "
                        "schema even after null-coerce; skipping",
                    )
                    self._scan_pos = end + 1
                    return None
            else:
                _log.warning(
                    "streaming dialog parse: beat object failed schema (%s); skipping",
                    exc,
                )
                self._scan_pos = end + 1
                return None
        self._scan_pos = end + 1
        return beat

    def _array_closed(self) -> bool:
        """Has the closing ``]`` of the beats array been seen?"""
        i = self._scan_pos
        n = len(self._buffer)
        while i < n and self._buffer[i] in ", \t\r\n":
            i += 1
        if i < n and self._buffer[i] == "]":
            self._scan_pos = i + 1
            return True
        return False

    def _find_matching_close(self, start: int) -> int | None:
        """Walk from ``start`` (which must be at ``{``) tracking
        string + escape state and bracket depth; return the index
        of the matching ``}``, or ``None`` when we hit end-of-buffer
        before closing.

        Handles: nested objects, strings containing ``{`` / ``}``,
        backslash-escaped quotes inside strings, and bare unicode
        escapes (which JSON allows but never affect bracket depth).
        """
        depth = 0
        in_string = False
        escape = False
        n = len(self._buffer)
        for i in range(start, n):
            ch = self._buffer[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        return None


async def stream_dialog_payload(
    chunks: AsyncIterator[str],
) -> AsyncIterator[LlmBeat | LlmDialogPayload]:
    """Convenience helper: consume an LLM-chunk stream and yield
    each ``LlmBeat`` as it parses, then yield the final
    ``LlmDialogPayload`` (containing all beats + options) when the
    stream ends.

    The caller consumes by ``isinstance``-checking the yielded
    item — beats stream first, the final payload arrives last.
    Useful when the foreground handler wants to emit per-beat
    s2c events as they parse.
    """
    parser = StreamingDialogParser()
    async for chunk in chunks:
        for beat in parser.feed(chunk):
            yield beat
    yield parser.finalize()

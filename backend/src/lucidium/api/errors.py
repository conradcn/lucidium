"""Structured error taxonomy.

Errors that reach the renderer are typed (``code``, ``message``,
``recoverable``); they are never raw stack traces. The taxonomy is small
on purpose — adding a code is a deliberate act, not a convenience.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

_log = logging.getLogger(__name__)


def new_correlation_id() -> str:
    """Short opaque id tying a client-facing message to a log entry."""
    return uuid4().hex[:8]


class ErrorCode(StrEnum):
    schema_error = "schema_error"
    provider_unreachable = "provider_unreachable"
    provider_validation = "provider_validation"
    # OpenRouter (and other OpenAI-compat providers) return 402 /
    # specific error bodies when the player's account has run out
    # of credits. We surface this as its own code so the renderer
    # can show an actionable banner ("add credits at openrouter.ai")
    # instead of the generic "provider unreachable" fallback.
    provider_credits = "provider_credits"
    not_found = "not_found"
    internal = "internal"
    # Renderer-originated, never emitted by the backend. The WebSocket
    # client synthesises an ``s2c/error`` with this code when a send is
    # dropped because the socket is down, so ``useOptimisticAction``
    # releases its pending guard instead of locking the UI. It lives in
    # the taxonomy because it travels on the s2c/error channel and the
    # renderer's generated ``ErrorCode`` union has to admit it.
    ws_disconnected = "ws_disconnected"


class StructuredError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    recoverable: bool = True


class LucidiumError(Exception):
    """Base class for errors that map to a structured renderer message."""

    def __init__(self, code: ErrorCode, message: str, *, recoverable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable

    def to_structured(self) -> StructuredError:
        return StructuredError(code=self.code, message=self.message, recoverable=self.recoverable)


class SchemaError(LucidiumError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.schema_error, message, recoverable=True)


class ProviderUnreachableError(LucidiumError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.provider_unreachable, message, recoverable=True)


class ProviderValidationError(LucidiumError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.provider_validation, message, recoverable=True)


class NotFoundError(LucidiumError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.not_found, message, recoverable=True)


class ProviderCreditsError(LucidiumError):
    """OpenRouter / OpenAI-compat provider rejected the call with a
    "you've run out of credits" signal. Distinct from
    ``ProviderUnreachableError`` because the fix is funding the
    account, not retrying or checking the network."""

    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.provider_credits, message, recoverable=True)


class SaveVersionError(SchemaError, ValueError):
    """A save written by a NEWER build than the one reading it.

    Recoverable in the only sense that matters to the player: nothing is
    broken and no data was touched, they just need a newer Lucidium. It
    is also a ``ValueError`` because the persistence layer raises it from
    a plain parse path whose callers already treat bad data that way.
    """

    def __init__(self, save_version: int, supported_version: int) -> None:
        super().__init__(
            "This save was created by a newer version of Lucidium "
            f"(save format {save_version}; this build understands up to "
            f"{supported_version}). Update Lucidium to open it — the save "
            "has been left untouched."
        )
        self.save_version = save_version
        self.supported_version = supported_version


def from_exception(exc: BaseException) -> StructuredError:
    """Best-effort conversion of an arbitrary exception to a renderer error.

    Unmapped exceptions never carry their ``str(exc)`` to the client:
    an ``OSError`` on the models directory would otherwise ship an
    absolute user path over the socket. The client gets the exception
    class name plus a correlation id; the full detail (message and
    traceback) goes to ``session.log`` at ERROR keyed by that id, so a
    bug report is still diagnosable from a screenshot.
    """
    if isinstance(exc, LucidiumError):
        return exc.to_structured()
    if isinstance(exc, PydanticValidationError):
        # Almost always a save (or other on-disk document) whose shape no
        # longer matches the models — a field added or removed across
        # builds, which ``extra="forbid"`` turns into a hard failure. The
        # player can act on that ("this file is from a different build"),
        # so it must not land in the ``internal`` bucket with
        # ``recoverable=False`` and verbatim pydantic text. The raw detail
        # (which quotes on-disk values, and can include paths) goes to the
        # log under a correlation id instead.
        ref = new_correlation_id()
        _log.error("validation error [ref %s]: %s", ref, exc)
        count = exc.error_count()
        fields = ", ".join(".".join(str(part) for part in err["loc"]) for err in exc.errors()[:3])
        detail = f" (fields: {fields})" if fields else ""
        return StructuredError(
            code=ErrorCode.schema_error,
            message=(
                f"This file doesn't match what this build of Lucidium expects — "
                f"{count} field(s) failed validation{detail}. It was most likely "
                f"written by a different version. Nothing was modified. (ref: {ref})"
            ),
            recoverable=True,
        )
    ref = new_correlation_id()
    _log.error("internal error [ref %s]: %r", ref, exc, exc_info=exc)
    return StructuredError(
        code=ErrorCode.internal,
        message=f"internal error [{type(exc).__name__}] (ref: {ref})",
        recoverable=False,
    )

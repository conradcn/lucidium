from __future__ import annotations

import httpx

from lucidium.api.errors import (
    ErrorCode,
    ProviderUnreachableError,
    SchemaError,
    from_exception,
)


def test_lucidium_error_round_trips() -> None:
    err = SchemaError("bad shape")
    payload = err.to_structured()
    assert payload.code == ErrorCode.schema_error
    assert payload.message == "bad shape"
    assert payload.recoverable is True


def test_from_exception_wraps_unknown() -> None:
    payload = from_exception(RuntimeError("kaboom"))
    assert payload.code == ErrorCode.internal
    assert "RuntimeError" in payload.message
    assert payload.recoverable is False


def test_from_exception_preserves_lucidium_subclasses() -> None:
    payload = from_exception(ProviderUnreachableError("provider down"))
    assert payload.code == ErrorCode.provider_unreachable
    assert payload.recoverable is True


def test_from_exception_handles_httpx_error_via_provider_class() -> None:
    # httpx errors are not LucidiumError; they should fall through to internal.
    payload = from_exception(httpx.ConnectError("refused"))
    assert payload.code == ErrorCode.internal

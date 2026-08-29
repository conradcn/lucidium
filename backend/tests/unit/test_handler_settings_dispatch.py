"""``c2s/settings/get`` and ``c2s/settings/validate_api_key`` through dispatch.

``settings/get`` is the handler that decides what the renderer learns
about the player's configuration, and the configuration contains a live,
billable ``llm.api_key``. The masking lives in a ``field_serializer`` on
``LlmSettings`` that only fires in JSON mode — so the assertion that
matters is on the bytes the handler actually emits, not on the model.

``settings/validate_api_key`` is the one handler here that makes an
outbound HTTP call. The offline gate in ``tests/conftest.py`` denies
every unrouted request; the ``http_router`` fixture is that same gate,
handed over so a test can open exactly one URL and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from lucidium.api.handlers import HandlerContext
from lucidium.api.messages import MessageType
from lucidium.domain.settings import LlmSettings, Settings

from .handler_harness import dispatch, make_registry, make_session, types_of

REAL_KEY = "sk-or-v1-totally-real-billable-key"
BASE_URL = "https://provider.invalid/v1"


def _settings_with_key() -> Settings:
    return Settings(llm=LlmSettings(api_key=REAL_KEY, base_url=BASE_URL))


# ---------------------------------------------------------------------------
# c2s/settings/get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settings_get_emits_a_replace_op_for_the_whole_settings_tree(
    tmp_app_data: Path,
) -> None:
    session = make_session(tmp_app_data)
    messages = await dispatch(
        make_registry(), HandlerContext(session=session), MessageType.c2s_settings_get
    )

    assert types_of(messages) == [MessageType.s2c_state_patch]
    ops = messages[0][1].ops
    assert len(ops) == 1
    assert ops[0].op == "replace"
    assert ops[0].path == "/settings"
    # Sanity: it really is the settings tree, not an empty stub.
    assert "llm" in ops[0].value
    assert ops[0].value["llm"]["base_url"] == Settings().llm.base_url


@pytest.mark.asyncio
async def test_settings_get_never_puts_the_api_key_on_the_wire(
    tmp_app_data: Path,
) -> None:
    """The abuse case: a configured key must not round-trip to the
    renderer. Asserted against the SERIALIZED frame, because that's what
    the WebSocket writes — checking ``ops[0].value`` alone would miss a
    ``SecretStr`` that only unmasks at encode time."""
    session = make_session(tmp_app_data, settings=_settings_with_key())
    # The key really is loaded — otherwise this test passes vacuously.
    assert session.settings.llm.api_key.get_secret_value() == REAL_KEY

    messages = await dispatch(
        make_registry(), HandlerContext(session=session), MessageType.c2s_settings_get
    )

    payload = messages[0][1]
    assert payload.ops[0].value["llm"]["api_key"] == ""
    encoded = payload.model_dump_json()
    assert REAL_KEY not in encoded
    assert "totally-real" not in encoded
    # And nothing smuggled it in under a different key.
    assert REAL_KEY not in json.dumps(payload.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# c2s/settings/validate_api_key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_api_key_reports_a_2xx_as_valid(
    tmp_app_data: Path, http_router: respx.MockRouter
) -> None:
    session = make_session(tmp_app_data)
    route = http_router.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_settings_validate_api_key,
        {"base_url": BASE_URL, "api_key": REAL_KEY},
    )

    assert types_of(messages) == [MessageType.s2c_settings_api_key_validation]
    reply = messages[0][1]
    assert reply.ok is True
    assert reply.status == "valid"
    # The key travels as a Bearer token, not a query parameter.
    assert route.called
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {REAL_KEY}"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [401, 403])
async def test_validate_api_key_reports_a_rejected_key_as_unauthorized(
    tmp_app_data: Path, code: int, http_router: respx.MockRouter
) -> None:
    session = make_session(tmp_app_data)
    http_router.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(code))

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_settings_validate_api_key,
        {"base_url": BASE_URL, "api_key": "sk-wrong"},
    )

    reply = messages[0][1]
    assert reply.ok is False
    assert reply.status == "unauthorized"


@pytest.mark.asyncio
async def test_validate_api_key_reports_a_transport_failure_as_unreachable(
    tmp_app_data: Path, http_router: respx.MockRouter
) -> None:
    session = make_session(tmp_app_data)
    http_router.get(f"{BASE_URL}/models").mock(side_effect=httpx.ConnectError("no route to host"))

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_settings_validate_api_key,
        {"base_url": BASE_URL, "api_key": REAL_KEY},
    )

    reply = messages[0][1]
    assert reply.ok is False
    assert reply.status == "unreachable"


@pytest.mark.asyncio
async def test_validate_api_key_reports_an_unexpected_status_as_unreachable(
    tmp_app_data: Path, http_router: respx.MockRouter
) -> None:
    session = make_session(tmp_app_data)
    http_router.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(503))

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_settings_validate_api_key,
        {"base_url": BASE_URL, "api_key": REAL_KEY},
    )

    reply = messages[0][1]
    assert reply.ok is False
    assert reply.status == "unreachable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "api_key"),
    [("", REAL_KEY), (BASE_URL, ""), (BASE_URL, "   ")],
)
async def test_validate_api_key_short_circuits_on_blank_input(
    tmp_app_data: Path, base_url: str, api_key: str
) -> None:
    """No route is registered, so the offline gate would fail the test if
    the handler reached the network before checking its inputs."""
    session = make_session(tmp_app_data)
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_settings_validate_api_key,
        {"base_url": base_url, "api_key": api_key},
    )

    reply = messages[0][1]
    assert reply.ok is False
    assert reply.status == "invalid_input"


@pytest.mark.asyncio
async def test_validate_api_key_does_not_mutate_stored_settings(
    tmp_app_data: Path, http_router: respx.MockRouter
) -> None:
    """Validation probes a CANDIDATE key. Nothing may be persisted — the
    player hasn't pressed Save yet."""
    session = make_session(tmp_app_data)
    before = session.settings.model_dump(mode="json")
    http_router.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json={}))

    await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_settings_validate_api_key,
        {"base_url": BASE_URL, "api_key": "sk-candidate"},
    )

    assert session.settings.model_dump(mode="json") == before
    assert session.settings.llm.api_key.get_secret_value() == ""

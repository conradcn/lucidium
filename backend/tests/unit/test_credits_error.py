"""Coverage for the OpenRouter "out of credits" detection.

The LLM client distinguishes credits exhaustion from generic
provider failures so the renderer can show an actionable banner
("top up at openrouter.ai/credits") instead of the catch-all
"backend unreachable, check the connection" message that fires
for transport / 5xx errors.

Three signal shapes the helper has to recognise:

  * HTTP 402 — the canonical Payment Required code.
  * HTTP 429 with body code ``insufficient_credits`` —
    OpenRouter's actual response when a funded account has
    burned its balance.
  * HTTP 4xx with a free-text "out of credits" / "insufficient
    credits" message — covers provider tweaks that don't update
    the structured ``code`` field.

Other 4xx (rate limit hits, malformed prompt, model not found)
must NOT trip credits detection — they keep their existing
ProviderValidationError path.
"""

from __future__ import annotations

import json

import httpx
import pytest

from lucidium.api.errors import ProviderCreditsError
from lucidium.providers.llm_client import _raise_for_credits_error


def _resp(status: int, body: object) -> httpx.Response:
    """Synthesize an httpx.Response with a JSON body for testing."""
    text = json.dumps(body) if isinstance(body, (dict, list)) else str(body)
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return httpx.Response(
        status_code=status,
        content=text.encode("utf-8"),
        request=request,
        headers={"content-type": "application/json"},
    )


def test_402_payment_required_raises() -> None:
    """Canonical HTTP signal — any 402 with any body counts."""
    response = _resp(402, {"error": {"message": "Account balance exhausted."}})
    with pytest.raises(ProviderCreditsError) as excinfo:
        _raise_for_credits_error(response)
    assert "Account balance exhausted" in str(excinfo.value)


def test_429_insufficient_credits_code() -> None:
    """OpenRouter's structured 429 with code=insufficient_credits."""
    response = _resp(
        429,
        {"error": {"code": "insufficient_credits", "message": "Add credits to continue."}},
    )
    with pytest.raises(ProviderCreditsError) as excinfo:
        _raise_for_credits_error(response)
    assert "Add credits" in str(excinfo.value)


def test_400_with_credits_phrasing() -> None:
    """Free-text 400 mentioning credits — provider may shift codes."""
    response = _resp(
        400,
        {"error": {"message": "You have no credits remaining on your account."}},
    )
    with pytest.raises(ProviderCreditsError):
        _raise_for_credits_error(response)


@pytest.mark.parametrize(
    "status,body",
    [
        (200, {"choices": []}),  # OK
        (429, {"error": {"message": "Rate limit hit. Try again."}}),  # not credits
        (404, {"error": {"message": "Model not found."}}),
        (400, {"error": {"message": "Invalid prompt format."}}),
        (500, {"error": {"message": "internal server error"}}),
    ],
)
def test_non_credits_responses_do_not_trip(status: int, body: object) -> None:
    """All other 2xx/4xx/5xx responses must fall through silently
    so the existing httpx.raise_for_status path keeps owning them."""
    response = _resp(status, body)
    # Should NOT raise.
    _raise_for_credits_error(response)


def test_non_json_body_falls_through() -> None:
    """Provider errors that return HTML (Cloudflare interstitials,
    proxy failures) must not crash the detector — fall through to
    the caller's raise_for_status."""
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(
        status_code=502,
        content=b"<html><body>502 Bad Gateway</body></html>",
        request=request,
        headers={"content-type": "text/html"},
    )
    _raise_for_credits_error(response)  # Should NOT raise.


def test_message_falls_back_to_default() -> None:
    """When the body has the credits signal but no human-readable
    message, the raised error gets a sensible default the renderer
    can display verbatim."""
    response = _resp(402, {"error": {"code": "insufficient_credits"}})
    with pytest.raises(ProviderCreditsError) as excinfo:
        _raise_for_credits_error(response)
    assert "openrouter.ai" in str(excinfo.value).lower()

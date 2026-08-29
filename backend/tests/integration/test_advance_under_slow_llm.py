"""Regression: a slow LLM provider must NOT cause a choice click to
be silently dropped.

The earlier failure mode: ``OpenAiCompatibleLlmClient`` had a 60 s
blanket timeout. A slower model (or queued request) would trip the
timeout, the handler raised ``ProviderUnreachableError``, the WS
server emitted ``s2c/error``, and on the renderer the optimistic
guard released — the player saw their pick "reset to the last
choice" with no visible explanation.

This test exercises the BACKEND half of that failure mode end-to-end:
it uses the REAL ``OpenAiCompatibleLlmClient`` (not the queued stub
from ``test_us1_flow``) talking to an httpx MockTransport that
delays responses to mimic a slow provider. The advance handler must
run to completion without raising and must yield a state/patch
carrying a new ``current_node_id``.

The frontend half lives in
``frontend/tests/unit/optimistic_guard_no_false_release.test.tsx``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import (
    Envelope,
    MessageType,
)
from lucidium.domain.character import Character
from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogOption,
    DialogTree,
)
from lucidium.domain.game import Game
from lucidium.domain.settings import LlmSettings, Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.session import Session
from lucidium.providers.llm_client import OpenAiCompatibleLlmClient


def _bootstrap_game() -> Game:
    """Hand-built game state with a single committed node carrying
    two options. The advance handler will look for a pre-generated
    child, find none, and call the LLM."""
    pc = Character(
        id="pc",
        is_player=True,
        name="Iris",
        description="archivist",
        gender="female",
        age=28,
        ethnicity="local",
        skin="pale",
        hair_color="auburn",
        hairstyle="braid",
        eye_color="grey",
        build="slight",
        bust="moderate",
        outfit="wool coat",
        pose="standing",
        expression="alert",
        seed=1,
    )
    n0 = DialogNode(
        id="n0",
        parent_id=None,
        chosen_option_id=None,
        text="A choice in front of you.",
        options=[
            DialogOption(id="o1", text="Walk left."),
            DialogOption(id="o2", text="Walk right."),
        ],
        state=DialogNodeState.committed,
        premise_hash="h0",
    )
    return Game(
        world=WorldState(
            game_name="Test",
            setting="harbor",
            genre="mystery",
            visual_style="ink",
        ),
        characters={"pc": pc},
        dialog_tree=DialogTree(
            nodes={"n0": n0},
            root_id="n0",
            committed_path=["n0"],
        ),
        environments={},
        current_node_id="n0",
        on_stage=[],
    )


def _llm_payload_text() -> str:
    payload = {
        "beats": [
            {
                "text": "The cobbles tilt under your boots.",
                "speaker_id": None,
                "entering_character_ids": [],
                "leaving_character_ids": [],
                "new_characters": [],
                "location_id": None,
                "location_prompt": None,
                "location_lighting": "",
                "character_changes": [],
            }
        ],
        "options": [],
    }
    return json.dumps(payload)


class _NullImage:
    async def generate(self, *_a, **_kw) -> bytes:
        # 1×1 transparent PNG — the asset pipeline expects bytes back
        # but the test never reads the file content.
        return bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )


def _make_slow_transport(delay: float, content: str) -> httpx.MockTransport:
    """httpx ``MockTransport`` that delays responses to simulate a
    slow LLM provider. The handler distinguishes streaming vs non-
    streaming requests from the ``stream`` body field — the
    storyteller uses streaming by default, the world-init path uses
    non-streaming."""

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(delay)
        body = json.loads(request.content.decode("utf-8") or "{}")
        if body.get("stream"):
            # Server-sent events with a single chunk + DONE marker.
            sse = (
                f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                content=sse.encode("utf-8"),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    return httpx.MockTransport(handler)


def _make_failing_transport(status: int) -> httpx.MockTransport:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=f"upstream returned {status}")

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_choice_click_completes_under_slow_llm(
    tmp_app_data: Path,
) -> None:
    """A slow LLM (5 s) must not cause the advance to silently drop.
    The handler must complete and the game's current_node_id must
    advance off the parent."""

    settings = Settings(
        llm=LlmSettings(
            base_url="http://fake-openrouter.test",
            model="x/fake",
            api_key="sk-fake",
            temperature=0.0,
        ),
    )
    transport = _make_slow_transport(5.0, _llm_payload_text())
    http = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
    )
    llm_client = OpenAiCompatibleLlmClient(settings.llm, http=http)

    session = Session(
        settings=settings,
        llm_client=llm_client,
        image_client=_NullImage(),
        saves_root=tmp_app_data / "saves",
    )
    session.install_game(_bootstrap_game())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    before_node = session.game.current_node_id
    out = []
    async for message in registry.dispatch(
        Envelope(
            type=MessageType.c2s_play_advance,
            payload={"option_id": "o1"},
        ),
        ctx,
    ):
        out.append(message)

    assert session.game is not None
    assert session.game.current_node_id != before_node, (
        f"current_node_id did not advance — slow LLM caused a silent drop. "
        f"Before: {before_node!r}, after: {session.game.current_node_id!r}"
    )
    patches = [payload for type_, payload in out if type_ == MessageType.s2c_state_patch]
    matching = [p for p in patches if any(op.path == "/current_node_id" for op in p.ops)]
    assert matching, "expected a state/patch updating /current_node_id"

    await http.aclose()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_choice_click_surfaces_error_when_llm_fails(
    tmp_app_data: Path,
) -> None:
    """If the LLM provider returns a hard error, the handler must
    raise ``LucidiumError`` (which the WS server converts to
    ``s2c/error``). The renderer's ``useOptimisticAction`` reads
    ``s2c/error`` to release the pending guard, so the player can
    retry instead of staring at a dead button."""

    settings = Settings(
        llm=LlmSettings(
            base_url="http://broken-openrouter.test",
            model="x/fake",
            api_key="sk-fake",
        ),
    )
    transport = _make_failing_transport(500)
    http = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0),
    )
    llm_client = OpenAiCompatibleLlmClient(settings.llm, http=http)

    session = Session(
        settings=settings,
        llm_client=llm_client,
        image_client=_NullImage(),
        saves_root=tmp_app_data / "saves",
    )
    session.install_game(_bootstrap_game())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    from lucidium.api.errors import LucidiumError

    with pytest.raises(LucidiumError):
        async for _ in registry.dispatch(
            Envelope(
                type=MessageType.c2s_play_advance,
                payload={"option_id": "o1"},
            ),
            ctx,
        ):
            pass

    await http.aclose()

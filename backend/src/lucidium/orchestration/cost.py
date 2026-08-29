"""Cost telemetry helpers.

A real provider would return ``usage: {prompt_tokens, completion_tokens}``
on every response. Our ``LlmClient`` protocol does not surface that yet,
so we estimate from text length (a 4-character-per-token heuristic) and
mark the estimate as such. When a provider client gains a real usage
field, it can call ``CostMeter.record_exact`` instead of ``record_estimate``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.game import CostTelemetry, Game

_CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass(frozen=True)
class CostDelta:
    tokens_in: int = 0
    tokens_out: int = 0
    image_calls: int = 0
    llm_calls: int = 0
    latency_ms: int = 0
    dollar_estimate: float = 0.0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def estimate_llm_call(*, prompt_chars: int, reply_chars: int, latency_ms: int) -> CostDelta:
    return CostDelta(
        tokens_in=max(1, prompt_chars // _CHARS_PER_TOKEN_ESTIMATE),
        tokens_out=max(1, reply_chars // _CHARS_PER_TOKEN_ESTIMATE),
        llm_calls=1,
        latency_ms=latency_ms,
    )


def estimate_image_call(*, latency_ms: int) -> CostDelta:
    return CostDelta(image_calls=1, latency_ms=latency_ms)


def apply_delta(telemetry: CostTelemetry, delta: CostDelta) -> CostTelemetry:
    return telemetry.model_copy(
        update={
            "tokens_in": telemetry.tokens_in + delta.tokens_in,
            "tokens_out": telemetry.tokens_out + delta.tokens_out,
            "image_calls": telemetry.image_calls + delta.image_calls,
            "llm_calls": telemetry.llm_calls + delta.llm_calls,
            "latency_ms_total": telemetry.latency_ms_total + delta.latency_ms,
            "dollar_estimate": telemetry.dollar_estimate + delta.dollar_estimate,
        }
    )


def fold_into_game(game: Game, delta: CostDelta) -> Game:
    return game.model_copy(update={"cost_telemetry": apply_delta(game.cost_telemetry, delta)})

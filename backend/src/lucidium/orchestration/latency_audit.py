"""Per-session timing recorder for turn-latency audits.

Records timestamps of the six events the audit cares about:

  * ``predictable``    — a node became available to pre-generate
                         (its parent committed with that option exposed).
                         For a free-text input there is no advance
                         prediction possible; the audit code synthesises
                         ``predictable`` = ``click`` for those turns.
  * ``click``          — c2s/play/advance or c2s/play/free_text entered
                         the handler.
  * ``llm_start``      — the LLM call for this (parent, option) pair
                         actually started (speculative OR foreground).
  * ``llm_end``        — that LLM call finished producing the FIRST beat
                         the player will see (chain head). Subsequent
                         beats may keep streaming but the first beat is
                         what releases the renderer.
  * ``displayed``      — the first ``s2c/state/patch`` and text-streaming
                         message for the new node went out on the wire.
                         (Frontend render latency lives below that and is
                         out of scope for the backend audit.)
  * ``speculation_*``  — bookkeeping: spawn / await-start / await-end /
                         cancel. Not used in the invariant but invaluable
                         for explaining a violation.

Disabled by default; ``Session.enable_latency_audit()`` flips the bit so
production calls record nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyEvent:
    ts: float
    name: str
    parent_id: str | None = None
    option_id: str | None = None
    node_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class LatencyAudit:
    """Append-only event log scoped to a single session."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.events: list[LatencyEvent] = []

    def enable(self) -> None:
        self.enabled = True

    def record(
        self,
        name: str,
        *,
        parent_id: str | None = None,
        option_id: str | None = None,
        node_id: str | None = None,
        **extra: Any,
    ) -> None:
        if not self.enabled:
            return
        self.events.append(
            LatencyEvent(
                ts=time.monotonic(),
                name=name,
                parent_id=parent_id,
                option_id=option_id,
                node_id=node_id,
                extra=extra,
            )
        )

    def clear(self) -> None:
        self.events.clear()

    # --- query helpers ----------------------------------------------------

    def by_name(self, name: str) -> list[LatencyEvent]:
        return [e for e in self.events if e.name == name]

    def first(
        self,
        name: str,
        *,
        parent_id: str | None = None,
        option_id: str | None = None,
        after: float | None = None,
    ) -> LatencyEvent | None:
        for e in self.events:
            if e.name != name:
                continue
            if parent_id is not None and e.parent_id != parent_id:
                continue
            if option_id is not None and e.option_id != option_id:
                continue
            if after is not None and e.ts < after:
                continue
            return e
        return None


__all__ = ["LatencyAudit", "LatencyEvent"]

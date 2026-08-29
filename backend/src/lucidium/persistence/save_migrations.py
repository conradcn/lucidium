"""Forward migrations for ``game.json`` payloads.

``Game`` is ``extra="forbid"``, so the moment a domain model gains or
loses a field every save already on disk becomes unloadable with a bare
Pydantic message. The fix is to bump :data:`lucidium.config.GAME_SCHEMA_VERSION`
and register a migration here that rewrites the old payload into the new
shape. ``load_save`` applies them in order, on the raw ``dict`` read from
the file, *before* validation — so a migration never has to construct a
model, only move keys around.

Adding the next one
-------------------
Say the current version is 1 and you are adding ``world.weather``:

1. Bump ``GAME_SCHEMA_VERSION`` to 2 in ``config.py``.
2. Write the upgrade and register it under the version it upgrades *from*::

       def _v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
           data["world"]["weather"] = "clear"
           return data

       MIGRATIONS[1] = _v1_to_v2

3. Leave the identity entry for the (new) current version in place.

Rules for a migration callable:

* It takes the decoded payload and returns the payload one version
  newer. Mutating and returning the same dict is fine — ``load_save``
  hands it a freshly parsed object that nothing else holds.
* It MUST NOT stamp ``schema_version``; ``load_save`` does that after the
  chain runs, from the registry key it advanced to.
* Every version between the oldest supported save and the current one
  needs an entry, or the chain stops early and validation fails with the
  original (accurate, if blunt) Pydantic error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import GAME_SCHEMA_VERSION

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def _identity(data: dict[str, Any]) -> dict[str, Any]:
    """No-op: a save already at the current version needs no rewriting.

    Registered so the current version is always present in the table —
    which is what lets ``load_save`` distinguish "nothing to do" from
    "this version is a gap in the chain"."""
    return data


# ``from_version -> upgrade to from_version + 1``. The entry for the
# current version is the identity above and stays that way until the
# version is bumped, at which point it becomes the real upgrade and a
# fresh identity is added for the new current version.
MIGRATIONS: dict[int, Migration] = {
    GAME_SCHEMA_VERSION: _identity,
}


def migrate(data: dict[str, Any], from_version: int) -> dict[str, Any]:
    """Run the registered chain from ``from_version`` up to the current one.

    Stops as soon as a version has no registered migration; the caller
    validates the result either way, so a gap surfaces as a normal
    validation failure rather than a silent success.
    """
    version = from_version
    while version < GAME_SCHEMA_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            break
        data = step(data)
        version += 1
    if version == GAME_SCHEMA_VERSION:
        # The terminal entry (normally the identity) runs exactly once,
        # whether the save arrived at the current version by migration or
        # was already there. It is the hook a future version can use to
        # normalise whatever the chain produced.
        terminal = MIGRATIONS.get(GAME_SCHEMA_VERSION)
        if terminal is not None:
            data = terminal(data)
    data["schema_version"] = version
    return data

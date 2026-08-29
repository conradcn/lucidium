"""Read/write the per-installation Settings file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ..config import settings_path
from ..domain.settings import REVEAL_SECRETS_CONTEXT_KEY, Settings
from .atomic import atomic_write_text

_log = logging.getLogger(__name__)


def load_settings(path: Path | None = None) -> Settings:
    """Load Settings, returning defaults if the file is missing.

    Tolerates UNKNOWN keys at any depth: when a feature gets
    reverted (e.g. an experimental audio field) but the player's
    on-disk settings.json still carries the now-removed key,
    Pydantic's ``extra="forbid"`` would refuse to deserialize and
    the WebSocket handshake would die before the player can even
    reach Settings to fix it. We strip unknown keys against the
    declared schema before validating, log them at WARNING level
    so a misconfigured save is visible, and rewrite the file so
    the next launch boots clean.

    Also tolerates KNOWN keys carrying values the schema rejects
    (an out-of-range number, an enum member that no longer exists).
    Stripping can't help there, and raising would reproduce exactly
    the lockout above: every WebSocket connection dies inside
    ``Session.__init__`` and the Settings screen — the only in-app
    fix — is unreachable. So the offending fields are dropped (their
    schema defaults apply), logged at WARNING like unknown keys, and
    the file is rewritten. If even the reduced payload won't
    validate, we fall all the way back to a default ``Settings``."""
    target = path or settings_path()
    if not target.exists():
        return Settings()
    raw_text = target.read_text(encoding="utf-8")
    try:
        return Settings.model_validate_json(raw_text)
    except Exception:
        pass
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        _log.warning(
            "settings file at %s is not valid JSON; using defaults",
            target,
        )
        return Settings()
    if not isinstance(raw, dict):
        _log.warning(
            "settings file at %s is not a JSON object; using defaults",
            target,
        )
        return Settings()
    cleaned, dropped = _strip_unknown_keys(Settings, raw)
    if dropped:
        _log.warning(
            "settings file at %s contained unknown keys (likely from a "
            "reverted feature); dropping and rewriting: %s",
            target,
            sorted(dropped),
        )
    settings, invalid = _validate_dropping_invalid(cleaned)
    if invalid:
        _log.warning(
            "settings file at %s contained known keys with values the "
            "schema rejects; dropping (defaults apply) and rewriting: %s",
            target,
            sorted(invalid),
        )
    if dropped or invalid:
        try:
            save_settings(settings, target)
        except Exception:
            _log.warning(
                "could not rewrite cleaned settings file at %s; "
                "next launch will repeat the migration",
                target,
                exc_info=True,
            )
    return settings


def save_settings(settings: Settings, path: Path | None = None) -> None:
    target = path or settings_path()
    # The ONE place that unmasks ``llm.api_key``: the engine has to be
    # able to read its own key back on the next launch. Every other
    # JSON dump of ``Settings`` — the state messages, the ``/settings``
    # patch echo — gets the masked empty string.
    serialized = settings.model_dump_json(
        indent=2,
        context={REVEAL_SECRETS_CONTEXT_KEY: True},
    )
    atomic_write_text(target, serialized + "\n")


def _validate_dropping_invalid(
    cleaned: dict[str, Any],
) -> tuple[Settings, list[str]]:
    """Validate ``cleaned``, dropping whatever pydantic objects to.

    Each round removes the fields named by ``exc.errors()[…]["loc"]``
    and retries, so the surviving keys still take effect and only the
    bad ones fall back to their defaults. Bounded by the number of
    errors we can actually act on: an error whose ``loc`` is empty
    (a model-level validator) or that points at an already-absent
    path can't be narrowed further, so we stop and return defaults
    rather than spin."""
    payload = cleaned
    invalid: list[str] = []
    while True:
        try:
            return Settings.model_validate(payload), invalid
        except ValidationError as exc:
            reduced = dict(payload)
            progressed = False
            for error in exc.errors():
                loc = tuple(error.get("loc") or ())
                if not loc:
                    continue
                if _drop_path(reduced, loc):
                    invalid.append(".".join(str(part) for part in loc))
                    progressed = True
            if not progressed:
                _log.warning(
                    "settings file could not be validated even after "
                    "dropping fields; using defaults",
                    exc_info=True,
                )
                return Settings(), invalid
            payload = reduced


def _drop_path(payload: dict[str, Any], loc: tuple[Any, ...]) -> bool:
    """Remove ``loc`` from a nested dict, copying only along the way.

    Returns False when the path isn't present as written (a
    discriminated-union ``loc`` carrying a tag element, an index into
    a list) — the caller treats that as "no progress"."""
    head, *rest = loc
    if not isinstance(head, str) or head not in payload:
        return False
    if not rest:
        del payload[head]
        return True
    child = payload[head]
    if not isinstance(child, dict):
        return False
    child = dict(child)
    if not _drop_path(child, tuple(rest)):
        return False
    payload[head] = child
    return True


def _strip_unknown_keys(
    model: type[BaseModel],
    raw: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Recursively drop keys that aren't declared on ``model``.

    Walks any nested key whose declared annotation is itself a
    ``BaseModel`` subclass and strips unknown keys at that depth
    too — covers the ``image`` / ``llm`` / ``user_profile`` /
    ``concurrency`` sub-models. Returns the cleaned dict and the
    flat list of dropped key paths (``image.embedded_ambient_audio``,
    etc.) for the caller to log."""
    dropped: list[str] = []
    cleaned = _strip_recursive(model, raw, prefix="", dropped=dropped)
    return cleaned, dropped


def _strip_recursive(
    model: type[BaseModel],
    raw: dict[str, Any],
    *,
    prefix: str,
    dropped: list[str],
) -> dict[str, Any]:
    fields = model.model_fields
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in fields:
            dropped.append(f"{prefix}{key}" if prefix else key)
            continue
        field_info = fields[key]
        annotation = field_info.annotation
        nested_model = _resolve_model_class(annotation)
        if nested_model is not None and isinstance(value, dict):
            out[key] = _strip_recursive(
                nested_model,
                value,
                prefix=f"{prefix}{key}." if prefix else f"{key}.",
                dropped=dropped,
            )
        else:
            out[key] = value
    return out


def _resolve_model_class(annotation: Any) -> type[BaseModel] | None:
    """If ``annotation`` is a BaseModel subclass (possibly wrapped
    in Optional / Union), return the class. Otherwise None.
    Handles ``Optional[X]`` and ``X | None`` shapes."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    args = getattr(annotation, "__args__", None)
    if args:
        for arg in args:
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None

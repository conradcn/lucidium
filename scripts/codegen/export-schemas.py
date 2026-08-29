"""Export Pydantic models to JSON Schema files.

Single source of truth: every Pydantic model declared in
``backend.src.lucidium.api.messages`` and ``backend.src.lucidium.domain.*``
is exported as a ``*.schema.json`` file consumed by:

  * ``shared-schemas/`` — the renderer's TypeScript codegen input
    (``npm --prefix frontend run codegen`` reads from here).
  * ``specs/001-ai-vn-engine/contracts/schemas/`` — documentation mirror.

This script is intentionally minimal at scaffolding time: it imports the
``lucidium`` package, walks the modules listed in ``EXPORT_MODULES`` for
``BaseModel`` subclasses, and writes one schema per class.

Usage:
    python scripts/codegen/export-schemas.py

Environment:
    LUCIDIUM_SCHEMA_OUT_DIR           override the ``shared-schemas/`` target
    LUCIDIUM_CONTRACT_SCHEMA_OUT_DIR  override the contracts mirror target
Both exist for the drift check in ``tests/smoke/test_pipeline.py``, which
exports into a scratch directory and diffs the result against the
committed tree instead of overwriting it.

Exit codes:
    0 success, schemas were (re)written.
    1 the ``lucidium`` package could not be imported.
    2 a model emitted an empty schema.

Constitution: Principle V (DRY) requires that schemas have one source of
truth. Do not introduce a second exporter; extend this one.
"""

from __future__ import annotations

import json
import os
import sys
from importlib import import_module
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_SRC = REPO_ROOT / "backend" / "src"

# Output roots. Both are overridable so a *drift check* can export into a
# scratch directory and diff it against the committed tree — a check that
# writes into the tree it then inspects can never fail by construction.
# Nothing but the drift test sets these; the default is the real layout.
SHARED_SCHEMAS_ENV_VAR = "LUCIDIUM_SCHEMA_OUT_DIR"
CONTRACTS_SCHEMAS_ENV_VAR = "LUCIDIUM_CONTRACT_SCHEMA_OUT_DIR"

SHARED_SCHEMAS = Path(
    os.environ.get(SHARED_SCHEMAS_ENV_VAR) or REPO_ROOT / "shared-schemas"
)
CONTRACTS_SCHEMAS = Path(
    os.environ.get(CONTRACTS_SCHEMAS_ENV_VAR)
    or REPO_ROOT / "specs" / "001-ai-vn-engine" / "contracts" / "schemas"
)

EXPORT_MODULES: tuple[str, ...] = (
    "lucidium.api.messages",
    "lucidium.domain.character",
    "lucidium.domain.world",
    "lucidium.domain.environment",
    "lucidium.domain.dialog",
    "lucidium.domain.game",
    "lucidium.domain.settings",
)

# Emitted alongside the schemas so the renderer can build a *typed*
# ``send()``: a discriminated union of message-type literal -> payload
# interface. The mapping is not derivable from the schema files alone
# (nothing in ``C2SPlayAdvance.schema.json`` says it is the payload of
# ``c2s/play/advance``), and re-deriving it from the class name in TS
# would be a second source of truth. So we dump the backend's own
# dispatch tables — see ``api/messages.py::C2S_PAYLOAD_BY_TYPE``.
MESSAGE_MAP_NAME = "message-map.json"


def _ensure_path() -> None:
    if str(BACKEND_SRC) not in sys.path:
        sys.path.insert(0, str(BACKEND_SRC))


def _require_pydantic() -> type:
    try:
        from pydantic import BaseModel
    except ImportError:
        print(
            "pydantic is not installed in this Python.\n"
            "  Run: .\\start.ps1 -Setup  (or: pip install -e backend[dev])",
            file=sys.stderr,
        )
        sys.exit(1)
    return BaseModel


def _is_model(obj: object, base_model: type) -> bool:
    return isinstance(obj, type) and issubclass(obj, base_model) and obj is not base_model


def _export_module(module_name: str, base_model: type) -> list[Path]:
    written: list[Path] = []
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        # The lucidium package itself is missing — this is fatal, not "skip".
        if exc.name and (exc.name == "lucidium" or exc.name.startswith("lucidium.")):
            print(
                f"failed to import {module_name}: {exc}\n"
                "  The 'lucidium' package is not installed. Run:\n"
                "    .\\start.ps1 -Setup",
                file=sys.stderr,
            )
            sys.exit(1)
        # A *missing optional dependency* of the module is also fatal —
        # silently skipping leaves the codegen step lying about success.
        print(
            f"failed to import {module_name}: {exc}\n"
            "  A dependency of the module is missing. Run:\n"
            "    .\\start.ps1 -Setup",
            file=sys.stderr,
        )
        sys.exit(1)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if not _is_model(attr, base_model):
            continue
        if attr.__module__ != module_name:
            continue  # re-exported, owned elsewhere
        schema = attr.model_json_schema()
        if not schema:
            print(f"empty schema for {module_name}.{attr_name}", file=sys.stderr)
            sys.exit(2)
        for target_dir in (SHARED_SCHEMAS, CONTRACTS_SCHEMAS):
            target_dir.mkdir(parents=True, exist_ok=True)
            out = target_dir / f"{attr_name}.schema.json"
            out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written.append(out)
    return written


def _export_message_map() -> list[Path]:
    """Dump ``MessageType -> payload class name`` for both directions."""
    messages = import_module("lucidium.api.messages")
    mapping = {
        direction: {
            message_type.value: model.__name__
            for message_type, model in getattr(messages, table).items()
        }
        for direction, table in (("c2s", "C2S_PAYLOAD_BY_TYPE"), ("s2c", "S2C_PAYLOAD_BY_TYPE"))
    }
    written: list[Path] = []
    for target_dir in (SHARED_SCHEMAS, CONTRACTS_SCHEMAS):
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / MESSAGE_MAP_NAME
        out.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(out)
    return written


def main() -> int:
    _ensure_path()
    base_model = _require_pydantic()
    total: list[Path] = []
    for module_name in EXPORT_MODULES:
        total.extend(_export_module(module_name, base_model))
    total.extend(_export_message_map())
    if not total:
        print(
            "wrote 0 schema files — no Pydantic models found. This usually means "
            "the lucidium package was imported from the wrong location.",
            file=sys.stderr,
        )
        return 2
    print(f"wrote {len(total)} schema files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

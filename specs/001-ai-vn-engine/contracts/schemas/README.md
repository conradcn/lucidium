# JSON Schema Exports

This directory holds the JSON Schema files exported from the canonical Pydantic v2 models in `backend/src/lucidium/api/messages.py` and `backend/src/lucidium/domain/`.

These files are **generated**. Do not edit them by hand. The build step that produces them lives in `scripts/codegen/export-schemas.py`.

The same schemas are mirrored under the repository-root `shared-schemas/` directory, where they are consumed by the renderer's TypeScript codegen step.

Consumers:
- `frontend/src/shared/generated/` — TypeScript types for the renderer (built via `json-schema-to-typescript`).
- `backend` itself — round-trip validation of incoming WebSocket messages against the exported schemas in tests.

When this plan is implemented, this directory will populate with files such as:

```text
Game.schema.json
WorldState.schema.json
Character.schema.json
DialogNode.schema.json
DialogTree.schema.json
Environment.schema.json
Settings.schema.json
ws/c2s.schema.json
ws/s2c.schema.json
```

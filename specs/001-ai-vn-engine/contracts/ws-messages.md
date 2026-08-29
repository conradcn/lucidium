# WebSocket Message Contract

**Feature**: 001-ai-vn-engine
**Source of truth**: `backend/src/lucidium/api/messages.py` (Pydantic v2). JSON Schemas are exported to `shared-schemas/` and mirrored into `specs/001-ai-vn-engine/contracts/schemas/` by `scripts/codegen/export-schemas.py`; TypeScript types are generated into `frontend/src/shared/generated/` by `npm --prefix frontend run codegen`.

The tables below enumerate **all** `MessageType` members and the payload models they dispatch to (`C2S_PAYLOAD_BY_TYPE` / `S2C_PAYLOAD_BY_TYPE`). When you add a message type, add a row here and re-run both codegen steps.

## Connection

- The Electron main process spawns the Python backend; the backend prints `LUCIDIUM_WS_PORT=<port>` on stdout once listening on `127.0.0.1`. Main passes the port to the renderer via `preload`.
- One WebSocket per app run. JSON-encoded text frames. UTF-8.
- Every message is a JSON object with a top-level `type: string` discriminator and a `payload: object` field.

## Versioning

- Top-level `protocol_version: int` is sent on the first server message after connect. Mismatch triggers an upgrade prompt and a clean shutdown of the renderer.
- Backwards-incompatible message changes bump `protocol_version`. Additive changes do not.

## Direction conventions

- `c2s/*` = client (renderer) to server (backend).
- `s2c/*` = server to client.

## Client-to-server messages

### Session and saves

| `type` | Payload | Purpose |
|---|---|---|
| `c2s/hello` | `{ protocol_version: int }` | Sent immediately on connect. Server replies with `s2c/hello`, then `s2c/state/full` if a session is loaded. |
| `c2s/saves/list` | `{}` | Request the list of saves for the Start Screen / Load Game. |
| `c2s/saves/continue` | `{}` | Load the most recent save. |
| `c2s/saves/load` | `{ save_id: string }` | Load a specific save. `save_id` is a bare directory name, never a path — the constraint at the wire boundary is what stops `saves/<id>` escaping the saves root. |
| `c2s/saves/rename` | `{ save_id: string, new_name: string }` | Rename a save's display name (`meta.json`), not its directory. |
| `c2s/saves/delete` | `{ save_id: string }` | Delete a save directory and everything under it. |
| `c2s/app/exit` | `{}` | Renderer is shutting down; the backend flushes saves and exits. |

### New-game interview

| `type` | Payload | Purpose |
|---|---|---|
| `c2s/new_game/start` | `{ preview_character_slug: string }` | Begin the interview. Returns the initial white-room state; the slug picks the placeholder preview portrait. |
| `c2s/new_game/answer` | `{ step: InterviewStep, answer: string, is_free_text: bool, pronouns: string }` | One answer per step (Setting, Genre, VisualStyle, CharacterDescription, Name). |
| `c2s/new_game/add_side_character` | `{ description: string }` | Add a one-line side-character stub on the confirmation screen. |
| `c2s/new_game/edit_side_character` | `{ character_id: string, description: string }` | Rewrite a side-character stub the player previously added. |
| `c2s/new_game/delete_side_character` | `{ character_id: string }` | Remove a side-character stub the player previously added. |
| `c2s/new_game/edit_review` | `{ field: string, value: string }` | Edit a previously-answered interview field from the Review screen. |
| `c2s/new_game/go_back` | `{}` | Step the interview one screen backwards (also sent from the Review screen). |
| `c2s/new_game/surprise_me` | `{}` | Skip the interview entirely; the backend asks the LLM to invent every answer. |
| `c2s/new_game/confirm` | `{ overrides: dict }` | Apply any edits made on the confirmation screen and start world generation. |

### Play

| `type` | Payload | Purpose |
|---|---|---|
| `c2s/play/advance` | `{ option_id: string \| null }` | `null` advances a "Continue" node; otherwise the chosen option's id. |
| `c2s/play/free_text` | `{ text: string }` | Submit free-text input. Triggers speculative invalidation via the premise hash. |
| `c2s/play/undo` | `{}` | Pop the most recent advance off the session's undo stack and restore the prior `Game` snapshot. |

### Editing

| `type` | Payload | Purpose |
|---|---|---|
| `c2s/edit/world` | `{ field: string, value: any }` | Edit a `WorldState` field. |
| `c2s/edit/character` | `{ character_id: string, field: string, value: any }` | Edit a `Character` field. |
| `c2s/edit/character/dismiss` | `{ character_id: string, reason: string }` | Manually remove a character from the live story. They drop off stage and out of every prompt; undo restores them. |
| `c2s/edit/character/show` | `{ character_id: string }` | Clear `removed` and place the character back on stage. |
| `c2s/edit/character/rerender` | `{ character_id: string }` | Re-trigger the portrait pipeline for a single character. |
| `c2s/edit/history` | `{ node_id: string, new_text: string }` | Edit committed dialog text. |
| `c2s/edit/history/delete` | `{ node_id: string }` | Delete a single committed beat; the chain is re-linked around it. |
| `c2s/edit/history/retcon` | `{ instructions: string }` | Apply a global LLM retcon across the entire committed history. |
| `c2s/edit/environment` | `{ environment_id: string, field: string, value: any }` | Edit an environment label/prompt. |
| `c2s/edit/environment/rerender` | `{ environment_id: string }` | Re-trigger the background pipeline for a single environment. |
| `c2s/edit/environment/apply` | `{ environment_id: string }` | Swap the current scene's backdrop to a different environment. |

### Settings, models, runtime

| `type` | Payload | Purpose |
|---|---|---|
| `c2s/settings/get` | `{}` | Fetch current settings. |
| `c2s/settings/update` | `{ patch: dict }` | Partial update; takes effect on the next generation. |
| `c2s/settings/validate_api_key` | `{ base_url: string, api_key: string }` | Ask the backend to verify that the key actually authenticates against the endpoint. |
| `c2s/embedded/list_models` | `{ models_dir: string }` | List the `.safetensors` / `.ckpt` files in the configured embedded-models directory. |
| `c2s/embedded/recommend_model` | `{ models_dir: string }` | Ask which base model the backend would download for *this* machine's hardware. |
| `c2s/embedded/download_model` | `{ key: string, models_dir: string }` | One-click download of a catalog checkpoint. `models_dir` is confined to the configured root. |
| `c2s/torch_overlay/status` | `{}` | Ask which torch runtime overlay is recommended, installed, and active. |
| `c2s/torch_overlay/install` | `{ flavor: string, activate: bool }` | Download + install (and by default activate) a torch runtime overlay flavor. |
| `c2s/music/inventory` | `{ base_url: string }` | Probe an ACE-Step server for its model inventory and reachability. |
| `c2s/music/regenerate` | `{ prompt: string }` | Re-render the live game's background music; an empty prompt reuses the scene-derived caption. |

## Server-to-client messages

| `type` | Payload | Purpose |
|---|---|---|
| `s2c/hello` | `{ protocol_version: int, has_save: bool }` | First message after `c2s/hello`. `has_save` controls Start Screen Continue-button visibility. |
| `s2c/saves/list` | `{ saves: list[SaveSummaryPayload] }` | Reply to `c2s/saves/list`. |
| `s2c/state/full` | `{ game: Game, settings: Settings }` | Sent on save load and after New Game completes. |
| `s2c/state/patch` | `{ ops: list[Patch] }` | RFC 6902-style JSON-Patch operations against the renderer's mirrored state. |
| `s2c/text/streaming` | `{ node_id: string, delta: string }` | Streamed text chunks for a generating node; the typewriter consumes these. |
| `s2c/text/complete` | `{ node_id: string, node: DialogNode \| null }` | The `delta` stream for `node_id` is finished; the node is `ready`. `node` carries the committed node when it changed. |
| `s2c/image/ready` | `{ kind: ImageKind, target_id: string, image_path: string, image_id: string }` | A new portrait/background asset is on disk. |
| `s2c/music/ready` | `{ audio_path: string, music_id: string, prompt: string }` | A new background-music track is ready for playback. |
| `s2c/error` | `{ code: ErrorCode, message: string, recoverable: bool }` | Surface-level error (backend unreachable, schema validation failure, …). |
| `s2c/notice` | `{ title: string, body: string, kind: NoticeKind }` | One-off modal pop-up — e.g. the storage-side age-correction notice, or an inert-content-filter warning. |
| `s2c/cost` | `{ delta: CostDelta }` | Per-turn cost telemetry; appended to the save. |
| `s2c/embedded/models` | `{ models_dir: string, models: list[string] }` | Reply to `c2s/embedded/list_models`; `models_dir` is the resolved absolute path. |
| `s2c/embedded/recommended_model` | `{ key: string, display_name: string, reason: string, models_dir: string, has_models: bool, approx_bytes: int }` | Reply to `c2s/embedded/recommend_model`. |
| `s2c/embedded/download_progress` | `{ key: string, display_name: string, stage: string, bytes_done: int, bytes_total: int \| null }` | Streamed progress during `c2s/embedded/download_model`. `bytes_total` is null when the server sends no Content-Length. |
| `s2c/settings/api_key_validation` | `{ ok: bool, status: string, message: string }` | Result of a `c2s/settings/validate_api_key` probe. |
| `s2c/torch_overlay/status` | `{ recommended: string, installed: list[string], active: string \| null, runtime_dir: string, activated: bool }` | Snapshot of the torch runtime overlay state. |
| `s2c/torch_overlay/progress` | `{ flavor: string, stage: string, bytes_done: int, bytes_total: int \| null }` | Streamed download/unpack progress during `c2s/torch_overlay/install`. |
| `s2c/music/inventory` | `{ base_url: string, ok: bool, models: list[string], error: string }` | Reply to `c2s/music/inventory`. `ok` is false with `error` set when the probe failed. |

## Required behaviors

- After `c2s/play/advance` or `c2s/play/free_text`, the server MUST send (in order):
  1. `s2c/state/patch` updating `current_node_id` and `on_stage`.
  2. Either `s2c/text/streaming` deltas followed by `s2c/text/complete` (when the node text is generated fresh), or no text events at all if the node was already `ready`.
  3. `s2c/image/ready` events as soon as any pending background or portrait completes.
- Any edit message (`c2s/edit/*`) MUST be acknowledged via a `s2c/state/patch` reflecting the edit, even if the patch is a no-op (so the renderer can clear pending UI state).
- A schema validation failure on any client message MUST elicit `s2c/error` with `recoverable: true` and no state change.
- Long-running installs and downloads (`c2s/torch_overlay/install`, `c2s/embedded/download_model`) MUST stream progress messages and MUST NOT block the socket. While a torch-overlay install is in flight, image generation refuses with a `torch_installing` error rather than silently falling back to the CPU overlay.

## Reliability semantics

- The server tolerates the client closing and reopening the socket. On reconnect the client sends `c2s/hello` again and the server resends `s2c/hello` plus, if a session is loaded, `s2c/state/full`.
- Outbound provider failures are transformed into structured `s2c/error` messages and never crash the WS connection.
- The server flushes the current save before responding to `c2s/app/exit`.

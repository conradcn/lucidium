# Phase 1 Data Model: AI-Driven Visual Novel Engine

**Feature**: 001-ai-vn-engine
**Date**: 2026-05-01

This document defines the persistent and in-memory entities. Field types are written in language-neutral form; the canonical Pydantic v2 schemas live in `backend/src/lucidium/domain/` and `backend/src/lucidium/api/messages.py` and are exported as JSON Schema for the renderer.

## Conventions

- IDs are opaque ULID-format strings unless noted.
- Timestamps are ISO-8601 UTC strings.
- "Optional" means the field may be `null`; "may be absent" means the field is omitted when not applicable.
- Hashes are SHA-256 hex unless noted.
- All entities are versioned via a top-level `schema_version: int` on the `Game` aggregate; bumping it requires a migration entry in `persistence/save_store.py`.

---

## Game (aggregate root)

The unit of "one playthrough." Persisted as `<save-id>/game.json` plus `<save-id>/meta.json` and `<save-id>/images/`.

| Field | Type | Notes |
|---|---|---|
| `id` | string (ULID) | Stable for the life of the save. |
| `schema_version` | int | Migration anchor. v1 starts at `1`. |
| `created_at` | timestamp | Set at New Game completion. |
| `world` | `WorldState` | Embedded. |
| `characters` | dict[character_id, `Character`] | Roster, including the player character. |
| `dialog_tree` | `DialogTree` | All committed nodes plus speculative children. |
| `environments` | dict[environment_id, `Environment`] | All generated backgrounds. |
| `current_node_id` | string | Pointer to the node the player is presently at. |
| `on_stage` | list[character_id] | Subset of `characters` that are currently on screen. |
| `cost_telemetry` | `CostTelemetry` | Cumulative tokens, latency, dollar estimate. |

Invariants:
- `current_node_id` is a key of `dialog_tree.nodes`.
- Every entry of `on_stage` is a key of `characters`.
- The player character is exactly one entry of `characters` flagged `is_player=true`.

---

## WorldState

| Field | Type | Notes |
|---|---|---|
| `game_name` | string | LLM-generated at New Game; user-editable. |
| `setting` | string | From the interview. |
| `genre` | string | From the interview. |
| `visual_style` | string | From the interview; feeds image prompts. |
| `overall_plot_direction` | string | Updated on each world-state refresh. |
| `player_intent` | `PlayerIntentForecast` | Refreshed by the world-state refresh task. |
| `active_plot_threads` | list[`PlotThread`] | Current focus of the story. |
| `dropped_plot_threads` | list[`PlotThread`] | May be reintroduced during a lull. |
| `summarizer_assessment` | string | Latest summarizer steering signal. |
| `prompt_history_clamp_chars` | int | Configurable cap on conversation-history length in prompts. Default `12000`. |

### PlayerIntentForecast

| Field | Type | Notes |
|---|---|---|
| `pace_preference` | enum {`faster`, `same`, `slower`} | Forecasted, weighted toward player text. |
| `tone_preference` | enum {`focus`, `chaos`, `unspecified`} | Forecasted. |
| `direction_signal` | enum {`stay_focused`, `reintroduce_thread`, `none`} | Steering injected into the next text-gen prompt. |
| `weighted_evidence` | list[`EvidenceQuote`] | Up to N most recent player-typed quotes that informed the forecast. |

### PlotThread

| Field | Type | Notes |
|---|---|---|
| `id` | string (ULID) | |
| `title` | string | Short label. |
| `summary` | string | A few sentences. |
| `last_referenced_node_id` | string \| null | For staleness detection. |

---

## Character

| Field | Type | Notes |
|---|---|---|
| `id` | string (ULID) | |
| `is_player` | bool | True for exactly one character per save. |
| `name` | string | |
| `description` | string | Traits and manner of speech only. Core text used for dialog generation. |
| `gender` | string | Free-form (player input or LLM). |
| `age` | int | Exact integer; transformed at image-prompt time via age-band rules. |
| `ethnicity` | string | |
| `skin` | string | |
| `hair_color` | string | |
| `hairstyle` | string | |
| `eye_color` | string | |
| `build` | string | |
| `bust` | string | |
| `outfit` | string | Most-frequently-changing attribute (along with pose/expression). |
| `pose` | string | |
| `expression` | string | |
| `facts` | list[`Fact`] | Maintained by the summarizer from history. |
| `images` | list[`CharacterImage`] | Most recent first. |
| `seed` | int (uint64) | Generated once at character creation; immutable. |
| `created_at` | timestamp | |

### Fact

| Field | Type | Notes |
|---|---|---|
| `id` | string (ULID) | |
| `text` | string | E.g., "lost a sister in the siege." |
| `confidence` | enum {`canon`, `inferred`} | Edits in the Characters tab become `canon`. |
| `source_node_id` | string \| null | Node that introduced the fact. |

### CharacterImage

| Field | Type | Notes |
|---|---|---|
| `id` | string (ULID) | |
| `path` | string | Relative path under `<save-id>/images/`. |
| `prompt_hash` | string (SHA-256) | Hash of the substituted prompt template + seed. |
| `attributes_snapshot` | dict[str, str] | Attribute values used to build the prompt. |
| `created_at` | timestamp | |

Validation rules:
- `seed` must be present after character creation; missing seed is a rejected output requiring repair.
- After character repair, all required string attributes must be non-empty; any missing field triggers an LLM repair task.

---

## DialogTree

| Field | Type | Notes |
|---|---|---|
| `nodes` | dict[node_id, `DialogNode`] | All committed and speculative nodes. |
| `root_id` | string | The first committed node after New Game completes. |
| `committed_path` | list[string] | Ordered node IDs the player has actually walked. The last entry is `current_node_id`. |

### DialogNode

| Field | Type | Notes |
|---|---|---|
| `id` | string (ULID) | |
| `parent_id` | string \| null | `null` only for the root. |
| `chosen_option_id` | string \| null | Which option (or free-text input) on the parent led here. `null` on root. |
| `speaker_id` | string \| null | character_id; `null` for narrator. |
| `text` | string \| null | `null` while still pending generation. |
| `options` | list[`DialogOption`] | Empty list implies a single "Continue." |
| `entering_character_ids` | list[string] | Characters that enter at this node. |
| `leaving_character_ids` | list[string] | Characters that leave at this node. |
| `new_character_descriptors` | dict[character_id, `Character`] | Full descriptor for any newly introduced character. |
| `location_id` | string \| null | environment_id; `null` if location is unchanged from the previous node. |
| `location_prompt` | string \| null | `null` if unchanged or reused (i.e., the existing environment's prompt is reused). |
| `character_changes` | list[`CharacterChange`] | Per-character attribute deltas applied at this node. |
| `state` | enum {`speculative`, `pending_text`, `ready`, `committed`, `invalidated`} | See state machine below. |
| `premise_hash` | string (SHA-256) | Hash of (parent_id, chosen_option_id, world snapshot vector). Drives free-text invalidation. |
| `generation_metadata` | `GenerationMetadata` | Tokens, latency, model id, prompt hash, seed parameters used. |

### DialogOption

| Field | Type | Notes |
|---|---|---|
| `id` | string (ULID) | |
| `text` | string | What the player sees on the button. |

### CharacterChange

| Field | Type | Notes |
|---|---|---|
| `character_id` | string | Must be on stage (or be entering this node). |
| `field` | enum mirroring `Character` mutable fields | `pose`, `expression`, `outfit`, `description`, ... |
| `new_value` | string | |

State machine for `DialogNode.state`:

```
                 advance() / commit
   speculative ---------------------+
        |                           |
  text gen succeeds                 v
        |                       committed
        v
   pending_text -- text gen succeeds --> ready
                                                     ^
   any state -- player free-text or edit obsoletes --|
   (-> invalidated)                                  |
                              (committed nodes are never invalidated)
```

Rules:
- A node is shown to the player only in `ready` or `committed` state. `speculative` and `pending_text` are scheduler-only.
- `invalidated` nodes are tombstoned, not deleted, so the renderer can resolve dangling references gracefully and so debugging can replay a turn.

---

## Environment

| Field | Type | Notes |
|---|---|---|
| `id` | string (ULID) | |
| `location_label` | string | Short human label (e.g., "the harbor at dusk"). |
| `prompt` | string | Image prompt the background was generated from. |
| `image_path` | string \| null | `null` while pending. |
| `prompt_hash` | string (SHA-256) | For deduplication and cache reuse. |
| `created_at` | timestamp | |

Validation:
- An `Environment` is reusable across nodes that emit the same `prompt_hash`. The dialog tree references environments by `id`, not by re-emitting the prompt.

---

## Generation Task

In-memory only (never persisted). Lives on the scheduler queue.

| Field | Type | Notes |
|---|---|---|
| `id` | string (ULID) | |
| `kind` | enum {`text_gen`, `char_repair`, `world_refresh`, `image_bg`, `image_portrait`, `image_portrait_regen`, `interview_options`, `interview_image`, `world_init`} | |
| `priority_class` | int | Smaller is higher priority. |
| `distance_from_current` | int | Secondary sort key (`0` for current node, increasing outward). |
| `target_id` | string | node_id, character_id, or environment_id depending on kind. |
| `premise_hash` | string \| null | For text-gen tasks; used by obsolescence. |
| `state` | enum {`queued`, `in_flight`, `done`, `cancelled`, `obsoleted`} | |
| `created_at` | timestamp | |

Rules (FR-013, FR-016, FR-017):
- A `queued` task whose target's `premise_hash` no longer matches the current state is `cancelled` before it can dispatch.
- An `in_flight` task is allowed to finish; its result is applied only if it is still more current than what it would replace, otherwise it is `obsoleted` (result discarded).

---

## Settings (per installation)

Stored in `<app-data>/settings.json`. Snapshotted into each `meta.json` at save creation; live edits write through to both.

| Field | Type | Notes |
|---|---|---|
| `llm` | `LlmSettings` | OpenAI-compatible endpoint config. |
| `image` | `ImageSettings` | ComfyUI endpoint config. |
| `typewriter_speed_chars_per_sec` | int | UI presentation. |
| `prompt_history_clamp_chars` | int | Default; per-save override allowed via `WorldState.prompt_history_clamp_chars`. |
| `concurrency` | `ConcurrencyLimits` | Per-provider in-flight caps. |

### LlmSettings

| Field | Type | Notes |
|---|---|---|
| `base_url` | string | Defaults to OpenRouter base URL. |
| `model` | string | Defaults to a Qwen-class model id. |
| `api_key` | `SecretStr` | Write-only. Held as a `SecretStr` so it can't reach a repr or log line; masked to `""` in every JSON dump except `settings.json` itself, so it never crosses the WebSocket or lands in a save. An empty `api_key` in a `c2s/settings/update` patch means "leave unchanged". |
| `temperature` | float | Default 0.8. |
| `max_tokens` | int | Default 1024. |

### ImageSettings

| Field | Type | Notes |
|---|---|---|
| `base_url` | string | Defaults to `http://127.0.0.1:8000`. |
| `portrait_workflow` | string | Path or name of the ComfyUI workflow JSON template. |
| `background_workflow` | string | Same. |

### ConcurrencyLimits

| Field | Type | Notes |
|---|---|---|
| `llm_max_in_flight` | int | Default 4. |
| `image_max_in_flight` | int | Default 2. |

---

## Save metadata

Stored in `<save-id>/meta.json` so the Start Screen and Load Game UI can list saves without parsing `game.json`.

| Field | Type | Notes |
|---|---|---|
| `id` | string (ULID) | Same as `Game.id`. |
| `name` | string | Editable. |
| `last_played_at` | timestamp | Updated on every commit. |
| `created_at` | timestamp | |
| `schema_version` | int | Mirrored from the game for migration scans. |
| `summary` | string | One-line summary of the most recent committed node, for the load-list. |

---

## Cross-entity invariants

1. The active environment shown in the renderer is `environments[current_node.location_id]`, falling back to the most recent committed ancestor with a non-null `location_id` if the current node hasn't set one (FR-035).
2. Every `Character` referenced by a dialog node, an option, a stage list, or a character-change record exists in `Game.characters`.
3. Every `Environment` referenced by a node exists in `Game.environments`.
4. `dialog_tree.committed_path[-1] == Game.current_node_id`.
5. Edits made via the Story panel update the source-of-truth field directly and bump the `premise_hash` of any descendant speculative nodes that depend on the changed field, marking them `invalidated` (FR-039 + FR-016 semantics).

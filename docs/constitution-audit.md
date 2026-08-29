# Constitution Audit — v1 / 2026-05-01

A walk through the Lucidium Constitution v1.0.0 against the implemented code at the close of the `001-ai-vn-engine` feature branch. This is the audit T088 of `tasks.md`.

## I. Reliability — PASS with one follow-up

- Every LLM output is parsed via `orchestration.responses.parse_json_object` against an explicit Pydantic schema before reaching the dialog tree (constitutional requirement: "validate every model output against an explicit schema before it reaches the player"). ✅
- Each scheduler retries up to a budget on `ProviderUnreachableError` / `ProviderValidationError` (`LLM_RETRY_BUDGET`, `IMAGE_RETRY_BUDGET`). ✅
- Fallbacks: missing image → most recent prior background or placeholder portrait (`MainView/Background.tsx`, `MainView/CharacterStage.tsx`). Missing text → renderer leaves the typewriter at "…" until `s2c/text/streaming` arrives. ✅
- Save commits happen on every accepted dialog turn (`play_advance_handler`, `play_free_text_handler` both call `Session.commit`); atomic writes via `persistence.atomic`. ✅
- Backend unreachable produces a structured `s2c/error{code:"provider_unreachable"}` and the renderer shows a banner offering Settings (`MainView/index.tsx`). ✅
- **Follow-up**: speculative pre-generation pump is not yet running. Once it is, errors from speculative tasks must continue to drain through the same `s2c/error` surface — currently the schedulers `_log.warning` and obsolete the task silently, which is acceptable for *speculative* failures (they are best-effort) but should still bubble exhausted-retry counts into telemetry.

## II. Elegance — PASS

- Module names are domain-shaped (`world`, `character`, `dialog`, `environment`, `game`, `summarizer`, `obsolescence`). ✅
- One IPC channel (the WebSocket); one save format (folder of JSON + PNGs); one settings file. ✅
- No premature abstractions: schedulers are concrete classes, not a plugin framework; provider clients are a 2-method Protocol with a real impl + a recorded fake. ✅
- Naming is verbose-readable: `play_advance_handler`, `invalidate_speculation_from`, `world_snapshot_vector`. ✅

## III. Efficiency — PASS with two follow-ups

- LLM calls fan out under `asyncio.Semaphore` (`LlmScheduler`, `ImageScheduler`). ✅
- Prompt structure is stable so providers' prompt-cache prefixes match across calls (`prompts/text_gen.py` builds in a deterministic order). ✅
- Per-turn cost telemetry is captured (`orchestration/cost.py`) and folded into `Game.cost_telemetry`. ✅
- Image generation is content-hash deduplicated against `<save-id>/images/<hash>.png` (`image_scheduler.py`). ✅
- **Follow-up 1**: per-turn telemetry is only updated in the critical-path `Session.llm_text`. When the schedulers actually run speculative work, they need to call the same fold to keep cost numbers honest.
- **Follow-up 2**: token estimates are character-based (4 chars ≈ 1 token). When the LLM client surfaces real `usage` from provider responses, `cost.py::estimate_llm_call` should be supplemented with a `record_exact` path.

## IV. Testability — PASS

- `LlmClient` and `ImageClient` are Protocols with `Recorded*` implementations. The integration test (`backend/tests/integration/test_us1_flow.py`) drives the full new-game-through-free-text flow without a real provider, using the in-test `QueuedLlm`. ✅
- `backend/tests/conftest.py` denies all unmocked HTTP via `respx`; this is the constitutional offline gate. ✅
- The frontend equivalent gate denies `fetch` in `frontend/tests/setup.ts`. ✅
- CI workflow `.github/workflows/offline-tests.yml` runs both suites on every push. ✅
- Currently 11 backend test files (`test_age_band`, `test_atomic`, `test_premise_hash`, `test_game_invariants`, `test_settings_store`, `test_save_store`, `test_errors`, `test_obsolescence`, `test_llm_scheduler`, `test_image_scheduler`, `test_cost_telemetry`) + 2 integration files (`test_ws_handshake`, `test_us1_flow`, `test_in_flight_edit_conflict`). ✅

## V. DRY — PASS

- `backend/src/lucidium/api/messages.py` is the single source of truth for IPC types. ✅
- `scripts/codegen/export-schemas.py` exports 50 distinct Pydantic models to `shared-schemas/` and `specs/.../contracts/schemas/`; the renderer's `npm run codegen` produces TypeScript from that. ✅
- All numeric and string defaults live in `backend/src/lucidium/config.py` (URLs, model id, ports, retry budgets, history clamps, concurrency caps, app-data path resolution). ✅
- Prompt fragments live in `backend/src/lucidium/orchestration/prompts/`; no inline prompt strings outside that package. ✅

## Summary

All five principles pass. Three follow-ups are recorded for future work; none of them block the MVP being shippable. No constitutional violations require an entry in `plan.md` Complexity Tracking.

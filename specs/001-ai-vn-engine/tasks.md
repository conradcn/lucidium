---

description: "Task list for the AI-Driven Visual Novel Engine"
---

# Tasks: AI-Driven Visual Novel Engine

**Input**: Design documents from `specs/001-ai-vn-engine/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Test tasks are included where the constitution mandates them (Principle IV — every non-trivial behavior must be exercisable without a live model call). Pure UI/cosmetic tasks do not have paired tests.

**Organization**: Tasks are grouped by user story (US1–US5 from spec.md). Setup, Foundational, and Polish phases are story-agnostic.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no incomplete dependencies).
- **[Story]**: User story label (US1, US2, US3, US4, US5).

## Path Conventions

Two top-level packages per `plan.md`: `backend/` (Python sidecar) and `frontend/` (Electron + React). Cross-cutting code lives in `scripts/codegen/` and `shared-schemas/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Repository scaffolding and tooling.

- [X] T001 Create the directory tree from `plan.md` (`backend/src/lucidium/{api,domain,orchestration/prompts,providers,persistence}`, `backend/tests/{unit,integration,fixtures}`, `frontend/{electron,src/{ws,shared/generated,state,screens/NewGameInterview,screens/MainView/StoryPanel,settings},tests/unit}`, `scripts/codegen/`, `shared-schemas/`).
- [X] T002 [P] Author `backend/pyproject.toml` declaring Python 3.11+ and pinning `pydantic>=2`, `websockets`, `httpx`, `pytest`, `pytest-asyncio`, `respx`, `ruff` (dev), `mypy` (dev), package layout `src/lucidium`.
- [X] T003 [P] Author `frontend/package.json` declaring `electron`, `react`, `react-dom`, `vite`, `@vitejs/plugin-react`, `zustand`, `typescript`, `vitest`, `@testing-library/react`, `eslint`, `prettier`, plus `dev`, `build`, `test`, `lint`, `format` scripts.
- [X] T004 [P] Add `backend/ruff.toml` and `backend/pyrightconfig.json` (or `mypy.ini`) configured for `src/` layout and the `src/lucidium` package.
- [X] T005 [P] Add `frontend/tsconfig.json`, `frontend/.eslintrc.cjs`, `frontend/.prettierrc`, and `frontend/vite.config.ts` (Vite + React plugin, output to `frontend/dist`).
- [X] T006 [P] Add a top-level `.gitignore` that excludes `backend/.venv/`, `frontend/node_modules/`, `frontend/dist/`, `shared-schemas/*.local.json`, and `%APPDATA%`-style local app-data files; commit `shared-schemas/.gitkeep`.
- [X] T007 [P] Add `scripts/codegen/export-schemas.py` as a stub that imports the eventual `lucidium.api.messages` and writes per-model `*.schema.json` files into both `shared-schemas/` and `specs/001-ai-vn-engine/contracts/schemas/`. Document its CLI in the file's module docstring.
- [X] T008 [P] Add an `npm` script `frontend/package.json -> "codegen"` that runs `json-schema-to-typescript` over `shared-schemas/*.schema.json` into `frontend/src/shared/generated/` and a top-level Makefile (or `tasks.ps1`) target that runs `python scripts/codegen/export-schemas.py && npm --prefix frontend run codegen`.
- [X] T009 Add a CI workflow at `.github/workflows/offline-tests.yml` that installs backend + frontend, runs `pytest backend/tests`, runs `npm --prefix frontend test`, and fails if either suite reaches the network (relies on `respx` and Vitest's fetch mock). Reference `.specify/memory/constitution.md` Principle IV in the workflow's top-level comment.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Single source of truth for IPC types, the WebSocket transport, the provider abstractions, and the persistence and config primitives that every user story builds on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T010 Author `backend/src/lucidium/config.py` declaring all constants in one place: default LLM base URL (OpenRouter), default LLM model id (Qwen-class placeholder), default ComfyUI base URL (`http://127.0.0.1:8000`), default `prompt_history_clamp_chars=12000`, default concurrency limits (`llm_max_in_flight=4`, `image_max_in_flight=2`), retry budgets, app-data directory resolution per OS, and the schema-version constant `GAME_SCHEMA_VERSION = 1`.
- [X] T011 [P] Author `backend/src/lucidium/persistence/atomic.py` with `atomic_write_text(path, content)` and `atomic_write_bytes(path, data)` using temp file + `os.replace`; cover with `backend/tests/unit/test_atomic.py`.
- [X] T012 Author `backend/src/lucidium/api/messages.py` defining the Pydantic v2 envelope (`type`, `payload`, `protocol_version`) and every `c2s/*` and `s2c/*` message model exactly as enumerated in `specs/001-ai-vn-engine/contracts/ws-messages.md`. This file is the single source of truth referenced by the constitution; no other file may redeclare these shapes.
- [X] T013 [P] Author `backend/src/lucidium/domain/character.py` with `Character`, `Fact`, `CharacterImage`, and `age_band(age:int)->str` rules ("twenty", "thirty", ...) per `data-model.md`; cover age-band rules with `backend/tests/unit/test_age_band.py`.
- [X] T014 [P] Author `backend/src/lucidium/domain/world.py` with `WorldState`, `PlayerIntentForecast`, `PlotThread`, `EvidenceQuote` per `data-model.md`.
- [X] T015 [P] Author `backend/src/lucidium/domain/environment.py` with `Environment` per `data-model.md`.
- [X] T016 [P] Author `backend/src/lucidium/domain/dialog.py` with `DialogTree`, `DialogNode`, `DialogOption`, `CharacterChange`, `GenerationMetadata`, the `DialogNode.state` enum, and the `premise_hash` helper (SHA-256 over (`parent_id`, `chosen_option_id`, world snapshot vector)). Cover the helper with `backend/tests/unit/test_premise_hash.py`.
- [X] T017 [US3] Author `backend/src/lucidium/domain/game.py` with the `Game` aggregate per `data-model.md`, including `schema_version`, invariants enforced via Pydantic validators (current_node_id ∈ dialog_tree.nodes; on_stage ⊆ characters; exactly one `is_player`); cover invariants with `backend/tests/unit/test_game_invariants.py`. (Foundational — depends on T013–T016.)
- [X] T018 Wire `scripts/codegen/export-schemas.py` to import every Pydantic model from T012–T017 and emit `*.schema.json` files into `shared-schemas/` and `specs/001-ai-vn-engine/contracts/schemas/`; commit the produced JSON Schemas; run `npm --prefix frontend run codegen` and commit the generated `frontend/src/shared/generated/*.ts` so the renderer build does not need Python.
- [X] T019 [P] Author `backend/src/lucidium/providers/llm_client.py` defining the `LlmClient` Protocol (`async def complete(prompt, *, model, temperature, max_tokens, stream) -> AsyncIterator[str]`), the `OpenAiCompatibleLlmClient` implementation against `httpx.AsyncClient`, and a `RecordedLlmClient` that reads from `backend/tests/fixtures/llm/` keyed by prompt hash + variant. Validate every model response shape via Pydantic before returning.
- [X] T020 [P] Author `backend/src/lucidium/providers/image_client.py` defining the `ImageClient` Protocol (`async def generate(workflow:str, params:dict, *, seed:int) -> bytes`), a ComfyUI implementation that templates the workflow JSON and polls for completion, and a `RecordedImageClient` that returns a fixture PNG keyed by prompt+seed.
- [X] T021 [P] Author `backend/src/lucidium/persistence/settings_store.py` with `load_settings()` and `save_settings(settings)` using the atomic helpers from T011 and the path resolution from T010; uses the `Settings` Pydantic model from T012; cover with `backend/tests/unit/test_settings_store.py`.
- [X] T022 [P] Author `backend/src/lucidium/persistence/save_store.py` with `list_saves()`, `load_save(save_id)`, `commit_save(game)`, `rename_save(save_id, name)`, `delete_save(save_id)`, `most_recent_save_id()`, and the on-disk layout from `data-model.md` (`game.json` + `meta.json` + `images/`). Schema-version migration scaffold included; cover round-trip and rename/delete with `backend/tests/unit/test_save_store.py`.
- [X] T023 Author `backend/src/lucidium/api/ws_server.py` exposing a single WebSocket endpoint on `127.0.0.1:<auto-port>`, printing `LUCIDIUM_WS_PORT=<port>` to stdout once listening, validating every inbound message against the Pydantic models from T012, and dispatching to handlers (handlers are stubs at this point). On any validation failure, emit `s2c/error` with `recoverable: true`. Cover hello/error/echo flow with `backend/tests/integration/test_ws_handshake.py`.
- [X] T024 [P] Author `backend/src/lucidium/app.py` as the process entrypoint that reads settings, constructs the `LlmClient` and `ImageClient` from settings (or the recorded fakes when `LUCIDIUM_OFFLINE=1` is set), and starts the WS server from T023. Wire structured logging to stderr; never log API keys.
- [X] T025 [P] Author `frontend/electron/main.ts` to spawn `python -m lucidium.app` (using `backend/.venv` on dev), capture stdout for `LUCIDIUM_WS_PORT=`, create the `BrowserWindow`, pass the port to the renderer via `preload.ts`, and forward window-close + app-quit to a clean `c2s/app/exit` followed by child-process termination.
- [X] T026 [P] Author `frontend/electron/preload.ts` to expose `window.lucidium.wsPort` and `window.lucidium.appDataPath` via `contextBridge`.
- [X] T027 [P] Author `frontend/src/ws/client.ts`: a typed WebSocket client using the generated message types from T018, with reconnect with jittered backoff, an inbound dispatch table, and a `send(type, payload)` helper. Cover with `frontend/tests/unit/ws_client.test.ts` (using a mock WS server).
- [X] T028 [P] Author `frontend/src/state/store.ts` (Zustand) holding `Game`, `Settings`, connection status, and reducers for `s2c/state/full` and `s2c/state/patch` (RFC 6902 ops). Cover patch reducer with `frontend/tests/unit/store.test.ts`.
- [X] T029 [P] Configure `backend/tests/conftest.py` to install a `respx` autouse fixture that fails any unmocked outbound HTTP, and to inject the recorded LLM/image clients by default. This is the offline gate the constitution requires.
- [X] T030 [P] Add an `s2c/error` taxonomy to `backend/src/lucidium/api/messages.py` (codes: `schema_error`, `provider_unreachable`, `provider_validation`, `internal`) and a centralized `errors.py` that converts exceptions into typed messages. Cover provider-unreachable mapping with `backend/tests/unit/test_errors.py`.

**Checkpoint**: Foundational complete. The renderer can launch Electron, spawn the backend, connect over WebSocket, exchange `c2s/hello`/`s2c/hello`, and validate every message — but no game logic runs yet. User stories can now begin.

---

## Phase 3: User Story 1 — Begin and play a brand-new AI-generated story (Priority: P1) 🎯 MVP

**Goal**: A player can press New Game, walk through the interview, reach the main view, and play through option-driven and free-text dialog with characters and backgrounds appearing on cue.

**Independent Test**: From a fresh install with the recorded providers (or live providers configured), press New Game, complete the interview using suggested options, and play through 10 consecutive nodes — at least one option-driven branch and one free-text moment — without the screen sitting on a blank background or missing-character state at the moment of presentation.

### Prompts and image-template assets

- [X] T031 [P] [US1] Author `backend/src/lucidium/orchestration/prompts/text_gen.py` defining the dialog-node prompt builder: takes a clamped conversation history, full attribute set of every on-stage character, base description (only) of off-stage characters, the summarizer assessment, and the steering signal; returns a system+user prompt pair. Constants live here, not inline.
- [X] T032 [P] [US1] Author `backend/src/lucidium/orchestration/prompts/world_refresh.py` defining the world-state-refresh prompt builder (player-intent forecast, plot direction, plot-thread maintenance, redundant-fact pruning), with explicit higher weight on player-typed text per FR-008.
- [X] T033 [P] [US1] Author `backend/src/lucidium/orchestration/prompts/character_repair.py` defining the malformed-descriptor repair prompt that fills missing schema fields from the current information.
- [X] T034 [P] [US1] Author `backend/src/lucidium/orchestration/prompts/interview.py` defining the LLM prompts for: 30 setting defaults, genre defaults, 30 visual-style defaults, character-description defaults, name defaults, side-character expansion, and `world_init` (Game Name, Overall Plot Direction, initial Active Plot Threads, opening dialog node).
- [X] T035 [P] [US1] Author `backend/src/lucidium/orchestration/prompts/image_prompts.py` defining the canonical portrait template and the canonical background template, both parameterized by attribute substitutions and a seed; `age_band` from T013 is applied here.
- [X] T036 [P] [US1] Author `backend/src/lucidium/orchestration/summarizer.py` exposing `update_facts(world, dialog_tree, characters) -> SummaryResult` that produces an updated character `facts` list and a summarizer assessment string; covers redundant-fact pruning and plot-thread maintenance.
- [X] T037 [P] [US1] Add ComfyUI workflow JSON templates `backend/src/lucidium/orchestration/prompts/comfy/portrait.workflow.json` and `backend/src/lucidium/orchestration/prompts/comfy/background.workflow.json` with KSampler `seed` field templated.

### Schedulers and obsolescence

- [X] T038 [US1] Author `backend/src/lucidium/orchestration/obsolescence.py` implementing the premise-hash matching rule from `data-model.md` plus `is_obsolete(task, game)` for each `kind`. Cover with `backend/tests/unit/test_obsolescence.py` (text-task invalidation by free-text input; image-task retention when character/environment unchanged; in-flight task allowed to finish per FR-013).
- [X] T039 [US1] Author `backend/src/lucidium/orchestration/llm_scheduler.py` with the priority classes `TEXT_NEXT > CHAR_REPAIR > WORLD_REFRESH`, distance-from-current as secondary key, an `asyncio.Semaphore`-bounded worker pool, and a `rescore()` pass after every state mutation. Tasks dispatch through `LlmClient`; results validated by Pydantic before commit. Cover ordering and rescore with `backend/tests/integration/test_llm_scheduler.py` using the recorded LLM client.
- [X] T040 [US1] Author `backend/src/lucidium/orchestration/image_scheduler.py` with priority classes `BG_NEAR > PORTRAIT_NEAR > CHAR_PROMPT_REGEN_SPEC`, distance-from-current secondary, semaphore-bounded workers, content-hash dedup against `<save-id>/images/<hash>.png`, and the `is_obsolete` check on dispatch. Cover with `backend/tests/integration/test_image_scheduler.py` using the recorded image client.
- [X] T041 [US1] Implement free-text invalidation in `backend/src/lucidium/orchestration/llm_scheduler.py` and `image_scheduler.py`: on `c2s/play/free_text`, walk the descendant tree from the current node and mark tasks `cancelled` or `obsoleted` per FR-016/FR-017. Cover with `backend/tests/integration/test_free_text_invalidation.py`.

### WebSocket handlers (US1 surface)

- [X] T042 [US1] Author `backend/src/lucidium/api/handlers.py::handle_new_game_start` to construct the white-room placeholder state, kick off the LLM call for Setting defaults, and emit `s2c/state/full` followed by `s2c/state/patch` as new defaults arrive.
- [X] T043 [US1] Extend `handlers.py` with `handle_new_game_answer` that records each step's answer and, when the player reaches Visual Style, dispatches the background LLM call for the next two questions' option lists plus the speculative character-replacement and background-replacement image jobs (FR-027).
- [X] T044 [US1] Extend `handlers.py` with `handle_new_game_confirm` that applies confirmation overrides, runs `world_init` (Game Name, Overall Plot Direction, initial Active Plot Threads, opening Dialog Node) via the LLM scheduler, generates the opening background via the image scheduler, commits the save, and emits `s2c/state/full` for the main view.
- [X] T045 [US1] Extend `handlers.py` with `handle_play_advance` that walks `current_node_id` to the chosen option's child (committing it), emits `s2c/state/patch`, dispatches text-streaming via `s2c/text/streaming` deltas if the next node was speculative-but-not-yet-streamed, and emits `s2c/text/complete` when done.
- [X] T046 [US1] Extend `handlers.py` with `handle_play_free_text` that creates a new child node from the player's text (high-priority `TEXT_NEXT` task), invokes free-text invalidation from T041, and emits the same advance sequence as T045.
- [X] T047 [US1] Wire `s2c/image/ready` emission from the image scheduler into the WS server so the renderer learns about new portraits and backgrounds without polling.

### Renderer (US1 surface)

- [X] T048 [P] [US1] Author `frontend/src/screens/StartScreen.tsx` with the New Game / Load Game / Options / Exit buttons. The Continue button is hidden in this story (US2 wires it up). Pressing New Game sends `c2s/new_game/start` and routes to the interview.
- [X] T049 [P] [US1] Author `frontend/src/screens/NewGameInterview/index.tsx` and `SettingStep.tsx`, `GenreStep.tsx`, `VisualStyleStep.tsx`, `CharacterStep.tsx`, `NameStep.tsx`, `ConfirmStep.tsx`. Each step renders the LLM-supplied options (5 randomized of 30 for Setting + Visual Style; full list for the rest), a free-text input, and a Continue button that dispatches `c2s/new_game/answer`. The interview shows the placeholder character + background until the WebSocket reports replacements.
- [X] T050 [P] [US1] Author `frontend/src/screens/MainView/Background.tsx` that picks the active background using FR-035 fallback (current node's background, or most recent prior committed node's background).
- [X] T051 [P] [US1] Author `frontend/src/screens/MainView/CharacterStage.tsx` rendering the on-stage roster, each with a name tag and a portrait (or placeholder portrait when the image is still pending).
- [X] T052 [P] [US1] Author `frontend/src/screens/MainView/Typewriter.tsx` consuming `s2c/text/streaming` deltas at the configured `typewriter_speed_chars_per_sec`. The component completes immediately when `s2c/text/complete` arrives if it has not already caught up.
- [X] T053 [P] [US1] Author `frontend/src/screens/MainView/InteractionPanel.tsx` rendering dialog text via `Typewriter`, the option buttons (or a single "Continue" when options is empty), and a free-text input. Buttons send `c2s/play/advance`; free-text sends `c2s/play/free_text`.
- [X] T054 [US1] Author `frontend/src/screens/MainView/index.tsx` composing Background + CharacterStage + InteractionPanel and wiring the top-of-screen "Story" and "Menu" affordances (Story toggles the side panel; the panel itself is implemented in US3).
- [X] T055 [US1] Author `frontend/src/main.tsx` to read `window.lucidium.wsPort`, instantiate the WS client from T027, hydrate the Zustand store from T028, and route between `StartScreen`, `NewGameInterview`, and `MainView` based on session state.
- [X] T056 [US1] Add an end-to-end Vitest suite at `frontend/tests/unit/main_view.test.tsx` that drives a recorded WS server through the full interview-to-tenth-node flow with the recorded providers; this covers the SC-002/SC-003/SC-009 acceptance criteria offline.

**Checkpoint**: User Story 1 is fully functional and demonstrably independent. A player can complete the interview and play through option-driven and free-text dialog with characters and backgrounds appearing on cue. **MVP shippable.**

---

## Phase 4: User Story 2 — Resume a previous story from the Start Screen (Priority: P2)

**Goal**: A returning player presses Continue (or uses Load Game to manage saves) and is dropped exactly back into their last session.

**Independent Test**: Play a US1 session through ≥5 nodes, exit the app, relaunch, press Continue, and verify the same node, characters, background, and world values. Then open Load Game, rename one save, delete another, relaunch, and verify both operations persisted.

### Backend

- [ ] T057 [US2] Extend `backend/src/lucidium/api/handlers.py` with `handle_saves_list` (replies with `s2c/saves/list` from `save_store.list_saves()`), `handle_saves_continue` (loads `most_recent_save_id` and emits `s2c/state/full`), `handle_saves_load`, `handle_saves_rename`, and `handle_saves_delete`. Cover with `backend/tests/integration/test_saves_handlers.py`.
- [ ] T058 [US2] Add per-turn save commit hook in `backend/src/lucidium/api/handlers.py::handle_play_advance` and `handle_play_free_text`: after the new node is committed in memory, write the save to disk via `save_store.commit_save` before emitting downstream events. Cover with `backend/tests/integration/test_per_turn_commit.py` (asserts every committed advance round-trips through disk).
- [ ] T059 [US2] In `backend/src/lucidium/api/ws_server.py::handle_app_exit`, flush the active save and the schedulers' in-flight committed work (in-flight speculative work is discarded per FR-016 and the spec's "save during active generation" edge case) before closing the socket.

### Renderer

- [ ] T060 [P] [US2] Update `frontend/src/screens/StartScreen.tsx` to query `c2s/saves/list` on mount and show Continue iff the response is non-empty. Continue dispatches `c2s/saves/continue`.
- [ ] T061 [P] [US2] Author `frontend/src/screens/LoadGameScreen.tsx` rendering the save list with name, last-played timestamp, and one-line summary; supports Load, Rename, and Delete with optimistic UI that reconciles on `s2c/state/patch`.
- [ ] T062 [US2] Update `frontend/src/main.tsx` routing to include `LoadGameScreen` and to handle the `s2c/state/full` arrival after Continue/Load.

**Checkpoint**: User Stories 1 and 2 both work independently. SC-006 is verifiable.

---

## Phase 5: User Story 3 — Inspect and edit live story state from the Story side panel (Priority: P2)

**Goal**: The player can open the Story panel, edit any field on any tab, and see edits propagate into subsequent generation.

**Independent Test**: During an active session, open Story, change a character's Description on the Characters tab, edit a History line, edit a World Info field. Advance two nodes; assert the new dialog reflects the edits. Reopen the panel; assert the edits persist.

### Backend

- [ ] T063 [US3] Extend `backend/src/lucidium/api/handlers.py` with `handle_edit_world`, `handle_edit_character`, `handle_edit_history`, `handle_edit_environment`. Each validates the patch via Pydantic, applies it through `Game`, persists the save, recomputes premise hashes for descendant speculative nodes per the cross-entity invariant (5) in `data-model.md`, marks newly-invalidated nodes via the obsolescence module from T038, and emits a `s2c/state/patch` reflecting the change. Cover with `backend/tests/integration/test_edit_propagation.py`.
- [ ] T064 [US3] In `backend/src/lucidium/orchestration/llm_scheduler.py`, when the world or a character's relevant attributes change mid-flight, mark in-flight tasks per the "conflicting edits in the side panel" edge case: the result is discarded on return and a fresh task is queued. Cover with `backend/tests/integration/test_in_flight_edit_conflict.py`.

### Renderer

- [X] T065 [P] [US3] Author `frontend/src/screens/MainView/StoryPanel/index.tsx` with tab navigation for History, World Info, Environments, Characters, Dialog Tree, Options. Highlight the active environment per FR-038.
- [X] T066 [P] [US3] Author `frontend/src/screens/MainView/StoryPanel/HistoryTab.tsx` rendering the committed dialog history as editable text fields. Edits dispatch `c2s/edit/history`.
- [X] T067 [P] [US3] Author `frontend/src/screens/MainView/StoryPanel/WorldInfoTab.tsx` exposing every WorldState field (game name, plot direction, intent forecast, active and dropped plot threads) as editable controls. Edits dispatch `c2s/edit/world`.
- [X] T068 [P] [US3] Author `frontend/src/screens/MainView/StoryPanel/EnvironmentsTab.tsx` listing every Environment with its label, prompt, and image; the active environment is visually highlighted; edits dispatch `c2s/edit/environment`.
- [X] T069 [P] [US3] Author `frontend/src/screens/MainView/StoryPanel/CharactersTab.tsx` listing every Character with all attributes, facts, and the latest portrait; every field editable; edits dispatch `c2s/edit/character`.
- [X] T070 [P] [US3] Author `frontend/src/screens/MainView/StoryPanel/DialogTreeTab.tsx` rendering the dialog tree (committed path bolded, speculative children dimmed, invalidated tombstones marked) with options editable.
- [X] T071 [P] [US3] Author `frontend/src/screens/MainView/StoryPanel/OptionsTab.tsx` linking to the Settings screen and exposing per-save overrides (e.g., `prompt_history_clamp_chars`).

**Checkpoint**: SC-005 verifiable. The engine is now a creative tool, not a black box.

---

## Phase 6: User Story 4 — Configure backend connections and presentation (Priority: P3)

**Goal**: The player can change LLM endpoint, image endpoint, model, and typewriter speed; changes apply on the next generation without restart.

**Independent Test**: Open Settings, change LLM base URL to an alternative endpoint and image port to a different value, confirm the next text-gen and next image-gen call hit the new endpoints (recorded fixtures distinguish endpoints). Drag typewriter slider; confirm the next node renders at the new speed.

### Backend

- [X] T072 [US4] Extend `backend/src/lucidium/api/handlers.py` with `handle_settings_get` and `handle_settings_update`. Settings update writes through `settings_store` and re-binds the `LlmClient` and `ImageClient` instances used by the schedulers without restarting the process. Cover with `backend/tests/integration/test_settings_live_update.py`.
- [X] T073 [US4] In `backend/src/lucidium/orchestration/llm_scheduler.py` and `image_scheduler.py`, accept a `provider_factory` callable so that a mid-flight settings update reroutes the next dispatch through the new provider. Cover with the same integration test (T072) by asserting two consecutive dispatches hit different endpoints.

### Renderer

- [X] T074 [P] [US4] Author `frontend/src/settings/SettingsScreen.tsx` with sections for LLM (base URL, model, API key — masked input), Image (base URL, portrait/background workflow paths), and Presentation (typewriter speed slider). Validates inputs locally and dispatches `c2s/settings/update`.
- [X] T075 [US4] Wire SettingsScreen entry from both StartScreen (Options button) and the Story panel's Options tab (T071).

**Checkpoint**: SC-010 verifiable.

---

## Phase 7: User Story 5 — Add custom side characters during the new-game interview (Priority: P3)

**Goal**: On the confirmation step, the player adds one-line side-character descriptions; each is fleshed into a full Character record and is available to enter scenes.

**Independent Test**: Add two one-line side characters in confirmation. Within the first 20 dialog nodes after entering the main view, both characters appear with full visual portraits consistent with their descriptions and visually distinct from the player character.

### Backend

- [X] T076 [US5] Extend `backend/src/lucidium/api/handlers.py` with `handle_new_game_add_side_character` that takes a one-line description, runs the side-character-expansion prompt (T034) through the LLM, validates the resulting `Character` against the schema, runs character-repair (T033) if any field is missing, generates the seed, queues a low-priority portrait job in the image scheduler, and adds the character to `Game.characters`. Cover with `backend/tests/integration/test_side_character_expansion.py`.

### Renderer

- [X] T077 [P] [US5] Update `frontend/src/screens/NewGameInterview/ConfirmStep.tsx` (from T049) to add a "Side Characters" section: a free-text input plus an Add button that dispatches `c2s/new_game/add_side_character`, and a list of already-added side characters shown by name (or a placeholder while expansion is in flight).

**Checkpoint**: All five user stories are independently functional.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Operational quality, observability, packaging, and the constitution's recurring obligations (cost telemetry, fallback paths, replayability).

- [X] T078 [P] Add per-turn cost telemetry capture in the LLM and image schedulers (tokens, latency, dollar estimate) and append into `Game.cost_telemetry`; emit `s2c/cost` deltas. Cover with `backend/tests/unit/test_cost_telemetry.py`.
- [X] T079 [P] Add scheduler debug overlay messages (`s2c/scheduler/status`) and a hidden `frontend/src/screens/MainView/SchedulerOverlay.tsx` toggled by a keyboard shortcut.
- [X] T080 [P] Add WS reconnection handling in `frontend/src/ws/client.ts` with jittered backoff, plus a renderer banner shown via Zustand when the connection is down. Cover with `frontend/tests/unit/ws_reconnect.test.ts`.
- [X] T081 [P] Implement the "backend unreachable" recovery banner in the renderer that surfaces `s2c/error{code:"provider_unreachable"}` with a Settings shortcut and confirms an autosave checkpoint per the spec's edge case.
- [X] T082 [P] Wire content-hash deduplication in `backend/src/lucidium/orchestration/image_scheduler.py` so that an identical (prompt, seed) pair reuses the existing `<save-id>/images/<hash>.png`. Cover with `backend/tests/unit/test_image_dedup.py`.
- [X] T083 [P] Capture per-turn random seeds and prompt parameters into `DialogNode.generation_metadata` per Constitution "Determinism" constraint; verify each committed node can be replayed by feeding metadata back into the prompt builder. Cover with `backend/tests/integration/test_turn_replay.py`.
- [X] T084 [P] Add `electron-builder` configuration in `frontend/package.json` to package Electron + Python sidecar (using `pyinstaller` for the backend) for Windows; document macOS and Linux build steps in `docs/packaging.md`.
- [X] T085 [P] Author `docs/operations.md` covering local app-data layout, save backup/restore, log locations, and the offline-test invocation; cross-link from `quickstart.md`.
- [ ] T086 Run the manual smoke checklist from `quickstart.md` end to end against the packaged build; record any regressions as new tickets.
- [X] T087 [P] Add accessibility passes: keyboard-only navigation through StartScreen, Interview, MainView, Story panel, and Settings; ensure focus order and ARIA labels are sane. Lightweight a11y tests in `frontend/tests/unit/a11y.test.tsx`.
- [X] T088 Final constitution audit: walk Principles I–V against the implemented code, file remediation tickets for any gap, and tag the release.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup. **Blocks all user stories.**
- **User Story 1 (Phase 3)**: Depends on Foundational. **MVP cut here.**
- **User Story 2 (Phase 4)**: Depends on Foundational. Works on top of US1's commit-per-turn rhythm but is independently testable against any save (recorded fixtures provide one).
- **User Story 3 (Phase 5)**: Depends on Foundational. Independent of US2; assumes US1 has produced live state (recorded fixtures provide a starting state).
- **User Story 4 (Phase 6)**: Depends on Foundational. Independent of US1/2/3.
- **User Story 5 (Phase 7)**: Depends on Foundational + the interview UI from US1's `ConfirmStep` (extends T049). Document this as the only cross-story dependency.
- **Polish (Phase 8)**: Depends on the user stories the team intends to ship.

### Within Each User Story

- Models before services before handlers before WS dispatch.
- Renderer screens after the WS messages they consume are in `messages.py` (Foundational).
- Tests live next to the unit they cover and run in CI on every change.

### Parallel Opportunities

- **Setup**: T002–T008 are all `[P]`.
- **Foundational**: T011, T013, T014, T015, T016, T019, T020, T021, T022, T024, T025, T026, T027, T028, T029, T030 are all `[P]`. T012 must precede T013–T017 (it declares the IPC types Pydantic the domain models reference for embedding). T018 depends on T012–T017.
- **User Story 1**: prompt files T031–T037 in parallel; renderer components T048–T053 in parallel; backend scheduler files T039 and T040 in parallel; T038 must precede T041; T042–T046 must run sequentially against `handlers.py`.
- **User Story 3**: T065–T071 are all `[P]` (separate files).
- **Polish**: T078–T085 and T087 are all `[P]`.

---

## Parallel Example: User Story 1 — Foundational kickoff

```text
# Once T012 (messages.py) is committed, fan out the domain models in parallel:
Task: "T013 Author backend/src/lucidium/domain/character.py with age-band rules"
Task: "T014 Author backend/src/lucidium/domain/world.py"
Task: "T015 Author backend/src/lucidium/domain/environment.py"
Task: "T016 Author backend/src/lucidium/domain/dialog.py with premise_hash helper"

# In parallel with the domain work, fan out the providers and persistence:
Task: "T019 Author backend/src/lucidium/providers/llm_client.py + Recorded fake"
Task: "T020 Author backend/src/lucidium/providers/image_client.py + Recorded fake"
Task: "T021 Author backend/src/lucidium/persistence/settings_store.py"
Task: "T022 Author backend/src/lucidium/persistence/save_store.py"
```

---

## Parallel Example: User Story 1 — Renderer kickoff

```text
# Once Foundational is done and T031–T037 produce stable prompts, the renderer
# can be built out in parallel against the recorded WS server:
Task: "T048 Author frontend/src/screens/StartScreen.tsx"
Task: "T049 Author frontend/src/screens/NewGameInterview/ (six step files)"
Task: "T050 Author frontend/src/screens/MainView/Background.tsx"
Task: "T051 Author frontend/src/screens/MainView/CharacterStage.tsx"
Task: "T052 Author frontend/src/screens/MainView/Typewriter.tsx"
Task: "T053 Author frontend/src/screens/MainView/InteractionPanel.tsx"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1 Setup → repo scaffolding green.
2. Phase 2 Foundational → IPC, WS, providers, persistence, codegen all green; constitution gate (offline tests) enforced in CI.
3. Phase 3 User Story 1 → fully playable from New Game through 10+ nodes against recorded providers; ship as MVP.
4. Validate against the spec's User Story 1 acceptance scenarios (1–6) before declaring done.

### Incremental Delivery

1. Setup + Foundational → release-candidate baseline.
2. + US1 → MVP demo.
3. + US2 → save resume; demo.
4. + US3 → side-panel editing; demo.
5. + US4 → settings; demo.
6. + US5 → side-character expansion; demo.
7. Polish → packaging + a11y + constitution audit; cut release.

### Parallel Team Strategy

Once Foundational completes, three streams can run in parallel:

1. Backend orchestration & schedulers (T031–T047, T057–T059, T063–T064, T072–T073, T076).
2. Renderer UI surfaces (T048–T056, T060–T062, T065–T071, T074–T075, T077).
3. Test/CI/packaging (T078–T088).

---

## Notes

- `[P]` ⇒ different files, no incomplete dependencies.
- Every test task in this plan is justified by Constitution Principle IV: behavior must be exercisable without a live model call. The recorded providers and `respx` gate make that enforceable.
- Commit cadence: after each user story phase checkpoint, at minimum.
- The only explicit cross-story dependency is **US5 → US1's `ConfirmStep`** (T077 extends T049). Document any other cross-story coupling in PR descriptions and refactor toward independence.
- Avoid: adding new Pydantic schemas outside `backend/src/lucidium/api/messages.py` and `backend/src/lucidium/domain/`; mirroring TS types by hand; introducing a second IPC channel.

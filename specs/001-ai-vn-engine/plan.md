# Implementation Plan: AI-Driven Visual Novel Engine

**Branch**: `001-ai-vn-engine` | **Date**: 2026-05-01 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/001-ai-vn-engine/spec.md`

## Summary

Lucidium is a single-player desktop visual novel engine that generates dialog, characters, and backgrounds on demand from configurable LLM and image-generation backends. A Python orchestration process owns the world state, dialog tree, and the two ahead-of-player schedulers (LLM and image); it exposes its state to the renderer over a single WebSocket. An Electron renderer hosts the UI (Start Screen, New Game interview, Main play view with a Story side panel) and is the only process the user launches — it spawns the backend as a child process and shuts it down on Exit. The IPC contract is generated from a single source of truth (Pydantic models in the backend → JSON Schema → TypeScript types in the renderer) so the two runtimes never drift.

## Technical Context

**Language/Version**: Python 3.11+ (backend); TypeScript on Node.js 20 LTS (Electron main + renderer).
**Primary Dependencies**:
- Backend: `pydantic` v2 (schemas + validation), `websockets` (server), `httpx` (async HTTP for OpenAI-compatible + ComfyUI), stdlib `asyncio` for concurrency, `pytest` + `pytest-asyncio` for tests, `respx` for HTTP recording/replay.
- Renderer: `electron`, `react` + `react-dom`, `vite` (dev server + build), `zustand` (lightweight state), `vitest` + `@testing-library/react` for tests.
- Schema codegen: Pydantic → JSON Schema → `json-schema-to-typescript`.

**Storage**: Local filesystem only.
- Per-save folder under the user's app-data directory (`%APPDATA%/Lucidium/saves/<save-id>/` on Windows, equivalent on macOS/Linux), each containing `game.json` (world + dialog tree + character roster), `images/` (PNG/WebP per asset), and `meta.json` (name, last-played, settings snapshot).
- One global `settings.json` for the installation.
- Atomic save commits via write-temp + `os.replace`.

**Testing**: pytest (backend, offline) + Vitest (renderer, jsdom) + a small Playwright smoke suite for end-to-end through Electron when a backend stub is acceptable. Recorded LLM/image fixtures live under `backend/tests/fixtures/`. Constitution requires every non-trivial behavior reachable without a live model call; that is enforced by a CI job that runs the suite with the providers replaced by `respx` mocks.

**Target Platform**: Windows 11 primary (development host), macOS 13+ and modern Linux secondary via Electron's cross-platform packaging.

**Project Type**: Desktop app (Electron renderer + Python sidecar backend).

**Performance Goals**:
- 95th-percentile time-to-first-character of generated dialog ≤ 1.5 s when the next node was pre-generated; ≤ 5 s for a fresh free-text response (median ≤ 5 s; cf. SC-007).
- Image scheduler keeps the visible node and the next two nodes' assets ready in steady state, with at most one consecutive fallback (cf. SC-008).
- Renderer paint within 100 ms of a WebSocket message under typical asset sizes (≤ 1 MB per portrait, ≤ 2 MB per background).

**Constraints**:
- Single-machine, no cloud dependency beyond the user-configured model endpoints.
- Backend speaks **one** WebSocket; HTTP is reserved for outbound calls to model providers.
- All shared types are generated; hand-mirroring between Python and TypeScript is forbidden by the constitution.
- Provider API keys read from `settings.json` only; never logged, never committed.
- Random seeds and prompt parameters captured per turn so a turn is replayable for debugging.

**Scale/Scope**:
- One concurrent player session per running app.
- Saves up to ~10,000 dialog nodes and ~50 characters per playthrough; assume save sizes up to ~500 MB on disk dominated by images.
- Up to 4 characters on stage simultaneously assumed for layout sizing (no hard data-model cap).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The Lucidium Constitution v1.0.0 defines five binding principles. Each is evaluated against this plan.

| # | Principle | Evaluation | Status |
|---|-----------|------------|--------|
| I | **Reliability** | All model outputs flow through Pydantic validation before reaching the dialog tree (FR-002, FR-003 schemas). Reject-and-retry loop in `orchestration.text_gen` with a bounded retry budget. Fallback paths defined for: missing image at present time (prior background, placeholder portrait), missing text at present time (signaled wait), backend unreachable (banner + autosave). Save commits after every accepted turn (atomic write). | **PASS** |
| II | **Elegance** | Domain-driven module names (`world`, `character`, `dialog`, `scheduler`). One IPC channel (WebSocket); one save format (folder of JSON + images); one settings file. No premature abstractions: schedulers are concrete classes, not a plugin framework. | **PASS** |
| III | **Efficiency** | Two-tier scheduler runs ahead of the player. LLM calls and image calls each fan out via `asyncio.gather`. Prompt prefixes (history, on-stage characters) sent in stable order to leverage provider prompt caching. Generated images cached in `save/images/` keyed by content hash so a re-run of the same prompt+seed reuses the file. Per-turn cost telemetry (tokens, latency, dollar estimate) recorded into the save and surfaced to settings. | **PASS** |
| IV | **Testability** | Provider clients sit behind `LlmClient` and `ImageClient` protocols. Tests inject fakes (`RecordedLlmClient`, `RecordedImageClient`) reading from `tests/fixtures/`. Scheduler logic, prompt construction, free-text invalidation, obsolescence detection, and persistence are all exercisable without network. CI runs the offline suite on every change. | **PASS** |
| V | **DRY** | Shared types between Python and TypeScript are generated from Pydantic via JSON Schema. Prompt fragments live in `backend/src/lucidium/prompts/` as named constants composed by builders. Model IDs, default endpoints, retry budgets, and history clamps are declared once in `config.py` and referenced. | **PASS** |

**Initial gate**: PASS. No violations to record. Proceeding to Phase 0.

**Post-design gate (re-evaluated after Phase 1)**: PASS. Phase 1 artifacts (data-model.md, contracts/, quickstart.md) preserve all five principles — no new model calls without fallbacks, schemas are the single IPC source of truth, and every cross-cutting concern (prompt fragments, config constants) lives in exactly one module.

## Project Structure

### Documentation (this feature)

```text
specs/001-ai-vn-engine/
├── plan.md                   # This file
├── research.md               # Phase 0 output
├── data-model.md             # Phase 1 output
├── quickstart.md             # Phase 1 output
├── contracts/
│   ├── ws-messages.md        # Human-readable WebSocket message contract
│   └── schemas/              # JSON Schemas exported from backend Pydantic models
│       └── README.md
└── checklists/
    └── requirements.md       # From /speckit-specify
```

### Source Code (repository root)

```text
backend/                                # Python sidecar
├── pyproject.toml
├── src/
│   └── lucidium/
│       ├── __init__.py
│       ├── app.py                      # Process entrypoint; starts WS server
│       ├── config.py                   # Defaults, paths, single source of constants
│       ├── api/
│       │   ├── ws_server.py            # Single WebSocket endpoint
│       │   ├── messages.py             # Pydantic message types (source of truth for IPC)
│       │   └── handlers.py             # Dispatch from message -> domain action
│       ├── domain/
│       │   ├── world.py                # WorldState
│       │   ├── character.py            # Character + age-band rules
│       │   ├── dialog.py               # DialogNode, DialogTree
│       │   ├── environment.py          # Environment (Location -> background)
│       │   └── game.py                 # Game aggregate (root)
│       ├── orchestration/
│       │   ├── llm_scheduler.py        # Priority queue + parallel LLM calls
│       │   ├── image_scheduler.py      # Priority queue + parallel image calls
│       │   ├── obsolescence.py         # Task validity rules (FR-013, FR-016/17)
│       │   ├── prompts/                # Shared prompt fragments + builders
│       │   │   ├── text_gen.py
│       │   │   ├── world_refresh.py
│       │   │   ├── character_repair.py
│       │   │   ├── interview.py
│       │   │   └── image_prompts.py
│       │   └── summarizer.py           # Facts + plot-thread maintenance
│       ├── providers/
│       │   ├── llm_client.py           # Protocol + OpenAI-compatible impl + recorded fake
│       │   └── image_client.py         # Protocol + ComfyUI impl + recorded fake
│       └── persistence/
│           ├── save_store.py           # CRUD on local saves
│           ├── settings_store.py
│           └── atomic.py               # Atomic write helpers
└── tests/
    ├── unit/
    ├── integration/                    # Scheduler + persistence with recorded providers
    └── fixtures/                       # Recorded LLM and image responses

frontend/                               # Electron renderer
├── package.json
├── electron/
│   ├── main.ts                         # Spawns Python backend, manages window lifecycle
│   └── preload.ts
├── src/
│   ├── main.tsx                        # React entrypoint
│   ├── ws/
│   │   └── client.ts                   # WebSocket client; reconnect; typed dispatch
│   ├── shared/
│   │   └── generated/                  # TS types generated from backend JSON Schema
│   ├── state/
│   │   └── store.ts                    # Zustand store mirroring world + dialog state
│   ├── screens/
│   │   ├── StartScreen.tsx
│   │   ├── NewGameInterview/
│   │   │   ├── index.tsx
│   │   │   ├── SettingStep.tsx
│   │   │   ├── GenreStep.tsx
│   │   │   ├── VisualStyleStep.tsx
│   │   │   ├── CharacterStep.tsx
│   │   │   ├── NameStep.tsx
│   │   │   └── ConfirmStep.tsx
│   │   └── MainView/
│   │       ├── index.tsx
│   │       ├── Background.tsx
│   │       ├── CharacterStage.tsx
│   │       ├── InteractionPanel.tsx
│   │       ├── Typewriter.tsx
│   │       └── StoryPanel/
│   │           ├── index.tsx
│   │           ├── HistoryTab.tsx
│   │           ├── WorldInfoTab.tsx
│   │           ├── EnvironmentsTab.tsx
│   │           ├── CharactersTab.tsx
│   │           ├── DialogTreeTab.tsx
│   │           └── OptionsTab.tsx
│   └── settings/
│       └── SettingsScreen.tsx
└── tests/
    └── unit/

scripts/
└── codegen/
    └── export-schemas.py               # Pydantic -> JSON Schema -> TS at build time

shared-schemas/                         # Generated artifact, checked in for reproducibility
└── *.schema.json
```

**Structure Decision**: Two top-level packages — `backend/` (Python sidecar) and `frontend/` (Electron + React) — joined by a generated `shared-schemas/` directory and a `scripts/codegen/` step that produces TypeScript types from the backend's Pydantic models. This satisfies the constitution's DRY mandate (one source of truth for IPC types) and Elegance mandate (domain-named modules, single WebSocket, single save format). The two-package layout matches the constitution's mandated runtime stack (Python orchestration + Electron renderer) and keeps the boundary explicit.

## Complexity Tracking

> No constitution violations. This section is intentionally empty.

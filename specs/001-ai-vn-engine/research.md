# Phase 0 Research: AI-Driven Visual Novel Engine

**Feature**: 001-ai-vn-engine
**Date**: 2026-05-01

## Scope

The Technical Context in `plan.md` had no `NEEDS CLARIFICATION` markers. This document records the design decisions that were made implicitly — the rationale behind each choice, and the alternatives that were considered and rejected. Each section is one decision; future readers should be able to trace any "why is it this way?" question to a row here.

---

## D1. Renderer-to-backend transport: single WebSocket

- **Decision**: One WebSocket connection between the Electron renderer and the Python sidecar, carrying a small set of versioned JSON message types. No HTTP from renderer to backend.
- **Rationale**: FR-004 already mandates WebSocket streaming for state updates. Adding HTTP for command dispatch would split the contract across two surfaces and complicate the IPC schema. A single bidirectional channel lets the backend push asset-ready notifications, partial text, and world-state diffs without the renderer polling, and lets the renderer issue commands (advance, free-text, edit, save, load) on the same socket. The Electron main process spawns the backend, learns the chosen port from the backend's stdout, and passes it to the renderer via `preload`.
- **Alternatives**:
  - **HTTP REST + Server-Sent Events**: Two surfaces; SSE is one-way; doesn't simplify anything.
  - **Embedded Python via PyO3 / pythonnet**: Removes IPC, but ties Electron build to a CPython binding and makes the orchestration unrunnable headless. Rejected for testability (Constitution IV).
  - **Stdio JSON-lines**: Works, but harder to debug and lacks browser-grade DevTools support compared to WebSockets viewed from the renderer.

## D2. IPC schema source of truth: Pydantic → JSON Schema → TypeScript

- **Decision**: Pydantic v2 models in `backend/src/lucidium/api/messages.py` are the single source of truth for IPC types. A build step exports each as JSON Schema into `shared-schemas/`; `json-schema-to-typescript` produces TypeScript types into `frontend/src/shared/generated/`.
- **Rationale**: Constitution V (DRY) explicitly forbids hand-mirroring. Pydantic gives us validation on the wire and the schema export for free. The generated TS file is checked in so the renderer build does not depend on Python being on the build host.
- **Alternatives**:
  - **Protocol Buffers**: Heavy for a pure-JSON local IPC; binary serialization adds value only when bandwidth matters (it doesn't, on localhost).
  - **TypeScript-first with `zod` and a Python codegen**: Inverts the source of truth, but the orchestration layer is the harder side to validate; it benefits more from being canonical.

## D3. Concurrency model in the backend: stdlib `asyncio`

- **Decision**: Plain `asyncio` task groups, with an async semaphore per provider to bound parallelism (e.g., 4 LLM in flight, 2 image in flight by default; configurable in `settings.json`).
- **Rationale**: Both LLM and image work are I/O-bound HTTP. `asyncio.gather` with bounded semaphores is the smallest correct primitive; it avoids the thread-pool/process-pool decision and keeps the scheduler logic linear and testable.
- **Alternatives**:
  - **Trio / AnyIO**: Better cancellation ergonomics, but adds a dependency for benefits we don't yet need. Revisit if cancellation gets gnarly during free-text invalidation.
  - **Threading**: Wastes threads on I/O-bound work and complicates shared-state access.

## D4. LLM provider abstraction: OpenAI-compatible only

- **Decision**: A single `LlmClient` protocol whose default implementation talks the OpenAI Chat Completions API, configured by base URL + API key + model. Defaults point at OpenRouter with a Qwen-class model. ComfyUI is the image counterpart.
- **Rationale**: FR-032 says LLM settings are OpenAI-style with OpenRouter defaults. Most viable LLM hosts speak the OpenAI dialect; constraining to it keeps the prompt builder and response validator simple and lets users swap providers without code changes.
- **Alternatives**:
  - **A pluggable LLM framework (LangChain/LiteLLM)**: Adds dependencies and abstraction without solving a problem we have. Constitution II (Elegance) rules it out.

## D5. Image provider abstraction: ComfyUI workflow templates

- **Decision**: One `ImageClient` whose default impl posts a parameterized ComfyUI workflow JSON (one workflow per asset class: portrait, background) to the configured ComfyUI HTTP endpoint, polls for completion, and downloads the resulting image file. Workflow templates live under `backend/src/lucidium/orchestration/prompts/` next to text prompts; per-character `seed` is injected into the workflow's KSampler node.
- **Rationale**: FR-015 mandates per-character seed stability. ComfyUI's workflow JSON exposes the seed; templating it gives exact control. Bundling the template with the prompt fragments centralizes "how we ask the models for this asset class," consistent with Constitution V.
- **Alternatives**:
  - **A1111-style /sdapi/v1/txt2img**: Simpler endpoint, but the user explicitly defaults to ComfyUI. Could be added later as a second `ImageClient` implementation if requested.
  - **External SaaS (Replicate, Fal)**: Off-default; can be added by writing a new `ImageClient` against the same protocol.

## D6. Save format: per-save folder of JSON + image files

- **Decision**: Each save is a directory `<app-data>/saves/<save-id>/` with `meta.json` (id, name, last-played, settings snapshot, schema version), `game.json` (world state + dialog tree + character roster + cost telemetry), and `images/<hash>.png` for every generated asset. Atomic writes via temp + `os.replace`.
- **Rationale**: Plain JSON is grep-able, version-controllable by the player, and trivially editable when something goes wrong (Constitution I, Reliability — "actionable diagnostics"). Folder layout keeps assets out of the JSON, so a single corrupt image cannot brick a save. Atomic rename guarantees either the previous save or the new save is on disk, never a half-written file.
- **Alternatives**:
  - **SQLite per save**: More robust to partial writes, but harder for a player to inspect or repair, and overkill for a one-writer workload.
  - **Single JSON with base64 images inline**: Bloats memory at load time and conflicts with the constitution's "smallest design that meets the requirement" (II).

## D7. Free-text invalidation strategy

- **Decision**: Each speculative dialog node is tagged with the *premise* it was generated from (a content hash of the parent node + the world state vector). When the player submits free text, the system commits a new node from the player's text under the parent and walks the tree marking any descendant whose premise hash no longer matches as invalidated; their generation tasks are cancelled if not yet started. In-flight image work is kept if the underlying character/environment state is still valid (FR-017), discarded otherwise.
- **Rationale**: Treating the premise as a hash gives a cheap, deterministic obsolescence test that the scheduler can apply uniformly to text and image tasks. It directly implements FR-013, FR-016, FR-017 without bespoke per-task logic.
- **Alternatives**:
  - **Generation IDs incremented on every state change**: Simpler counter, but invalidates more than necessary (any state change flushes everything; FR-017 wants partial retention).
  - **Manual diffing on each task**: Fragile and duplicates logic per task class.

## D8. Scheduler priority model

- **Decision**: Each scheduler maintains an ordered priority queue keyed by class (`TEXT_NEXT > CHAR_REPAIR > WORLD_REFRESH` for LLM; `BG_NEAR > PORTRAIT_NEAR > CHAR_PROMPT_REGEN_SPEC` for image), with distance-from-current-node as the secondary key. A worker pool drains the queue under the per-provider semaphore. Re-prioritization on state change uses a "rescore" pass that re-sorts pending tasks before the next dispatch.
- **Rationale**: The user explicitly enumerated priority orders; encoding them as ordered classes makes the implementation self-documenting and the priorities user-visible (showing the queue in a debug overlay is straightforward). Distance-from-current as secondary key makes "work outward from the current node" trivially correct.
- **Alternatives**:
  - **Per-task numeric priority**: Equivalent expressive power, but priority is implicit in a magic number rather than a class name. Loses readability.

## D9. Per-character image identity: seed + canonical prompt template

- **Decision**: At character creation, a 64-bit random seed is rolled and stored on the character. Every portrait is generated by applying the canonical portrait template, substituting current attribute values (transformed Age via the age-band rule, plus pose, expression, outfit, etc.), with the stored seed pinned. The seed never changes for the life of the character; only attribute substitutions change.
- **Rationale**: Stable diffusion identity reuse via shared seed + minimally-changing prompt is the established practice for character consistency. FR-015 mandates exactly this.
- **Alternatives**:
  - **Reference-image conditioning (IP-Adapter / FaceID)**: Far better identity fidelity but requires a specific ComfyUI workflow and additional model weights; out of scope for v1, can be added by upgrading the portrait workflow without changing the data model.

## D10. Renderer state synchronization: server-authoritative store

- **Decision**: The backend is authoritative for world + dialog state. The renderer holds a Zustand store that is initialized from a `state/full` snapshot on connection and incrementally updated by `state/patch` messages. Renderer-initiated edits go to the backend, are validated, and are echoed back as patches; the renderer never mutates its own copy independently of the server.
- **Rationale**: The orchestration logic depends on a single consistent state for prompt construction. Server-authoritative is the simplest model that prevents desync, and matches the "WebSocket is the source of truth" framing in FR-004 / FR-039.
- **Alternatives**:
  - **CRDT-style local mutation with eventual reconciliation**: Massive overkill for one renderer with one writer.

## D11. Save autosave cadence

- **Decision**: Save commits happen after every accepted dialog node and after every confirmed Story-panel edit. There is no time-based autosave. "Continue" loads the most recent commit by `last-played` timestamp.
- **Rationale**: Constitution I requires that no failure can lose more than the in-flight turn. Per-turn commits achieve that with the lowest possible bookkeeping.
- **Alternatives**:
  - **Periodic timer-based autosave**: Adds a background task and creates the question "what counts as in-progress." Per-turn commit is simpler and stricter.

## D12. Testing strategy: recorded-fixture providers + offline CI gate

- **Decision**: Both `LlmClient` and `ImageClient` ship with `Recorded*` implementations that read from `backend/tests/fixtures/`. A `record` mode (off by default) re-captures fixtures from a live provider; a `replay` mode (default in CI) reads only from disk. CI runs `pytest` in replay mode and fails on any HTTP call that escapes the fakes (enforced by `respx`).
- **Rationale**: Constitution IV mandates every non-trivial behavior be exercisable without live model calls. Recorded fixtures are how AI-orchestration projects meet that bar.
- **Alternatives**:
  - **Stubbed-deterministic fakes only**: Simpler but less realistic; recorded fixtures catch real schema/edge-case bugs that handcrafted stubs miss.

---

## Open Questions (none required to start implementation)

- Whether to ship a portrait-identity ComfyUI workflow that uses IP-Adapter, or stay with seed-only for v1. **Tentative answer**: seed-only for v1; revisit after the first round of player feedback on character consistency (SC-004).
- Whether to expose per-turn cost telemetry in the UI in v1 or hold it for a later release. **Tentative answer**: capture into the save in v1, surface in UI in a later release.

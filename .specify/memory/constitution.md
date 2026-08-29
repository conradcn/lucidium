<!--
SYNC IMPACT REPORT
==================
Version change: (template) → 1.0.0
Bump rationale: Initial ratification of the Lucidium Constitution. No prior
versioned constitution existed (file held only template placeholders), so this
is treated as the first MAJOR release rather than a 0.x preview.

Modified principles:
  - [PRINCIPLE_1_NAME] → I. Reliability
  - [PRINCIPLE_2_NAME] → II. Elegance
  - [PRINCIPLE_3_NAME] → III. Efficiency
  - [PRINCIPLE_4_NAME] → IV. Testability
  - [PRINCIPLE_5_NAME] → V. DRY (Don't Repeat Yourself)

Added sections:
  - Technology & Architectural Constraints (replaces [SECTION_2_*])
  - Development Workflow & Quality Gates (replaces [SECTION_3_*])
  - Governance (concrete amendment + versioning policy)

Removed sections: none (all template slots resolved or intentionally retitled).

Templates requiring updates:
  - ✅ .specify/templates/plan-template.md — "Constitution Check" gate is
    generic ("Gates determined based on constitution file") and remains
    compatible; the gate will be exercised by /speckit-plan against the five
    principles below. No edit required.
  - ✅ .specify/templates/spec-template.md — No constitution-specific
    references; compatible as-is.
  - ✅ .specify/templates/tasks-template.md — Task categorization (Setup,
    Foundational, per-story, Polish) accommodates testability and DRY
    principles without changes. Compatible as-is.
  - ✅ .specify/templates/checklist-template.md — Generic; compatible.
  - ✅ .claude/skills/speckit-*/ — Command skills are agent-agnostic;
    no outdated agent-name references found.

Follow-up TODOs: none. RATIFICATION_DATE set to 2026-05-01 per user
direction (project initialization day).
-->

# Lucidium Constitution

Lucidium is an infinite visual novel built on-the-fly via AI orchestration.
The constitution below governs how the project is designed, built, reviewed,
and evolved. Every contributor and every automated agent operating in this
repository MUST treat these rules as binding unless an amendment is recorded
through the Governance process.

## Core Principles

### I. Reliability

The runtime experience MUST degrade gracefully, never catastrophically.
Because Lucidium generates narrative and assets on demand from non-deterministic
AI providers, every orchestration call MUST assume the upstream may fail, time
out, return malformed content, or violate schema. The system MUST:

- Validate every model output against an explicit schema before it reaches
  the player; reject-and-retry rather than render unchecked content.
- Provide a deterministic fallback path (cached content, prior turn, or a
  graceful "the muse is silent" state) for every user-visible action.
- Persist player progress after each accepted turn so a crash, network
  failure, or provider outage cannot lose more than the in-flight turn.
- Surface failures with actionable diagnostics — never silent swallowing.

**Rationale**: An infinite novel is worthless if a single 500 from a model
provider ends the story. Reliability is the precondition for the product to
exist at all.

### II. Elegance

Code, prompts, and player-facing surfaces MUST be simple, readable, and
free of incidental complexity. Concretely:

- Prefer the smallest design that meets the requirement; YAGNI is the
  default. New abstractions require a justified, present need (not a
  hypothetical future one).
- Function and module boundaries MUST reflect domain concepts (scene,
  beat, character, choice, asset) rather than implementation accidents.
- Public interfaces (Python module APIs, IPC messages between Python and
  Electron, prompt contracts) MUST be documented in one place and have one
  obvious correct usage.
- Naming is part of elegance: identifiers MUST read as English; clever
  abbreviations are forbidden.

**Rationale**: Lucidium's complexity ceiling is set by the orchestration
graph, not the plumbing. Elegant plumbing keeps cognitive budget free for
the hard parts.

### III. Efficiency

Latency and cost MUST be treated as first-class product constraints, not
afterthoughts. The system MUST:

- Cache deterministically-derivable content (embeddings, summaries, asset
  variants, prompt prefixes via provider prompt caching) by default; a new
  call to a model is a design choice that requires a reason.
- Pipeline orchestration steps so the player perceives streaming progress;
  blocking the UI on a multi-step chain is a defect.
- Track per-turn token, latency, and dollar cost; regressions beyond an
  agreed budget block release until justified or fixed.
- Prefer the smallest capable model for each subtask; escalate to a larger
  model only when a measured quality gap demands it.

**Rationale**: An infinite novel has unbounded calls; without efficiency
discipline the product is either too slow to enjoy or too expensive to run.

### IV. Testability

Every non-trivial behavior MUST be exercisable without a live model call.
The system MUST:

- Separate orchestration logic from model I/O behind seams that accept a
  fake/recorded model client, so logic is unit-testable offline.
- Provide recorded fixtures (or equivalent replay mechanism) for at least
  the canonical happy-path and one failure path of each orchestration step.
- Treat any code path that cannot be reached without a real model call as a
  defect; refactor until it can be.
- Run the offline test suite in CI on every change; a red suite blocks
  merge.

Tests SHOULD be written alongside or before the code they verify when the
behavior is non-obvious; strict TDD is encouraged but not mandated, because
exploratory work on prompts is a legitimate exception.

**Rationale**: AI orchestration is the part of the system most prone to
silent regression. Without testability, every change is a gamble and the
Reliability principle is unenforceable.

### V. DRY (Don't Repeat Yourself)

Each piece of knowledge — a schema, a prompt fragment, a cost constant, a
narrative invariant — MUST have a single authoritative source.

- Shared schemas/types between Python and Electron MUST be generated from
  one source of truth, not hand-mirrored.
- Prompt fragments reused across orchestration steps MUST live in a shared
  prompt library, not be copy-pasted.
- Configuration (model IDs, budgets, feature flags) MUST be defined once
  and referenced; magic numbers and string literals scattered across files
  are a defect.

The Rule of Three applies to behavior, not to data: a literal string
duplicated in two places is fine if it represents two genuinely independent
facts; a duplicated invariant is not.

**Rationale**: Drift between mirrored definitions is the most common source
of "impossible" bugs in dual-runtime apps. DRY at the knowledge level
prevents the bug class entirely.

## Technology & Architectural Constraints

- **Runtime stack**: Python (orchestration, model I/O, persistence) +
  Electron (renderer, player UX). Cross-process communication MUST go
  through a single, versioned IPC contract.
- **Language versions**: Python ≥ 3.11 and a current LTS Node.js for
  Electron. Version bumps require a recorded amendment only if they break
  contributor tooling.
- **Dependencies**: Prefer the standard library and small, well-maintained
  packages. Adding a new runtime dependency requires (a) a stated need,
  (b) a license check, and (c) a note in the PR description.
- **Data**: Player saves and generated content MUST be stored locally by
  default; cloud sync, if added, is opt-in.
- **Secrets**: Provider API keys MUST never be committed and MUST be
  loaded from per-user configuration the player controls.
- **Determinism**: Random seeds and model parameters used for a given turn
  MUST be recorded with the save so a turn can be replayed for debugging.

## Development Workflow & Quality Gates

- **Spec Kit flow is canonical**: Significant features go through
  `/speckit-specify` → `/speckit-clarify` (when ambiguous) → `/speckit-plan`
  → `/speckit-tasks` → `/speckit-implement`. Skipping a phase requires a
  note in the PR explaining why.
- **Constitution Check**: `/speckit-plan` MUST evaluate the plan against
  the five principles above before Phase 0 research and again after Phase 1
  design. Violations go in the plan's Complexity Tracking table with a
  justification, or the plan is revised.
- **Review**: Pre-publication, the project has been developed solo and most
  changes land directly on `master` — the rule below describes the review
  bar, not a branch-protection setting that is currently enforced. At open-
  source publication, branch protection on `master` MUST be enabled so that
  every change from that point on merges via pull request. Whether the
  change arrives as a PR or a direct commit, the same bar applies: the
  author (and any reviewer) MUST verify (1) tests cover the new behavior
  offline, (2) no duplicated knowledge was introduced, (3) any new model
  call has a fallback path.
- **CI**: Offline tests, lint, and type checks MUST pass before merge.
- **Telemetry of cost & latency**: Changes to the orchestration graph MUST
  report measured cost/latency deltas in the PR description.

## Governance

- **Authority**: This constitution supersedes ad-hoc conventions. When a
  README, doc, or comment conflicts with the constitution, the constitution
  wins and the conflicting text MUST be updated.
- **Amendments**: Any contributor may propose an amendment via PR editing
  this file. The PR MUST (a) state the rationale, (b) propose the new
  version per the policy below, and (c) update the Sync Impact Report at
  the top of this file.
- **Versioning policy** (semantic):
  - **MAJOR**: Removing or redefining a principle, or making a previously
    permitted practice forbidden (or vice versa) in a backward-incompatible
    way.
  - **MINOR**: Adding a new principle or section, or materially expanding
    an existing one.
  - **PATCH**: Wording clarifications, typo fixes, non-semantic edits.
- **Compliance review**: Maintainers SHOULD audit the repository against
  this constitution at least once per release cycle. Findings become PRs,
  not silent fixes.
- **Runtime guidance**: Day-to-day contributor guidance lives in
  `CLAUDE.md` and the `.specify/` templates; both MUST stay consistent
  with this file.

**Version**: 1.0.0 | **Ratified**: 2026-05-01 | **Last Amended**: 2026-05-01

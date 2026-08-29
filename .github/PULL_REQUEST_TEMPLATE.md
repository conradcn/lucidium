<!--
Thanks for contributing. Keep this short — delete sections that do not apply.
-->

## What & why

<!-- What changes, and what problem it solves. Link the issue: Fixes #123 -->

## How it was verified

<!--
Which suites you ran and what they said. Name the commands, e.g.:
  backend:  uv run pytest -m "not embedded_live"
  frontend: npm test  /  npm run test:e2e
  types:    npm run build  /  uv run mypy
-->

## Checklist

- [ ] I have read and agree to the [Contributor License Agreement](../CLA.md).
- [ ] Tests cover the new behavior and pass **offline** (no network, no
      real model calls in the default suite).
- [ ] No duplicated knowledge — schemas, prompts and constants have a
      single source of truth rather than a copy.
- [ ] Any new model call has a fallback path when the model is absent,
      slow, or returns something unusable.
- [ ] Lint and type checks pass.
- [ ] No machine-local, generated, or build artifacts are committed
      (`git status --porcelain` is clean after a build + test run).

## Spec-kit phase

<!--
This repo follows /speckit-specify → /speckit-clarify → /speckit-plan →
/speckit-tasks → /speckit-implement. If you skipped a phase, say which and
why. For small fixes, "n/a — bug fix" is a fine answer.
-->

## Cost & latency

<!--
Required only if this changes the orchestration graph: the measured
cost/latency delta, and how you measured it. Otherwise: "no orchestration
change".
-->

# Contributing to Lucidium

Thanks for your interest. Bug reports, fixes and features are all welcome.

## Licensing of contributions

Lucidium is MIT licensed (see [LICENSE](LICENSE)), and contributions are
**inbound = outbound**: what you submit is licensed to the project under the
same MIT terms it publishes under.

In addition, every contribution requires agreement to the
[Contributor License Agreement](CLA.md). It is short, you keep your copyright,
and its purpose is to let the project relicense or dual-license later without
having to track down every past contributor. Agree by checking the CLA box in
the pull request template, or by commenting "I have read the CLA and I agree
to it" on your pull request. You only need to do this once.

Pull requests without CLA agreement cannot be merged.

## Before you open a pull request

- **Discuss anything large first.** Open an issue for new features or
  refactors so the design can be agreed before you spend time on it. Small
  bug fixes can go straight to a PR.
- **Read the active plan.** This repo is spec-driven — see
  [CLAUDE.md](CLAUDE.md) and the plan it points at for the architecture,
  project structure and conventions.
- **Run the suites offline.** `./tasks.ps1` / `./tasks.sh` wrap the backend
  pytest suite, the frontend vitest suite and the Playwright end-to-end
  tests. The default suite must pass with no network and no real model calls.
- **Work through the checklist** in the pull request template — single source
  of truth for schemas and prompts, a fallback path for every new model call,
  lint and types clean, and no generated or machine-local artifacts committed.

## Safeguards

Lucidium enforces hard limits at the prompt, storage and output layers — see
[SAFETY.md](SAFETY.md). Contributions that weaken, bypass or remove those
limits will not be merged, including changes that only do so incidentally. If
your change touches that path, say so explicitly in the PR description and
explain how the limits still hold.

Found a way around a safeguard? That is a security report, not a pull
request — email **conradcn@gmail.com** and see
[.github/SECURITY.md](.github/SECURITY.md). Please do not open a public issue
for it.

## Conduct

Participation is governed by the
[Code of Conduct](.github/CODE_OF_CONDUCT.md).

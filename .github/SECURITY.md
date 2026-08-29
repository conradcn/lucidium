# Security Policy

This document covers **software vulnerabilities** in Lucidium. For
reports about *generated content* — safeguard bypasses, output-filter
false negatives, content the engine should refuse but does not — see
[SAFETY.md](../SAFETY.md) instead.

## Supported versions

Lucidium is pre-1.0 and ships from `master`. Only the latest release
and the current `master` receive security fixes. Older tagged builds
are not patched; upgrade instead.

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Report privately by either route:

- GitHub → the repository's **Security** tab → *Report a vulnerability*
  (private security advisory).
- Email the maintainer at **conradcn@gmail.com** with `[lucidium
  security]` in the subject.

Include what you have: affected version or commit, a description of
the issue, reproduction steps or a proof of concept, and the impact
you believe it has. A minimal reproduction is far more valuable than
a scanner report.

## What to expect

- **Acknowledgement** within 7 days.
- **Initial assessment** — whether we can reproduce it and a rough
  severity — within 14 days.
- **Fix or mitigation plan** communicated once assessed. Timelines
  depend on severity; this is a small volunteer-maintained project.
- **Credit** in the release notes for the fix, unless you ask to stay
  anonymous.

We ask that you give us a reasonable window to ship a fix before
disclosing publicly. We will not pursue legal action against anyone
acting in good faith under this policy.

## Threat model

Lucidium is a **local, single-player desktop application**: an Electron
frontend and a Python backend that runs on the player's own machine.
There is no Lucidium-operated server and no multi-user trust boundary.
The interesting attack surface is therefore:

**In scope**

- Remote code execution or arbitrary file access from opening a
  malicious save file, story pack, character card, or other
  user-supplied asset.
- Path traversal or sandbox escape in asset resolution — anything
  that lets content reach outside its intended directory.
- Renderer-to-main privilege escalation in Electron: context-isolation
  or preload-bridge weaknesses, unsafe IPC handlers, `nodeIntegration`
  regressions.
- Unauthenticated or overly permissive access to the local backend
  HTTP/IPC surface from other processes or from the network.
- Leaking secrets — API keys, tokens, save contents — into logs,
  crash reports, telemetry, or outbound requests.
- Insecure download or update paths: unverified model/binary fetches,
  missing TLS, or tampering opportunities during first-run setup.
- Dependency vulnerabilities with a demonstrated path to exploitation
  in Lucidium's actual usage.

**Out of scope**

- Anything requiring an attacker who already has code execution or an
  interactive session as the same OS user.
- The player deliberately configuring the engine against themselves —
  loading their own untrusted models, pointing at a hostile endpoint,
  or editing files under the app data directory by hand.
- Generated-content policy issues (→ [SAFETY.md](../SAFETY.md)).
- Vulnerabilities in third-party models, model weights, or their
  licensing terms.
- Missing hardening with no demonstrated impact, and automated scanner
  output submitted without analysis.

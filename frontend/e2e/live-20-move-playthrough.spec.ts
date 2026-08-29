/**
 * Live LLM + image-gen 20-move playthrough — driven through the
 * actual Electron UI, not the backend console.
 *
 * Replaces an earlier pytest-based version that drove the handler
 * registry directly. This Playwright spec is closer to "what the
 * player does":
 *
 *   1. Launch real Electron with the real Python backend.
 *   2. Click through the new-game interview (Setting → Visual
 *      Style → Genre → Character → Name → Begin).
 *   3. 20 turns. Each turn picks one of three input modes per
 *      ``CHOICE_PATTERN``:
 *        * ``B`` — click the FIRST presented option (or Continue).
 *        * ``L`` — click the LAST presented option (or Continue).
 *        * ``F`` — submit a free-text input.
 *   4. After each turn, assert the dialog has actually advanced:
 *      ``window.__lucidium_current_node_id`` changes AND the
 *      visible dialog text becomes non-empty.
 *
 * What this catches that pytest-driving-the-registry didn't:
 *   * Renderer-side issues: typewriter not flushing, optimistic
 *     guard not releasing, click-to-advance race.
 *   * IPC: the WebSocket envelope shape, state/patch ordering.
 *   * Electron asset-protocol fetches for portrait images.
 *
 * Setup:
 *   * ``frontend/dist-electron/main.js`` — built via
 *     ``npx tsc -p frontend/tsconfig.electron.json``.
 *   * ``frontend/dist/index.html`` — built via
 *     ``npm --prefix frontend run build``.
 *   * ``backend/.venv`` — backend deps installed.
 *   * ``settings.json`` populated with a real LLM ``api_key``.
 *
 * Skip gates:
 *   * Auto-skip when any of the above is missing.
 *   * ``LUCIDIUM_SKIP_LIVE=1`` skips unconditionally.
 *
 * Wall-clock budget: ~5–15 min on a 4090, depending on LLM latency.
 */

import path from "node:path";
import os from "node:os";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";

import { _electron as electron, expect, test, type Page } from "@playwright/test";

// Single-instance guard. The live test launches a real Electron app
// + a real Python backend that bind to the user's local LLM /
// image / audio settings — running two of them at once spawns two
// game windows fighting for the same GPU and the same WebSocket
// port. The lock is a tiny PID file in the OS temp dir; on test
// start we check whether the recorded PID is still alive (cross-
// platform: ``process.kill(pid, 0)`` throws ``ESRCH`` for dead
// PIDs and is a no-op for live ones), and if it is, the second
// invocation fails fast with a clear message instead of opening
// a competing game window.
const LIVE_E2E_LOCK_PATH = path.join(
  os.tmpdir(), "lucidium-live-e2e.lock",
);

function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    // ESRCH = no such process. EPERM = process exists but we can't
    // signal it (still alive — a different user's process). Any
    // other error means the runtime can't tell, so treat as alive
    // to err on the safe side.
    if (code === "ESRCH") return false;
    return true;
  }
}

function acquireLiveE2eLock(): void {
  if (existsSync(LIVE_E2E_LOCK_PATH)) {
    let priorPid: number | null = null;
    try {
      priorPid = Number.parseInt(
        readFileSync(LIVE_E2E_LOCK_PATH, "utf-8").trim(), 10,
      );
    } catch {
      priorPid = null;
    }
    if (priorPid && Number.isFinite(priorPid) && isPidAlive(priorPid)) {
      throw new Error(
        `another live e2e test is already running (PID ${priorPid}); `
        + `refusing to launch a second Electron + backend instance. `
        + `If you're sure no other run is in flight, delete `
        + `${LIVE_E2E_LOCK_PATH} and retry.`,
      );
    }
    // Stale lock from a previous crashed run — overwrite below.
  }
  mkdirSync(path.dirname(LIVE_E2E_LOCK_PATH), { recursive: true });
  writeFileSync(LIVE_E2E_LOCK_PATH, String(process.pid), "utf-8");
}

function releaseLiveE2eLock(): void {
  try {
    if (!existsSync(LIVE_E2E_LOCK_PATH)) return;
    const recordedPid = Number.parseInt(
      readFileSync(LIVE_E2E_LOCK_PATH, "utf-8").trim(), 10,
    );
    // Only delete the lock if it's ours — a parallel run that
    // somehow squeaked past acquireLiveE2eLock() shouldn't have
    // its lock deleted out from under it on our exit.
    if (recordedPid === process.pid) {
      unlinkSync(LIVE_E2E_LOCK_PATH);
    }
  } catch {
    // Best-effort cleanup; a stuck lock self-clears via the
    // PID-alive check on the next run.
  }
}

const repoRoot = path.resolve(__dirname, "..", "..");
const electronMain = path.join(repoRoot, "frontend", "dist-electron", "main.js");
const distIndex = path.join(repoRoot, "frontend", "dist", "index.html");
const venvPython =
  process.platform === "win32"
    ? path.join(repoRoot, "backend", ".venv", "Scripts", "python.exe")
    : path.join(repoRoot, "backend", ".venv", "bin", "python");

// Default 20 turns; override via ``LIVE_E2E_TURNS=N`` for shorter
// audit runs (e.g. 5 turns when capturing AI-task patterns without
// paying for the full 20-turn budget). The choice pattern is sliced
// to match — clamped to its own length.
const TURNS_TO_PLAY = (() => {
  const raw = process.env.LIVE_E2E_TURNS;
  if (!raw) return 20;
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n) || n <= 0) return 20;
  return Math.min(n, 20);
})();
// One char per turn. ``B`` = first option, ``L`` = last option,
// ``F`` = free-text input. Mixing forces both code paths through
// the test on every run. Sliced to ``TURNS_TO_PLAY`` so a 5-turn
// run still hits B/F/B/L/B (covers all three input modes).
const CHOICE_PATTERN = "BFBLBFBFBLBFBLBFBFBL".slice(0, TURNS_TO_PLAY);

const FREE_TEXT_INPUTS = [
  "I scan the room for something useful.",
  "I ask about the missing lighthouse keeper.",
  "I leave without saying anything.",
  "I follow the figure into the alley.",
  "I look up at the sky and listen.",
  "I check my pockets.",
];

// Per-turn deadline. The first few turns can be slow because the
// embedded SDXL pipeline loads on first portrait render; later
// turns are LLM-bound (~5–15s per beat batch).
const PER_TURN_TIMEOUT_MS = 4 * 60_000;

test.describe("Live LLM 20-move playthrough", () => {
  test.skip(
    !existsSync(electronMain),
    `Electron main not built. Run: npx tsc -p ${path.join(
      "frontend", "tsconfig.electron.json",
    )}`,
  );
  test.skip(
    !existsSync(distIndex),
    "Renderer dist not built. Run: npm --prefix frontend run build",
  );
  test.skip(
    !existsSync(venvPython),
    "Backend venv missing. Run: ./start.ps1 -Setup (or ./start.sh --setup)",
  );
  test.skip(
    process.env.LUCIDIUM_SKIP_LIVE === "1",
    "Live e2e disabled by LUCIDIUM_SKIP_LIVE=1",
  );

  test.setTimeout(30 * 60_000);

  test("walk new-game then 20 mixed-input turns through the UI", async () => {
    acquireLiveE2eLock();
    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: { ...process.env, LUCIDIUM_LOAD_DIST: "1" },
      timeout: 60_000,
    });
    const turnLog: string[] = [];
    // Fail-fast tripwires. Set when ANY backend traceback or
    // s2c/error message lands; checked by ``assertNoFatalError``
    // between every meaningful step. Without this, a backend
    // crash stalls the test for the full poll-timeout (often
    // 10 minutes per ``waitForBeatText`` call) before pytest /
    // Playwright finally times out and reports the *symptom*
    // instead of the actual error visible in the stderr stream.
    const fatals: string[] = [];
    try {
      const window = await electronApp.firstWindow({ timeout: 30_000 });
      // Pipe renderer + backend output into the test stdout so a
      // failure leaves a usable trace. Also scan for fatal-pattern
      // lines and tip them onto ``fatals`` so the next polling
      // step can abort the test.
      window.on("console", (msg) => {
        const text = msg.text();
        console.log(`[renderer] ${msg.type()} ${text}`);
        // s2c/error frames with "internal" code mean a Python
        // exception escaped the handler — almost always fatal
        // for the playthrough since the engine is now in an
        // unknown state.
        if (
          /s2c\/error[^\n]*\b(internal|provider_unreachable|provider_validation)\b/i
            .test(text)
        ) {
          fatals.push(`renderer s2c/error: ${text}`);
        }
      });
      window.on("pageerror", (err) => {
        console.log(`[renderer pageerror] ${err.message}`);
        fatals.push(`renderer pageerror: ${err.message}`);
      });
      electronApp.process().stdout?.on("data", (chunk) =>
        process.stdout.write(`[main-stdout] ${chunk}`),
      );
      electronApp.process().stderr?.on("data", (chunk) => {
        const text = chunk.toString();
        process.stderr.write(`[main-stderr] ${text}`);
        // Conservative fatal detection. The backend's structured
        // logger writes WARNINGs that ALSO carry tracebacks for
        // best-effort failures (e.g. music gen with no ACE-Step
        // server reachable, asset rendering hiccups). Those are
        // documented non-fatal and the engine recovers — matching
        // a bare "Traceback" prefix here would false-positive
        // on every run.
        //
        // Match patterns that ACTUALLY indicate the engine's
        // handler chain crashed:
        //   * The standard handler-error log line shape:
        //     ``ERROR lucidium.api.{ws_server,handlers,…}:`` —
        //     the backend's logging config ONLY raises ERROR
        //     for unhandled exceptions in handler dispatch.
        //   * The WS-server's own "unexpected handler failure"
        //     phrase.
        //   * ``NameError`` / ``ImportError`` etc with the
        //     standard Python prefix (these are programming
        //     errors that should never reach production logs).
        if (
          /\bERROR\s+lucidium\.api\.(ws_server|handlers)\b/.test(text)
          || /unexpected handler failure/.test(text)
        ) {
          fatals.push(`backend handler error: ${text.slice(0, 500)}`);
        }
        // Python class:message error shapes that reach stderr —
        // a real handler crash always shows up as one of these
        // alongside the WS-server ERROR line above. Listed
        // separately for paranoia: even if the logging-shape
        // pattern misses, the literal exception class will
        // catch it.
        if (
          /^\s*(NameError|ImportError|SyntaxError):\s/m.test(text)
        ) {
          fatals.push(`backend python error: ${text.slice(0, 500)}`);
        }
      });

      const assertNoFatalError = (where: string): void => {
        if (fatals.length === 0) return;
        // Surface every captured error so the test report has
        // the full root-cause trail, not just the first one.
        const summary = fatals
          .slice(0, 5)
          .map((f, i) => `  [${i + 1}] ${f}`)
          .join("\n");
        throw new Error(
          `aborting at ${where}: ${fatals.length} fatal error(s) detected:\n${summary}`,
        );
      };

      // Connection-banner check — fail fast if the backend won't
      // come up on the WS port.
      await expect(window.getByTestId("start-screen")).toBeVisible({
        timeout: 30_000,
      });
      await expect(window.getByTestId("connection-banner")).toHaveCount(0, {
        timeout: 30_000,
      });
      assertNoFatalError("after start-screen visible");

      // Dismiss the startup safety warning. It's a modal dialog
      // that intercepts pointer events on every launch (per-
      // session, not per-install) so the New Game button below
      // is unclickable until the player acknowledges. The
      // ``I understand`` button has autofocus so this is the
      // shape a real player follows.
      const startupWarning = window.getByTestId("startup-warning");
      if (await startupWarning.count()) {
        await window
          .getByRole("button", { name: "I understand" })
          .click();
        await expect(startupWarning).toHaveCount(0, { timeout: 5_000 });
      }
      assertNoFatalError("after startup-warning dismissed");

      // ---- New-game interview ---------------------------------
      await window.getByRole("button", { name: "New Game" }).click();
      const interviewSteps = [
        "setting", "visual_style", "genre",
        "character_description", "name",
      ];
      for (const stepName of interviewSteps) {
        await window
          .locator(".option-grid button")
          .first()
          .waitFor({ state: "visible", timeout: 120_000 });
        // The slide-away animation briefly disables option
        // buttons mid-swap; wait until they're enabled before
        // clicking.
        await expect(window.locator(".option-grid button").first())
          .toBeEnabled({ timeout: 5_000 });
        await window.locator(".option-grid button").first().click();
        turnLog.push(`interview ${stepName}: clicked first option`);
        assertNoFatalError(`after interview step ${stepName}`);
      }

      await expect(window.getByTestId("interview-confirm"))
        .toBeVisible({ timeout: 120_000 });
      assertNoFatalError("after interview-confirm visible");
      await window.getByRole("button", { name: "Begin" }).click();
      const beginClickedAt = Date.now();
      // Use a polling wait instead of plain ``toBeVisible`` so a
      // backend traceback during world_init aborts immediately
      // instead of stalling for the full 600s timeout. Capture
      // a screenshot + DOM testid snapshot every poll so a hang
      // shows the actual stuck UI in the test report instead of
      // just timing out with no breadcrumbs.
      let lastDiagAt = 0;
      let lastDiagSummary = "";
      try {
        await expect
          .poll(
            async () => {
              assertNoFatalError("waiting for main-view after Begin");
              const count = await window.getByTestId("main-view").count();
              if (count > 0) return count;
              // Periodic diagnostic snapshot (every ~10s) so a
              // multi-minute hang surfaces what the renderer is
              // actually stuck on. Cheap — a single DOM walk.
              const now = Date.now();
              if (now - lastDiagAt > 10_000) {
                lastDiagAt = now;
                const summary = await window.evaluate(() => {
                  const ids = Array.from(
                    document.querySelectorAll("[data-testid]"),
                  )
                    .slice(0, 12)
                    .map((el) => el.getAttribute("data-testid"))
                    .filter(Boolean);
                  const headings = Array.from(
                    document.querySelectorAll("h1, h2, h3"),
                  )
                    .slice(0, 4)
                    .map((el) => el.textContent?.trim() ?? "");
                  return JSON.stringify({ ids, headings });
                });
                if (summary !== lastDiagSummary) {
                  lastDiagSummary = summary;
                  const elapsed = Math.round((now - beginClickedAt) / 1000);
                  turnLog.push(
                    `  waiting for main-view (${elapsed}s after Begin): ${summary}`,
                  );
                }
              }
              return 0;
            },
            { timeout: 600_000, intervals: [500, 1_000, 2_000] },
          )
          .toBeGreaterThan(0);
      } catch (err) {
        // Surface the last DOM snapshot in the failure message so
        // the report points at the actual hung screen.
        const elapsed = Math.round((Date.now() - beginClickedAt) / 1000);
        throw new Error(
          `main-view never appeared (waited ${elapsed}s after Begin). `
          + `Last visible state: ${lastDiagSummary || "(no snapshot)"}. `
          + `Underlying: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
      turnLog.push("entered main view");

      // Wait for the very first beat to type its text in. The
      // typewriter writes at typewriter_speed; we just need
      // SOME chars to land before we start counting turns.
      await waitForBeatText(window, "", assertNoFatalError);

      // ---- 20-turn playthrough --------------------------------
      let lastNodeId = await getCurrentNodeId(window);
      let lastText = await getDialogText(window);

      for (let i = 0; i < TURNS_TO_PLAY; i++) {
        const kind = CHOICE_PATTERN[i] as "B" | "L" | "F";
        const startedAt = Date.now();

        if (kind === "F") {
          const text = FREE_TEXT_INPUTS[i % FREE_TEXT_INPUTS.length]!;
          turnLog.push(`turn ${i + 1}/${TURNS_TO_PLAY} kind=F text=${text!}`);
          await submitFreeText(window, text!);
        } else {
          // B = first, L = last. The button locator covers BOTH
          // option buttons (when node has options) AND the
          // continue glyph (when there are no options at the
          // end of a batch).
          await window
            .locator(".main-view .options button")
            .first()
            .waitFor({ state: "visible", timeout: PER_TURN_TIMEOUT_MS });
          const btnLoc =
            kind === "B"
              ? window.locator(".main-view .options button").first()
              : window.locator(".main-view .options button").last();
          await expect(btnLoc).toBeEnabled({ timeout: 30_000 });
          const btnText = (await btnLoc.textContent())?.trim() || "(continue)";
          turnLog.push(
            `turn ${i + 1}/${TURNS_TO_PLAY} kind=${kind} text=${btnText}`,
          );
          await btnLoc.click();
        }
        assertNoFatalError(`after turn ${i + 1} click/submit`);

        // Assert: node advances + new text lands within the budget.
        await expect
          .poll(
            async () => {
              // Tripwire BEFORE the slow polling work. A backend
              // error during this turn should kill the test in
              // the next poll cycle, not after the full
              // PER_TURN_TIMEOUT_MS budget expires.
              assertNoFatalError(`during turn ${i + 1} poll`);
              const nodeId = await getCurrentNodeId(window);
              const text = await getDialogText(window);
              if (nodeId === lastNodeId) return "node-not-advanced";
              if (!text || text === lastText) return "text-not-landed";
              return "ok";
            },
            {
              timeout: PER_TURN_TIMEOUT_MS,
              intervals: [500, 1_500, 3_000],
            },
          )
          .toBe("ok");

        lastNodeId = await getCurrentNodeId(window);
        lastText = await getDialogText(window);
        const elapsed = Date.now() - startedAt;
        turnLog.push(
          `  turn ${i + 1} ok elapsed=${elapsed}ms node=${lastNodeId}`,
        );
      }

      // Sanity: completed 20 turns, dialog still visible.
      await expect(window.getByTestId("main-view")).toBeVisible();
      const finalText = await getDialogText(window);
      expect(finalText).toBeTruthy();
    } finally {
      // Dump the turn log no matter what — without it, a failure
      // mid-run leaves no breadcrumbs.
      console.log("===== 20-move playthrough turn log =====");
      for (const line of turnLog) console.log(line);
      console.log("========================================");
      await electronApp.close();
      releaseLiveE2eLock();
    }
  });
});

// ---------- helpers ---------------------------------------------------------

/** Read the renderer's exposed ``__lucidium_current_node_id``. The
 *  state-patch handler in ``app/client.ts`` mirrors current_node_id
 *  to this global on every patch — making it the cheapest way for an
 *  e2e test to assert "the dialog actually advanced" without poking
 *  at the store directly. */
async function getCurrentNodeId(window: Page): Promise<string> {
  return await window.evaluate(() => {
    const w = window as unknown as { __lucidium_current_node_id?: string };
    return w.__lucidium_current_node_id ?? "";
  });
}

/** Read the visible dialog text from the InteractionPanel. The
 *  Typewriter renders into ``.text``; what we see is what the
 *  player sees. */
async function getDialogText(window: Page): Promise<string> {
  return await window.evaluate(() => {
    const el = document.querySelector(".main-view .text");
    return (el?.textContent ?? "").trim();
  });
}

/** Wait until the dialog text is non-empty AND different from the
 *  ``previous`` value. Used to confirm the typewriter has at least
 *  started rendering the new beat. ``assertNoFatalError`` is the
 *  per-test tripwire — fatal errors during the poll abort early
 *  rather than stalling the full timeout. */
async function waitForBeatText(
  window: Page,
  previous: string,
  assertNoFatalError: (where: string) => void,
): Promise<void> {
  await expect
    .poll(
      async () => {
        assertNoFatalError("during waitForBeatText poll");
        const t = await getDialogText(window);
        if (!t || t === previous) return "pending";
        return "ok";
      },
      { timeout: 5 * 60_000, intervals: [500, 1_000, 2_000] },
    )
    .toBe("ok");
}

/** Submit a free-text input into the foreground InteractionPanel. */
async function submitFreeText(window: Page, text: string): Promise<void> {
  const input = window.locator(".main-view .free-text input");
  await input.waitFor({ state: "visible", timeout: PER_TURN_TIMEOUT_MS });
  await expect(input).toBeEnabled({ timeout: 30_000 });
  await input.fill(text);
  // The Submit button is in the same .free-text container; click
  // it rather than relying on Enter so the test exercises the
  // explicit-click path. (Enter also submits — covered by the
  // unit tests in InteractionPanel.)
  await window
    .locator(".main-view .free-text button")
    .filter({ hasText: "Submit" })
    .click();
}

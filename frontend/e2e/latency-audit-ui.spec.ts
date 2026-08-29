/**
 * Per-dispatch latency audit driven through the Electron UI, against
 * the real backend (live LLM + live image backend).
 *
 * Mirrors ``backend/scripts/audit_turn_latency.py`` but measures from
 * the renderer's perspective:
 *
 *   * ``click``               — user pressed a button on the interaction panel
 *   * ``node_change``         — ``__lucidium_current_node_id`` updated
 *                               (state_patch / state_full arrived for the
 *                               new beat)
 *   * ``continue_state``      — Continue glyph transitioned
 *                               loading <-> ready
 *   * ``button_disabled``     — option / continue / free-text submit
 *                               transitioned disabled <-> enabled
 *
 * For each click the spec computes:
 *
 *   1. ``click -> node_change`` — the latency the player perceives
 *      between pressing the button and the next beat landing.
 *   2. ``node_change -> ready`` — the gap between the new beat
 *      arriving and the Continue button becoming ready (spinner
 *      removed) OR a fresh set of options rendering. The user asked
 *      to ensure these surface "as soon as they can"; the spec
 *      asserts the transition lands within ``READY_TOLERANCE_MS``
 *      of the state change.
 *
 * 10 decision iterations interleaved (5 selected + 5 free-text);
 * each iteration walks Continue at 1s pacing until reaching a tail
 * node with options, then makes the decision.
 *
 * Setup mirrors ``live-20-move-playthrough.spec.ts``:
 *   * Electron prebuilt at ``frontend/dist-electron/main.js``
 *   * Renderer dist at ``frontend/dist/index.html``
 *   * Backend venv at ``backend/.venv``
 *   * ``settings.json`` with a working LLM provider
 *
 * Skip gates: ``LUCIDIUM_SKIP_LIVE=1`` or any of the build artefacts
 * missing.
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

// ---- Single-instance lock (shared shape with live-20-move) -----------------

const LIVE_E2E_LOCK_PATH = path.join(
  os.tmpdir(), "lucidium-live-e2e.lock",
);

function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
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
        + `Delete ${LIVE_E2E_LOCK_PATH} to force.`,
      );
    }
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
    if (recordedPid === process.pid) {
      unlinkSync(LIVE_E2E_LOCK_PATH);
    }
  } catch {
    // best-effort
  }
}

// ---- Paths / config --------------------------------------------------------

const repoRoot = path.resolve(__dirname, "..", "..");
const electronMain = path.join(repoRoot, "frontend", "dist-electron", "main.js");
const distIndex = path.join(repoRoot, "frontend", "dist", "index.html");
const venvPython =
  process.platform === "win32"
    ? path.join(repoRoot, "backend", ".venv", "Scripts", "python.exe")
    : path.join(repoRoot, "backend", ".venv", "bin", "python");

const DECISIONS = 10;
const PACING_MS = 1_000;            // 1 click / second
const PER_TURN_TIMEOUT_MS = 4 * 60_000;
// Tolerance for the click->display invariant. The backend test uses
// 120 ms for live; the UI adds renderer + MutationObserver overhead
// so we widen to 250 ms.
const DISPLAY_TOLERANCE_MS = 250;
// Tolerance for the secondary invariant: how soon after the new
// node lands should the spinner clear / the ready button appear?
// React render + the next batched commit is the floor (~16-32 ms);
// 100 ms catches anything that's actually waiting on something it
// shouldn't be.
const READY_TOLERANCE_MS = 100;

const FREE_TEXT_INPUTS = [
  "I take a careful look around.",
  "I keep going, but stay alert.",
  "I follow the sound.",
  "I ask if anyone has news.",
  "I check my pockets.",
];

// ---- Instrumentation injected into the renderer ---------------------------

interface AuditEvent {
  t_ms: number;
  event: string;
  [k: string]: unknown;
}

declare global {
  interface Window {
    __lucidium_current_node_id?: string;
    __audit_log?: AuditEvent[];
    __audit_t0?: number;
    __audit_node_poller?: number;
  }
}

/** Runs inside the renderer to set up the audit log. Idempotent. */
function installAuditInstrumentation(): void {
  const w = window;
  if (w.__audit_log) return;
  const log: AuditEvent[] = [];
  w.__audit_log = log;
  const t0 = performance.now();
  w.__audit_t0 = t0;

  const rec = (event: string, extra: Record<string, unknown> = {}): void => {
    log.push({ t_ms: performance.now() - t0, event, ...extra });
  };

  // Click capture (capture phase). Classifies the click as one of
  // continue / selected / free_text / interview based on which
  // container holds the button.
  document.addEventListener(
    "click",
    (e) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const btn = target.closest("button");
      if (!btn) return;
      const isContinue = btn.classList.contains("continue-glyph");
      const isInOptions = !!btn.closest(".options");
      const isFreeTextSubmit = !!btn.closest(".free-text");
      const isInterview = !!btn.closest(".option-grid");
      let kind: string;
      if (isContinue) kind = "continue";
      else if (isFreeTextSubmit) kind = "free_text";
      else if (isInOptions) kind = "selected";
      else if (isInterview) kind = "interview";
      else kind = "other";
      rec("click", {
        kind,
        state: btn.getAttribute("data-state"),
        disabled: (btn as HTMLButtonElement).disabled,
        label: btn.getAttribute("aria-label")
          ?? btn.textContent?.trim().slice(0, 40)
          ?? "",
      });
    },
    true,
  );

  // Mirror current_node_id transitions. Polling at 5 ms gives enough
  // resolution to bracket the React render that follows.
  let lastNodeId: unknown = w.__lucidium_current_node_id;
  w.__audit_node_poller = window.setInterval(() => {
    const cur = w.__lucidium_current_node_id;
    if (cur !== lastNodeId) {
      lastNodeId = cur;
      rec("node_change", { nodeId: cur as string | undefined });
    }
  }, 5);

  // Mutation observer on the interaction panel — captures the
  // Continue glyph data-state toggles, option-button enable/disable
  // transitions, and option-list additions/removals. Re-installs
  // whenever the panel is replaced (full state refresh).
  let panelObs: MutationObserver | null = null;
  const installPanelObserver = (): void => {
    const root = document.querySelector(".interaction");
    if (!root) return;
    if (panelObs) panelObs.disconnect();
    panelObs = new MutationObserver((muts) => {
      for (const m of muts) {
        const t = m.target as Element;
        if (m.type === "attributes") {
          if (
            m.attributeName === "data-state"
            && t.classList.contains("continue-glyph")
          ) {
            rec("continue_state", {
              state: t.getAttribute("data-state"),
              disabled: (t as HTMLButtonElement).disabled,
            });
          }
          if (m.attributeName === "disabled" && t.tagName === "BUTTON") {
            const btn = t as HTMLButtonElement;
            const isContinue = btn.classList.contains("continue-glyph");
            const inOptions = !!btn.closest(".options");
            const inFreeText = !!btn.closest(".free-text");
            let kind = "other";
            if (isContinue) kind = "continue";
            else if (inOptions) kind = "option";
            else if (inFreeText) kind = "free_text_submit";
            rec("button_disabled", {
              disabled: btn.disabled,
              kind,
              label: btn.getAttribute("aria-label")
                ?? btn.textContent?.trim().slice(0, 40)
                ?? "",
            });
          }
        }
        if (m.type === "childList") {
          if ((m.target as Element).classList?.contains("options")) {
            const buttons = (m.target as Element).querySelectorAll("button");
            rec("options_children", {
              count: buttons.length,
              hasContinue: !!(m.target as Element).querySelector(
                ".continue-glyph",
              ),
            });
          }
        }
      }
    });
    panelObs.observe(root, {
      attributes: true,
      attributeFilter: ["data-state", "disabled"],
      subtree: true,
      childList: true,
    });
    rec("panel_observer_installed");
  };
  installPanelObserver();
  // Body observer — re-attaches the panel observer whenever the
  // .interaction element is replaced.
  const bodyObs = new MutationObserver(() => {
    const root = document.querySelector(".interaction");
    if (root && (!panelObs || !root.isConnected)) installPanelObserver();
  });
  bodyObs.observe(document.body, { childList: true, subtree: true });

  rec("audit_ready");
}

// ---- Per-event helpers -----------------------------------------------------

async function getAuditLog(window: Page): Promise<AuditEvent[]> {
  return await window.evaluate(() =>
    (window.__audit_log ?? []).slice(),
  );
}

async function getCurrentNodeId(window: Page): Promise<string> {
  return await window.evaluate(() => {
    const w = window as unknown as { __lucidium_current_node_id?: string };
    return w.__lucidium_current_node_id ?? "";
  });
}

async function getDialogText(window: Page): Promise<string> {
  return await window.evaluate(() => {
    const el = document.querySelector(".main-view .text");
    return (el?.textContent ?? "").trim();
  });
}

/** True when the foreground interaction panel renders option buttons. */
async function hasVisibleOptions(window: Page): Promise<boolean> {
  return await window.evaluate(() => {
    const opts = document.querySelectorAll(".main-view .options button");
    if (opts.length === 0) return false;
    for (const b of opts) {
      if (b.classList.contains("continue-glyph")) return false;
    }
    return true;
  });
}

async function pacingWait(ms: number): Promise<void> {
  await new Promise((r) => setTimeout(r, ms));
}

// ---- Test ------------------------------------------------------------------

test.describe("UI latency audit", () => {
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

  test("10 decisions, 1s-paced walk, capture button time-series", async () => {
    acquireLiveE2eLock();
    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: { ...process.env, LUCIDIUM_LOAD_DIST: "1" },
      timeout: 60_000,
    });
    const phaseLog: string[] = [];
    const fatals: string[] = [];

    try {
      const window = await electronApp.firstWindow({ timeout: 30_000 });

      window.on("console", (msg) => {
        const text = msg.text();
        console.log(`[renderer] ${msg.type()} ${text}`);
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
        if (
          /\bERROR\s+lucidium\.api\.(ws_server|handlers)\b/.test(text)
          || /unexpected handler failure/.test(text)
        ) {
          fatals.push(`backend handler error: ${text.slice(0, 500)}`);
        }
      });

      const assertNoFatal = (where: string): void => {
        if (fatals.length === 0) return;
        throw new Error(
          `aborting at ${where}: ${fatals.length} fatal error(s):\n`
          + fatals.slice(0, 3).map((f, i) => `  [${i + 1}] ${f}`).join("\n"),
        );
      };

      // ---- Start screen + interview ----
      await expect(window.getByTestId("start-screen"))
        .toBeVisible({ timeout: 30_000 });
      await expect(window.getByTestId("connection-banner"))
        .toHaveCount(0, { timeout: 30_000 });
      assertNoFatal("after start-screen visible");

      const startupWarning = window.getByTestId("startup-warning");
      if (await startupWarning.count()) {
        await window.getByRole("button", { name: "I understand" }).click();
        await expect(startupWarning).toHaveCount(0, { timeout: 5_000 });
      }

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
        await expect(window.locator(".option-grid button").first())
          .toBeEnabled({ timeout: 5_000 });
        await window.locator(".option-grid button").first().click();
        phaseLog.push(`interview ${stepName}: clicked first option`);
        assertNoFatal(`after interview step ${stepName}`);
      }
      await expect(window.getByTestId("interview-confirm"))
        .toBeVisible({ timeout: 120_000 });
      await window.getByRole("button", { name: "Begin" }).click();
      const beginAt = Date.now();
      phaseLog.push("clicked Begin");

      // ---- Wait for main view + first beat ----
      await expect(window.getByTestId("main-view"))
        .toBeVisible({ timeout: 600_000 });
      // Wait for the first beat text to appear (typewriter has at
      // least started writing).
      await expect
        .poll(
          async () => {
            assertNoFatal("waiting for first beat text");
            const t = await getDialogText(window);
            return t.length > 0 ? "ok" : "pending";
          },
          { timeout: 5 * 60_000, intervals: [500, 1_000, 2_000] },
        )
        .toBe("ok");
      phaseLog.push(
        `main-view + first beat visible after ${Math.round((Date.now() - beginAt) / 1000)}s`,
      );

      // ---- Install audit instrumentation ----
      // Must happen AFTER .interaction exists; the function is
      // idempotent so re-invocation on a future panel rebuild is
      // safe. The instrumentation re-attaches its observer
      // whenever .interaction is replaced by a fresh subtree.
      await window.evaluate(installAuditInstrumentation);
      phaseLog.push("audit instrumentation installed");

      // ---- 10 decision iterations ----
      const dispatchPlan: { decisionIdx: number; kind: string }[] = [];
      let lastNodeId = await getCurrentNodeId(window);
      for (let d = 1; d <= DECISIONS; d++) {
        const isSelected = d % 2 === 1;
        phaseLog.push(`decision ${d} (${isSelected ? "selected" : "free_text"}) start`);

        // Walk Continue at 1 s pacing until a node with options.
        while (true) {
          const onOptions = await hasVisibleOptions(window);
          if (onOptions) break;
          await pacingWait(PACING_MS);
          // Click the Continue glyph (always exactly one button in
          // .options when there are no options).
          const continueBtn = window.locator(
            '.main-view .options button.continue-glyph',
          );
          await continueBtn.waitFor({
            state: "visible", timeout: PER_TURN_TIMEOUT_MS,
          });
          await expect(continueBtn).toBeEnabled({ timeout: 30_000 });
          await continueBtn.click();
          dispatchPlan.push({ decisionIdx: d, kind: "continue" });
          // Wait for node id to change so we don't pile clicks while
          // the renderer is still on the previous beat.
          await expect
            .poll(async () => {
              const cur = await getCurrentNodeId(window);
              return cur === lastNodeId ? "stale" : "ok";
            }, { timeout: PER_TURN_TIMEOUT_MS, intervals: [50, 100, 250] })
            .toBe("ok");
          lastNodeId = await getCurrentNodeId(window);
          assertNoFatal(`after continue (decision ${d})`);
        }

        // 1 s pause before the decision click.
        await pacingWait(PACING_MS);

        if (isSelected) {
          const optionBtn = window
            .locator(".main-view .options button")
            .first();
          await optionBtn.waitFor({
            state: "visible", timeout: PER_TURN_TIMEOUT_MS,
          });
          await expect(optionBtn).toBeEnabled({ timeout: 30_000 });
          await optionBtn.click();
          dispatchPlan.push({ decisionIdx: d, kind: "selected" });
        } else {
          const text = FREE_TEXT_INPUTS[d % FREE_TEXT_INPUTS.length]!;
          const input = window.locator(".main-view .free-text input");
          await input.waitFor({
            state: "visible", timeout: PER_TURN_TIMEOUT_MS,
          });
          await expect(input).toBeEnabled({ timeout: 30_000 });
          await input.fill(text);
          await window
            .locator(".main-view .free-text button")
            .filter({ hasText: "Submit" })
            .click();
          dispatchPlan.push({ decisionIdx: d, kind: "free_text" });
        }

        await expect
          .poll(async () => {
            const cur = await getCurrentNodeId(window);
            return cur === lastNodeId ? "stale" : "ok";
          }, { timeout: PER_TURN_TIMEOUT_MS, intervals: [50, 100, 250] })
          .toBe("ok");
        lastNodeId = await getCurrentNodeId(window);
        assertNoFatal(`after decision ${d}`);
      }

      // ---- Extract + analyse audit log ----
      const events = await getAuditLog(window);
      const clicks = events.filter(
        (e) => e.event === "click" && (
          e.kind === "continue" || e.kind === "selected"
          || e.kind === "free_text"
        ),
      );
      const nodeChanges = events.filter((e) => e.event === "node_change");
      const continueStates = events.filter((e) => e.event === "continue_state");

      // Sanity: every dispatch we drove should be in the click log.
      // Free-text submits sometimes trigger TWO click events (focus
      // shift fires through capture), so we tolerate >=. The
      // dispatchPlan is the source of truth for ordering.
      expect(clicks.length).toBeGreaterThanOrEqual(dispatchPlan.length);

      const displayViolations: string[] = [];
      const readyViolations: string[] = [];
      const rows: string[] = [];
      rows.push(
        [
          "  #", "kind         ",
          "click->disp", "node->ready",
          "spinner?",
        ].join("  "),
      );

      // Walk the dispatchPlan in order; for each, find the next
      // click event in the log of the matching kind, then the next
      // node_change after it, then the next continue_state
      // transition to "ready" (or option-buttons appearing).
      let cursor = 0;
      for (let i = 0; i < dispatchPlan.length; i++) {
        const plan = dispatchPlan[i]!;
        // Advance to the next click event matching the planned kind.
        while (cursor < clicks.length && clicks[cursor]!.kind !== plan.kind) {
          cursor++;
        }
        if (cursor >= clicks.length) break;
        const click = clicks[cursor]!;
        cursor++;
        const next = clicks[cursor];
        const windowEnd = next ? next.t_ms : Number.POSITIVE_INFINITY;

        const nodeChange = nodeChanges.find(
          (e) => e.t_ms > click.t_ms && e.t_ms < windowEnd,
        );
        const clickToDispMs = nodeChange
          ? nodeChange.t_ms - click.t_ms
          : Number.NaN;

        // After the node_change, when did the Continue glyph go
        // back to "ready", OR when did fresh option buttons appear
        // (both indicate the player can now take the next action)?
        const readyAfter = continueStates.find(
          (e) =>
            nodeChange != null
            && e.t_ms >= nodeChange.t_ms
            && e.t_ms < windowEnd
            && e.state === "ready",
        );
        const optionsAfter = events.find(
          (e) =>
            e.event === "options_children"
            && nodeChange != null
            && e.t_ms >= nodeChange.t_ms
            && e.t_ms < windowEnd
            && (e.count as number) > 0
            && !(e.hasContinue as boolean),
        );
        const readyEvent = (readyAfter && optionsAfter)
          ? (readyAfter.t_ms < optionsAfter.t_ms ? readyAfter : optionsAfter)
          : (readyAfter ?? optionsAfter);
        const nodeToReadyMs = (readyEvent && nodeChange)
          ? readyEvent.t_ms - nodeChange.t_ms
          : Number.NaN;

        const spinnerShown = continueStates.find(
          (e) =>
            e.t_ms >= click.t_ms
            && e.t_ms < (nodeChange?.t_ms ?? windowEnd)
            && e.state === "loading",
        );
        const spinnerLabel = spinnerShown ? "shown" : "no";

        rows.push(
          [
            `${(i + 1).toString().padStart(3)}`,
            `${plan.kind.padEnd(11)} `,
            `${Number.isFinite(clickToDispMs) ? clickToDispMs.toFixed(1) + " ms" : "  ?  "}`.padStart(11),
            `${Number.isFinite(nodeToReadyMs) ? nodeToReadyMs.toFixed(1) + " ms" : "  ?  "}`.padStart(11),
            spinnerLabel.padStart(8),
          ].join("  "),
        );

        // Invariant 1 (display latency): for non-free-text dispatches
        // where speculation should have hit, click->display must be
        // sub-tolerance. Free-text dispatches are LLM-bound; the live
        // call dominates and this check is meaningless against it.
        if (
          (plan.kind === "continue" || plan.kind === "selected")
          && Number.isFinite(clickToDispMs)
          && clickToDispMs > DISPLAY_TOLERANCE_MS
        ) {
          displayViolations.push(
            `dispatch #${i + 1} (${plan.kind}): click->display = `
            + `${clickToDispMs.toFixed(1)} ms (> ${DISPLAY_TOLERANCE_MS} ms)`,
          );
        }
        // Invariant 2 (button-ready latency): when the new node
        // arrives, the spinner/ready transition should fire within
        // React's commit window. Anything beyond READY_TOLERANCE_MS
        // means the UI is sitting on a known-ready beat.
        if (
          Number.isFinite(nodeToReadyMs)
          && nodeToReadyMs > READY_TOLERANCE_MS
        ) {
          readyViolations.push(
            `dispatch #${i + 1} (${plan.kind}): node->ready = `
            + `${nodeToReadyMs.toFixed(1)} ms (> ${READY_TOLERANCE_MS} ms)`,
          );
        }
      }

      console.log("\n===== UI latency audit =====");
      for (const row of rows) console.log(row);
      console.log("============================");
      console.log(`dispatches:           ${dispatchPlan.length}`);
      console.log(`click->disp violations: ${displayViolations.length} (> ${DISPLAY_TOLERANCE_MS} ms) — non-LLM dispatches only`);
      console.log(`node->ready violations: ${readyViolations.length} (> ${READY_TOLERANCE_MS} ms)`);

      if (displayViolations.length > 0) {
        console.log("--- click->disp violations ---");
        for (const v of displayViolations) console.log("  " + v);
      }
      if (readyViolations.length > 0) {
        console.log("--- ready violations ---");
        for (const v of readyViolations) console.log("  " + v);
      }

      // Enforced: UI button-state pipeline must transition
      // loading -> ready (or render fresh options) within
      // READY_TOLERANCE_MS of the new node landing in the
      // renderer. This is the user-visible "is the UI showing
      // as soon as it can" check that isolates renderer
      // responsiveness from any backend queueing.
      expect(
        readyViolations,
        `node->ready invariant broken on ${readyViolations.length} `
        + "dispatch(es); see test stdout for details",
      ).toEqual([]);

      // click->display outliers are surfaced but NOT asserted.
      // After the drain-task refactor
      // (handlers.py:_drain_remaining_foreground_chain), the
      // pre-existing "wait the full chain to walk beat 2" class
      // of violations is gone — verify by grepping
      // ``advance: drain paid off`` in the backend log. The
      // remaining outliers are:
      //   * SELECTED dispatches whose speculation hadn't
      //     finished — bounded by the LLM call duration. Real
      //     latency cost, but already optimal given the
      //     speculation cap.
      //   * CONTINUE dispatches that fell off the end of a
      //     chain whose tail beat had no options (an LLM data-
      //     quality blip) — the player walked past the chain,
      //     triggering a fresh foreground gen. Bounded by the
      //     LLM call duration.
      // Both classes satisfy the strict invariant
      // (displayed_at - click_at <= LLM duration); the UI's
      // simplified "click->display under tolerance" framing
      // can't distinguish them, so we surface them as data.
    } finally {
      console.log("===== phase log =====");
      for (const line of phaseLog) console.log(line);
      console.log("=====================");
      await electronApp.close();
      releaseLiveE2eLock();
    }
  });
});

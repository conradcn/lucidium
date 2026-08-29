/**
 * Speculation-cycle stress tests.
 *
 * The user reported the VN freezing during play. The backend's
 * speculative tree-generation pipeline (handlers.py:_speculate_branch /
 * _ensure_speculative_branches / _await_speculation) creates one
 * background task per option after every commit; if any of those tasks
 * hangs, falls into a race, or is awaited at the wrong moment, the
 * foreground play_advance handler can stall and the UI freezes.
 *
 * These tests drive multiple advance/free-text cycles through the
 * mocked WebSocket and assert two invariants that the freeze would
 * violate:
 *   1. Every player click reaches a NEW dialog node (state.current_node_id
 *      changes) within a small time budget.
 *   2. The interaction panel's option/Continue/Submit buttons never stay
 *      disabled past the next state/full snapshot — i.e., the optimistic
 *      guard always resets per-node.
 *
 * The mock keeps a per-node id sequence so successive `c2s/play/advance`
 * calls always produce a fresh, distinct node — exactly what real
 * speculation would deliver after the player picks each option.
 */

import { test, expect, type Page } from "@playwright/test";

import { APP_URL, installMockWs, installHappyPathMock } from "./helpers";

/* -------------------------------------------------------------------- */
/* Helpers                                                              */
/* -------------------------------------------------------------------- */

async function getCurrentNodeId(page: Page): Promise<string | null> {
  return page.evaluate(() => {
    const w = window as unknown as { __lucidium_current_node_id?: string | null };
    return w.__lucidium_current_node_id ?? null;
  });
}

async function waitForNodeChange(
  page: Page,
  previous: string | null,
  timeout = 3000,
): Promise<string> {
  await page.waitForFunction(
    (prev) => {
      const w = window as unknown as { __lucidium_current_node_id?: string | null };
      return w.__lucidium_current_node_id != null && w.__lucidium_current_node_id !== prev;
    },
    previous,
    { timeout },
  );
  const next = await getCurrentNodeId(page);
  if (!next) throw new Error("expected a non-null current_node_id after advance");
  return next;
}

async function countSent(page: Page, type: string): Promise<number> {
  return page.evaluate(
    (t) => window.__lucidium.sent.filter((m) => m.type === t).length,
    type,
  );
}

/* -------------------------------------------------------------------- */
/* Mock setup — chain-walking play_advance                              */
/* -------------------------------------------------------------------- */

interface ChainOptions {
  /** Milliseconds to delay each c2s/play/advance reply. 0 = instant. */
  advanceDelayMs?: number;
  /**
   * Inject extra speculative state/full updates that arrive *between*
   * the player's clicks. Simulates the backend pushing speculative
   * chains in flight.
   */
  speculationDelayMs?: number;
}

/**
 * Install a mock that lets the player advance indefinitely. Each
 * c2s/play/advance call moves the current_node_id to a freshly-minted
 * id and includes that node in the dialog_tree with two options. A
 * c2s/play/free_text call does the same. Both echo the player's pick
 * back through the standard s2c sequence (state/patch + state/full).
 */
async function installInfiniteChainMock(
  page: Page,
  opts: ChainOptions = {},
): Promise<void> {
  await installMockWs(page);
  const advanceDelayMs = opts.advanceDelayMs ?? 0;
  const speculationDelayMs = opts.speculationDelayMs ?? 0;

  const baseGameJson = JSON.stringify(makeGameSnapshot("n0"));
  const settingsJson = JSON.stringify(BASE_SETTINGS);

  await page.addInitScript({
    content: `
(() => {
  const advanceDelayMs = ${advanceDelayMs};
  const speculationDelayMs = ${speculationDelayMs};
  let turn = 0;
  let lastNodeId = "n0";
  const settings = ${settingsJson};
  const baseGame = ${baseGameJson};

  function buildGame(currentId, optionIds) {
    const game = JSON.parse(JSON.stringify(baseGame));
    game.dialog_tree.nodes[currentId] = {
      id: currentId,
      parent_id: lastNodeId === currentId ? null : lastNodeId,
      chosen_option_id: null,
      speaker_id: null,
      text: "Beat for " + currentId + ".",
      options: optionIds.map((oid) => ({ id: oid, text: "Option " + oid })),
      entering_character_ids: [],
      leaving_character_ids: [],
      new_characters: [],
      location_id: null,
      location_prompt: null,
      character_changes: [],
      state: "committed",
      premise_hash: "h-" + currentId,
      generation_metadata: { model: null, prompt_hash: null, seed_parameters: {}, tokens_in: 0, tokens_out: 0, latency_ms: 0 },
    };
    game.current_node_id = currentId;
    game.dialog_tree.committed_path = [...game.dialog_tree.committed_path, currentId]
      .filter((id, i, arr) => arr.indexOf(id) === i);
    return game;
  }

  // Pre-seed the dialog tree with the opening node (n0) so c2s/saves/continue
  // returns a sensible game.
  baseGame.dialog_tree.nodes["n0"] = {
    id: "n0",
    parent_id: null,
    chosen_option_id: null,
    speaker_id: null,
    text: "The harbor wakes slow.",
    options: [{ id: "n0-a", text: "Walk to the inn." }, { id: "n0-b", text: "Climb the cliff." }],
    entering_character_ids: [], leaving_character_ids: [], new_characters: [],
    location_id: null, location_prompt: null, character_changes: [],
    state: "committed", premise_hash: "h-n0",
    generation_metadata: { model: null, prompt_hash: null, seed_parameters: {}, tokens_in: 0, tokens_out: 0, latency_ms: 0 },
  };
  baseGame.dialog_tree.committed_path = ["n0"];
  baseGame.current_node_id = "n0";

  function nextStepReplies() {
    turn += 1;
    const nodeId = "n" + turn;
    const optAId = "n" + turn + "-a";
    const optBId = "n" + turn + "-b";
    lastNodeId = nodeId;
    const game = buildGame(nodeId, [optAId, optBId]);
    return [
      { type: "s2c/state/patch", payload: { ops: [{ op: "replace", path: "/current_node_id", value: nodeId }] } },
      { type: "s2c/text/streaming", payload: { node_id: nodeId, delta: game.dialog_tree.nodes[nodeId].text } },
      { type: "s2c/text/complete", payload: { node_id: nodeId } },
      { type: "s2c/state/full", payload: { game, settings } },
    ];
  }

  function delayedReplies(replies, delay) {
    if (delay <= 0) return replies;
    // Wrap each reply with a setTimeout of the given delay — applied
    // by mock-ws.js via the existing 5ms cadence; we extend that.
    return replies.map((r, i) => ({ ...r, __delayMs: delay + i * 5 }));
  }

  window.__lucidium.handlers["c2s/hello"] = function () {
    return [{ type: "s2c/hello", payload: { protocol_version: 1, has_save: true } }];
  };
  window.__lucidium.handlers["c2s/saves/list"] = function () {
    return [{ type: "s2c/saves/list", payload: { saves: [{
      id: "save-1", name: "Test", last_played_at: "2026-05-01T00:00:00Z",
      created_at: "2026-05-01T00:00:00Z", schema_version: 1, summary: "test"
    }] } }];
  };
  window.__lucidium.handlers["c2s/saves/continue"] = function () {
    return [{ type: "s2c/state/full", payload: { game: baseGame, settings } }];
  };
  window.__lucidium.handlers["c2s/saves/load"] = function () {
    return [{ type: "s2c/state/full", payload: { game: baseGame, settings } }];
  };
  window.__lucidium.handlers["c2s/play/advance"] = function () {
    const replies = nextStepReplies();
    const out = delayedReplies(replies, advanceDelayMs);

    // Simulate the backend kicking off speculative branches: a delayed
    // unsolicited state/full that adds a speculative child to the tree.
    if (speculationDelayMs > 0) {
      setTimeout(() => {
        const game = JSON.parse(JSON.stringify(baseGame));
        // No-op speculation push — the renderer should ignore old
        // node ids that don't match its current_node_id.
        window.__lucidium.pushFromServer({
          type: "s2c/state/full",
          payload: { game, settings }
        });
      }, speculationDelayMs);
    }
    return out;
  };
  window.__lucidium.handlers["c2s/play/free_text"] = function (payload) {
    return delayedReplies(nextStepReplies(), advanceDelayMs);
  };
  window.__lucidium.handlers["c2s/settings/get"] = function () {
    return [{ type: "s2c/state/patch", payload: { ops: [{ op: "replace", path: "/settings", value: settings }] } }];
  };

  // ---- New-Game interview path -------------------------------------
  // The renderer drives the player through five steps: Setting,
  // Genre, Visual Style, Character Description, Name, then Confirm.
  // c2s/new_game/start surfaces hardcoded setting options. Each
  // c2s/new_game/answer transitions to the next step; the final
  // answer (name) hands the player to confirm. c2s/new_game/confirm
  // emits a state/full that drops the player on n0 — the same
  // baseGame the chain-walking advance handler operates on.
  window.__lucidium.handlers["c2s/new_game/start"] = function () {
    return [{
      type: "s2c/state/patch",
      payload: { ops: [{
        op: "replace", path: "/interview",
        value: { step: "setting", setting_options: ["stone harbor", "neon city", "deep wood"] },
      }] },
    }];
  };
  // Reordered interview: setting → visual_style → genre → char → name → confirm.
  const stepNext = {
    setting: ["visual_style", "/interview/visual_style_options", ["ink", "anime", "noir"]],
    visual_style: ["genre", "/interview/genre_options", ["Mystery", "Romance", "Horror"]],
    genre: ["character_description", "/interview/character_description_options", ["wry archivist"]],
    character_description: ["name", "/interview/name_options", ["Iris Vale", "Hale Stone"]],
    name: ["confirm", "", []],
  };
  window.__lucidium.handlers["c2s/new_game/answer"] = function (payload) {
    const step = payload.step;
    const nx = stepNext[step];
    if (!nx) return [];
    const fieldName = step === "name" ? "character_name" : step;
    const ops = [
      { op: "replace", path: "/interview/" + fieldName, value: payload.answer },
      { op: "replace", path: "/interview/step", value: nx[0] },
    ];
    if (nx[1]) ops.push({ op: "replace", path: nx[1], value: nx[2] });
    return [{ type: "s2c/state/patch", payload: { ops } }];
  };
  window.__lucidium.handlers["c2s/new_game/add_side_character"] = function () {
    return [];  // not used in these tests; just answer empty
  };
  window.__lucidium.handlers["c2s/new_game/confirm"] = function () {
    return [{ type: "s2c/state/full", payload: { game: baseGame, settings } }];
  };
})();
`,
  });
}

const BASE_SETTINGS = {
  llm: {
    base_url: "https://openrouter.ai/api/v1",
    model: "x",
    api_key: "",
    temperature: 0.0,
    max_tokens: 1024,
  },
  image: {
    base_url: "http://127.0.0.1:8000",
    portrait_workflow: "p",
    background_workflow: "b",
  },
  typewriter_speed_chars_per_sec: 1000,
  prompt_history_clamp_chars: 12000,
  concurrency: { llm_max_in_flight: 4, image_max_in_flight: 2 },
  // These specs model an already-configured install. main.tsx routes
  // to the first-run wizard whenever this flag is falsy (and a missing
  // field is falsy), so without it every flow below lands on "Welcome
  // to Lucidium" and the start-screen buttons are torn out from under
  // Playwright mid-click ("element was detached from the DOM"). Same
  // reasoning as the happy-path mock in e2e/helpers.ts.
  first_time_setup_complete: true,
};

function makeGameSnapshot(currentNodeId: string): Record<string, unknown> {
  return {
    id: "g1",
    schema_version: 1,
    world: {
      game_name: "Test",
      setting: "harbor",
      genre: "mystery",
      visual_style: "ink",
      overall_plot_direction: "",
      active_plot_threads: [],
      dropped_plot_threads: [],
      summarizer_assessment: "",
      prompt_history_clamp_chars: 12000,
      player_intent: {
        pace_preference: "same",
        tone_preference: "unspecified",
        direction_signal: "none",
        weighted_evidence: [],
      },
    },
    characters: {},
    dialog_tree: {
      nodes: {},
      root_id: currentNodeId,
      committed_path: [currentNodeId],
    },
    environments: {},
    current_node_id: currentNodeId,
    on_stage: [],
    cost_telemetry: {
      tokens_in: 0,
      tokens_out: 0,
      image_calls: 0,
      llm_calls: 0,
      latency_ms_total: 0,
      dollar_estimate: 0,
    },
  };
}

/* -------------------------------------------------------------------- */
/* Entry flows                                                          */
/* -------------------------------------------------------------------- */

type EntryFlow = "continue" | "new-game" | "load-game";

/** Drive the renderer from the start screen to a main-view on n0,
 *  using one of the three entry paths a real player might take.
 *  Returns once main-view is visible and current_node_id is "n0".
 */
async function enterMainView(page: Page, flow: EntryFlow): Promise<void> {
  if (flow === "continue") {
    await page.getByRole("button", { name: "Continue" }).click();
  } else if (flow === "load-game") {
    await page.getByRole("button", { name: "Load Game" }).click();
    await expect(page.getByTestId("load-game-screen")).toBeVisible();
    // The mock returns one save with name "Test"; click its Load.
    await page.getByRole("button", { name: "Load" }).first().click();
  } else {
    // new-game: walk the 5 interview answers (new ordering: setting →
    // visual_style → genre → character_description → name → confirm).
    await page.getByRole("button", { name: "New Game" }).click();
    await expect(page.getByTestId("interview-setting")).toBeVisible({ timeout: 5000 });
    await page.getByRole("button", { name: "stone harbor" }).click();
    await expect(page.getByTestId("interview-visual_style")).toBeVisible();
    await page.getByRole("button", { name: "ink" }).click();
    await expect(page.getByTestId("interview-genre")).toBeVisible();
    await page.getByRole("button", { name: "Mystery", exact: true }).click();
    await expect(page.getByTestId("interview-character_description")).toBeVisible();
    await page.getByRole("button", { name: "wry archivist" }).click();
    await expect(page.getByTestId("interview-name")).toBeVisible();
    await page.getByRole("button", { name: "Iris Vale" }).click();
    await expect(page.getByTestId("interview-confirm")).toBeVisible();
    await page.getByRole("button", { name: "Begin" }).click();
  }
  await expect(page.getByTestId("main-view")).toBeVisible({ timeout: 10_000 });
  // Wait until the renderer has the actual game state on n0 — without
  // this the first option click can race the state/full and target
  // a placeholder.
  await page.waitForFunction(
    () => {
      const w = window as unknown as { __lucidium_current_node_id?: string | null };
      return w.__lucidium_current_node_id === "n0";
    },
    { timeout: 10_000 },
  );
}

/* -------------------------------------------------------------------- */
/* Tests                                                                */
/* -------------------------------------------------------------------- */

const FLOWS: EntryFlow[] = ["continue", "new-game", "load-game"];

for (const flow of FLOWS) {
  test.describe(`Speculation cycles — entered via ${flow}`, () => {
    test(`[${flow}] 10 consecutive option clicks each advance to a fresh node within 3s`, async ({ page }) => {
      await installInfiniteChainMock(page);
      await page.goto(APP_URL);
      await enterMainView(page, flow);

      let prevNodeId = await getCurrentNodeId(page);
      expect(prevNodeId).toBe("n0");

      for (let turn = 1; turn <= 10; turn++) {
        const optionLocator = page
          .locator(".main-view .options button")
          .first();
        await expect(optionLocator).toBeEnabled({ timeout: 3000 });
        await optionLocator.click();

        const nextId = await waitForNodeChange(page, prevNodeId, 3000);
        expect(nextId).toBe(`n${turn}`);
        prevNodeId = nextId;
      }
    });

    test(`[${flow}] UI never gets stuck with disabled buttons after rapid sequential advances`, async ({ page }) => {
      await installInfiniteChainMock(page);
      await page.goto(APP_URL);
      await enterMainView(page, flow);

      for (let turn = 0; turn < 6; turn++) {
        const previous = await getCurrentNodeId(page);
        const option = page.locator(".main-view .options button").first();
        await expect(option).toBeEnabled();
        await option.click();
        await waitForNodeChange(page, previous, 3000);
        const fresh = page.locator(".main-view .options button").first();
        await expect(fresh).toBeEnabled({ timeout: 3000 });
      }
    });

    test(`[${flow}] free-text mid-stream after a regular advance still produces a new node`, async ({ page }) => {
      await installInfiniteChainMock(page);
      await page.goto(APP_URL);
      await enterMainView(page, flow);

      let previous = await getCurrentNodeId(page);
      for (let i = 0; i < 2; i++) {
        await page.locator(".main-view .options button").first().click();
        previous = await waitForNodeChange(page, previous, 3000);
      }

      const input = page.locator(".main-view .free-text input");
      await input.fill("Iris pries up the floorboard.");
      await page.locator(".main-view .free-text button", { hasText: "Submit" }).click();
      await waitForNodeChange(page, previous, 3000);
    });

    test(`[${flow}] rapid double-click on the same option only fires one advance`, async ({ page }) => {
      await installInfiniteChainMock(page, { advanceDelayMs: 80 });
      await page.goto(APP_URL);
      await enterMainView(page, flow);

      const advancesBefore = await countSent(page, "c2s/play/advance");

      await page.evaluate(() => {
        const btn = document.querySelector(
          ".main-view .options button",
        ) as HTMLButtonElement | null;
        if (!btn) return 0;
        for (let i = 0; i < 5; i++) {
          try {
            btn.click();
          } catch {
            break;
          }
        }
        return 5;
      });

      await waitForNodeChange(page, "n0", 3000);

      const advances = await countSent(page, "c2s/play/advance");
      expect(advances - advancesBefore).toBe(1);
    });

    test(`[${flow}] speculation update arriving mid-advance does not freeze the UI`, async ({ page }) => {
      await installInfiniteChainMock(page, { speculationDelayMs: 100 });
      await page.goto(APP_URL);
      await enterMainView(page, flow);

      let previous = await getCurrentNodeId(page);
      for (let turn = 1; turn <= 5; turn++) {
        await page.locator(".main-view .options button").first().click();
        previous = await waitForNodeChange(page, previous, 3000);
        await page.waitForTimeout(150);
      }
    });

    test(`[${flow}] alternating option / free-text / option does not stall`, async ({ page }) => {
      await installInfiniteChainMock(page);
      await page.goto(APP_URL);
      await enterMainView(page, flow);

      let previous = await getCurrentNodeId(page);
      const sequence: Array<"option" | "freetext"> = [
        "option",
        "freetext",
        "option",
        "option",
        "freetext",
        "option",
      ];
      for (const action of sequence) {
        if (action === "option") {
          await page.locator(".main-view .options button").first().click();
        } else {
          await page.locator(".main-view .free-text input").fill("Iris waits a moment.");
          await page
            .locator(".main-view .free-text button", { hasText: "Submit" })
            .click();
        }
        previous = await waitForNodeChange(page, previous, 3000);
      }
    });

    test(`[${flow}] with a slow advance reply, the click-to-disabled-then-re-enabled cycle still completes`, async ({ page }) => {
      await installInfiniteChainMock(page, { advanceDelayMs: 500 });
      await page.goto(APP_URL);
      await enterMainView(page, flow);

      let previous = await getCurrentNodeId(page);
      for (let turn = 0; turn < 3; turn++) {
        const button = page.locator(".main-view .options button").first();
        await button.click();
        previous = await waitForNodeChange(page, previous, 5000);
        await expect(
          page.locator(".main-view .options button").first(),
        ).toBeEnabled({ timeout: 3000 });
      }
    });
  });
}

test.describe("Speculation cycles — happy-path mock smoke", () => {
  test("happy-path mock: 5 advances do not corrupt option-button state", async ({ page }) => {
    // Sanity check against the existing happy-path mock — the
    // current_node_id ping-pongs n1↔n2, but option buttons should
    // still toggle enabled/disabled cleanly across turns. If the
    // freeze were a renderer-level guard bug, this would catch it
    // even without speculation simulation.
    await installHappyPathMock(page, { hasSave: true });
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByTestId("main-view")).toBeVisible();

    for (let turn = 0; turn < 5; turn++) {
      const button = page.locator(".main-view .options button").first();
      await expect(button).toBeEnabled({ timeout: 3000 });
      await button.click();
      await page.waitForTimeout(200);
      await expect(
        page.locator(".main-view .options button").first(),
      ).toBeEnabled({ timeout: 3000 });
    }
  });
});

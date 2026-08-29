import { readFileSync } from "node:fs";
import path from "node:path";

import type { Page } from "@playwright/test";

const MOCK_SOURCE = readFileSync(path.join(__dirname, "mock-ws.js"), "utf-8");

/**
 * The app entry URL every browser-based spec should navigate to.
 *
 * Carries ``skipWarning=1``, the renderer's automation flag for the
 * startup safety modal (src/main.tsx). Without it the modal mounts as a
 * focus-trapping ``aria-modal`` overlay over the whole app and swallows
 * pointer events, so the first ``click()`` in a spec fails with
 * "<div role=dialog ...> intercepts pointer events" — or, worse, races
 * the modal's mount and passes intermittently.
 *
 * The Electron-based specs get the same behaviour by launching the app
 * with ``--skip-warning`` instead; main.ts turns that flag into this
 * query parameter.
 *
 * Specs that are ABOUT the warning must navigate to "/" instead, so
 * they see it. See startup-warning.spec.ts.
 */
export const APP_URL = "/?skipWarning=1";

/**
 * Install the WebSocket mock into the page before any other script
 * runs. ``addInitScript`` runs on every navigation, so this must be
 * called BEFORE ``page.goto`` in each test.
 */
export async function installMockWs(page: Page): Promise<void> {
  await page.addInitScript({ content: MOCK_SOURCE });
}

export interface MockGameOptions {
  hasSave?: boolean;
}

/**
 * Install the WebSocket mock + a complete happy-path handler registry
 * via an init script. Both run before any other page code, so the
 * renderer's first ``c2s/hello`` is answered correctly. Idempotent.
 */
export async function installHappyPathMock(
  page: Page,
  options: MockGameOptions = {},
): Promise<void> {
  const hasSave = options.hasSave ?? false;
  await installMockWs(page);
  await page.addInitScript({ content: buildHandlersScript(hasSave) });
}

/**
 * Replace a single handler at runtime. Use only after ``page.goto``.
 */
export async function setHandler(
  page: Page,
  type: string,
  body: string,
): Promise<void> {
  await page.evaluate(
    ({ type, body }) => {
       
      window.__lucidium.handlers[type] = new Function("payload", body) as never;
    },
    { type, body },
  );
}

/**
 * Push an unsolicited ``s2c/*`` event. Useful for asserting renderer
 * reactions to backend-pushed updates (image-ready, scheduler-status).
 */
export async function pushFromServer(
  page: Page,
  envelope: { type: string; payload: Record<string, unknown> },
): Promise<void> {
  await page.evaluate((env) => window.__lucidium.pushFromServer(env), envelope);
}

function buildHandlersScript(hasSave: boolean): string {
  // The script body is a top-level IIFE — it runs once on every
  // navigation, after mock-ws.ts has installed __lucidium.
  return `
(() => {
  const hasSave = ${JSON.stringify(hasSave)};
  const settings = {
    llm: {
      base_url: "https://openrouter.ai/api/v1",
      model: "qwen/qwen-2.5-72b-instruct",
      api_key: "",
      temperature: 0.8,
      max_tokens: 1024,
    },
    image: {
      base_url: "http://127.0.0.1:8000",
      portrait_workflow: "portrait.workflow.json",
      background_workflow: "background.workflow.json",
    },
    typewriter_speed_chars_per_sec: 60,
    prompt_history_clamp_chars: 12000,
    concurrency: { llm_max_in_flight: 4, image_max_in_flight: 2 },
    // The happy-path mock represents an ALREADY-CONFIGURED app. main.tsx
    // routes to the first-run wizard whenever this flag is falsy, and a
    // missing field is falsy - so without it every spec built on this
    // mock lands on "Welcome to Lucidium" instead of the start screen,
    // and the start-screen buttons get torn out from under Playwright
    // ("element was detached from the DOM"). Specs that WANT the wizard
    // override it with false in their own init script (see
    // first-time-setup-screenshot.spec.ts).
    //
    // NOTE: this object is emitted inside a template literal, so keep
    // backticks and dollar-brace interpolations out of these comments.
    first_time_setup_complete: true,
  };

  const openingGame = {
    id: "g1",
    schema_version: 1,
    world: {
      game_name: "Embers of the Veil",
      setting: "stone harbor at dawn",
      genre: "occult mystery",
      visual_style: "ink wash",
      overall_plot_direction: "Find the missing archivist before the eclipse.",
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
    characters: {
      c1: {
        id: "c1",
        is_player: true,
        name: "Iris",
        description: "wry archivist",
        gender: "female",
        age: 28,
        ethnicity: "local",
        skin: "pale",
        hair_color: "auburn",
        hairstyle: "braid",
        eye_color: "grey",
        build: "slight",
        bust: "moderate",
        outfit: "wool coat",
        pose: "standing",
        expression: "alert",
        facts: [],
        images: [],
        seed: 42,
      },
      c2: {
        id: "c2",
        is_player: false,
        name: "Hale",
        description: "weathered keeper",
        gender: "male",
        age: 52,
        ethnicity: "local",
        skin: "tan",
        hair_color: "grey",
        hairstyle: "short",
        eye_color: "brown",
        build: "stocky",
        bust: "n/a",
        outfit: "oilskin",
        pose: "leaning",
        expression: "watchful",
        facts: [],
        // No images yet — exercises the placeholder silhouette path
        // so screenshots reflect the asset-pending state every run
        // starts in (portrait generation is best-effort/async).
        images: [],
        seed: 7,
      },
    },
    dialog_tree: {
      nodes: {
        n1: {
          id: "n1",
          parent_id: null,
          chosen_option_id: null,
          speaker_id: null,
          text: "The harbor wakes slow.",
          options: [
            { id: "o1", text: "Walk to the archive." },
            { id: "o2", text: "Find Hale at the inn." },
          ],
          entering_character_ids: [],
          leaving_character_ids: [],
          new_character_descriptors: {},
          location_id: null,
          location_prompt: null,
          character_changes: [],
          state: "committed",
          premise_hash: "h0",
          generation_metadata: {
            model: null,
            prompt_hash: null,
            seed_parameters: {},
            tokens_in: 0,
            tokens_out: 0,
            latency_ms: 0,
          },
        },
      },
      root_id: "n1",
      committed_path: ["n1"],
    },
    environments: {},
    current_node_id: "n1",
    // Per FR-034a the player (c1) is NOT a stage actor; only c2 (Hale)
    // gets rendered on-stage in the mock fixtures.
    on_stage: ["c2"],
    cost_telemetry: {
      tokens_in: 0,
      tokens_out: 0,
      image_calls: 0,
      llm_calls: 0,
      latency_ms_total: 0,
      dollar_estimate: 0,
    },
  };

  const advancedGame = JSON.parse(JSON.stringify(openingGame));
  advancedGame.dialog_tree.nodes.n2 = {
    id: "n2",
    parent_id: "n1",
    chosen_option_id: "o1",
    speaker_id: null,
    text: "She turns the brass key in the archive door.",
    options: [],
    entering_character_ids: [],
    leaving_character_ids: [],
    new_character_descriptors: {},
    location_id: null,
    location_prompt: null,
    character_changes: [],
    state: "committed",
    premise_hash: "h1",
    generation_metadata: {
      model: "qwen/qwen-2.5-72b-instruct",
      prompt_hash: null,
      seed_parameters: { temperature: 0.8 },
      tokens_in: 0,
      tokens_out: 0,
      latency_ms: 0,
    },
  };
  advancedGame.dialog_tree.committed_path = ["n1", "n2"];
  advancedGame.current_node_id = "n2";

  // New step order (FR-027b): Setting → Visual Style → Genre →
  // Character Description → Name → Confirm. Genre is hard-coded so
  // its options ride along with the visual_style answer; the LLM-
  // driven steps each carry options that the real backend would
  // either include inline (prefetch already done) or emit later via
  // a follow-up state/patch (prefetch still in flight). The mock
  // returns inline options here — see the slow-LLM specs for the
  // deferred case.
  const stepNext = {
    setting: ["visual_style", "/interview/visual_style_options", ["ink", "anime", "noir"]],
    visual_style: [
      "genre",
      "/interview/genre_options",
      [
        "Power fantasy", "Coming of age", "Slice of life", "Tragedy",
        "Romance", "Comedy", "Mystery", "Thriller", "Horror", "Noir",
        "Wholesome", "Bittersweet", "Heroic", "Existential", "Dark academia",
        "Cozy",
      ],
    ],
    genre: [
      "character_description",
      "/interview/character_description_options",
      ["wry archivist", "retired bounty hunter"],
    ],
    character_description: [
      "name",
      "/interview/name_options",
      ["Iris Vale", "Hale Stone", "Mira Quill"],
    ],
    name: ["confirm", "", []],
  };

  window.__lucidium.handlers = {
    "c2s/hello": function () {
      return [{ type: "s2c/hello", payload: { protocol_version: 1, has_save: hasSave } }];
    },
    "c2s/saves/list": function () {
      return [{ type: "s2c/saves/list", payload: { saves: [] } }];
    },
    "c2s/saves/continue": function () {
      return [{ type: "s2c/state/full", payload: { game: openingGame, settings } }];
    },
    "c2s/saves/load": function () {
      return [{ type: "s2c/state/full", payload: { game: openingGame, settings } }];
    },
    "c2s/saves/rename": function () {
      return [{ type: "s2c/saves/list", payload: { saves: [] } }];
    },
    "c2s/saves/delete": function () {
      return [{ type: "s2c/saves/list", payload: { saves: [] } }];
    },
    "c2s/new_game/start": function () {
      return [
        {
          type: "s2c/state/patch",
          payload: {
            ops: [
              {
                op: "replace",
                path: "/interview/setting_options",
                value: ["stone harbor", "neon city", "deep wood"],
              },
            ],
          },
        },
      ];
    },
    "c2s/new_game/answer": function (payload) {
      const step = payload.step;
      const next = stepNext[step];
      if (!next) return [];
      // Backend stores the player's name under "character_name", not
      // "name" — keep the mock aligned so the Review step renders it.
      const fieldName = step === "name" ? "character_name" : step;
      const ops = [
        { op: "replace", path: "/interview/" + fieldName, value: payload.answer },
        { op: "replace", path: "/interview/step", value: next[0] },
      ];
      if (next[1]) ops.push({ op: "replace", path: next[1], value: next[2] });
      return [{ type: "s2c/state/patch", payload: { ops } }];
    },
    "c2s/new_game/add_side_character": function () {
      return [
        {
          type: "s2c/state/patch",
          payload: {
            ops: [
              {
                op: "add",
                path: "/interview/side_characters/sc1",
                value: { id: "sc1", name: "Hale Stone" },
              },
            ],
          },
        },
      ];
    },
    "c2s/new_game/confirm": function () {
      return [{ type: "s2c/state/full", payload: { game: openingGame, settings } }];
    },
    "c2s/play/advance": function () {
      return [
        {
          type: "s2c/state/patch",
          payload: { ops: [{ op: "replace", path: "/current_node_id", value: "n2" }] },
        },
        {
          type: "s2c/text/streaming",
          payload: { node_id: "n2", delta: "She turns the brass key in the archive door." },
        },
        { type: "s2c/text/complete", payload: { node_id: "n2" } },
        { type: "s2c/state/full", payload: { game: advancedGame, settings } },
      ];
    },
    "c2s/play/free_text": function () {
      return [
        {
          type: "s2c/state/patch",
          payload: { ops: [{ op: "replace", path: "/current_node_id", value: "n1" }] },
        },
        { type: "s2c/text/complete", payload: { node_id: "n1" } },
      ];
    },
    "c2s/edit/world": function (payload) {
      return [
        {
          type: "s2c/state/patch",
          payload: {
            ops: [{ op: "replace", path: "/world/" + payload.field, value: payload.value }],
          },
        },
      ];
    },
    "c2s/edit/character": function (payload) {
      return [
        {
          type: "s2c/state/patch",
          payload: {
            ops: [
              {
                op: "replace",
                path: "/characters/" + payload.character_id + "/" + payload.field,
                value: payload.value,
              },
            ],
          },
        },
      ];
    },
    "c2s/edit/history": function (payload) {
      return [
        {
          type: "s2c/state/patch",
          payload: {
            ops: [
              {
                op: "replace",
                path: "/dialog_tree/nodes/" + payload.node_id + "/text",
                value: payload.new_text,
              },
            ],
          },
        },
      ];
    },
    "c2s/edit/environment": function (payload) {
      return [
        {
          type: "s2c/state/patch",
          payload: {
            ops: [
              {
                op: "replace",
                path: "/environments/" + payload.environment_id + "/" + payload.field,
                value: payload.value,
              },
            ],
          },
        },
      ];
    },
    "c2s/settings/get": function () {
      return [
        {
          type: "s2c/state/patch",
          payload: { ops: [{ op: "replace", path: "/settings", value: settings }] },
        },
      ];
    },
    "c2s/settings/update": function () {
      return [
        {
          type: "s2c/state/patch",
          payload: { ops: [{ op: "replace", path: "/settings", value: settings }] },
        },
      ];
    },
    "c2s/app/exit": function () {
      return [];
    },
  };
})();
`;
}

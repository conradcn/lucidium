import { describe, expect, it, beforeEach } from "vitest";

import { initialInterview, useLucidiumStore } from "../../src/state/store";
import { gameFixture, settingsFixture } from "../fixtures";

describe("useLucidiumStore", () => {
  beforeEach(() => {
    useLucidiumStore.setState({
      game: null,
      settings: null,
      interview: initialInterview(),
      hasSave: false,
      status: "connecting",
    });
  });

  it("applyFullState replaces game and settings", () => {
    useLucidiumStore
      .getState()
      .applyFullState(
        gameFixture({ id: "g1" }),
        settingsFixture({ typewriter_speed_chars_per_sec: 60 }),
      );
    const state = useLucidiumStore.getState();
    expect(state.game).toEqual({ id: "g1" });
    expect(state.settings).toEqual({ typewriter_speed_chars_per_sec: 60 });
  });

  it("applyPatch replaces a leaf field", () => {
    useLucidiumStore
      .getState()
      .applyFullState(gameFixture({ world: { game_name: "Old" } }), settingsFixture({}));
    useLucidiumStore.getState().applyPatch([
      { op: "replace", path: "/world/game_name", value: "New" },
    ]);
    expect(useLucidiumStore.getState().game?.world.game_name).toBe("New");
  });

  it("applyPatch removes a field", () => {
    useLucidiumStore
      .getState()
      .applyFullState(
        gameFixture({ world: { summarizer_assessment: "x" } }),
        settingsFixture({}),
      );
    useLucidiumStore
      .getState()
      .applyPatch([{ op: "remove", path: "/world/summarizer_assessment" }]);
    expect(useLucidiumStore.getState().game?.world.summarizer_assessment).toBeUndefined();
  });

  it("applyPatch routes /interview ops to the interview slice", () => {
    useLucidiumStore.getState().applyFullState(gameFixture({}), settingsFixture({}));
    useLucidiumStore.getState().applyPatch([
      { op: "replace", path: "/interview/step", value: "genre" },
      { op: "replace", path: "/interview/genre_options", value: ["Noir"] },
    ]);
    const { interview, game } = useLucidiumStore.getState();
    expect(interview.step).toBe("genre");
    expect(interview.genre_options).toEqual(["Noir"]);
    // The interview slice must not leak into the game mirror.
    expect((game as unknown as Record<string, unknown>).interview).toBeUndefined();
  });

  it("applyPatch handles per-entry ops on interview maps", () => {
    useLucidiumStore.getState().applyPatch([
      { op: "add", path: "/interview/side_characters/sc1", value: { id: "sc1", name: "Hale" } },
      { op: "add", path: "/interview/side_characters/sc2", value: { id: "sc2", name: "Mira" } },
    ]);
    useLucidiumStore.getState().applyPatch([
      { op: "replace", path: "/interview/side_characters/sc1", value: { id: "sc1", name: "Hale Stone" } },
      { op: "remove", path: "/interview/side_characters/sc2" },
    ]);
    expect(useLucidiumStore.getState().interview.side_characters).toEqual({
      sc1: { id: "sc1", name: "Hale Stone" },
    });
  });

  it("replace on /interview swaps the whole slice, null resets it", () => {
    useLucidiumStore
      .getState()
      .applyPatch([{ op: "replace", path: "/interview/setting", value: "stone harbor" }]);
    useLucidiumStore
      .getState()
      .applyPatch([{ op: "replace", path: "/interview", value: { step: "name" } }]);
    expect(useLucidiumStore.getState().interview).toEqual({ step: "name" });
    useLucidiumStore
      .getState()
      .applyPatch([{ op: "replace", path: "/interview", value: null }]);
    expect(useLucidiumStore.getState().interview).toEqual({ step: "setting" });
  });

  it("applyPatch gives the interview slice a fresh identity so subscribers fire", () => {
    const before = useLucidiumStore.getState().interview;
    useLucidiumStore
      .getState()
      .applyPatch([{ op: "replace", path: "/interview/step", value: "genre" }]);
    expect(useLucidiumStore.getState().interview).not.toBe(before);
    expect(before.step).toBe("setting");
  });

  it("resetInterview drops stale answers from an aborted onboarding", () => {
    useLucidiumStore.getState().applyPatch([
      { op: "replace", path: "/interview/step", value: "confirm" },
      { op: "replace", path: "/interview/setting", value: "neon city" },
    ]);
    useLucidiumStore.getState().resetInterview();
    expect(useLucidiumStore.getState().interview).toEqual({ step: "setting" });
  });
});

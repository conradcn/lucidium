/**
 * ``applyPatch`` routing for the ``/settings`` slice.
 *
 * The settings branch of ``store.ts`` has three distinct behaviours and
 * none of them had a test — even though commit 280573c already had to
 * fix a bug on this exact seam (a top-level slice replace being applied
 * as if the path were relative to the slice itself):
 *
 *   1. ``replace /settings`` swaps the slice wholesale.
 *   2. ``replace /settings`` with ``value: null`` coalesces to ``{}`` —
 *      the renderer must never end up with a null settings mirror,
 *      because every screen reads ``settings?.x`` off a live object.
 *   3. ``/settings/<key>`` strips the ``/settings`` prefix and applies
 *      the op relative to the slice, leaving ``game`` untouched.
 */

import { afterEach, describe, expect, it } from "vitest";

import { initialInterview, useLucidiumStore } from "../../src/state/store";

afterEach(() => {
  useLucidiumStore.setState({
    game: null,
    settings: null,
    interview: initialInterview(),
  });
});

describe("store applyPatch /settings routing", () => {
  it("replaces the settings slice wholesale on /settings", () => {
    useLucidiumStore.setState({
      settings: { typewriter_speed_chars_per_sec: 20 } as never,
    });

    useLucidiumStore.getState().applyPatch([
      {
        op: "replace",
        path: "/settings",
        value: { typewriter_speed_chars_per_sec: 60, llm: { model: "m" } },
      },
    ]);

    // Wholesale — the old key is gone, not merged over.
    expect(useLucidiumStore.getState().settings).toEqual({
      typewriter_speed_chars_per_sec: 60,
      llm: { model: "m" },
    });
  });

  it("coalesces a null /settings replace to an empty object", () => {
    useLucidiumStore.setState({ settings: { audio: { music_volume: 1 } } as never });

    useLucidiumStore
      .getState()
      .applyPatch([{ op: "replace", path: "/settings", value: null }]);

    // NOT null: screens read ``settings?.audio`` off a live object and
    // ``touchedSettings`` still has to publish the cleared slice.
    expect(useLucidiumStore.getState().settings).toEqual({});
    expect(useLucidiumStore.getState().settings).not.toBeNull();
  });

  it("strips the /settings prefix on a nested path", () => {
    useLucidiumStore.setState({
      settings: { llm: { model: "old" }, typewriter_speed_chars_per_sec: 20 } as never,
      game: { id: "g1", world: { game_name: "Keep" } } as never,
    });

    useLucidiumStore.getState().applyPatch([
      { op: "replace", path: "/settings/llm/model", value: "new" },
      { op: "replace", path: "/settings/typewriter_speed_chars_per_sec", value: 90 },
    ]);

    const state = useLucidiumStore.getState();
    expect(state.settings).toEqual({
      llm: { model: "new" },
      typewriter_speed_chars_per_sec: 90,
    });
    // A settings-only batch must not republish (or clobber) the game
    // slice — the prefix strip happens instead of the game-relative
    // fallback, not in addition to it.
    expect(state.game).toEqual({ id: "g1", world: { game_name: "Keep" } });
  });

  it("does not mutate the previous settings object in place", () => {
    const before = { llm: { model: "old" } };
    useLucidiumStore.setState({ settings: before as never });

    useLucidiumStore
      .getState()
      .applyPatch([{ op: "replace", path: "/settings/llm/model", value: "new" }]);

    // structuredClone on first touch — React bails out of a re-render
    // when the slice identity is unchanged, so an in-place mutation
    // would land in the store but never reach the screen.
    expect(before.llm.model).toBe("old");
    expect(useLucidiumStore.getState().settings).not.toBe(before);
  });

  it("removes a settings key without touching the rest of the slice", () => {
    useLucidiumStore.setState({
      settings: { llm: { model: "m", api_key: "secret" } } as never,
    });

    useLucidiumStore
      .getState()
      .applyPatch([{ op: "remove", path: "/settings/llm/api_key" }]);

    expect(useLucidiumStore.getState().settings).toEqual({ llm: { model: "m" } });
  });

  it("applies a mixed batch to the right slices", () => {
    useLucidiumStore.setState({
      settings: { llm: { model: "old" } } as never,
      game: { id: "g1", world: { game_name: "Old" } } as never,
    });

    useLucidiumStore.getState().applyPatch([
      { op: "replace", path: "/settings/llm/model", value: "new" },
      { op: "replace", path: "/world/game_name", value: "New" },
      { op: "replace", path: "/interview/step", value: "genre" },
    ]);

    const state = useLucidiumStore.getState();
    expect(state.settings).toEqual({ llm: { model: "new" } });
    expect(state.game).toEqual({ id: "g1", world: { game_name: "New" } });
    expect(state.interview.step).toBe("genre");
  });
});

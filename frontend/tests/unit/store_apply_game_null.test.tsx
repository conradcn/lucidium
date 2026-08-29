/**
 * Regression: clicking Menu → New Game in succession briefly
 * flashed the previous save's main view because the
 * ``/game = null`` state-patch the backend emits at the start
 * of a fresh new-game flow wasn't applied at the top level —
 * the store's patch handler treated it as relative to
 * ``game`` and ended up with ``state.game.game = null`` while
 * leaving the loaded save in place.
 *
 * After the fix, the store handler routes ``/game`` (and any
 * future top-level keys) through a dedicated branch that
 * replaces ``state.game`` wholesale.
 */

import { afterEach, describe, expect, it } from "vitest";

import { useLucidiumStore } from "../../src/state/store";

afterEach(() => {
  useLucidiumStore.setState({ game: null, settings: null });
});

describe("store applyPatch /game replace", () => {
  it("replaces state.game with null on a /game = null patch", () => {
    // Simulate the post-Continue, mid-Menu state: store has a
    // populated game.
    const loadedGame = {
      id: "g-old",
      current_node_id: "n-old",
      world: { game_name: "Loaded Save" },
    };
    useLucidiumStore.setState({ game: loadedGame as never });
    expect(useLucidiumStore.getState().game).not.toBeNull();

    // Apply the patch the backend's c2s/new_game/start emits.
    useLucidiumStore.getState().applyPatch([
      { op: "replace", path: "/game", value: null },
    ]);

    expect(useLucidiumStore.getState().game).toBeNull();
  });

  it("nested /world/foo patch still mutates inside game", () => {
    // Sanity: the existing convention for game-relative paths
    // (no /game prefix) still works.
    useLucidiumStore.setState({
      game: { world: { game_name: "old" } } as never,
    });
    useLucidiumStore.getState().applyPatch([
      { op: "replace", path: "/world/game_name", value: "new" },
    ]);
    const game = useLucidiumStore.getState().game as
      | { world: { game_name: string } }
      | null;
    expect(game?.world.game_name).toBe("new");
  });

  it("/game = null clears even when other game-relative ops follow", () => {
    // The /game = null replace happens at the start of a fresh
    // new-game flow. If a later op in the same patch envelope
    // tried to mutate the (now-null) game, the store should
    // promote the null back to a fresh empty object rather
    // than reach back to the now-stale state.game from BEFORE
    // the null replace fired. Otherwise the loaded save would
    // bleed back in via the subsequent op's clone path.
    useLucidiumStore.setState({
      game: {
        id: "old", world: { game_name: "Loaded" }, current_node_id: "n-old",
      } as never,
    });
    useLucidiumStore.getState().applyPatch([
      { op: "replace", path: "/game", value: null },
      { op: "replace", path: "/current_node_id", value: "n-new" },
    ]);
    const after = useLucidiumStore.getState().game as
      | { id?: string; current_node_id?: string; world?: { game_name?: string } }
      | null;
    // The old save's identity (id, world.game_name) MUST be gone.
    expect(after?.id).toBeUndefined();
    expect(after?.world?.game_name).toBeUndefined();
    // The trailing op's value did land somewhere parseable.
    expect(after?.current_node_id).toBe("n-new");
  });
});

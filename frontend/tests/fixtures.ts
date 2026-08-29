/**
 * Typed fixture builders for the store mirror.
 *
 * Unit tests exercise one slice of the save at a time and deliberately
 * omit the rest, so they cannot hand the store a complete ``Game``.
 * ``gameFixture`` accepts a deep-partial of the *generated* type: field
 * names and value unions are still checked (a renamed Pydantic field or
 * a bogus ``state: "reddy"`` fails to compile), only presence is
 * relaxed. The single unchecked step is confined to this file.
 */

import type { Game, Settings } from "../src/state/store";

/** Recursively optional. Homomorphic mapped types preserve arrays and
 *  tuples, so ``DeepPartial<Fact[]>`` is still an array type. */
export type DeepPartial<T> = T extends (infer U)[]
  ? DeepPartial<U>[]
  : T extends object
    ? { [K in keyof T]?: DeepPartial<T[K]> }
    : T;

export function gameFixture(partial: DeepPartial<Game>): Game {
  return partial as Game;
}

export function settingsFixture(partial: DeepPartial<Settings>): Settings {
  return partial as Settings;
}

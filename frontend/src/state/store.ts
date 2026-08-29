/**
 * Zustand store mirroring backend state.
 *
 * The renderer is *not* authoritative. ``s2c/state/full`` replaces the
 * mirror; ``s2c/state/patch`` applies RFC 6902-style ops. Local edits
 * round-trip through the backend before re-appearing here.
 *
 * Patch routing: a path that starts with ``/settings`` updates the
 * settings slice, ``/interview`` the new-game interview slice, and
 * everything else the game slice. All three go through the single
 * ``applyOp`` below — there is no second JSON-Patch implementation.
 */

import { create } from "zustand";

import type { Game } from "../shared/generated/Game.schema";
import type { Settings } from "../shared/generated/Settings.schema";
import type { PatchOp } from "../shared/generated/S2CStatePatch.schema";
import type { S2CImageReady } from "../shared/generated/S2CImageReady.schema";
import type { CharacterImage } from "../shared/generated/Game.schema";

export type WsStatus = "connecting" | "open" | "closed" | "errored";

// Re-exported so consumers keep importing the wire shapes from the
// store (their historical home) while the declarations live in the
// generated bindings. Renaming a field in the Pydantic model now
// breaks compilation here instead of failing silently at runtime.
export type { Game, Settings, PatchOp };
export type ImageReady = S2CImageReady;

/**
 * Re-require an ``id`` the generated types mark optional.
 *
 * Pydantic fields with a ``default_factory`` (every domain ``id``)
 * export as *not required*, so json-schema-to-typescript emits
 * ``id?: string``. The backend stamps them before the mirror ever
 * leaves the process, and the entities in question live in maps keyed
 * by that very id — so at the renderer they are always present.
 */
export type WithId<T> = T & { id: string };

/**
 * New-game onboarding scratch state.
 *
 * Unlike ``game`` / ``settings`` this has NO Pydantic counterpart —
 * the backend patches it by path off ``session.interview``. Declaring
 * it here rather than inside a screen means a contributor adding a
 * field edits one place, and the patch stream reaches it through the
 * same ``applyOp`` as everything else.
 */
export interface InterviewState {
  step: string;
  setting?: string;
  genre?: string;
  visual_style?: string;
  character_description?: string;
  character_name?: string;
  pronouns?: string;
  setting_options?: string[];
  genre_options?: string[];
  visual_style_options?: string[];
  character_description_options?: string[];
  name_options?: string[];
  side_characters?: Record<string, { id: string; name: string }>;
  // Bundled-then-restyled stage assets shown behind the interview
  // questions (FR-027). Path strings come straight from the backend.
  dream_guide_image_path?: string;
  white_room_image_path?: string;
  // Preview assets pushed as the player answers: the backdrop in
  // their chosen visual style (after ``visual_style``) and their own
  // character portrait (after ``character_description``).
  preview_background_path?: string;
  pc_portrait_path?: string;
}

/** A fresh interview — the first step, no answers. */
export function initialInterview(): InterviewState {
  return { step: "setting" };
}

export interface NoticeEntry {
  id: string;
  title: string;
  body: string;
  kind: "info" | "warning";
}

interface LucidiumState {
  game: Game | null;
  settings: Settings | null;
  interview: InterviewState;
  hasSave: boolean;
  status: WsStatus;
  /** Pending pop-ups surfaced to the user one at a time. The
   *  player dismisses each via ``dismissNotice``; until then
   *  the first entry is rendered as a modal that blocks
   *  interaction. Used for safety / policy events
   *  (``s2c/notice``) — informational only, NOT a substitute
   *  for connection / provider errors. */
  notices: NoticeEntry[];
  setStatus: (status: WsStatus) => void;
  setHasSave: (hasSave: boolean) => void;
  applyFullState: (game: Game, settings: Settings) => void;
  applyPatch: (ops: PatchOp[]) => void;
  /** Drop any stale interview scratch state from a previously
   *  aborted onboarding, before the backend's reset patch lands. */
  resetInterview: () => void;
  applyImageReady: (event: ImageReady) => void;
  pushNotice: (notice: Omit<NoticeEntry, "id">) => void;
  dismissNotice: (id: string) => void;
}

export const useLucidiumStore = create<LucidiumState>((set, get) => ({
  game: null,
  settings: null,
  interview: initialInterview(),
  hasSave: false,
  status: "connecting",
  notices: [],
  setStatus: (status) => set({ status }),
  resetInterview: () => set({ interview: initialInterview() }),
  setHasSave: (hasSave) => set({ hasSave }),
  applyFullState: (game, settings) => set({ game, settings }),
  pushNotice: (notice) => {
    const id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? (crypto as Crypto).randomUUID()
        : `notice-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    set((state) => ({ notices: [...state.notices, { ...notice, id }] }));
  },
  dismissNotice: (id) =>
    set((state) => ({ notices: state.notices.filter((n) => n.id !== id) })),
  applyImageReady: (event) => {
    const state = get();
    const game = state.game;
    if (!game) return;
    if (event.kind === "portrait") {
      const characters = game.characters;
      if (!characters) return;
      const target = characters[event.target_id];
      if (!target) return;
      const images = target.images ?? [];
      // Already at the front — nothing to do. Avoids a re-render
      // storm if the same event lands twice (state/full + image_ready
      // for the same render is the normal path).
      if (images[0]?.path === event.image_path) return;
      // Pull the image to the front if it's already in the list,
      // otherwise prepend it. Either way the renderer's images[0]
      // read picks up the freshest render.
      const filtered = images.filter((img) => img?.path !== event.image_path);
      // A placeholder entry standing in for the authoritative one the
      // next ``s2c/state/full`` carries: the image_ready event has no
      // attribute snapshot, and prompt_hash is unknown here (the id is
      // used as a non-colliding stand-in). Only ``path`` is read by
      // the renderer before the real record lands.
      const fresh: CharacterImage = {
        attributes_snapshot: {},
        id: event.image_id,
        path: event.image_path,
        prompt_hash: event.image_id,
      };
      const nextChars = { ...characters };
      nextChars[event.target_id] = { ...target, images: [fresh, ...filtered] };
      set({ game: { ...game, characters: nextChars } });
      return;
    }
    // Background: the target_id is an environment id. Update its
    // image_path so the Background component picks up the new render
    // even when the slow state/full hasn't landed yet.
    const environments = game.environments;
    if (!environments) return;
    const env = environments[event.target_id];
    if (!env) return;
    if (env.image_path === event.image_path) return;
    const nextEnvs = { ...environments };
    nextEnvs[event.target_id] = { ...env, image_path: event.image_path };
    set({ game: { ...game, environments: nextEnvs } });
  },
  applyPatch: (ops) => {
    const state = get();
    // ``gameValue`` mirrors ``state.game`` but lets us track the
    // distinction between "we replaced game wholesale" (e.g.,
    // /game = null on new-game start) and "we mutated nested
    // fields under /game" — both flip the touchedGame flag, but
    // only the wholesale-replace path skips the structured-clone
    // step. The previous shape applied ``/game = null`` as if
    // the path were relative to game itself, so it set
    // ``state.game.game = null`` and the loaded save lingered
    // in the store; the renderer's start→interview transition
    // then briefly flashed the OLD save's main view because
    // game.current_node_id still resolved.
    let gameValue: unknown = state.game ?? {};
    let settings: Settings = state.settings ?? {};
    let interview: InterviewState = state.interview;
    let touchedGame = false;
    let touchedSettings = false;
    let touchedInterview = false;

    for (const op of ops) {
      // Whole-tree replace: ``replace`` on ``/interview`` swaps the
      // slice atomically. The backend uses this on c2s/new_game/start
      // to drop stale answers from a prior aborted onboarding.
      if (op.path === "/interview" && op.op === "replace") {
        interview = { ...((op.value as InterviewState | null) ?? initialInterview()) };
        touchedInterview = true;
        continue;
      }
      if (op.path.startsWith("/interview/")) {
        if (!touchedInterview) {
          interview = structuredClone(interview);
          touchedInterview = true;
        }
        applyOp(asJson(interview), {
          ...op,
          path: op.path.replace(/^\/interview/, ""),
        });
        continue;
      }
      if (op.path === "/settings" && op.op === "replace") {
        settings = (op.value as Settings | null) ?? {};
        touchedSettings = true;
        continue;
      }
      if (op.path.startsWith("/settings/")) {
        if (!touchedSettings) {
          settings = structuredClone(settings);
          touchedSettings = true;
        }
        applyOp(asJson(settings), { ...op, path: op.path.replace(/^\/settings/, "") });
        continue;
      }
      // Top-level /game replace — applied wholesale (most often
      // ``value: null`` on c2s/new_game/start to clear the
      // loaded save's state out of the store).
      if (op.path === "/game" && op.op === "replace") {
        gameValue = op.value;
        touchedGame = true;
        continue;
      }
      // Any other path — treated as relative to ``game`` per
      // the existing convention. Backend never prefixes these
      // with ``/game`` (paths look like ``/world/foo``,
      // ``/characters/c1/x``, ``/current_node_id``).
      if (!touchedGame) {
        // First op of this batch — clone state.game as the base.
        gameValue = structuredClone(state.game ?? {});
        touchedGame = true;
      } else if (typeof gameValue !== "object" || gameValue === null) {
        // A previous op in this same batch wholesale-replaced
        // game (e.g. ``/game = null``). The current op is
        // game-relative — promote the cleared value to a
        // fresh empty object rather than reach back to
        // ``state.game``, otherwise the wholesale replace's
        // intent (drop the loaded save) would be undone.
        gameValue = {};
      }
      applyOp(asJson(gameValue as object), op);
    }

    const next: Partial<LucidiumState> = {};
    // The backend is authoritative for the mirror's shape; ops it sent
    // are assumed to keep ``game`` a Game (or null, on a wholesale
    // clear). Validating that per-op is the backend's job, not the
    // renderer's.
    if (touchedGame) next.game = gameValue as Game | null;
    if (touchedSettings) next.settings = settings;
    if (touchedInterview) next.interview = interview;
    if (Object.keys(next).length > 0) set(next);
  },
}));

/**
 * View a typed slice of the mirror as plain JSON.
 *
 * RFC 6902 paths are computed by the backend at runtime, so no static
 * type can describe the target of an arbitrary op — this is the single
 * place the mirror is deliberately untyped. ``applyOp`` mutates in
 * place and never changes object identity, so the ``Game`` /
 * ``Settings`` type on the slice survives the round trip.
 */
function asJson(value: object): Record<string, unknown> {
  return value as Record<string, unknown>;
}

function applyOp(target: Record<string, unknown>, op: PatchOp): void {
  const segments = op.path.split("/").filter((segment) => segment.length > 0);
  if (segments.length === 0) return;
  const last = segments[segments.length - 1] as string;
  const parents = segments.slice(0, -1);
  let cursor: Record<string, unknown> = target;
  for (const segment of parents) {
    const next = cursor[segment];
    if (next === undefined || next === null) {
      const fresh: Record<string, unknown> = {};
      cursor[segment] = fresh;
      cursor = fresh;
    } else if (typeof next === "object") {
      // Walking an arbitrary RFC 6902 path — see ``asJson`` above for
      // why the mirror is viewed as plain JSON in this function.
      cursor = next as Record<string, unknown>;
    } else {
      return;
    }
  }
  if (op.op === "remove") {
    delete cursor[last];
  } else {
    cursor[last] = op.value;
  }
}

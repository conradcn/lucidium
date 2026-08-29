import type { JSX } from "react";
import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

// Bundle the title fonts the start-menu carousel cycles through.
// Each pair in menuManifest.generated.ts names one of these in
// ``title_font`` plus a system-font fallback chain.
import "@fontsource/playfair-display/600.css";
import "@fontsource/playfair-display/700.css";
import "@fontsource/cinzel/600.css";
import "@fontsource/cinzel/700.css";
import "@fontsource/caveat/700.css";
import "@fontsource/quicksand/500.css";
import "@fontsource/cormorant-garamond/400.css";
import "@fontsource/cormorant-garamond/500.css";
import "@fontsource/cormorant-garamond/600.css";
import "@fontsource/bebas-neue/400.css";

import { ConnectionBanner } from "./app/ConnectionBanner";
import { MusicPlayer } from "./app/MusicPlayer";
import { NoticeModal } from "./app/NoticeModal";
import { assetUrl } from "./app/assetUrl";
import { ensureClient, send } from "./app/client";
import { useMenuCycle } from "./app/useMenuCycle";
import { useLucidiumStore } from "./state/store";
import type { AudioSettings } from "./shared/generated/Settings.schema";
import { FirstTimeSetup } from "./screens/FirstTimeSetup";
import { GpuAccelStatus } from "./screens/GpuAccelPrompt";
import { LoadGameScreen } from "./screens/LoadGameScreen";
import { LoadingScreen } from "./screens/LoadingScreen";
import { MainView } from "./screens/MainView";
import { activeGuidePair, MenuCarousel } from "./screens/MenuCarousel";
import { NewGameInterview } from "./screens/NewGameInterview";
import { SettingsScreen } from "./settings/SettingsScreen";
import { StartScreen } from "./screens/StartScreen";
import { StartupWarning } from "./screens/StartupWarning";

export type Phase =
  | "wizard"
  | "start"
  | "interview"
  | "loading"
  | "main"
  | "settings"
  | "load";

/**
 * Whether this launch was told to skip the startup safety modal.
 *
 * Set by ``?skipWarning=1`` on the URL, which Electron's main process
 * adds when it was itself launched with ``--skip-warning`` (see
 * electron/main.ts). Browser-only Playwright specs, which have no
 * Electron process to pass a flag to, put the parameter on the URL
 * directly.
 *
 * FOR AUTOMATION ONLY. The modal is intentionally unskippable for
 * players: it re-shows on every launch with no opt-out, because a
 * warning you can permanently silence stops being read. Nothing in the
 * shipped UI sets this, it never persists, and it has to be re-supplied
 * on every single launch. It exists because the modal is a
 * pointer-event-blocking ``aria-modal`` overlay that otherwise has to be
 * clicked away at the top of every end-to-end test — noise, and a race
 * against the modal's own mount.
 *
 * Read once at module scope: the value cannot change within a session,
 * and reading it in a hook would re-parse the URL on every render.
 */
const startupWarningSkipped: boolean = (() => {
  try {
    return new URLSearchParams(window.location.search).get("skipWarning") === "1";
  } catch {
    // Non-browser/SSR-ish contexts (jsdom without a location, etc.):
    // fail closed, i.e. show the warning.
    return false;
  }
})();

/**
 * Progressive interview preview state. Filled in piece by piece:
 *   * background_path — after the player picks Visual Style
 *   * guide_path — after the player picks Character Description
 *     (the player-character portrait replaces the dream guide)
 *
 * Either field may be ``null`` while the corresponding render is
 * still in flight. ``null`` for guide_path with a non-null
 * background means "hide the cycling dream guide" (per spec — the
 * guide vanishes the moment the chosen-style backdrop lands).
 */
interface PreviewAssets {
  background_path: string | null;
  guide_path: string | null;
}

// Exported for unit tests: the phase machine below is the one piece of
// the renderer whose failure mode is a blank screen, and Playwright
// specs are too heavy to run on it in CI. Nothing in the app imports
// ``App`` — the module-scope ``createRoot`` render at the bottom is
// still the only production entry point.
export function App(): JSX.Element {
  const [phase, setPhase] = useState<Phase>("start");
  const [previousPhase, setPreviousPhase] = useState<Phase>("start");
  const game = useLucidiumStore((s) => s.game);
  const settings = useLucidiumStore((s) => s.settings);
  // Track whether we've already routed off the welcome screen this
  // session. Without it the first-time-setup gate would re-fire
  // every time the player navigated back to ``start`` after
  // settings had loaded — re-flipping the wizard on a returning
  // player who briefly visits a screen with empty/loading settings.
  const [routedFromSettings, setRoutedFromSettings] = useState(false);
  // Startup warning: shown on EVERY launch, not just first run.
  // Dismissed per-session — re-displaying every time is the point;
  // the warning loses meaning if the player can opt out of seeing it.
  // The sole exception is the automation flag (see
  // ``startupWarningSkipped``), which starts this pre-acknowledged.
  const [warningAcknowledged, setWarningAcknowledged] = useState(
    startupWarningSkipped,
  );
  // GPU-acceleration status surface. The backend auto-provisions a
  // GPU torch overlay on startup; ``GpuAccelStatus`` is a non-blocking
  // banner that REFLECTS that flow (download progress, then a
  // relaunch-needed notice) — it never asks the player to initiate
  // anything, and renders nothing when there's nothing to provision.
  // Carousel cycle owned at App level so it persists across the
  // start → interview → start round trip — the dream guide visible
  // when the player clicks New Game stays the same throughout the
  // interview steps.
  const cycle = useMenuCycle();
  // Captured ONCE at the moment the player clicks New Game, then
  // held constant for the whole interview. Recomputing this every
  // render (it would change every 10 s as the carousel ticks) would
  // re-fire NewGameInterview's mount effect and reset the interview
  // back to step 1.
  const [frozenPair, setFrozenPair] = useState<ReturnType<typeof activeGuidePair>>(null);
  // Filled in once the backend's pre-render after visual_style
  // completes (s2c sends paths via assetUrl-style URLs).
  const [previewAssets, setPreviewAssets] = useState<PreviewAssets | null>(null);

  useEffect(() => {
    const client = ensureClient();
    // Send ``c2s/settings/get`` once the connection is OPEN, not at
    // mount time. ``ensureClient()`` returns synchronously while the
    // socket is still in CONNECTING; a ``send`` on a non-open socket
    // is silently dropped (see ws/client.ts) so an immediate call
    // never reached the backend and the wizard / start routing never
    // had settings to read. Hooking on ``s2c/hello`` (which the
    // server emits on every fresh connection) re-fires correctly on
    // reconnects too.
    const off = client.on("s2c/hello", () => {
      send("c2s/settings/get");
    });
    return off;
  }, []);

  // Route to the wizard exactly once per session, on the first
  // settings payload that says ``first_time_setup_complete`` is
  // false. Players who finish the wizard see the flag flip and
  // never get bounced back; players who hand-edited settings to
  // pre-complete it skip the wizard entirely.
  //
  // Done as a render-phase adjustment rather than in an effect (see
  // https://react.dev/learn/you-might-not-need-an-effect): React
  // re-runs App with the routed phase before anything is painted, so
  // a first-run player never sees a frame of the start screen before
  // the wizard takes over.
  if (!routedFromSettings && settings) {
    setRoutedFromSettings(true);
    if (!settings["first_time_setup_complete"]) {
      setPhase("wizard");
    }
  }

  // Reset the override + frozen pair when leaving the interview path
  // so the carousel resumes cycling next time the player visits
  // start/new and the next New Game click captures a fresh pair.
  // Also a render-phase adjustment, for the same reason.
  const [phaseSeenForReset, setPhaseSeenForReset] = useState<Phase>(phase);
  if (phaseSeenForReset !== phase) {
    setPhaseSeenForReset(phase);
    if (phase === "start" || phase === "main" || phase === "load") {
      setPreviewAssets(null);
      setFrozenPair(null);
    }
  }

  // Snapshot the current node id when the player ENTERS the load
  // picker. While they're browsing saves we don't want the auto-
  // route to bounce them out just because there's already an
  // active game in memory. The snapshot is used ONLY for "load",
  // not for "loading" — see below.
  const loadPhaseEntryNodeId = useRef<string | null>(null);
  useEffect(() => {
    if (phase !== "load") return;
    // Read the node id imperatively from the store instead of closing
    // over the rendered ``game``: the snapshot must be taken ONLY on
    // entry to the load picker. Listing ``game?.current_node_id`` as a
    // dependency (which is what exhaustive-deps would otherwise want)
    // would re-snapshot every time a background state/full landed
    // while the player was still browsing saves.
    const nodeId = useLucidiumStore.getState().game?.current_node_id;
    loadPhaseEntryNodeId.current = nodeId ?? null;
  }, [phase]);

  useEffect(() => {
    if (!game) return;
    // Phases that EXPECT to land on main once the scene is ready:
    // "loading" (post-confirm or post-Continue) and "load" (pick a
    // save). Going to "start" is an explicit Menu click and must not
    // bounce the user back to main.
    if (phase !== "loading" && phase !== "load") return;
    // Single gate: the backend has set current_node_id and the dialog
    // node has text. Image readiness is NO LONGER a gate — we drop
    // the player onto the main view immediately and let portraits /
    // backgrounds swap in as their s2c/image/ready events arrive.
    if (!game.current_node_id) return;
    // ``load`` (picker open): only route when the id has CHANGED
    // from the one in scope when the picker opened. Stops a
    // pre-existing loaded game from auto-routing the user out of
    // the picker before they've selected a save.
    //
    // ``loading``: ALWAYS route as soon as we have a valid node id.
    // The earlier shape compared against a snapshot here too, but
    // that broke the start → menu → Continue path: after the first
    // commit the on-disk save's ``current_node_id`` matches the
    // in-memory game's, so the equality check returned early and
    // ``loading`` stuck forever. ``loading`` is only ever entered
    // by an explicit user action (Continue / new-game-confirm /
    // surprise-me) where we KNOW we want to show the next scene.
    if (phase === "load" && game.current_node_id === loadPhaseEntryNodeId.current) {
      return;
    }
    setPreviousPhase(phase);
    setPhase("main");
  }, [game, phase]);

  const goToSettings = (): void => {
    setPreviousPhase(phase);
    setPhase("settings");
  };

  // Capture the currently-visible carousel pair AT CLICK TIME and
  // hold it for the entire interview. Doing this in the click handler
  // (rather than reading on every render) is what keeps the interview
  // from resetting back to step 1 when the carousel ticks.
  const startNewGame = (): void => {
    setFrozenPair(activeGuidePair(cycle));
    setPhase("interview");
  };

  const backFromSettings = (): void => {
    setPhase(previousPhase === "settings" ? "start" : previousPhase);
  };

  // The carousel renders behind ANY screen that uses the menu
  // backdrop. start + interview + loading share the same atmospheric
  // layer — it persists across the phase change so the dream guide /
  // player-character portrait doesn't teleport. Loading specifically
  // shows the chosen-style backdrop + PC portrait the interview
  // already pinned, with just a spinner overlaid on top.
  // Wizard sits on its own dark backdrop — no carousel behind it.
  // Otherwise the menu pair would tick under the welcome card and
  // distract from the first impression.
  const showCarousel =
    phase === "start" || phase === "interview" || phase === "loading";
  const overrideForCarousel = previewAssets
    ? {
        // Backend hands us absolute filesystem paths; convert through
        // the lucidium-asset scheme so Electron can fetch them. In
        // tests assetUrl is a no-op for already-URL inputs. Either
        // field may be null while its render is still in flight.
        background: previewAssets.background_path
          ? assetUrl(previewAssets.background_path) ?? previewAssets.background_path
          : null,
        guide: previewAssets.guide_path
          ? assetUrl(previewAssets.guide_path) ?? previewAssets.guide_path
          : null,
        slug: "preview",
      }
    : null;

  // Compute the active background-music URL by phase. ``main``
  // plays the world-specific generated track (or null while the
  // backend is rendering one); every other phase plays the
  // bundled MainTheme. Crossfading lives inside MusicPlayer so
  // the menu → game → menu transitions never pop. Vite serves
  // ``frontend/public/`` at the site root so the literal
  // ``/main-menu/MainTheme.mp3`` URL fetches the bundled file
  // both in dev and from a packaged Electron build.
  const worldMusicPath = game?.world?.music_path ?? null;
  // Relative path (no leading slash). Vite's ``base: "./"`` ships
  // ``index.html`` with relative URLs, and this audio fetch needs
  // to follow the same convention — a leading-slash URL resolves
  // to ``file:///`` (drive root) in the packaged Electron build,
  // not the asar root, so the menu theme silently 404s in
  // production. The bundled file lives at
  // ``frontend/public/main-menu/MainTheme.mp3`` → vite copies it
  // into ``dist/main-menu/`` → electron-builder packs it into
  // ``app.asar/dist/main-menu/``, which the renderer reaches via
  // a path relative to its loaded ``index.html``.
  const musicUrl = phase === "main"
    ? (worldMusicPath ? assetUrl(worldMusicPath) ?? worldMusicPath : null)
    : "main-menu/MainTheme.mp3";
  // Audio settings — read from the live store snapshot. Defaults
  // mirror the backend defaults so the first frame before
  // settings load doesn't blast the player at full volume.
  const audioSrc: AudioSettings = settings?.audio ?? {};
  const masterVolume = typeof audioSrc.master_volume === "number"
    ? audioSrc.master_volume : 0.7;
  const musicVolume = typeof audioSrc.music_volume === "number"
    ? audioSrc.music_volume : 0.4;
  const musicEnabled = Boolean(
    settings?.music?.enabled,
  );
  // ``music.enabled`` is the master switch for ALL music — both
  // the server-side generated track in-game and the bundled menu
  // theme. Earlier the menu theme played unconditionally on the
  // theory that bundled audio is "free" (no LLM / GPU cost), but
  // a player who turned music off didn't expect the menu to keep
  // singing. Treat the toggle as an audio-on-or-off intent that
  // applies wherever the engine plays music, regardless of source.
  const musicPlayerEnabled = musicEnabled;

  return (
    <>
      <ConnectionBanner />
      <MusicPlayer
        url={musicUrl}
        masterVolume={masterVolume}
        musicVolume={musicVolume}
        enabled={musicPlayerEnabled}
      />
      <main data-testid={`phase-${phase}`}>
        {showCarousel ? (
          <MenuCarousel cycle={cycle} override={overrideForCarousel} />
        ) : null}
        {renderPhase(
          phase,
          setPhase,
          goToSettings,
          backFromSettings,
          cycle,
          frozenPair,
          setPreviewAssets,
          startNewGame,
        )}
      </main>
      {/* Modal sits OVER everything — wizard, start screen, even
          a mid-load splash. Players read it before they can poke
          at anything else. Per-session: re-shown on every launch. */}
      {warningAcknowledged ? null : (
        <StartupWarning onAcknowledge={() => setWarningAcknowledged(true)} />
      )}
      {/* Automatic GPU-acceleration status. NON-BLOCKING: a small
          banner that reflects the backend's auto-provisioning
          (download progress, then a dismissible relaunch notice).
          Mounted across all phases (after first-time setup + the
          startup warning so it doesn't stack on either) so it stays
          visible while the player keeps using the app on the CPU
          overlay. Renders nothing unless provisioning is actually
          happening — a no-op on machines already on a GPU build or
          with no GPU. */}
      {warningAcknowledged && settings?.["first_time_setup_complete"] ? (
        <GpuAccelStatus />
      ) : null}
      {/* Safety / policy pop-ups (s2c/notice). Sit above everything
          else, including the startup warning, because they fire
          mid-game and need to register before the player keeps
          playing. */}
      <NoticeModal />
    </>
  );
}

// Exported alongside ``App`` so the switch can be exercised for every
// Phase value — including the unreachable-by-construction ``default``,
// whose whole job is to render something rather than nothing.
export function renderPhase(
  phase: Phase,
  setPhase: (p: Phase) => void,
  goToSettings: () => void,
  backFromSettings: () => void,
  cycle: ReturnType<typeof useMenuCycle>,
  frozenPair: ReturnType<typeof activeGuidePair>,
  setPreviewAssets: (assets: PreviewAssets | null) => void,
  startNewGame: () => void,
): JSX.Element {
  switch (phase) {
    case "wizard":
      return <FirstTimeSetup onDone={() => setPhase("start")} />;
    case "start":
      return (
        <StartScreen
          cycle={cycle}
          onNewGame={startNewGame}
          onLoadGame={() => setPhase("load")}
          onSettings={goToSettings}
          // Continue is optimistic: transition to the loading screen
          // immediately. The auto-route above advances to main once
          // the backend's state/full lands.
          onContinue={() => setPhase("loading")}
          // Surprise Me is also optimistic — transition to loading
          // BEFORE the c2s message goes out so the loading-phase
          // node-id snapshot fires while game is still null/stale.
          // When the backend's state/full arrives with a fresh
          // current_node_id, the auto-route advances to main.
          onSurpriseMe={() => setPhase("loading")}
        />
      );
    case "interview":
      return (
        <NewGameInterview
          frozenPair={frozenPair}
          // Live carousel pair updates as the backdrop ticks
          // through pairs — used only for the interview's border
          // colour so it tracks whatever the player is currently
          // looking at, instead of staying frozen to the pair
          // they clicked New Game on. ``frozenPair`` still seeds
          // the mount effect (its deps must stay stable to avoid
          // resetting the interview to step 1 every 10 s).
          livePair={activeGuidePair(cycle)}
          onPreviewAssets={setPreviewAssets}
          onLeftToMainView={() => setPhase("loading")}
          // Go back on the Review screen: hop straight to the start
          // (main-menu) phase. The interview's mount effect already
          // resets its local snapshot on remount; the backend
          // wipes its InterviewState in the c2s/new_game/go_back
          // handler so a re-entered New Game lands on a clean slate.
          onCancel={() => setPhase("start")}
        />
      );
    case "loading":
      return <LoadingScreen />;
    case "main":
      return (
        <MainView
          onOpenStory={() => undefined}
          onOpenMenu={() => setPhase("start")}
          onOpenSettings={goToSettings}
        />
      );
    case "load":
      return <LoadGameScreen onBack={() => setPhase("start")} />;
    case "settings":
      return <SettingsScreen onBack={backFromSettings} />;
    default:
      return (
        <StartScreen
          cycle={cycle}
          onNewGame={startNewGame}
          onLoadGame={() => setPhase("load")}
          onSettings={goToSettings}
          onContinue={() => setPhase("loading")}
        />
      );
  }
}

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

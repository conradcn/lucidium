/**
 * The renderer's phase machine (``src/main.tsx``).
 *
 * ``App`` decides which screen exists at all. When it picks wrong the
 * failure mode is a blank window with no console error, which is the
 * hardest kind of regression to notice — and until now the only thing
 * exercising it was the Playwright suite, which CI does not run.
 *
 * Every screen is stubbed down to a marker div plus buttons that fire
 * the callbacks ``renderPhase`` wires up, so what is under test is
 * purely the routing: which phase each transition lands on, which
 * effect auto-routes, and what ``previousPhase`` means on the way back
 * out of Settings.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";

// Runs before the module imports below. ``src/main`` renders itself
// into ``#root`` at module scope, so the node has to exist first, and
// ``startupWarningSkipped`` reads the URL exactly once at module scope.
vi.hoisted(() => {
  window.history.replaceState({}, "", "/?skipWarning=1");
  document.body.innerHTML = '<div id="root"></div>';
});

// --- stubs -----------------------------------------------------------
// Each screen becomes a marker plus the buttons the phase machine
// needs; none of the real screens' data dependencies are involved.

// A function *declaration*, not a const: ``vi.mock`` calls are hoisted
// above this file's imports, and their factories must be able to reach
// it. (The factories themselves run lazily, once ``src/main`` is
// imported, so the JSX runtime is loaded by then.)
function stub(testid: string, buttons: Record<string, string> = {}) {
  return function Stub(props: Record<string, unknown>) {
    return (
      <div data-testid={testid}>
        {Object.entries(buttons).map(([label, prop]) => (
          <button
            key={label}
            data-testid={`${testid}-${label}`}
            onClick={() => (props[prop] as ((...a: unknown[]) => void) | undefined)?.()}
          >
            {label}
          </button>
        ))}
      </div>
    );
  };
}

vi.mock("../../src/app/ConnectionBanner", () => ({ ConnectionBanner: () => null }));
vi.mock("../../src/app/MusicPlayer", () => ({
  MusicPlayer: (props: { url: string | null; enabled: boolean }) => (
    <div data-testid="music" data-url={props.url ?? ""} data-enabled={String(props.enabled)} />
  ),
}));
vi.mock("../../src/app/NoticeModal", () => ({ NoticeModal: () => null }));
vi.mock("../../src/app/client", () => ({
  ensureClient: () => ({ on: () => () => undefined }),
  send: () => undefined,
}));
vi.mock("../../src/screens/MenuCarousel", () => ({
  MenuCarousel: () => <div data-testid="carousel" />,
  activeGuidePair: () => ({ slug: "pair-1" }),
}));
vi.mock("../../src/app/useMenuCycle", () => ({ useMenuCycle: () => ({ index: 0 }) }));
vi.mock("../../src/screens/StartupWarning", () => ({
  StartupWarning: stub("startup-warning", { ack: "onAcknowledge" }),
}));
vi.mock("../../src/screens/GpuAccelPrompt", () => ({
  GpuAccelStatus: () => <div data-testid="gpu-status" />,
}));
vi.mock("../../src/screens/FirstTimeSetup", () => ({
  FirstTimeSetup: stub("wizard", { done: "onDone" }),
}));
vi.mock("../../src/screens/StartScreen", () => ({
  StartScreen: stub("start-screen", {
    new: "onNewGame",
    load: "onLoadGame",
    settings: "onSettings",
    continue: "onContinue",
    surprise: "onSurpriseMe",
  }),
}));
vi.mock("../../src/screens/NewGameInterview", () => ({
  NewGameInterview: stub("interview", {
    left: "onLeftToMainView",
    cancel: "onCancel",
  }),
}));
vi.mock("../../src/screens/LoadingScreen", () => ({
  LoadingScreen: () => <div data-testid="loading-screen" />,
}));
vi.mock("../../src/screens/MainView", () => ({
  MainView: stub("main-view", { menu: "onOpenMenu", settings: "onOpenSettings" }),
}));
vi.mock("../../src/screens/LoadGameScreen", () => ({
  LoadGameScreen: stub("load-screen", { back: "onBack" }),
}));
vi.mock("../../src/settings/SettingsScreen", () => ({
  SettingsScreen: stub("settings-screen", { back: "onBack" }),
}));

import { App, renderPhase, type Phase } from "../../src/main";
import { initialInterview, useLucidiumStore } from "../../src/state/store";
import { gameFixture, settingsFixture } from "../fixtures";

// Importing ``src/main`` mounts the real entry point into ``#root``.
// That is deliberate — it proves the module's top level survives an
// import — but it must not be visible to the queries below, so detach
// the container. The React root keeps rendering into a node that is no
// longer in the document.
document.getElementById("root")?.remove();

/** Reset the mirror to a returning player who has finished setup. */
function resetStore(settings: Record<string, unknown> = {
  first_time_setup_complete: true,
}): void {
  useLucidiumStore.setState({
    game: null,
    settings: settingsFixture(settings as never),
    interview: initialInterview(),
    hasSave: false,
    status: "open",
    notices: [],
  });
}

/** The phase is published on ``<main data-testid="phase-X">``. */
function currentPhase(): string {
  const el = document.querySelector("main[data-testid^='phase-']");
  return el?.getAttribute("data-testid")?.replace("phase-", "") ?? "";
}

const click = (testid: string): void => {
  act(() => {
    screen.getByTestId(testid).click();
  });
};

/** Land a game in the mirror the way ``s2c/state/full`` does. */
const landGame = (nodeId: string | null): void => {
  act(() => {
    useLucidiumStore.setState({
      game: gameFixture({ id: "g1", current_node_id: nodeId ?? undefined } as never),
    });
  });
};

beforeEach(() => {
  resetStore();
});

afterEach(() => {
  cleanup();
});

describe("App phase machine", () => {
  it("opens on the start screen with the carousel behind it", () => {
    render(<App />);
    expect(currentPhase()).toBe("start");
    expect(screen.getByTestId("start-screen")).toBeTruthy();
    expect(screen.getByTestId("carousel")).toBeTruthy();
    // ``?skipWarning=1`` was on the URL at module scope.
    expect(screen.queryByTestId("startup-warning")).toBeNull();
  });

  it("routes a player who has not finished setup to the wizard, once", () => {
    resetStore({ first_time_setup_complete: false });
    render(<App />);
    expect(currentPhase()).toBe("wizard");

    // Finishing the wizard lands on start...
    click("wizard-done");
    expect(currentPhase()).toBe("start");

    // ...and a later settings payload must NOT bounce the player back,
    // even while the flag is still false (the gate is once-per-session).
    act(() => {
      useLucidiumStore.setState({
        settings: settingsFixture({ first_time_setup_complete: false } as never),
      });
    });
    expect(currentPhase()).toBe("start");
  });

  it("start → New Game → interview → loading → main", () => {
    render(<App />);
    click("start-screen-new");
    expect(currentPhase()).toBe("interview");

    click("interview-left");
    expect(currentPhase()).toBe("loading");
    expect(screen.getByTestId("loading-screen")).toBeTruthy();

    // The backend's state/full is what actually promotes loading→main.
    landGame("n1");
    expect(currentPhase()).toBe("main");
    expect(screen.getByTestId("main-view")).toBeTruthy();
  });

  it("stays on loading until the game has a current node id", () => {
    render(<App />);
    click("start-screen-continue");
    expect(currentPhase()).toBe("loading");

    // A game with no node yet is not a scene — routing now would show
    // an empty main view.
    landGame(null);
    expect(currentPhase()).toBe("loading");

    landGame("n1");
    expect(currentPhase()).toBe("main");
  });

  it("Surprise Me takes the same optimistic loading path", () => {
    render(<App />);
    click("start-screen-surprise");
    expect(currentPhase()).toBe("loading");
    landGame("n2");
    expect(currentPhase()).toBe("main");
  });

  it("the load picker does not auto-route on the already-loaded game", () => {
    // A save is already in memory when the picker opens.
    landGame("n-old");
    render(<App />);
    click("start-screen-load");
    expect(currentPhase()).toBe("load");

    // Same node id — the player is still browsing; do not eject them.
    landGame("n-old");
    expect(currentPhase()).toBe("load");

    // A DIFFERENT node id means their pick landed.
    landGame("n-new");
    expect(currentPhase()).toBe("main");
  });

  it("Back from the load picker returns to start", () => {
    render(<App />);
    click("start-screen-load");
    click("load-screen-back");
    expect(currentPhase()).toBe("start");
  });

  it("Settings returns to whichever phase opened it", () => {
    render(<App />);
    click("start-screen-settings");
    expect(currentPhase()).toBe("settings");
    click("settings-screen-back");
    expect(currentPhase()).toBe("start");

    // From main, Back goes to main — not to the menu.
    click("start-screen-continue");
    landGame("n1");
    expect(currentPhase()).toBe("main");
    click("main-view-settings");
    expect(currentPhase()).toBe("settings");
    click("settings-screen-back");
    expect(currentPhase()).toBe("main");
  });

  it("Menu from the main view does not bounce straight back to main", () => {
    render(<App />);
    click("start-screen-continue");
    landGame("n1");
    click("main-view-menu");
    // ``start`` is an explicit Menu click; the auto-route must ignore
    // the still-loaded game.
    expect(currentPhase()).toBe("start");
    landGame("n1");
    expect(currentPhase()).toBe("start");
  });

  it("cancelling the interview returns to start", () => {
    render(<App />);
    click("start-screen-new");
    click("interview-cancel");
    expect(currentPhase()).toBe("start");
  });

  it("shows the carousel only on the menu-backdrop phases", () => {
    render(<App />);
    expect(screen.queryByTestId("carousel")).toBeTruthy(); // start
    click("start-screen-new");
    expect(screen.queryByTestId("carousel")).toBeTruthy(); // interview
    click("interview-left");
    expect(screen.queryByTestId("carousel")).toBeTruthy(); // loading
    landGame("n1");
    expect(screen.queryByTestId("carousel")).toBeNull(); // main
    click("main-view-settings");
    expect(screen.queryByTestId("carousel")).toBeNull(); // settings
  });

  it("plays the bundled menu theme off the game phases and the world track on main", () => {
    useLucidiumStore.setState({
      settings: settingsFixture({
        first_time_setup_complete: true,
        music: { enabled: true },
      } as never),
    });
    render(<App />);
    expect(screen.getByTestId("music").getAttribute("data-url")).toBe(
      "main-menu/MainTheme.mp3",
    );
    expect(screen.getByTestId("music").getAttribute("data-enabled")).toBe("true");

    click("start-screen-continue");
    act(() => {
      useLucidiumStore.setState({
        game: gameFixture({
          id: "g1",
          current_node_id: "n1",
          world: { music_path: "C:/tmp/world.mp3" },
        } as never),
      });
    });
    expect(currentPhase()).toBe("main");
    expect(screen.getByTestId("music").getAttribute("data-url")).not.toBe(
      "main-menu/MainTheme.mp3",
    );
  });

  it("mounts the GPU status banner only after setup is complete", () => {
    resetStore({ first_time_setup_complete: false });
    render(<App />);
    expect(screen.queryByTestId("gpu-status")).toBeNull();
    click("wizard-done");
    act(() => {
      useLucidiumStore.setState({
        settings: settingsFixture({ first_time_setup_complete: true } as never),
      });
    });
    expect(screen.queryByTestId("gpu-status")).toBeTruthy();
  });
});

describe("renderPhase", () => {
  const noop = (): void => undefined;
  const call = (phase: Phase) =>
    renderPhase(
      phase,
      noop,
      noop,
      noop,
      { index: 0 } as never,
      null,
      noop,
      noop,
    );

  it.each([
    ["wizard", "wizard"],
    ["start", "start-screen"],
    ["interview", "interview"],
    ["loading", "loading-screen"],
    ["main", "main-view"],
    ["load", "load-screen"],
    ["settings", "settings-screen"],
  ] as [Phase, string][])("renders a screen for %s", (phase, testid) => {
    render(<div>{call(phase)}</div>);
    expect(screen.getByTestId(testid)).toBeTruthy();
  });

  it("falls back to the start screen for an unknown phase", () => {
    // The default arm exists so a Phase value added without a case
    // still renders SOMETHING — a blank window is the failure this
    // whole file is here to catch.
    render(<div>{call("bogus" as Phase)}</div>);
    expect(screen.getByTestId("start-screen")).toBeTruthy();
  });
});

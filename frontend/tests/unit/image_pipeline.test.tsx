/**
 * Component-level half of the image-pipeline test suite.
 *
 * Drives the renderer surfaces that consume disk paths and confirms
 * each one (a) uses the lucidium-asset:// scheme and (b) round-trips
 * the original disk path through the data-image-path attribute so the
 * end-to-end Playwright tests can match what the backend handed down
 * with what landed in the DOM.
 *
 * The assertions deliberately treat raw ``file://`` as a regression —
 * Electron 31 blocks cross-origin file:// requests issued from a
 * file:// page, which is exactly how shipped builds run.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, cleanup } from "@testing-library/react";

import { ASSET_SCHEME } from "../../src/app/assetUrl";
import { useLucidiumStore } from "../../src/state/store";
import { gameFixture } from "../fixtures";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(),
}));

import { Background } from "../../src/screens/MainView/Background";
import { CharacterStage } from "../../src/screens/MainView/CharacterStage";
import { InterviewStage } from "../../src/screens/NewGameInterview/InterviewStage";

const PORTRAIT_PATH = "C:/Users/Iris/portraits/hale.png";
const ENV_PATH = "C:/Users/Iris/environments/harbor.png";
const DREAM_GUIDE_PATH = "C:/lucidium/placeholders/dream_guide.png";
const WHITE_ROOM_PATH = "C:/lucidium/placeholders/white_room.png";

afterEach(() => cleanup());

describe("CharacterStage", () => {
  beforeEach(() => {
    useLucidiumStore.setState({
      game: gameFixture({
        current_node_id: "n1",
        on_stage: ["c1"],
        characters: {
          c1: {
            id: "c1",
            is_player: false,
            name: "Hale",
            images: [{ id: "img-c1", path: PORTRAIT_PATH }],
          },
        },
      }),
    });
  });

  it("renders the portrait through the lucidium-asset scheme", () => {
    const { container } = render(<CharacterStage />);
    const img = container.querySelector("img");
    expect(img).not.toBeNull();
    const src = img!.getAttribute("src") ?? "";
    expect(src.startsWith(`${ASSET_SCHEME}://`)).toBe(true);
    expect(src).not.toMatch(/^file:/);
  });

  it("never reaches the DOM with a raw file:// scheme", () => {
    const { container } = render(<CharacterStage />);
    expect(container.innerHTML).not.toMatch(/file:\/\//);
  });
});

describe("Background", () => {
  beforeEach(() => {
    useLucidiumStore.setState({
      game: gameFixture({
        current_node_id: "n1",
        environments: { e1: { id: "e1", image_path: ENV_PATH } },
        dialog_tree: {
          nodes: { n1: { id: "n1", parent_id: null, location_id: "e1" } },
          committed_path: ["n1"],
          root_id: "n1",
        },
      }),
    });
  });

  it("uses the lucidium-asset scheme in background-image", () => {
    const { container } = render(<Background />);
    const div = container.querySelector(".background");
    const style = div!.getAttribute("style") ?? "";
    expect(style).toContain(`${ASSET_SCHEME}://`);
    expect(style).not.toMatch(/file:\/\//);
    expect(div!.getAttribute("data-image-path")).toBe(ENV_PATH);
  });

  it("renders nothing for backgrounds when no environment is active", () => {
    useLucidiumStore.setState({
      game: gameFixture({
        current_node_id: null,
        environments: {},
        dialog_tree: { nodes: {}, committed_path: [], root_id: null },
      }),
    });
    const { container } = render(<Background />);
    const div = container.querySelector(".background");
    // Element still rendered so the gradient backdrop CSS shows; just
    // no inline style URL.
    expect(div!.getAttribute("style") ?? "").not.toContain("url(");
  });
});

describe("InterviewStage (dream guide / white room)", () => {
  it("renders both bundled assets when the snapshot includes them", () => {
    const { container } = render(
      <InterviewStage
        whiteRoomPath={WHITE_ROOM_PATH}
        dreamGuidePath={DREAM_GUIDE_PATH}
      />,
    );

    const guide = container.querySelector(
      '[data-testid="interview-dream-guide"]',
    ) as HTMLImageElement | null;
    expect(guide).not.toBeNull();
    expect(guide!.getAttribute("src") ?? "").toContain(`${ASSET_SCHEME}://`);
    expect(guide!.getAttribute("data-image-path")).toBe(DREAM_GUIDE_PATH);

    const backdrop = container.querySelector(
      '[data-testid="interview-white-room"]',
    );
    expect(backdrop).not.toBeNull();
    const style = backdrop!.getAttribute("style") ?? "";
    expect(style).toContain(`${ASSET_SCHEME}://`);
    expect(backdrop!.getAttribute("data-image-path")).toBe(WHITE_ROOM_PATH);
  });

  it("renders nothing if both paths are blank", () => {
    const { container } = render(<InterviewStage />);
    expect(container.querySelector(".interview-stage")).toBeNull();
  });

  it("survives one path missing without crashing", () => {
    const { container } = render(
      <InterviewStage dreamGuidePath={DREAM_GUIDE_PATH} />,
    );
    expect(
      container.querySelector('[data-testid="interview-dream-guide"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="interview-white-room"]'),
    ).toBeNull();
  });
});

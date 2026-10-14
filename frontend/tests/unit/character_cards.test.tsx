/**
 * SillyTavern character-card UI.
 *
 *   1. ``CardLibrary`` lists what ``s2c/cards/list`` reports, filters,
 *      hands the picked file to ``onAdd``, and surfaces an
 *      ``s2c/error`` that lands while its add is in flight (a failed add
 *      otherwise has no visible effect).
 *   2. The New Game review step adds cards via
 *      ``c2s/new_game/add_card_character`` and marks card rows.
 *   3. The Characters tab adds cards mid-story (with the on-stage
 *      choice) and exports any character with ``export_card``.
 *   4. Settings commits ``character_cards_dir`` on blur.
 *
 * Same in-memory bus mock as ``gpu_acceleration.test.tsx`` so tests
 * can push S2C payloads.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render } from "@testing-library/react";

const listeners = new Map<string, Set<(p: unknown) => void>>();
function emit(type: string, payload: unknown): void {
  act(() => {
    for (const fn of listeners.get(type) ?? []) fn(payload);
  });
}

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({
    on: (type: string, fn: (p: unknown) => void) => {
      let set = listeners.get(type);
      if (!set) {
        set = new Set();
        listeners.set(type, set);
      }
      set.add(fn);
      return () => set?.delete(fn);
    },
  })),
}));

import { send } from "../../src/app/client";
import { CardLibrary } from "../../src/app/CardLibrary";
import { CharactersTab } from "../../src/screens/MainView/StoryPanel/CharactersTab";
import { ConfirmStep } from "../../src/screens/NewGameInterview/ConfirmStep";
import { SettingsScreen } from "../../src/settings/SettingsScreen";
import { useLucidiumStore } from "../../src/state/store";
import { gameFixture, settingsFixture } from "../fixtures";

const LISTING = {
  cards_dir: "C:\\Users\\me\\cards",
  cards: [
    {
      file: "Seraphine.png",
      name: "Seraphine",
      creator: "sora",
      tags: ["fantasy", "elf"],
      summary: "A forest guardian.",
      spec: "chara_card_v2",
      thumbnail_path: "C:\\saves\\_card_thumbs\\abc.png",
      has_lucidium_data: false,
    },
    {
      file: "Mira.png",
      name: "Mira",
      tags: ["Lucidium"],
      summary: "A smuggler.",
      spec: "chara_card_v3",
      thumbnail_path: "",
      has_lucidium_data: true,
    },
  ],
  errors: [{ file: "broken.json", reason: "file is not valid JSON" }],
};

function sentOf(type: string): unknown[][] {
  return vi.mocked(send).mock.calls.filter((c) => c[0] === type);
}

afterEach(() => {
  cleanup();
  listeners.clear();
});

beforeEach(() => {
  vi.mocked(send).mockClear();
});

describe("CardLibrary", () => {
  it("requests the listing and renders cards, badges and read failures", () => {
    const { getByTestId, getByText, container } = render(
      <CardLibrary onAdd={() => undefined} busy={false} />,
    );
    expect(sentOf("c2s/cards/list")).toHaveLength(1);
    expect(getByText("Loading cards...")).toBeTruthy();

    emit("s2c/cards/list", LISTING);
    expect(getByTestId("card-library-card-Seraphine.png").textContent).toContain(
      "by sora",
    );
    expect(getByTestId("card-library-card-Mira.png").textContent).toContain("Lucidium");
    const img = container.querySelector(".card-library__thumb") as HTMLImageElement;
    expect(img.src).toMatch(/^lucidium-asset:\/\/local\//);
    expect(getByText("1 file could not be read")).toBeTruthy();
    expect(container.textContent).toContain("C:\\Users\\me\\cards");
  });

  it("filters by name, tag and creator", () => {
    const { getByTestId, queryByTestId, getByText } = render(
      <CardLibrary onAdd={() => undefined} busy={false} />,
    );
    emit("s2c/cards/list", LISTING);
    const filter = getByTestId("card-library-filter");
    fireEvent.change(filter, { target: { value: "ELF" } });
    expect(queryByTestId("card-library-card-Seraphine.png")).not.toBeNull();
    expect(queryByTestId("card-library-card-Mira.png")).toBeNull();
    fireEvent.change(filter, { target: { value: "nobody" } });
    expect(getByText(/No cards match/)).toBeTruthy();
  });

  it("shows the empty-folder guidance", () => {
    const { container } = render(<CardLibrary onAdd={() => undefined} busy={false} />);
    emit("s2c/cards/list", { cards_dir: "/cards", cards: [], errors: [] });
    expect(container.textContent).toContain("No character cards yet");
  });

  it("passes the file to onAdd and shows an error raised during the add", () => {
    const onAdd = vi.fn();
    const { getByLabelText, getByRole, queryByRole, rerender } = render(
      <CardLibrary onAdd={onAdd} busy={false} />,
    );
    emit("s2c/cards/list", LISTING);

    // An error that isn't ours (no add in flight) is ignored.
    emit("s2c/error", { code: "internal", message: "unrelated" });
    expect(queryByRole("alert")).toBeNull();

    fireEvent.click(getByLabelText("Add Seraphine"));
    expect(onAdd).toHaveBeenCalledWith("Seraphine.png");
    rerender(<CardLibrary onAdd={onAdd} busy={true} />);
    expect(getByRole("status").textContent).toContain("adapting the card");

    emit("s2c/error", { code: "provider_unreachable", message: "LLM is down" });
    rerender(<CardLibrary onAdd={onAdd} busy={false} />);
    expect(getByRole("alert").textContent).toBe("LLM is down");
  });
});

describe("ConfirmStep — add from character card", () => {
  it("opens the library and sends add_card_character", () => {
    const { getByTestId, getByLabelText } = render(
      <ConfirmStep snapshot={{ setting: "x", side_characters: {} }} onConfirmed={() => undefined} />,
    );
    fireEvent.click(getByTestId("side-character-from-card"));
    emit("s2c/cards/list", LISTING);
    fireEvent.click(getByLabelText("Add Seraphine"));
    expect(sentOf("c2s/new_game/add_card_character")).toEqual([
      ["c2s/new_game/add_card_character", { file: "Seraphine.png" }],
    ]);
  });

  it("marks rows that came from a card", () => {
    const { getByTestId } = render(
      <ConfirmStep
        snapshot={{
          side_characters: {
            a: { id: "a", name: "Seraphine", from_card: true },
            b: { id: "b", name: "the keeper" },
          },
        }}
        onConfirmed={() => undefined}
      />,
    );
    expect(getByTestId("side-character-row-a").textContent).toContain("card");
    expect(getByTestId("side-character-row-b").querySelector(".side-character-row__card")).toBeNull();
  });
});

function gameState() {
  const character = (id: string, name: string, isPlayer: boolean) => ({
    id,
    is_player: isPlayer,
    name,
    description: "d",
    kind: "human" as const,
    physical_description: "",
    gender: "male", age: 50, ethnicity: "local",
    skin: "tan", hair_color: "grey", hairstyle: "short",
    eye_color: "brown", build: "stocky", bust: "n/a",
    outfit: "oilskin", pose: "leaning", expression: "watchful",
    facts: [], images: [], seed: 7,
  });
  return {
    game: gameFixture({
      id: "g1",
      schema_version: 1,
      current_node_id: "n1",
      characters: { p1: character("p1", "Iris", true), c2: character("c2", "Hale", false) },
      dialog_tree: { nodes: {}, root_id: "n1", committed_path: [] },
      environments: {},
      on_stage: ["c2"],
    }),
    settings: settingsFixture({}),
    hasSave: true,
    status: "open" as const,
  };
}

function openCard(container: HTMLElement, name: string): void {
  const header = Array.from(container.querySelectorAll("h3")).find((h) =>
    h.textContent?.includes(name),
  );
  if (!header) throw new Error(`${name} header not found`);
  // The player card starts open; NPC cards start closed.
  if (!header.textContent?.includes("▾")) fireEvent.click(header);
}

describe("CharactersTab — cards", () => {
  it("adds a card mid-story, honouring the on-stage checkbox", () => {
    useLucidiumStore.setState(gameState());
    const { getByTestId, getByLabelText } = render(<CharactersTab />);
    fireEvent.click(getByTestId("characters-add-from-card"));
    emit("s2c/cards/list", LISTING);
    fireEvent.click(getByLabelText(/Place on stage now/));
    fireEvent.click(getByLabelText("Add Mira"));
    expect(sentOf("c2s/edit/character/add_from_card")).toEqual([
      ["c2s/edit/character/add_from_card", { file: "Mira.png", on_stage: false }],
    ]);
  });

  it("exports NPCs and the player, then reports where the card went", () => {
    useLucidiumStore.setState(gameState());
    const { container, getAllByText } = render(<CharactersTab />);
    openCard(container, "Hale");
    openCard(container, "Iris");
    const buttons = getAllByText("Export card");
    expect(buttons).toHaveLength(2);
    fireEvent.click(buttons[0]!);
    const exports = sentOf("c2s/edit/character/export_card");
    expect(exports).toHaveLength(1);
    const id = (exports[0]![1] as { character_id: string }).character_id;

    emit("s2c/cards/exported", {
      character_id: id,
      file: "Card.png",
      path: "C:\\Users\\me\\cards\\Card.png",
    });
    expect(container.textContent).toContain("Saved card to C:\\Users\\me\\cards\\Card.png");
  });
});

describe("SettingsScreen — character cards folder", () => {
  it("shows the resolved default and commits a typed folder on blur", () => {
    useLucidiumStore.setState({
      settings: {
        llm: { base_url: "", model: "", api_key: "", temperature: 0.8, max_tokens: 1024 },
        image: { backend: "embedded", embedded_models_dir: "" },
        music: { enabled: false },
        audio: { master_volume: 0.7, music_volume: 0.4 },
        user_profile: {},
        character_cards_dir: "",
      },
    });
    const { getByTestId } = render(<SettingsScreen onBack={() => undefined} />);
    expect(sentOf("c2s/cards/list").length).toBeGreaterThan(0);
    emit("s2c/cards/list", LISTING);
    const input = getByTestId("character-cards-dir") as HTMLInputElement;
    expect(input.placeholder).toBe("default: C:\\Users\\me\\cards");
    expect(getByTestId("character-cards-section").textContent).toContain("2 cards found");

    // Typing alone doesn't commit; blur does.
    fireEvent.change(input, { target: { value: "  D:\\SillyTavern\\characters  " } });
    expect(
      sentOf("c2s/settings/update").filter(
        (c) => "character_cards_dir" in (c[1] as { patch: object }).patch,
      ),
    ).toHaveLength(0);
    fireEvent.blur(input);
    expect(sentOf("c2s/settings/update")).toContainEqual([
      "c2s/settings/update",
      { patch: { character_cards_dir: "D:\\SillyTavern\\characters" } },
    ]);
  });
});

import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { useLucidiumStore } from "../../../state/store";
import { CharactersTab } from "./CharactersTab";
import { DialogTreeTab } from "./DialogTreeTab";
import { EnvironmentsTab } from "./EnvironmentsTab";
import { HistoryTab } from "./HistoryTab";
import { MusicTab } from "./MusicTab";
import { OptionsTab } from "./OptionsTab";
import { WorldInfoTab } from "./WorldInfoTab";

type TabKey =
  | "history" | "world" | "environments" | "characters"
  | "music" | "tree" | "options";

interface Props {
  onClose: () => void;
  onOpenSettings: () => void;
}

// Labels are short on purpose — the story panel is 30 rem wide
// and the tabs share that horizontal space equally. "Scenes" is
// what the tab actually shows (one entry per location the
// engine has built); "Environments" was the internal class name
// that didn't fit visually. The Music tab only renders when the
// player has music gen enabled in settings — without that toggle
// there's nothing to list or regenerate.
const ALL_TABS: { key: TabKey; label: string }[] = [
  { key: "history", label: "History" },
  { key: "world", label: "World" },
  { key: "environments", label: "Scenes" },
  { key: "characters", label: "Cast" },
  { key: "music", label: "Music" },
  { key: "tree", label: "Tree" },
  { key: "options", label: "Options" },
];

// Module-level scroll-position cache + last-active-tab, both keyed
// at module scope so they survive the brief unmount that happens
// when the player goes Story → Settings → back. The StoryPanel is
// destroyed and recreated; we restore both the same tab and the
// same scrollTop on the next mount.
const scrollMemory: Map<TabKey, number> = new Map();
let lastActiveTab: TabKey = "history";
// Reading it during render is fine; the WRITE goes through this
// module-level helper rather than assigning ``lastActiveTab`` straight
// from the component body, which is what ``react-hooks/globals``
// (rightly) flags. Same module-scope-cache role as ``scrollMemory``.
function rememberActiveTab(next: TabKey): void {
  lastActiveTab = next;
}

export function StoryPanel({ onClose, onOpenSettings }: Props): JSX.Element {
  const settings = useLucidiumStore((s) => s.settings) as
    | { music?: { enabled?: boolean } }
    | null;
  const musicEnabled = Boolean(settings?.music?.enabled);
  const tabs = ALL_TABS.filter((t) => t.key !== "music" || musicEnabled);
  const initialTab = tabs.some((t) => t.key === lastActiveTab)
    ? lastActiveTab
    : "history";
  const [tab, setTabRaw] = useState<TabKey>(initialTab);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  // Where the player actually left the active tab. We keep it here
  // instead of reading ``body.scrollTop`` when the effect tears down:
  // by then React has already swapped the tab's content out (or the
  // panel is unmounting), the body is short again, and the browser has
  // clamped scrollTop to fit. Saving that clamped number is how the
  // remembered offset used to decay on every switch — 800 became 595,
  // then 390, then eventually 0.
  const desiredScrollRef = useRef(0);
  // Set while we write scrollTop ourselves. The scroll event our own
  // write produces must not be mistaken for the player scrolling: a
  // restore into content that hasn't grown yet lands clamped, and
  // recording that would throw the saved offset away.
  const restoringRef = useRef(false);
  const setTab = (next: TabKey): void => {
    rememberActiveTab(next);
    setTabRaw(next);
  };

  // Restore the scroll position when this panel mounts or when the tab
  // changes, and save it again for the tab we're leaving. Without this
  // the player loses their place after every Settings round-trip.
  //
  // Restore and persist share one effect on purpose: the ordering
  // between "save where we were" and "aim at the new tab's offset"
  // matters, and splitting them makes it hinge on React's
  // cleanup/setup interleaving between two effects.
  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;
    const target = scrollMemory.get(tab) ?? 0;
    desiredScrollRef.current = target;

    let frame = 0;
    const clearRestoring = (): void => {
      restoringRef.current = false;
    };
    // Browsers clamp scrollTop to the scrollable range, so a restore
    // attempted before the body's content (history beats, auto-growing
    // textareas, images) has laid out silently lands short. ``apply``
    // reports whether it actually got there.
    const apply = (): boolean => {
      if (Math.abs(body.scrollTop - target) > 1) {
        restoringRef.current = true;
        body.scrollTop = target;
        // Scroll events are dispatched in the frame's scroll steps,
        // which run before animation-frame callbacks — so our own
        // event has already been swallowed by the time this runs.
        if (typeof requestAnimationFrame === "function") {
          cancelAnimationFrame(frame);
          frame = requestAnimationFrame(clearRestoring);
        } else {
          clearRestoring();
        }
      }
      return Math.abs(body.scrollTop - target) <= 1;
    };

    const onScroll = (): void => {
      if (restoringRef.current) return;
      desiredScrollRef.current = body.scrollTop;
      scrollMemory.set(tab, body.scrollTop);
    };
    body.addEventListener("scroll", onScroll);

    const reached = apply();
    // ResizeObserver is missing in jsdom (used by unit tests); there
    // the one-shot apply above is all we get, which is fine — the unit
    // suite doesn't exercise scroll restore.
    let observer: ResizeObserver | undefined;
    if (!reached && typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(() => {
        // Retry once the content can host the saved offset, then stop:
        // past that point the player's own scrolling owns the body.
        if (body.scrollHeight - body.clientHeight < target) return;
        if (apply()) observer?.disconnect();
      });
      // Watching the body alone only catches its own box changing (a
      // scrollbar appearing, the window resizing). Content growing
      // taller changes the children's height, not the body's, so the
      // children have to be watched too.
      observer.observe(body);
      for (const child of Array.from(body.children)) observer.observe(child);
    }

    return () => {
      if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame);
      restoringRef.current = false;
      observer?.disconnect();
      body.removeEventListener("scroll", onScroll);
      scrollMemory.set(tab, desiredScrollRef.current);
    };
  }, [tab]);

  return (
    <aside className="story-panel" data-testid="story-panel">
      <div className="story-panel__head">
        <div className="header">
          <h2>Story</h2>
          <button onClick={onClose}>Close</button>
        </div>
        <div className="tabs">
          {tabs.map((entry) => (
            <button
              key={entry.key}
              className={tab === entry.key ? "active" : undefined}
              onClick={() => setTab(entry.key)}
            >
              {entry.label}
            </button>
          ))}
        </div>
      </div>
      <div
        className="story-panel__body"
        ref={bodyRef}
        data-testid="story-panel-body"
      >
        {tab === "history" && <HistoryTab />}
        {tab === "world" && <WorldInfoTab />}
        {tab === "environments" && <EnvironmentsTab />}
        {tab === "characters" && <CharactersTab />}
        {tab === "music" && musicEnabled && <MusicTab />}
        {tab === "tree" && <DialogTreeTab />}
        {tab === "options" && <OptionsTab onOpenSettings={onOpenSettings} />}
      </div>
    </aside>
  );
}

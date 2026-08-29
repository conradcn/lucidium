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
  const setTab = (next: TabKey): void => {
    rememberActiveTab(next);
    setTabRaw(next);
  };

  // Restore scroll position when this panel mounts or when the tab
  // changes. Without this the player loses their place after every
  // Settings round-trip.
  //
  // Browsers clamp scrollTop to scrollHeight, so a naive one-shot
  // restore at mount-time fails when the body's content (HistoryTab
  // beats, etc) hasn't laid out yet. Retry through a ResizeObserver
  // until either the saved offset is reached or the tab is changed.
  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;
    const target = scrollMemory.get(tab) ?? 0;
    const apply = (): void => {
      if (body.scrollTop !== target) body.scrollTop = target;
    };
    apply();
    // ResizeObserver is missing in jsdom (used by unit tests). The
    // production fallback is one-shot apply; the test harness doesn't
    // exercise scroll-restore directly so this is fine.
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      // Once content has grown enough to host the target scroll,
      // re-apply. After that the scroll listener takes over.
      if (body.scrollHeight - body.clientHeight >= target) apply();
    });
    observer.observe(body);
    return () => observer.disconnect();
  }, [tab]);

  // Persist scroll position on unmount (for the round-trip case) and
  // on any scroll event (so the latest position is what gets restored).
  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;
    const onScroll = (): void => {
      scrollMemory.set(tab, body.scrollTop);
    };
    body.addEventListener("scroll", onScroll);
    return () => {
      scrollMemory.set(tab, body.scrollTop);
      body.removeEventListener("scroll", onScroll);
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

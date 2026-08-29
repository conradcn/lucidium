import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { send } from "../../../app/client";
import { useLucidiumStore } from "../../../state/store";
import type { AudioSettings } from "../../../shared/generated/Settings.schema";

interface Props {
  onOpenSettings: () => void;
}

interface AudioDraft {
  master_volume: number;
  music_volume: number;
}

/**
 * In-game options tab — replicates the most common global
 * settings (audio mixer) so the player can tweak them without
 * leaving the run. Settings that need a full form (LLM keys,
 * image backend, profile editing) still live in the dedicated
 * Settings screen, reachable from the Open Settings button.
 *
 * The audio sliders push debounced ``c2s/settings/update``
 * patches the same way the global Settings screen does — both
 * write into the same ``settings.audio`` keys, so the
 * MusicPlayer (which reads from the store's settings.audio)
 * reacts identically regardless of which panel the player
 * dragged the slider from.
 */
export function OptionsTab({ onOpenSettings }: Props): JSX.Element {
  const settings = useLucidiumStore((s) => s.settings);
  const world = useLucidiumStore((s) => s.game?.world ?? null);
  const [draft, setDraft] = useState<number>(world?.prompt_history_clamp_chars ?? 12000);
  const [audio, setAudio] = useState<AudioDraft>({
    master_volume: 0.7,
    music_volume: 0.4,
  });

  // Hydrate audio from live settings whenever they change. The
  // sliders track the store as the source of truth so a change
  // made in the global Settings screen reflects here without a
  // remount. Done with the render-phase "adjust state when the data
  // changes" pattern (https://react.dev/learn/you-might-not-need-an-effect)
  // rather than an effect, so React re-runs this component with the
  // hydrated sliders before painting instead of cascading a second
  // committed render.
  // Seeded with ``null``, never with the first ``settings`` value: if
  // the payload is already in the store when this tab mounts, the very
  // first render must still hydrate the sliders.
  const [seenSettings, setSeenSettings] = useState<typeof settings | null>(null);
  if (seenSettings !== settings) {
    setSeenSettings(settings);
    if (settings) {
      const audioSrc: AudioDraft | AudioSettings = settings.audio ?? {};
      setAudio((prev) => ({ ...prev, ...audioSrc }));
    }
  }

  // Debounced live push — same 150 ms window as the global
  // Settings screen. The MusicPlayer reads from the store, so
  // the volume reacts within ~150 ms of slider drag. No "Save"
  // button: audio adjustments should feel immediate.
  const audioPatchTimerRef = useRef<number | null>(null);
  const pushAudioLive = (next: AudioDraft): void => {
    if (audioPatchTimerRef.current !== null) {
      window.clearTimeout(audioPatchTimerRef.current);
    }
    audioPatchTimerRef.current = window.setTimeout(() => {
      send("c2s/settings/update", { patch: { audio: next } });
      audioPatchTimerRef.current = null;
    }, 150);
  };
  useEffect(() => {
    return () => {
      if (audioPatchTimerRef.current !== null) {
        window.clearTimeout(audioPatchTimerRef.current);
      }
    };
  }, []);

  // Autosave the prompt-history-clamp value on a slightly
  // longer 500 ms window — the user is typing a number, not
  // dragging a slider, and a single keystroke shouldn't
  // round-trip per character. Per-tab cleanup flushes the
  // pending timer on unmount.
  const clampTimerRef = useRef<number | null>(null);
  const scheduleClampSave = (next: number): void => {
    if (clampTimerRef.current !== null) {
      window.clearTimeout(clampTimerRef.current);
    }
    clampTimerRef.current = window.setTimeout(() => {
      send("c2s/edit/world", {
        field: "prompt_history_clamp_chars",
        value: next,
      });
      clampTimerRef.current = null;
    }, 500);
  };
  useEffect(() => {
    return () => {
      if (clampTimerRef.current !== null) {
        window.clearTimeout(clampTimerRef.current);
      }
    };
  }, []);

  return (
    <div className="options-tab">
      <h3 style={{ marginTop: 0 }}>Audio</h3>
      <label
        style={{ display: "block", marginBottom: "0.6rem" }}
        htmlFor="options-master-volume"
      >
        <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
          Master volume ({Math.round(audio.master_volume * 100)}%)
        </div>
        <input
          id="options-master-volume"
          type="range"
          min={0}
          max={100}
          value={Math.round(audio.master_volume * 100)}
          onChange={(e) => {
            const next = { ...audio, master_volume: Number(e.target.value) / 100 };
            setAudio(next);
            pushAudioLive(next);
          }}
          style={{ width: "100%" }}
        />
      </label>
      <label style={{ display: "block" }} htmlFor="options-music-volume">
        <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
          Music volume ({Math.round(audio.music_volume * 100)}%)
        </div>
        <input
          id="options-music-volume"
          type="range"
          min={0}
          max={100}
          value={Math.round(audio.music_volume * 100)}
          onChange={(e) => {
            const next = { ...audio, music_volume: Number(e.target.value) / 100 };
            setAudio(next);
            pushAudioLive(next);
          }}
          style={{ width: "100%" }}
        />
      </label>

      <hr style={{ margin: "1.2rem 0" }} />

      <h3 style={{ marginTop: 0 }}>Per-save</h3>
      <label style={{ display: "block" }}>
        <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
          Prompt history clamp (characters)
        </div>
        <input
          type="number"
          min={500}
          max={200000}
          step={500}
          value={draft}
          onChange={(e) => {
            const n = Number(e.target.value);
            setDraft(n);
            scheduleClampSave(n);
          }}
        />
      </label>

      <hr style={{ margin: "1.2rem 0" }} />

      <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
        Full configuration — LLM, image backend, music server,
        profile, mature-content gate — lives in the global
        Settings screen.
      </p>
      <button onClick={onOpenSettings}>Open Settings</button>
    </div>
  );
}

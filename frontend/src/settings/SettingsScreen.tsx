import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { ensureClient, send } from "../app/client";
import {
  RelaunchNotice,
  TORCH_FLAVORS,
  TorchOverlayProgress,
  flavorLabel,
  useTorchOverlay,
} from "../app/TorchOverlayPanel";
import { useApiKeyValidation } from "../app/useApiKeyValidation";
import { useLucidiumStore } from "../state/store";

interface Props {
  onBack: () => void;
}

interface LlmDraft {
  base_url: string;
  model: string;
  api_key: string;
  temperature: number;
  max_tokens: number;
}

interface ImageDraft {
  backend: "comfyui" | "embedded";
  base_url: string;
  portrait_workflow: string;
  background_workflow: string;
  embedded_models_dir: string;
  embedded_model_name: string;
  embedded_character_model_name: string;
  embedded_environment_model_name: string;
  embedded_face_detail: boolean;
  // Master switch for the backend's automatic GPU-overlay
  // provisioning. Default true (mirrors ImageSettings on the
  // backend): on startup the engine background-downloads the
  // recommended torch build itself. Turn off to require a manual
  // "Download" in the GPU acceleration control below.
  auto_gpu_provision: boolean;
}

interface EmbeddedModelsResponse {
  models_dir: string;
  models: string[];
}

interface MusicDraft {
  enabled: boolean;
  base_url: string;
  model_name: string;
  clip_seconds: number;
  inference_steps: number;
  guidance_scale: number;
  local_gpu_coordination: boolean;
}

interface AudioDraft {
  master_volume: number;
  music_volume: number;
}

interface UserProfileDraft {
  likes: string[];
  dislikes: string[];
  notes: string[];
  summarizer_likes: string[];
  summarizer_dislikes: string[];
  summarizer_notes: string[];
  // Per-tag hit counters — keyed by case-folded tag. Tags
  // missing from the dict default to ``CERTAINTY_THRESHOLD``
  // for backward-compat with old saves whose entries pre-date
  // the certainty mechanism (those should keep showing).
  summarizer_likes_scores: Record<string, number>;
  summarizer_dislikes_scores: Record<string, number>;
  summarizer_notes_scores: Record<string, number>;
}

// Strength threshold below which a summarizer-inferred tag is
// hidden from the player's Settings panel — the engine still
// tracks it, but a single (or few) noisy LLM inference doesn't
// surface to the player as "confirmed taste" until enough
// independent re-inferences accumulate. Mirrors
// ``TRAIT_SURFACE_THRESHOLD`` on the backend.
const SURFACE_THRESHOLD = 1.0;

function tagStrength(
  tag: string,
  scores: Record<string, number> | undefined,
): number {
  const key = tag.trim().toLocaleLowerCase();
  if (!key) return 0;
  // Default to the threshold so legacy saves whose tags pre-date
  // the strength scheme stay visible. New inferences arrive at
  // INITIAL_TRAIT_STRENGTH (0.1) and have to be reinforced ~9
  // more times to cross the gate.
  return scores?.[key] ?? SURFACE_THRESHOLD;
}

function aboveThreshold(
  entries: string[],
  scores: Record<string, number> | undefined,
): string[] {
  return entries.filter((tag) => tagStrength(tag, scores) >= SURFACE_THRESHOLD);
}

/**
 * The Settings UI shows ONLY above-threshold inferred tags,
 * but the saved profile keeps the below-threshold ones too.
 * When the player edits the visible list (deleting a tag, or
 * editing one — additions are out-of-scope; the inferred
 * bucket is read-only for adds), we have to stitch the edited
 * visible subset back together with the hidden tail so the
 * persisted list isn't accidentally wiped of the
 * still-tracking-but-not-yet-shown tags.
 *
 * Approach: the relative order of below-threshold tags is
 * preserved (they ride at the END of the saved list) and
 * the above-threshold tags are replaced wholesale by the
 * visible-edited list. A tag deleted from the visible list
 * disappears from the saved list — the parent
 * ``settings_update`` handler then routes the deletion into
 * the matching ``dismissed_*`` list, exactly as before.
 */
function stitchInferredEdits(
  saved: string[],
  scores: Record<string, number> | undefined,
  visibleNext: string[],
): string[] {
  const hiddenTail = saved.filter(
    (tag) => tagStrength(tag, scores) < SURFACE_THRESHOLD,
  );
  return [...visibleNext, ...hiddenTail];
}

export function SettingsScreen({ onBack }: Props): JSX.Element {
  const settings = useLucidiumStore((s) => s.settings);

  const [llm, setLlm] = useState<LlmDraft>({
    base_url: "",
    model: "",
    api_key: "",
    temperature: 0.8,
    max_tokens: 1024,
  });
  const apiKeyValidation = useApiKeyValidation();
  const [image, setImage] = useState<ImageDraft>({
    backend: "comfyui",
    base_url: "",
    portrait_workflow: "",
    background_workflow: "",
    embedded_models_dir: "",
    embedded_model_name: "",
    embedded_character_model_name: "",
    embedded_environment_model_name: "",
    embedded_face_detail: false,
    auto_gpu_provision: true,
  });
  // Live response from c2s/embedded/list_models. The "models" array
  // populates the model-name dropdown; ``models_dir`` is the
  // resolved-absolute path the backend's ``resolve_models_dir``
  // returned, shown to the user as a tooltip so they know where to
  // drop additional checkpoints.
  const [embeddedModels, setEmbeddedModels] = useState<EmbeddedModelsResponse>({
    models_dir: "",
    models: [],
  });
  const [music, setMusic] = useState<MusicDraft>({
    enabled: false,
    base_url: "http://127.0.0.1:8001",
    model_name: "acestep-v15-turbo",
    clip_seconds: 60,
    inference_steps: 27,
    guidance_scale: 15,
    local_gpu_coordination: true,
  });
  // Last response from c2s/music/inventory. ``ok`` is the
  // connection-probe result; ``models`` is what populates the
  // model dropdown (empty list = renderer falls back to a free-
  // text input). ``base_url`` echoes what was probed so a stale
  // response from a previous URL doesn't overwrite a fresh one.
  // ``probing`` covers the in-flight gap between send + reply
  // so the UI can disable Test Connection during the round-trip.
  const [musicInventory, setMusicInventory] = useState<{
    base_url: string;
    ok: boolean;
    models: string[];
    error: string;
    probing: boolean;
    probed: boolean;
  }>({
    base_url: "",
    ok: false,
    models: [],
    error: "",
    probing: false,
    probed: false,
  });
  const [typewriter, setTypewriter] = useState<number>(60);
  const [matureContent, setMatureContent] = useState<boolean>(false);
  const [narratorStyle, setNarratorStyle] = useState<string>("conversational");
  // Global audio mixing controls. master × per-channel multiply
  // into the renderer's HTMLAudioElement.volume. Defaults match
  // the backend's AudioSettings — quiet enough not to drown the
  // storyteller on first launch.
  const [audio, setAudio] = useState<AudioDraft>({
    master_volume: 0.7,
    music_volume: 0.4,
  });
  // Master toggle for the summarizer's profile-inference step.
  // Defaults to TRUE so a fresh install learns from play; the
  // checkbox lets the player turn that off if they want a
  // playthrough that doesn't accumulate cross-save tags.
  const [learnUserProfile, setLearnUserProfile] = useState<boolean>(true);
  const [profile, setProfile] = useState<UserProfileDraft>({
    likes: [],
    dislikes: [],
    notes: [],
    summarizer_likes: [],
    summarizer_dislikes: [],
    summarizer_notes: [],
    summarizer_likes_scores: {},
    summarizer_dislikes_scores: {},
    summarizer_notes_scores: {},
  });

  // Settings haven't arrived yet — flips true on first
  // s2c/state/full or s2c/state/patch carrying /settings.
  // Used to gate the embedded-models listing request: without
  // the gate, we'd fire one list_models with the default empty
  // dir BEFORE settings arrive AND a second list_models with
  // the user's configured dir AFTER, and their replies could
  // land out of order — stale empty-dir reply overwriting the
  // populated configured-dir reply, leaving the dropdown
  // claiming "no models" even when there are some on disk.
  const settingsLoadedRef = useRef(false);

  useEffect(() => {
    const client = ensureClient();
    const offEmbedded = client.on("s2c/embedded/models", (p) => {
      setEmbeddedModels({ models_dir: p.models_dir, models: p.models ?? [] });
    });
    send("c2s/settings/get");
    const offMusic = client.on("s2c/music/inventory", (p) => {
      setMusicInventory({
        base_url: p.base_url,
        ok: p.ok,
        models: Array.isArray(p.models) ? p.models : [],
        error: p.error ?? "",
        probing: false,
        probed: true,
      });
    });
    return () => {
      offEmbedded();
      offMusic();
    };
  }, []);

  // Whenever the embedded backend is selected, ask the server which
  // .safetensors files exist in the configured directory. Re-runs
  // when the user edits ``embedded_models_dir`` so the dropdown
  // tracks live edits before the user clicks Save.
  //
  // Gated on ``settingsLoadedRef``: we MUST NOT fire this request
  // before the settings/get reply lands, because ``image`` is
  // initialised with empty defaults and a list_models call against
  // those defaults is wasted at best and at worst races a later
  // configured-dir reply (out-of-order replies leave the dropdown
  // showing "no models found" even when there are some on disk).
  useEffect(() => {
    if (image.backend !== "embedded") return;
    if (!settingsLoadedRef.current) return;
    send("c2s/embedded/list_models", {
      models_dir: image.embedded_models_dir,
    });
  }, [image.backend, image.embedded_models_dir]);

  // Auto-probe the ACE-Step server when music is enabled. We
  // debounce against ``base_url`` edits — the user types a host
  // and we hit the server when they pause. ``probing`` flips to
  // ``true`` synchronously so the Test Connection button reads
  // as in-flight without a flash.
  useEffect(() => {
    if (!music.enabled) return;
    const url = music.base_url.trim();
    if (!url) return;
    const handle = window.setTimeout(() => {
      setMusicInventory((prev) => ({ ...prev, probing: true }));
      send("c2s/music/inventory", { base_url: url });
    }, 400);
    return () => window.clearTimeout(handle);
  }, [music.enabled, music.base_url]);

  const probeMusicNow = (): void => {
    setMusicInventory((prev) => ({ ...prev, probing: true }));
    send("c2s/music/inventory", { base_url: music.base_url.trim() });
  };

  // Audio sliders feel live — every drag pushes a debounced
  // partial settings patch so the MusicPlayer (which reads the
  // store's settings.audio) reacts within ~150 ms instead of
  // waiting for the player to click Save & close. Save still
  // persists the rest of the form; this side channel handles
  // ONLY the two volume fields. Debounce window is short enough
  // that users perceive it as immediate but long enough that a
  // continuous drag doesn't spam the WS.
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

  // Declared up here (before the settings-hydrate effect that
  // reads it) so the player-typed-buckets merge below can branch
  // on "do we have unsaved changes pending?". Initialized null —
  // the autosave effect below sets / clears it. Read-only from
  // the hydrate-from-settings effect.
  const autosaveTimerRef = useRef<number | null>(null);
  // Player-typed likes/dislikes/notes are locally authoritative from the
  // moment the player edits them until the backend echoes the SAME values
  // back. The autosave-timer flag alone only covered the 350 ms debounce
  // window; a settings echo that arrived AFTER the timer fired but BEFORE
  // our own save round-tripped (e.g. a summarizer-driven profile refresh)
  // could still clobber a just-typed entry — the reported "profile not
  // maintained" reset. This ref extends the protection across that
  // in-flight window and is cleared only once the echo confirms.
  const profileDirtyRef = useRef(false);

  // Settings have arrived — release the gate on the list_models
  // effect so it can fire with the user's actual configured directory
  // (not the empty-default). Kept in an effect of its own: ref writes
  // belong outside render.
  useEffect(() => {
    if (!settings) return;
    settingsLoadedRef.current = true;
  }, [settings]);

  // Hydrate every draft from the settings payload. This is a
  // render-phase adjustment rather than an effect (see
  // https://react.dev/learn/you-might-not-need-an-effect): React
  // re-runs this screen with the hydrated drafts before painting, so
  // the player never sees a frame of empty/default fields when the
  // settings payload lands.
  // Seeded with ``null``, never with the first ``settings`` value: if
  // the payload is already in the store when this screen mounts (the
  // common case — the app fetches settings at startup) the very first
  // render must still hydrate.
  const [seenSettings, setSeenSettings] = useState<typeof settings | null>(null);
  if (seenSettings !== settings && settings) {
    setSeenSettings(settings);
    const llmSrc = (settings["llm"] ?? {}) as Partial<LlmDraft>;
    const imageSrc = (settings["image"] ?? {}) as Partial<ImageDraft>;
    const musicSrc = (settings["music"] ?? {}) as Partial<MusicDraft>;
    setLlm((prev) => ({ ...prev, ...llmSrc }));
    setImage((prev) => ({ ...prev, ...imageSrc }));
    setMusic((prev) => ({ ...prev, ...musicSrc }));
    const audioSrc = (settings["audio"] ?? {}) as Partial<AudioDraft>;
    setAudio((prev) => ({ ...prev, ...audioSrc }));
    setTypewriter(Number(settings["typewriter_speed_chars_per_sec"] ?? 60));
    setMatureContent(Boolean(settings["mature_content"] ?? false));
    setNarratorStyle(String(settings["narrator_style"] ?? "conversational"));
    setLearnUserProfile(Boolean(settings["learn_user_profile"] ?? true));
    const profileSrc = (settings["user_profile"] ?? {}) as Partial<UserProfileDraft>;
    setProfile((prev) => {
      // Player-typed buckets (likes / dislikes / notes) are
      // locally-authoritative from the moment the player edits
      // them until the backend echoes the SAME values back. The
      // earlier shape gated only on the autosave DEBOUNCE timer;
      // that wiped a like the player had just typed when a
      // summarizer-driven settings echo arrived after the timer
      // fired but before our own save round-tripped (the
      // summarizer mutates summarizer_*, persists profile, the
      // engine emits the refreshed settings, and the frontend
      // reset clobbered the in-flight typed addition). Now we
      // also honour ``profileDirtyRef`` — set on every typed edit
      // (see the likes/dislikes/notes onChange handlers) and
      // cleared only once the backend echo matches — so the
      // protection spans the whole in-flight window. Backend's
      // summarizer_* fields are still taken verbatim (those ARE
      // backend-authoritative — the local UI never edits them).
      const arr = (v: unknown): string[] | null =>
        Array.isArray(v) ? [...v as string[]] : null;
      const bLikes = arr(profileSrc.likes) ?? [];
      const bDislikes = arr(profileSrc.dislikes) ?? [];
      const bNotes = arr(profileSrc.notes) ?? [];
      const same = (a: string[], b: string[]): boolean =>
        a.length === b.length && a.every((v, i) => v === b[i]);
      // Round-trip landed: the echo matches our typed buckets, so
      // drop the dirty flag and let future backend updates apply.
      if (
        profileDirtyRef.current &&
        same(prev.likes, bLikes) &&
        same(prev.dislikes, bDislikes) &&
        same(prev.notes, bNotes)
      ) {
        profileDirtyRef.current = false;
      }
      const editing =
        autosaveTimerRef.current !== null || profileDirtyRef.current;
      return {
        likes: editing ? prev.likes : bLikes,
        dislikes: editing ? prev.dislikes : bDislikes,
        notes: editing ? prev.notes : bNotes,
        summarizer_likes: arr(profileSrc.summarizer_likes) ?? [],
        summarizer_dislikes: arr(profileSrc.summarizer_dislikes) ?? [],
        summarizer_notes: arr(profileSrc.summarizer_notes) ?? [],
        summarizer_likes_scores: profileSrc.summarizer_likes_scores ?? {},
        summarizer_dislikes_scores: profileSrc.summarizer_dislikes_scores ?? {},
        summarizer_notes_scores: profileSrc.summarizer_notes_scores ?? {},
      };
    });
  }

  // Autosave: every field mutation kicks off a debounced
  // ``c2s/settings/update`` round-trip. The dedicated
  // audio-slider live-push from earlier is still in place
  // (~150 ms cadence so the MusicPlayer reacts to drag) — this
  // covers EVERYTHING ELSE at a slightly relaxed 350 ms window
  // so a player rapidly toggling the visual_style dropdown or
  // typing in the LLM model field doesn't fire a fresh patch
  // on every keystroke. ``initialMountRef`` skips the first
  // post-hydrate render so loading the page doesn't echo a
  // no-op patch back to the backend.
  // ``autosaveTimerRef`` is declared earlier — see the hydrate-
  // from-settings effect for why. The autosave logic itself is
  // unchanged: ``set`` on edit, ``clear`` on fire.
  const initialMountRef = useRef(true);
  useEffect(() => {
    if (initialMountRef.current) {
      initialMountRef.current = false;
      return;
    }
    if (autosaveTimerRef.current !== null) {
      window.clearTimeout(autosaveTimerRef.current);
    }
    autosaveTimerRef.current = window.setTimeout(() => {
      send("c2s/settings/update", {
        patch: {
          llm,
          image,
          music,
          audio,
          typewriter_speed_chars_per_sec: typewriter,
          mature_content: matureContent,
          narrator_style: narratorStyle,
          learn_user_profile: learnUserProfile,
          // Send the WHOLE profile (not a partial patch) — the
          // settings_update handler does shallow merge per-key,
          // and a partial profile would leave stale list
          // entries the player just deleted.
          user_profile: profile,
        },
      });
      autosaveTimerRef.current = null;
    }, 350);
  }, [
    llm, image, music, audio, typewriter, matureContent,
    narratorStyle, learnUserProfile, profile,
  ]);
  // On unmount, flush any pending patch synchronously so the
  // last edit doesn't get dropped if the player navigates back
  // before the debounce window closes. The cleanup closure
  // captures values via ``latestRef`` so it sees the most
  // recent state — a cleanup that closed over the initial
  // render's values would flush a no-op patch instead.
  const latestRef = useRef({
    llm, image, music, audio, typewriter, matureContent,
    narratorStyle, learnUserProfile, profile,
  });
  // Synced in an effect, not during render: refs must not be written
  // while rendering. Declared BEFORE the unmount-flush effect below,
  // so the ref is always current by the time that effect's cleanup
  // reads it.
  useEffect(() => {
    latestRef.current = {
      llm, image, music, audio, typewriter, matureContent,
      narratorStyle, learnUserProfile, profile,
    };
  });
  useEffect(() => {
    return () => {
      if (autosaveTimerRef.current !== null) {
        window.clearTimeout(autosaveTimerRef.current);
        const v = latestRef.current;
        send("c2s/settings/update", {
          patch: {
            llm: v.llm,
            image: v.image,
            music: v.music,
            audio: v.audio,
            typewriter_speed_chars_per_sec: v.typewriter,
            mature_content: v.matureContent,
            narrator_style: v.narratorStyle,
            learn_user_profile: v.learnUserProfile,
            user_profile: v.profile,
          },
        });
      }
    };
  }, []);

  return (
    <div className="interview settings-screen" data-testid="settings-screen">
      <div className="page-frame">
        <div className="page-content">
          <h2 className="page-title">Settings</h2>

          {/* Wrapping div lets the settings sections lay out in a
              two-column grid (see .settings-sections in styles.css)
              while the title and button-row stay full-width above
              and below. */}
          <div className="settings-sections">
          <section className="settings-section">
            <h3>LLM backend (OpenAI-compatible)</h3>
            <Field label="Base URL">
              <input
                type="text"
                value={llm.base_url}
                onChange={(e) => setLlm({ ...llm, base_url: e.target.value })}
              />
            </Field>
            <Field label="Model">
              <input
                type="text"
                value={llm.model}
                onChange={(e) => setLlm({ ...llm, model: e.target.value })}
              />
            </Field>
            <Field label="API key">
              <input
                type="password"
                value={llm.api_key}
                onChange={(e) => {
                  setLlm({ ...llm, api_key: e.target.value });
                  // Clear any stale badge the moment the player
                  // starts editing — otherwise an old "valid" /
                  // "unauthorized" badge would sit next to in-
                  // progress input and read as the verdict on it.
                  apiKeyValidation.reset();
                }}
                onBlur={() => {
                  const key = llm.api_key.trim();
                  if (!key) {
                    apiKeyValidation.reset();
                    return;
                  }
                  apiKeyValidation.validate(llm.base_url, key);
                }}
                data-testid="settings-api-key-input"
              />
              <ApiKeyBadge state={apiKeyValidation.state} />
            </Field>
            <Field label="Temperature" narrow>
              <input
                type="number"
                step="0.05"
                min="0"
                max="2"
                value={llm.temperature}
                onChange={(e) => setLlm({ ...llm, temperature: Number(e.target.value) })}
              />
            </Field>
            <Field label="Max tokens" narrow>
              <input
                type="number"
                min="32"
                max="32000"
                value={llm.max_tokens}
                onChange={(e) => setLlm({ ...llm, max_tokens: Number(e.target.value) })}
              />
            </Field>
          </section>

          <section className="settings-section">
            <h3>Image backend</h3>
            <Field label="Backend">
              <select
                value={image.backend}
                onChange={(e) =>
                  setImage({
                    ...image,
                    backend: e.target.value as ImageDraft["backend"],
                  })
                }
              >
                <option value="comfyui">ComfyUI (external server)</option>
                <option value="embedded">Embedded (in-process diffusers)</option>
              </select>
            </Field>
            {image.backend === "comfyui" ? (
              <>
                <Field label="Base URL">
                  <input
                    type="text"
                    value={image.base_url}
                    onChange={(e) =>
                      setImage({ ...image, base_url: e.target.value })
                    }
                  />
                </Field>
                <Field label="Portrait workflow">
                  <input
                    type="text"
                    value={image.portrait_workflow}
                    onChange={(e) =>
                      setImage({ ...image, portrait_workflow: e.target.value })
                    }
                  />
                </Field>
                <Field label="Background workflow">
                  <input
                    type="text"
                    value={image.background_workflow}
                    onChange={(e) =>
                      setImage({
                        ...image,
                        background_workflow: e.target.value,
                      })
                    }
                  />
                </Field>
              </>
            ) : (
              <>
                <p className="settings-field-hint" style={{ marginTop: 0 }}>
                  <strong>Recommended:</strong>{" "}
                  <a
                    href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Z-Image-Turbo
                  </a>
                  {" "}— 9-step renders, sharp anatomy, and CPU offload
                  keeps peak VRAM low enough for 1536×1024 backgrounds
                  on a 24 GiB card. Drop{" "}
                  <code>zImageTurbo_turbo.safetensors</code> (or any
                  file whose name contains <code>z-image</code> /{" "}
                  <code>zimage</code>) into the models directory and
                  pick it below; Lucidium routes Z-Image filenames to
                  the Z-Image pipeline automatically and pre-loads the
                  Qwen3 text encoder + VAE from HuggingFace on first
                  launch.
                  <br />
                  <strong>Qwen-Image</strong> is also supported — drop a
                  transformer checkpoint whose name contains{" "}
                  <code>qwen-image</code> / <code>qwen_image</code> (the
                  Comfy-Org fp8 repackage, or the one-click download) and
                  Lucidium routes it to the Qwen pipeline, fetching the
                  Qwen2.5-VL text encoder + VAE from HuggingFace on first
                  launch. On Qwen, a character change (new outfit, pose,
                  expression…) is rendered as an <em>img2img edit</em> of
                  that character's previous portrait rather than a fresh
                  generation, so identity and framing stay consistent.
                  It runs in seconds on a 24 GiB card (e.g. a 4090): the
                  loader quantizes the transformer to fp8 (ComfyUI's trick
                  for the 20B model) and fuses the Lightning few-step
                  distill. First load is slow (downloads + fuses the LoRA);
                  renders after that are fast.
                  <br />
                  <strong>Also supported:</strong> SDXL-family
                  checkpoints (SDXL 1.0, SDXL Turbo, Pony XL,
                  Illustrious). SD 1.5 / SD 3.x checkpoints load but
                  render badly at SDXL's resolution. SDXL starter:{" "}
                  <a
                    href="https://civitai.com/models/112902"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    DreamShaper XL on civitai
                  </a>
                  .
                </p>
                <Field label="Models directory">
                  <input
                    type="text"
                    placeholder={
                      embeddedModels.models_dir
                        ? `default: ${embeddedModels.models_dir}`
                        : "leave blank for the per-user default"
                    }
                    value={image.embedded_models_dir}
                    onChange={(e) =>
                      setImage({
                        ...image,
                        embedded_models_dir: e.target.value,
                      })
                    }
                    title={
                      embeddedModels.models_dir
                        ? `Resolved: ${embeddedModels.models_dir}`
                        : ""
                    }
                  />
                </Field>
                <Field label="Model (default)">
                  {embeddedModels.models.length > 0 ? (
                    <select
                      value={image.embedded_model_name}
                      onChange={(e) =>
                        setImage({
                          ...image,
                          embedded_model_name: e.target.value,
                        })
                      }
                    >
                      <option value="">
                        (auto-pick first available)
                      </option>
                      {embeddedModels.models.map((name) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <span className="settings-field-hint">
                      No models found in{" "}
                      {embeddedModels.models_dir || "the configured directory"}.
                      Drop a <code>.safetensors</code> file into that
                      folder. Recommended:{" "}
                      <a
                        href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Z-Image-Turbo
                      </a>{" "}
                      (best balance of speed + headroom). SDXL
                      alternatives:{" "}
                      <a
                        href="https://huggingface.co/stabilityai/sdxl-turbo/tree/main"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        SDXL Turbo
                      </a>{" "}
                      /{" "}
                      <a
                        href="https://civitai.com/models/112902"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        DreamShaper XL
                      </a>
                      . (Lucidium does not bundle or auto-download
                      weights — see SAFETY.md.)
                    </span>
                  )}
                </Field>
                {/*
                  Per-workflow model overrides. Empty string = "use the
                  default model above". Picking different files for
                  characters and environments keeps two pipelines warm
                  in VRAM when there's headroom; on a low-VRAM card
                  the embedded client evicts the older pipeline on
                  OOM and reloads on the next render of that type.
                  Models that aren't tuned for both — e.g., an
                  anime character checkpoint and a photorealistic
                  scene checkpoint — are why this split exists.
                */}
                {/*
                  Face-detail toggle: runs a second SDXL img2img pass
                  on the centered face region of every character
                  render, with face_prompt as its only prompt and a
                  feathered alpha-blend back onto the body. The
                  embedded equivalent of ComfyUI's FaceDetailer node.
                  Roughly doubles per-render time on character
                  workflows; off by default so existing players
                  don't see a silent latency regression.
                */}
                <label className="settings-field settings-field--checkbox">
                  <input
                    type="checkbox"
                    checked={image.embedded_face_detail}
                    onChange={(e) =>
                      setImage({
                        ...image,
                        embedded_face_detail: e.target.checked,
                      })
                    }
                  />
                  <span>
                    <span className="settings-field-label">
                      Face detail pass
                    </span>
                    <span className="settings-field-hint">
                      Runs a second SDXL img2img pass over the face region
                      using the per-beat face prompt. Sharper expressions
                      and tighter facial detail; roughly doubles render
                      time per character. <strong>SDXL only</strong> —
                      on Z-Image-Turbo the toggle is a no-op (Z-Image
                      already renders faces cleanly at 1024² without a
                      second pass), and the face prompt is folded into
                      the body prompt automatically.
                    </span>
                  </span>
                </label>
                {embeddedModels.models.length > 0 ? (
                  <>
                    <Field label="Character model">
                      <select
                        value={image.embedded_character_model_name}
                        onChange={(e) =>
                          setImage({
                            ...image,
                            embedded_character_model_name: e.target.value,
                          })
                        }
                      >
                        <option value="">
                          (use default model above)
                        </option>
                        {embeddedModels.models.map((name) => (
                          <option key={name} value={name}>
                            {name}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <Field label="Environment model">
                      <select
                        value={image.embedded_environment_model_name}
                        onChange={(e) =>
                          setImage({
                            ...image,
                            embedded_environment_model_name: e.target.value,
                          })
                        }
                      >
                        <option value="">
                          (use default model above)
                        </option>
                        {embeddedModels.models.map((name) => (
                          <option key={name} value={name}>
                            {name}
                          </option>
                        ))}
                      </select>
                    </Field>
                  </>
                ) : null}
              </>
            )}
          </section>

          <GpuAccelerationSection
            autoProvision={image.auto_gpu_provision}
            onAutoProvisionChange={(next) =>
              setImage((prev) => ({ ...prev, auto_gpu_provision: next }))
            }
          />

          <section className="settings-section">
            <h3>Background music (ACE-Step)</h3>
            <p className="settings-field-hint" style={{ marginTop: 0 }}>
              Generates an instrumental loop per scene via a local
              ACE-Step server (default <code>127.0.0.1:8001</code>).
              Off by default — rendering takes 30-90 s per swap.
            </p>
            <label className="settings-field settings-field--checkbox">
              <input
                type="checkbox"
                checked={music.enabled}
                onChange={(e) =>
                  setMusic({ ...music, enabled: e.target.checked })
                }
              />
              <span>
                <span className="settings-field-label">Enable background music</span>
              </span>
            </label>
            <Field label="ACE-Step server URL">
              <input
                type="text"
                value={music.base_url}
                onChange={(e) =>
                  setMusic({ ...music, base_url: e.target.value })
                }
              />
            </Field>
            <Field label="Connection">
              <div style={{
                display: "flex", gap: 8, alignItems: "center",
                flexWrap: "wrap",
              }}>
                <button
                  type="button"
                  onClick={probeMusicNow}
                  disabled={musicInventory.probing || !music.enabled}
                >
                  {musicInventory.probing ? "Testing…" : "Test connection"}
                </button>
                {musicInventory.probed && !musicInventory.probing && (
                  musicInventory.ok ? (
                    <span style={{ color: "#7ec77e" }}>
                      Connected — {musicInventory.models.length} model
                      {musicInventory.models.length === 1 ? "" : "s"} available
                    </span>
                  ) : (
                    <span style={{ color: "#c77e7e" }}>
                      {musicInventory.error || "Not reachable"}
                    </span>
                  )
                )}
              </div>
            </Field>
            <Field label="Model version">
              {musicInventory.ok && musicInventory.models.length > 0 ? (
                /*
                  Dropdown is fed by /v1/model_inventory. We add
                  the current music.model_name as an extra entry
                  if the server doesn't list it (e.g. user pasted
                  in a fork name) so the saved value isn't
                  silently overwritten when the dropdown's first
                  option becomes the selected value.
                */
                <select
                  value={music.model_name}
                  onChange={(e) =>
                    setMusic({ ...music, model_name: e.target.value })
                  }
                >
                  {!musicInventory.models.includes(music.model_name) &&
                    music.model_name && (
                      <option value={music.model_name}>
                        {music.model_name} (custom)
                      </option>
                    )}
                  {musicInventory.models.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={music.model_name}
                  onChange={(e) =>
                    setMusic({ ...music, model_name: e.target.value })
                  }
                  placeholder="acestep-v15-turbo"
                />
              )}
            </Field>
            <Field label={`Clip length (${music.clip_seconds.toFixed(0)}s)`}>
              <input
                type="range"
                min="15"
                max="240"
                step="5"
                value={music.clip_seconds}
                onChange={(e) =>
                  setMusic({
                    ...music,
                    clip_seconds: Number(e.target.value),
                  })
                }
                style={{ width: "100%" }}
              />
            </Field>
            <Field label={`Inference steps (${music.inference_steps})`} narrow>
              <input
                type="number"
                min="10"
                max="100"
                value={music.inference_steps}
                onChange={(e) =>
                  setMusic({
                    ...music,
                    inference_steps: Number(e.target.value),
                  })
                }
              />
            </Field>
            <Field label={`Guidance scale (${music.guidance_scale.toFixed(1)})`} narrow>
              <input
                type="number"
                step="0.5"
                min="1"
                max="30"
                value={music.guidance_scale}
                onChange={(e) =>
                  setMusic({
                    ...music,
                    guidance_scale: Number(e.target.value),
                  })
                }
              />
            </Field>
            {/*
              GPU coordination toggle. On by default — assumes the
              ACE-Step server runs on the same machine / same GPU
              as the embedded SDXL pipeline, so they need to take
              turns. Disable when ACE-Step lives elsewhere (remote
              host, separate GPU); the engine then doesn't shuttle
              SDXL weights off the GPU on every music swap.
            */}
            <label className="settings-field settings-field--checkbox">
              <input
                type="checkbox"
                checked={music.local_gpu_coordination}
                onChange={(e) =>
                  setMusic({
                    ...music,
                    local_gpu_coordination: e.target.checked,
                  })
                }
              />
              <span>
                <span className="settings-field-label">
                  Local GPU coordination
                </span>
                <span className="settings-field-hint">
                  ACE-Step and the embedded SDXL pipeline take turns
                  on the GPU. Image renders pause until music
                  finishes; music swaps wait for the active image
                  render. Turn off when ACE-Step runs on a different
                  machine or different GPU.
                </span>
              </span>
            </label>
          </section>

          <section className="settings-section">
            <h3>Audio</h3>
            <Field label={`Master volume (${Math.round(audio.master_volume * 100)}%)`}>
              <input
                type="range"
                min="0"
                max="100"
                value={Math.round(audio.master_volume * 100)}
                onChange={(e) => {
                  const next = {
                    ...audio,
                    master_volume: Number(e.target.value) / 100,
                  };
                  setAudio(next);
                  pushAudioLive(next);
                }}
                style={{ width: "100%" }}
              />
            </Field>
            <Field label={`Music volume (${Math.round(audio.music_volume * 100)}%)`}>
              <input
                type="range"
                min="0"
                max="100"
                value={Math.round(audio.music_volume * 100)}
                onChange={(e) => {
                  const next = {
                    ...audio,
                    music_volume: Number(e.target.value) / 100,
                  };
                  setAudio(next);
                  pushAudioLive(next);
                }}
                style={{ width: "100%" }}
              />
            </Field>
          </section>

          <section className="settings-section">
            <h3>Presentation</h3>
            <Field label={`Typewriter speed (${typewriter} chars/sec)`}>
              <input
                type="range"
                min="10"
                max="300"
                value={typewriter}
                onChange={(e) => setTypewriter(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </Field>
            <Field label="Narrator style">
              <input
                type="text"
                value={narratorStyle}
                onChange={(e) => setNarratorStyle(e.target.value)}
                placeholder="conversational"
                spellCheck={false}
              />
              <div
                className="settings-field-hint"
                style={{ marginTop: "0.4rem" }}
              >
                Free-form. Affects narration prose only — option
                buttons / character names / structured fields stay
                neutral. Suggestions:{" "}
                {[
                  "conversational",
                  "literary",
                  "informal",
                  "hard-boiled noir",
                  "pirate",
                  "fairy tale",
                  "gothic",
                  "lyrical",
                  "terse and sparse",
                ].map((s, i, arr) => (
                  <span key={s}>
                    <button
                      type="button"
                      onClick={() => setNarratorStyle(s)}
                      style={{
                        background: "transparent",
                        color: narratorStyle === s ? "var(--accent)" : "inherit",
                        border: "1px solid var(--border-soft)",
                        borderRadius: "var(--r-md, 0.4rem)",
                        padding: "0.05rem 0.45rem",
                        cursor: "pointer",
                        fontSize: "0.85em",
                        textDecoration: narratorStyle === s ? "underline" : "none",
                      }}
                    >
                      {s}
                    </button>
                    {i < arr.length - 1 ? " " : ""}
                  </span>
                ))}
              </div>
            </Field>
          </section>

          <section className="settings-section">
            <h3>Content</h3>
            <label className="settings-field settings-field--checkbox">
              <input
                type="checkbox"
                checked={matureContent}
                onChange={(e) => setMatureContent(e.target.checked)}
              />
              <span>
                <span className="settings-field-label">Mature content</span>
                <span className="settings-field-hint">
                  Allow the LLM to write explicit themes and language. Off by default.
                </span>
              </span>
            </label>
            {/* Hard-policy notice: mature mode does NOT unlock
                minor-sexual content. The engine has independent
                age-floor + storage-correction + output-filter
                guards that this checkbox cannot turn off. The
                wording mirrors SAFETY.md so a reader of either
                document gets the same message. */}
            <div
              className="settings-field-notice"
              role="note"
              data-testid="mature-content-notice"
              style={{
                marginTop: "0.5rem",
                padding: "0.6rem 0.75rem",
                borderLeft: "3px solid var(--accent-warning, #c97a3a)",
                background: "rgba(201, 122, 58, 0.08)",
                fontSize: "0.85rem",
                lineHeight: 1.45,
              }}
            >
              <strong>Mature mode does NOT unlock:</strong>
              <ul style={{ margin: "0.35rem 0 0 1.1rem", padding: 0 }}>
                <li>
                  Sexual content involving minors. Any character
                  described or visualised as under 18 in a sexual
                  context is blocked. This guard is independent of
                  this checkbox and cannot be turned off.
                </li>
                <li>
                  Likeness-based depictions of real people.
                </li>
                <li>
                  Anything prohibited by the licenses of the image
                  weights you load.
                </li>
              </ul>
              <div style={{ marginTop: "0.4rem" }}>
                See{" "}
                <a
                  href="https://github.com/anthropics/lucidium/blob/main/SAFETY.md"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  SAFETY.md
                </a>{" "}
                for the full policy and how to report a bypass.
              </div>
            </div>
          </section>

          <section className="settings-section">
            <h3>Player profile — typed by you</h3>
            <p className="settings-field-hint" style={{ marginTop: 0 }}>
              Cross-save tags the engine uses to tune tone and pacing.
              These are entries YOU type; the engine treats them as
              first-party intent.
            </p>
            <ProfileBucket
              label="Likes"
              entries={profile.likes}
              onChange={(next) => {
                profileDirtyRef.current = true;
                setProfile((prev) => ({ ...prev, likes: next }));
              }}
              placeholder="e.g. slow-burn investigation, dialog over action"
            />
            <ProfileBucket
              label="Dislikes"
              entries={profile.dislikes}
              onChange={(next) => {
                profileDirtyRef.current = true;
                setProfile((prev) => ({ ...prev, dislikes: next }));
              }}
              placeholder="e.g. tonal whiplash, time pressure"
            />
            <ProfileBucket
              label="Notes"
              entries={profile.notes}
              onChange={(next) => {
                profileDirtyRef.current = true;
                setProfile((prev) => ({ ...prev, notes: next }));
              }}
              placeholder="any other observation"
            />
          </section>

          <section className="settings-section">
            <h3>Inferred by the engine</h3>
            <p className="settings-field-hint" style={{ marginTop: 0 }}>
              Tags the summarizer learned from your past play. Kept
              separate from the typed entries above so you can see
              what the engine thinks vs. what you've stated, but you
              can edit or delete any of these — the summarizer respects
              the edits and stops re-inferring entries you removed.
            </p>
            {/*
              Master toggle. When unchecked, the summarizer still
              runs (it does facts / plot / drift work that the
              storyteller depends on) but its profile-inference
              step's output gets discarded server-side — the
              ``summarizer_*`` lists below stop growing. The lists
              themselves remain editable; existing entries stick.
            */}
            <label className="settings-field settings-field--checkbox">
              <input
                type="checkbox"
                checked={learnUserProfile}
                onChange={(e) => setLearnUserProfile(e.target.checked)}
              />
              <span>
                <span className="settings-field-label">
                  Learn from my play
                </span>
                <span className="settings-field-hint">
                  When on, the engine adds new tags to the inferred
                  buckets below as it watches what you choose.
                  When off, those buckets stop growing — what you've
                  already stated and what was already inferred both
                  stay, and the storyteller still reads them, but
                  no new inferences land.
                </span>
              </span>
            </label>
            {/* Each inferred list is filtered to tags above
                the certainty threshold — single-sighting LLM
                guesses stay tracked internally (the storyteller
                still sees them via the merged profile) but
                don't surface here until reinforced. Edits
                operate on the visible (above-threshold) subset;
                writing back stitches the visible list together
                with the hidden tail so the dismiss-tracking
                logic on the backend keeps working. */}
            <ProfileBucket
              label="Likes (inferred)"
              entries={aboveThreshold(profile.summarizer_likes, profile.summarizer_likes_scores)}
              onChange={(visibleNext) =>
                setProfile((prev) => ({
                  ...prev,
                  summarizer_likes: stitchInferredEdits(
                    prev.summarizer_likes,
                    prev.summarizer_likes_scores,
                    visibleNext,
                  ),
                }))
              }
              placeholder="(empty — populates after a few reinforced sightings)"
            />
            <ProfileBucket
              label="Dislikes (inferred)"
              entries={aboveThreshold(profile.summarizer_dislikes, profile.summarizer_dislikes_scores)}
              onChange={(visibleNext) =>
                setProfile((prev) => ({
                  ...prev,
                  summarizer_dislikes: stitchInferredEdits(
                    prev.summarizer_dislikes,
                    prev.summarizer_dislikes_scores,
                    visibleNext,
                  ),
                }))
              }
              placeholder="(empty — populates after a few reinforced sightings)"
            />
            <ProfileBucket
              label="Notes (inferred)"
              entries={aboveThreshold(profile.summarizer_notes, profile.summarizer_notes_scores)}
              onChange={(visibleNext) =>
                setProfile((prev) => ({
                  ...prev,
                  summarizer_notes: stitchInferredEdits(
                    prev.summarizer_notes,
                    prev.summarizer_notes_scores,
                    visibleNext,
                  ),
                }))
              }
              placeholder="(empty — populates after a few reinforced sightings)"
            />
          </section>
          </div>

          <div className="button-row">
            {/* Settings autosave on every change, so there's
                nothing to "save" — the only action here is
                "I'm done editing, take me back". */}
            <button onClick={onBack}>Done</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({
  label,
  children,
  narrow,
}: {
  label: string;
  children: React.ReactNode;
  narrow?: boolean;
}): JSX.Element {
  return (
    <label className={`settings-field${narrow ? " settings-field--narrow" : ""}`}>
      <div className="settings-field-label">{label}</div>
      {children}
    </label>
  );
}

/** Inline status line shown beneath the API-key input. Visible after
 * the first onBlur-triggered validation; goes away when the player
 * starts editing the key again. Colour is driven by the status tag
 * so the green / red signal is parseable at a glance without reading. */
function ApiKeyBadge({
  state,
}: {
  state: import("../app/useApiKeyValidation").ApiKeyValidationState;
}): JSX.Element | null {
  if (state.status === "idle") return null;
  const color =
    state.status === "valid"
      ? "var(--ok, #4ea36b)"
      : state.status === "pending"
        ? "var(--muted, #8a8676)"
        : "var(--accent-warning, #c97a3a)";
  return (
    <div
      className="settings-field-hint"
      data-testid="api-key-validation-badge"
      data-status={state.status}
      style={{ color, marginTop: "0.25rem" }}
    >
      {state.message || state.status}
    </div>
  );
}

/**
 * Editable list of short tags. Each row shows the tag text + an
 * "x" delete button; the row at the bottom is a free input for
 * adding a new tag (Enter or click +).
 */
function ProfileBucket({
  label,
  entries,
  onChange,
  placeholder,
}: {
  label: string;
  entries: string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
}): JSX.Element {
  const [draft, setDraft] = useState("");
  const submit = (): void => {
    const tag = draft.trim();
    if (!tag) return;
    if (entries.some((e) => e.toLowerCase() === tag.toLowerCase())) {
      setDraft("");
      return;
    }
    onChange([...entries, tag]);
    setDraft("");
  };
  return (
    <div className="settings-field profile-bucket">
      <div className="settings-field-label">{label}</div>
      <ul className="profile-bucket__list">
        {entries.map((entry, idx) => (
          // Index-only key. Earlier ``${entry}-${idx}`` mixed the
          // INPUT'S CURRENT VALUE into the key, so every keystroke
          // produced a new key, React unmounted the ``<li>`` and
          // remounted a new one, the input element was a fresh DOM
          // node, and focus was lost. With an index-only key the
          // same input element persists across edits — only its
          // ``value`` prop updates, focus stays put.
          <li key={idx} className="profile-bucket__row">
            <input
              type="text"
              value={entry}
              onChange={(e) => {
                const next = [...entries];
                next[idx] = e.target.value;
                onChange(next);
              }}
              aria-label={`${label} entry ${idx + 1}`}
            />
            <button
              type="button"
              className="profile-bucket__delete"
              onClick={() =>
                onChange(entries.filter((_, i) => i !== idx))
              }
              aria-label={`Remove ${entry}`}
            >
              ×
            </button>
          </li>
        ))}
        <li className="profile-bucket__row profile-bucket__row--new">
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={placeholder}
            aria-label={`Add ${label.toLowerCase()} entry`}
          />
          <button
            type="button"
            className="profile-bucket__add"
            onClick={submit}
            disabled={draft.trim().length === 0}
            aria-label={`Add ${label.toLowerCase()}`}
          >
            +
          </button>
        </li>
      </ul>
    </div>
  );
}

/**
 * GPU acceleration (torch runtime overlay) control.
 *
 * The frozen build ships only the CPU torch overlay, so image
 * generation runs on the CPU until the player downloads a GPU-correct
 * build (CUDA / ROCm / DirectML / XPU). This control shows the active
 * + recommended flavor and lets the player install / switch / re-
 * download any flavor, reusing the shared progress + relaunch UI from
 * ``TorchOverlayPanel``.
 *
 * The select defaults to the recommended flavor so the common path
 * ("yes, install what you suggested") is a single click. Re-selecting
 * the already-installed flavor re-downloads it — useful if a partial
 * download left a broken overlay.
 *
 * The backend AUTO-PROVISIONS this overlay on startup by default (see
 * the ``auto_gpu_provision`` toggle); this manual control stays for
 * when auto is off, or when the player wants a specific flavor /
 * re-download.
 */
function GpuAccelerationSection({
  autoProvision,
  onAutoProvisionChange,
}: {
  autoProvision: boolean;
  onAutoProvisionChange: (next: boolean) => void;
}): JSX.Element {
  const { state, requestStatus, install } = useTorchOverlay();
  // The flavor the select is pointed at. ``null`` until the status
  // reply lands, then defaults to the recommended flavor.
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => {
    requestStatus();
    // Fire-once probe on mount; the hook handles its own
    // (re)subscription.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Seed the select with the recommendation once it arrives, but don't
  // clobber a choice the player has already made. No effect needed:
  // ``picked === null`` already MEANS "the player hasn't chosen", and
  // resolving the fallback chain at render time gives exactly the same
  // select value the effect used to copy into state.
  const target = picked ?? state.recommended ?? "cpu";
  const alreadyActive = target === state.active;
  const alreadyInstalled = state.installed.includes(target);
  const buttonLabel = alreadyInstalled
    ? alreadyActive
      ? `Re-download ${flavorLabel(target)}`
      : `Switch to ${flavorLabel(target)}`
    : `Download ${flavorLabel(target)}`;

  return (
    <section className="settings-section" data-testid="gpu-acceleration-section">
      <h3>GPU acceleration</h3>
      <p className="settings-field-hint" style={{ marginTop: 0 }}>
        Image generation runs on a torch runtime that has to match your
        hardware. The app ships with a CPU build that works everywhere but is
        slow; download a GPU build (CUDA for NVIDIA, ROCm for AMD, DirectML or
        XPU for Intel) for much faster renders. A switch takes effect on the
        next launch.
      </p>
      {/* Auto-provision master switch. On (the default) the backend
          background-downloads the recommended GPU overlay on startup
          and the player never touches the controls below — they just
          see a progress banner + a relaunch notice. Off reverts to the
          manual "Download" flow using the select + button below. */}
      <label className="settings-field settings-field--checkbox">
        <input
          type="checkbox"
          checked={autoProvision}
          onChange={(e) => onAutoProvisionChange(e.target.checked)}
          data-testid="gpu-auto-provision-toggle"
        />
        <span>
          <span className="settings-field-label">
            Automatically download GPU acceleration
          </span>
          <span className="settings-field-hint">
            When on, Lucidium detects your GPU on startup and downloads the
            matching torch build in the background, then asks you to restart.
            Turn off to choose a flavor and download it manually below.
          </span>
        </span>
      </label>
      <Field label="Active runtime">
        <span className="settings-field-hint" style={{ marginTop: 0 }}>
          {state.active ? flavorLabel(state.active) : "none yet (using CPU)"}
          {state.recommended
            ? ` — recommended for this machine: ${flavorLabel(state.recommended)}`
            : ""}
        </span>
      </Field>
      <Field label="Install / switch runtime">
        <select
          value={target}
          onChange={(e) => setPicked(e.target.value)}
          disabled={state.installing}
          title={state.runtimeDir ? `Overlay root: ${state.runtimeDir}` : ""}
        >
          {TORCH_FLAVORS.map((f) => (
            <option key={f} value={f}>
              {flavorLabel(f)}
              {f === state.recommended ? " (recommended)" : ""}
              {state.installed.includes(f) ? " — installed" : ""}
              {f === state.active ? " — active" : ""}
            </option>
          ))}
        </select>
      </Field>
      <div className="button-row" style={{ marginTop: "0.4rem" }}>
        <button
          type="button"
          onClick={() => install(target)}
          disabled={state.installing}
          data-testid="gpu-acceleration-install"
        >
          {state.installing ? "Installing…" : buttonLabel}
        </button>
      </div>
      {state.installing && state.progress ? (
        <TorchOverlayProgress progress={state.progress} />
      ) : null}
      {state.activatedFlavor ? (
        <RelaunchNotice flavor={state.activatedFlavor} />
      ) : null}
      {state.error ? (
        <p
          className="settings-field-hint"
          style={{ color: "#c77e7e" }}
          data-testid="gpu-acceleration-error"
        >
          {state.error}
        </p>
      ) : null}
    </section>
  );
}

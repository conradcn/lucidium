import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { ensureClient, send } from "../app/client";
import {
  type ApiKeyValidationState,
  useApiKeyValidation,
} from "../app/useApiKeyValidation";
import { useLucidiumStore } from "../state/store";

interface Props {
  onDone: () => void;
}

type Step = "welcome" | "llm" | "image" | "done";
type Backend = "comfyui" | "embedded";

interface EmbeddedModelsResponse {
  models_dir: string;
  models: string[];
}

// The hardware-picked base model the backend would download. Fills the
// "<highlight>" in the wizard copy and the key the download button sends.
interface RecommendedModel {
  key: string;
  display_name: string;
  reason: string;
  has_models: boolean;
  approx_bytes: number;
}

// Civitai search pre-filtered to the three base-model families the
// embedded pipeline can load. Shared by the "here" link and the
// empty-state hint so a refactor can't let them drift apart.
const CIVITAI_COMPATIBLE_SEARCH =
  "https://civitai.com/models?types=Checkpoint" +
  "&baseModels=SDXL+1.0&baseModels=Pony&baseModels=Z-Image";

// One-click download lifecycle. ``running`` carries live byte progress;
// ``done`` names the file that landed (auto-selected as the checkpoint).
type DownloadState =
  | { status: "idle" }
  | { status: "running"; stage: string; bytesDone: number; bytesTotal: number | null }
  | { status: "error"; message: string }
  | { status: "done"; filename: string | null };

/**
 * First-run wizard. Fires the first time the engine starts on a
 * machine where ``Settings.first_time_setup_complete`` is still
 * false. Walks the player through:
 *
 *   1. an OpenRouter API key (skippable — they can set it from the
 *      Settings screen later);
 *   2. an image backend choice (ComfyUI vs embedded diffusers, with
 *      a model picker on the embedded path that uses the same
 *      ``c2s/embedded/list_models`` round-trip the Settings screen
 *      uses).
 *
 * Final step writes BOTH the chosen settings AND the
 * ``first_time_setup_complete`` flag in a single
 * ``c2s/settings/update`` call so a crash mid-wizard doesn't lock
 * the player out (re-running the wizard is the right behaviour
 * when something failed midway).
 */
export function FirstTimeSetup({ onDone }: Props): JSX.Element {
  const settings = useLucidiumStore((s) => s.settings);
  const [step, setStep] = useState<Step>("welcome");
  const [apiKey, setApiKey] = useState("");
  // Default to the embedded backend on first run. The bundled
  // package ships torch + CUDA + diffusers, so the engine can
  // render images out of the box without the player setting up a
  // separate ComfyUI process. Players who DO have ComfyUI can
  // flip this from Settings; the Choose-your-backend screen still
  // shows both options for that case.
  const [backend, setBackend] = useState<Backend>("embedded");
  const [embeddedModelsDir, setEmbeddedModelsDir] = useState("");
  const [embeddedModelName, setEmbeddedModelName] = useState("");
  const [embeddedModels, setEmbeddedModels] = useState<EmbeddedModelsResponse>({
    models_dir: "",
    models: [],
  });
  const [recommended, setRecommended] = useState<RecommendedModel | null>(null);
  const [download, setDownload] = useState<DownloadState>({ status: "idle" });
  // Snapshot of the model list taken when a download starts, so the
  // refreshed inventory diff tells us which file just landed (to
  // auto-select it) without the renderer hard-coding catalog filenames.
  const preDownloadModels = useRef<string[]>([]);

  // Subscribe to the embedded-models reply at mount so the dropdown
  // populates whenever the user picks the embedded backend (even
  // before they edit the dir).
  useEffect(() => {
    const client = ensureClient();
    const off = client.on("s2c/embedded/models", (payload) => {
      const p = payload as EmbeddedModelsResponse;
      setEmbeddedModels({ models_dir: p.models_dir, models: p.models });
    });
    return off;
  }, []);

  // Hardware recommendation reply — drives the "<highlight>" and the
  // download button's target.
  useEffect(() => {
    const client = ensureClient();
    const off = client.on("s2c/embedded/recommended_model", (payload) => {
      setRecommended(payload as RecommendedModel);
    });
    return off;
  }, []);

  // Live download progress. The terminal success arrives as a refreshed
  // ``s2c/embedded/models`` (handled by the effect below); failure comes
  // through ``s2c/error``.
  useEffect(() => {
    const client = ensureClient();
    const offProgress = client.on("s2c/embedded/download_progress", (p) => {
      setDownload((prev) =>
        prev.status === "running"
          ? {
              status: "running",
              stage: p.stage ?? "",
              bytesDone: p.bytes_done ?? 0,
              bytesTotal: p.bytes_total ?? null,
            }
          : prev,
      );
    });
    const offError = client.on("s2c/error", (p) => {
      setDownload((prev) =>
        prev.status === "running"
          ? { status: "error", message: p.message || "Download failed." }
          : prev,
      );
    });
    return () => {
      offProgress();
      offError();
    };
  }, []);

  // A refreshed model list that arrives mid-download IS the success
  // signal (the handler re-lists once the file is saved). Diff against
  // the pre-download snapshot to find the new checkpoint and auto-select
  // it so the player can finish without touching the dropdown.
  useEffect(() => {
    setDownload((prev) => {
      if (prev.status !== "running") return prev;
      const added = embeddedModels.models.filter(
        (m) => !preDownloadModels.current.includes(m),
      );
      const filename = added[0] ?? null;
      if (filename) setEmbeddedModelName(filename);
      return { status: "done", filename };
    });
  }, [embeddedModels]);

  const startDownload = (): void => {
    preDownloadModels.current = embeddedModels.models;
    setDownload({ status: "running", stage: "resolving", bytesDone: 0, bytesTotal: null });
    send("c2s/embedded/download_model", {
      key: recommended?.key ?? "",
      models_dir: embeddedModelsDir,
    });
  };

  // Pre-fill from existing settings — if the player happens to have
  // hand-edited settings.json before first launch, don't blow it away.
  //
  // Render-phase adjustment rather than an effect (see
  // https://react.dev/learn/you-might-not-need-an-effect): the wizard
  // re-runs with the pre-filled fields before painting, so the player
  // never sees a frame of empty inputs when settings land.
  // Seeded with ``null``, never with the first ``settings`` value: if
  // the payload is already in the store when the wizard mounts, the
  // very first render must still pre-fill.
  const [seenSettings, setSeenSettings] = useState<typeof settings | null>(null);
  if (seenSettings !== settings) {
    setSeenSettings(settings);
    if (settings) {
      const llmSrc = settings.llm;
      if (llmSrc?.api_key) setApiKey(llmSrc.api_key);
      const imageSrc = settings.image;
      if (imageSrc?.backend) setBackend(imageSrc.backend);
      if (imageSrc?.embedded_models_dir)
        setEmbeddedModelsDir(imageSrc.embedded_models_dir);
      if (imageSrc?.embedded_model_name)
        setEmbeddedModelName(imageSrc.embedded_model_name);
    }
  }

  // Whenever the embedded backend is selected (or its directory
  // changes), ask the server which checkpoints exist there.
  useEffect(() => {
    if (backend !== "embedded") return;
    if (step !== "image") return;
    send("c2s/embedded/list_models", { models_dir: embeddedModelsDir });
    // Ask which base model this machine would download — fills the
    // "<highlight>" copy and the button's target even before the player
    // touches anything.
    send("c2s/embedded/recommend_model", { models_dir: embeddedModelsDir });
  }, [backend, embeddedModelsDir, step]);

  const finish = (): void => {
    send("c2s/settings/update", {
      patch: {
        llm: apiKey ? { api_key: apiKey } : {},
        image: {
          backend,
          embedded_models_dir: embeddedModelsDir,
          embedded_model_name: embeddedModelName,
        },
        first_time_setup_complete: true,
      },
    });
    onDone();
  };

  return (
    <div className="interview first-time-setup" data-testid="first-time-setup">
      <div className="page-frame">
        <div className="page-content">
          {step === "welcome" ? (
            <WelcomeStep onNext={() => setStep("llm")} />
          ) : null}
          {step === "llm" ? (
            <LlmStep
              apiKey={apiKey}
              onApiKey={setApiKey}
              baseUrl={
                settings?.llm?.base_url ?? ""
              }
              onBack={() => setStep("welcome")}
              onNext={() => setStep("image")}
            />
          ) : null}
          {step === "image" ? (
            <ImageStep
              backend={backend}
              onBackend={setBackend}
              modelsDir={embeddedModelsDir}
              onModelsDir={setEmbeddedModelsDir}
              modelName={embeddedModelName}
              onModelName={setEmbeddedModelName}
              embeddedModels={embeddedModels}
              recommended={recommended}
              download={download}
              onDownload={startDownload}
              onBack={() => setStep("llm")}
              onNext={() => setStep("done")}
            />
          ) : null}
          {step === "done" ? (
            <DoneStep
              backend={backend}
              hasKey={apiKey.trim().length > 0}
              onBack={() => setStep("image")}
              onFinish={finish}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}

function WelcomeStep({ onNext }: { onNext: () => void }): JSX.Element {
  return (
    <>
      <h2 className="page-title">Welcome to Lucidium</h2>
      <p>
        Lucidium is an AI-driven visual novel engine. It needs an LLM provider
        to write narration, and an image generator to render the scenes you
        explore. This quick setup walks you through both.
      </p>
      <p>
        You can change every choice later from the Settings screen.
      </p>
      <div className="button-row">
        <button onClick={onNext} autoFocus>
          Begin setup
        </button>
      </div>
    </>
  );
}

function LlmStep({
  apiKey,
  onApiKey,
  baseUrl,
  onBack,
  onNext,
}: {
  apiKey: string;
  onApiKey: (v: string) => void;
  baseUrl: string;
  onBack: () => void;
  onNext: () => void;
}): JSX.Element {
  const apiKeyValidation = useApiKeyValidation();
  return (
    <>
      <h2 className="page-title">Step 1 of 2: LLM provider</h2>
      <p>
        Lucidium uses an OpenAI-compatible chat-completions endpoint.{" "}
        <a
          href="https://openrouter.ai/keys"
          target="_blank"
          rel="noopener noreferrer"
        >
          OpenRouter
        </a>{" "}
        is a good default — it routes a single key to dozens of models, gives
        you a free tier on several of them, and is what Lucidium ships pointed
        at out of the box.
      </p>
      <ol className="setup-steps">
        <li>
          Open{" "}
          <a
            href="https://openrouter.ai/keys"
            target="_blank"
            rel="noopener noreferrer"
          >
            openrouter.ai/keys
          </a>{" "}
          and sign up (Google / GitHub login is fine).
        </li>
        <li>
          Click <b>Create Key</b>, give it any name (e.g. "lucidium"), and
          copy the resulting <code>sk-or-…</code> string.
        </li>
        <li>Paste the key below.</li>
      </ol>
      <label className="settings-field">
        <div className="settings-field-label">OpenRouter API key</div>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => {
            onApiKey(e.target.value);
            // Clear the badge on edit — a stale verdict next to in-
            // progress input reads as the answer for what's there now.
            apiKeyValidation.reset();
          }}
          onBlur={() => {
            const key = apiKey.trim();
            if (!key) {
              apiKeyValidation.reset();
              return;
            }
            // Fall back to OpenRouter's standard base URL if settings
            // haven't loaded yet — the wizard is the user's first
            // launch, so settings.llm.base_url usually IS this exact
            // value but the validation must work even on the very
            // first paint before the s2c/state/full has landed.
            const url = baseUrl.trim() || "https://openrouter.ai/api/v1";
            apiKeyValidation.validate(url, key);
          }}
          placeholder="sk-or-…"
          spellCheck={false}
          autoComplete="off"
          data-testid="first-time-api-key-input"
        />
        <ApiKeyBadge state={apiKeyValidation.state} />
        <div className="settings-field-hint">
          Optional — leave blank to skip and configure it later from Settings.
          Without a key the engine can't generate any text.
        </div>
      </label>
      <div className="button-row">
        <button onClick={onBack}>Back</button>
        <button onClick={onNext} autoFocus>
          {apiKey.trim() ? "Next" : "Skip for now"}
        </button>
      </div>
    </>
  );
}

function ApiKeyBadge({
  state,
}: {
  state: ApiKeyValidationState;
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

function ImageStep({
  backend,
  onBackend,
  modelsDir,
  onModelsDir,
  modelName,
  onModelName,
  embeddedModels,
  recommended,
  download,
  onDownload,
  onBack,
  onNext,
}: {
  backend: Backend;
  onBackend: (b: Backend) => void;
  modelsDir: string;
  onModelsDir: (v: string) => void;
  modelName: string;
  onModelName: (v: string) => void;
  embeddedModels: EmbeddedModelsResponse;
  recommended: RecommendedModel | null;
  download: DownloadState;
  onDownload: () => void;
  onBack: () => void;
  onNext: () => void;
}): JSX.Element {
  return (
    <>
      <h2 className="page-title">Step 2 of 2: Image generator</h2>
      <p>
        Lucidium can render scenes two ways. The bundled package
        runs <strong>Embedded</strong> out of the box — leave this
        on the default unless you already have ComfyUI installed
        and want to reuse it.
      </p>
      <div className="setup-choice-grid">
        <ChoiceCard
          active={backend === "embedded"}
          onSelect={() => onBackend("embedded")}
          title="Embedded (recommended)"
          body={
            <>
              Lucidium loads an SDXL-family checkpoint via{" "}
              <code>diffusers</code> directly inside the engine. No
              extra process to run; torch + CUDA ship in the
              package. You supply the checkpoint yourself —
              Lucidium does not bundle or auto-download model
              weights. The next step has direct download links.
            </>
          }
        />
        <ChoiceCard
          active={backend === "comfyui"}
          onSelect={() => onBackend("comfyui")}
          title="ComfyUI (external)"
          body={
            <>
              You run{" "}
              <a
                href="https://github.com/comfyanonymous/ComfyUI"
                target="_blank"
                rel="noopener noreferrer"
              >
                ComfyUI
              </a>{" "}
              as a separate process; Lucidium talks to it over
              HTTP. Best when you already have ComfyUI set up
              with a checkpoint and custom nodes (Impact Pack,
              RMBG) installed. Default port 8188.
            </>
          }
        />
      </div>
      {backend === "embedded" ? (
        <>
          <ModelDownloadGuidance
            recommended={recommended}
            download={download}
            onDownload={onDownload}
            hasModels={embeddedModels.models.length > 0}
          />
          <label className="settings-field">
            <div className="settings-field-label">Models directory</div>
            <input
              type="text"
              value={modelsDir}
              onChange={(e) => onModelsDir(e.target.value)}
              placeholder={
                embeddedModels.models_dir
                  ? `default: ${embeddedModels.models_dir}`
                  : "leave blank for the per-user default"
              }
              spellCheck={false}
            />
            <div className="settings-field-hint">
              Folder where the engine looks for SDXL{" "}
              <code>.safetensors</code> checkpoints.
            </div>
          </label>
          <label className="settings-field">
            <div className="settings-field-label">Checkpoint</div>
            {embeddedModels.models.length > 0 ? (
              <select
                value={modelName}
                onChange={(e) => onModelName(e.target.value)}
              >
                <option value="">(auto-pick first available)</option>
                {embeddedModels.models.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            ) : (
              <div className="settings-field-hint">
                No checkpoints found in{" "}
                <code>
                  {embeddedModels.models_dir || "the configured directory"}
                </code>
                . Use the one-click download above, or grab an SDXL /
                Pony / Z-Image <code>.safetensors</code> checkpoint from{" "}
                <a
                  href={CIVITAI_COMPATIBLE_SEARCH}
                  target="_blank"
                  rel="noopener noreferrer"
                  data-testid="civitai-compatible-search-link-empty"
                >
                  Civitai's compatible-checkpoint search
                </a>{" "}
                and drop the file into that folder.
              </div>
            )}
          </label>
        </>
      ) : null}
      <div className="button-row">
        <button onClick={onBack}>Back</button>
        <button onClick={onNext} autoFocus>
          Next
        </button>
      </div>
    </>
  );
}

function formatGb(bytes: number | null | undefined): string {
  if (!bytes || bytes <= 0) return "";
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

/**
 * The image-model guidance + one-click download for the embedded
 * backend. Renders the recommendation copy (with the hardware-picked
 * model highlighted, a Civitai "here" link for the curious, and a
 * download button), then swaps the button for a live progress bar /
 * success / error state as the download runs.
 *
 * The one-click offer is the empty-directory bootstrap: once a
 * checkpoint exists we drop the button and show a lighter "you're set"
 * note, but keep the visual-style guidance + Civitai pointer so the
 * player still knows they can swap in any fine-tune.
 */
function ModelDownloadGuidance({
  recommended,
  download,
  onDownload,
  hasModels,
}: {
  recommended: RecommendedModel | null;
  download: DownloadState;
  onDownload: () => void;
  hasModels: boolean;
}): JSX.Element {
  // Fall back to "SDXL" for the highlight until the hardware probe
  // replies — it's the safe middle recommendation, so the copy reads
  // sensibly even on the first paint.
  const highlight = recommended?.display_name ?? "SDXL";
  // Offer the auto-download only when the directory is still empty
  // (the user's "if it starts with an empty models directory" rule)
  // and we're not already mid/post download.
  const offerDownload = !hasModels;

  return (
    <div className="model-guidance" data-testid="model-download-guidance">
      <p>
        The image model you use will dramatically change Lucidium's
        visual style. Z-Image and SDXL are both great starting points,
        but there are hundreds of alternatives{" "}
        <a
          href={CIVITAI_COMPATIBLE_SEARCH}
          target="_blank"
          rel="noopener noreferrer"
          data-testid="civitai-compatible-search-link"
        >
          here
        </a>{" "}
        that you may like to explore. Based on your hardware, a fine
        tune of{" "}
        <span className="model-guidance__highlight" data-testid="recommended-model">
          {highlight}
        </span>{" "}
        will probably give you the best experience. The application can
        handle any SDXL, Pony, Z-Image, or Qwen-Image fine-tune.
        {offerDownload ? (
          <>
            {" "}
            Or, if you prefer, you can simply{" "}
            <button
              type="button"
              className="model-guidance__download-link"
              onClick={onDownload}
              disabled={download.status === "running"}
              data-testid="download-recommended-model"
            >
              download the recommend base model
            </button>{" "}
            for your hardware.
          </>
        ) : null}
      </p>
      {offerDownload ? (
        <DownloadStatus download={download} recommended={recommended} />
      ) : (
        <div className="settings-field-hint" data-testid="model-already-installed">
          You already have a checkpoint installed — pick it below, or
          drop another <code>.safetensors</code> file into the models
          directory to switch styles.
        </div>
      )}
    </div>
  );
}

function DownloadStatus({
  download,
  recommended,
}: {
  download: DownloadState;
  recommended: RecommendedModel | null;
}): JSX.Element | null {
  if (download.status === "idle") {
    const size = formatGb(recommended?.approx_bytes);
    return (
      <div className="settings-field-hint">
        Downloads straight from the model's upstream HuggingFace repo
        {size ? ` (~${size} in total)` : ""}; its license applies. Nothing
        is fetched until you click.
        {size
          ? " That figure covers the checkpoint plus the text encoder and" +
            " VAE this model fetches the first time it renders — check it" +
            " against your data plan before starting."
          : ""}
      </div>
    );
  }
  if (download.status === "running") {
    const { bytesDone, bytesTotal, stage } = download;
    const pct =
      bytesTotal && bytesTotal > 0
        ? Math.min(100, Math.round((bytesDone / bytesTotal) * 100))
        : null;
    return (
      <div className="model-guidance__progress" data-testid="download-progress">
        <div className="model-guidance__progress-track">
          <div
            className={`model-guidance__progress-bar${
              pct === null ? " model-guidance__progress-bar--indeterminate" : ""
            }`}
            style={pct === null ? undefined : { width: `${pct}%` }}
          />
        </div>
        <div className="settings-field-hint">
          {stage === "downloading"
            ? `Downloading… ${formatGb(bytesDone)}${
                bytesTotal ? ` / ${formatGb(bytesTotal)}` : ""
              }${pct === null ? "" : ` (${pct}%)`}`
            : "Preparing download…"}
        </div>
      </div>
    );
  }
  if (download.status === "error") {
    return (
      <div
        className="settings-field-hint model-guidance__error"
        role="alert"
        data-testid="download-error"
      >
        {download.message} You can retry, or download a checkpoint
        manually from the Civitai link above.
      </div>
    );
  }
  // done
  return (
    <div
      className="settings-field-hint model-guidance__success"
      role="status"
      data-testid="download-success"
    >
      ✓ {download.filename ? `${download.filename} ` : "Model "}downloaded
      and selected — you're ready to play.
    </div>
  );
}

function DoneStep({
  backend,
  hasKey,
  onBack,
  onFinish,
}: {
  backend: Backend;
  hasKey: boolean;
  onBack: () => void;
  onFinish: () => void;
}): JSX.Element {
  return (
    <>
      <h2 className="page-title">Ready to play</h2>
      <p>You picked:</p>
      <ul className="setup-summary">
        <li>
          <b>LLM:</b>{" "}
          {hasKey
            ? "OpenRouter key configured"
            : "no API key yet — set one from Settings before starting a game"}
        </li>
        <li>
          <b>Image generator:</b>{" "}
          {backend === "embedded"
            ? "Embedded (in-process diffusers)"
            : "ComfyUI (external server)"}
        </li>
      </ul>
      <p>
        Click <b>Finish</b> to save these choices and head to the main menu.
        Every field is editable from Settings later.
      </p>
      <div className="button-row">
        <button onClick={onBack}>Back</button>
        <button onClick={onFinish} autoFocus>
          Finish
        </button>
      </div>
    </>
  );
}

function ChoiceCard({
  active,
  onSelect,
  title,
  body,
}: {
  active: boolean;
  onSelect: () => void;
  title: string;
  body: React.ReactNode;
}): JSX.Element {
  return (
    <button
      type="button"
      className={`setup-choice${active ? " setup-choice--active" : ""}`}
      onClick={onSelect}
      aria-pressed={active}
    >
      <div className="setup-choice__title">{title}</div>
      <div className="setup-choice__body">{body}</div>
    </button>
  );
}

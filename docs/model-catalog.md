# The embedded model catalog and download flow

Image generation defaults to the **embedded** backend: diffusers plus an
SDXL-family checkpoint loaded inside the engine process
(`ImageBackend.embedded`, the default for `ImageSettings.backend`). That
requires a checkpoint on disk, and Lucidium ships none — per `SAFETY.md`, no
image-generation weights are bundled. The catalog is how a player gets one
without leaving the app.

Code: `backend/src/lucidium/providers/embedded_models.py`.

## Where checkpoints live

`ImageSettings.embedded_models_dir` names the directory; an empty string falls
back to `config.py::embedded_models_dir()` — `<app-data>/models/image`.
`resolve_models_dir` is the single place that fallback rule lives, and it takes
an `allowed_root`: any directory that arrived over the WebSocket is confined to
the root the player already configured, so a download request can at most
narrow to a subdirectory rather than naming (say) the Startup folder. A
violation raises `ModelsDirOutsideRootError`.

`list_models` returns the top-level `.safetensors` / `.ckpt` files, sorted, and
an empty list when the directory does not exist yet. `pick_default_model` uses
`embedded_model_name` when it matches a file and otherwise takes the first file
alphabetically; `None` means "empty directory", which the caller turns into an
actionable error pointing at the download instructions.

## The catalog

`MODEL_CATALOG` maps a key to a `ModelSpec` (`hf_repo`, `hf_filename`,
`local_filename`, `approx_bytes`, `aux_approx_bytes`, `sha256`). Weights come
straight from the upstream public HuggingFace repo via the stable `resolve/main`
URL using plain `urllib` — Lucidium neither hosts nor relicenses them.

Sizes and digests are the upstream git-lfs pointer values
(`<repo>/raw/main/<file>` serves the pointer; its `oid sha256:` is the content
hash). `total_approx_bytes` — checkpoint **plus** the components fetched at
first render — is what the UI quotes, because that is what the player pays for
on a metered connection.

| Key | Model | Checkpoint | + first render | Total |
|---|---|---|---|---|
| `sdxl` | SDXL base 1.0 | ~6.94 GB | — | ~6.94 GB |
| `sdxl-turbo` | SDXL Turbo (few-step, fp16) | ~6.94 GB | — | ~6.94 GB |
| `z-image-turbo` | Z-Image Turbo (bf16 transformer) | ~12.3 GB | ~8.2 GB | ~20.5 GB |
| `qwen-image` | Qwen-Image, pre-distilled fp8 transformer | ~20.4 GB | ~16.8 GB | ~37.3 GB |
| `krea-2-turbo` | Krea 2 Turbo, fp8-scaled transformer | ~13.1 GB | ~9.1 GB | ~22.3 GB |

`local_filename` is chosen deliberately, not cosmetically: the loader's cheap
pre-filter keys off substrings in the filename (`z-image`/`zimage`, `qwen`,
`krea`, plus `Distill`/`Turbo` to select the few-step recipe) before it reads
the safetensors header. Renaming a downloaded file can send it down the wrong
pipeline.

For Z-Image, Qwen-Image and Krea 2 only the transformer is downloaded here; the
text encoder and VAE are fetched from their own upstream repos lazily at first
render. Z-Image and Krea 2 use Comfy-Org repackages — the official diffusers
repo has no single-file checkpoint for the former, and Krea's own repo is
licence-gated.

## Recommendation

`recommend_model` is pure given two signals — the torch overlay flavor
(`torch_overlay.recommend_flavor()`) and total VRAM
(`embedded_image_client.detect_total_vram_gb()`) — so the decision table is
unit-testable without hardware:

| Condition | Recommendation |
|---|---|
| CUDA/ROCm and VRAM ≥ 40 GB | `qwen-image` |
| CUDA/ROCm and VRAM ≥ 16 GB | `z-image-turbo` |
| Any other GPU (DirectML, Intel Arc), or CUDA/ROCm with unknown or 8–16 GB | `sdxl` |
| GPU with *known* < 8 GB VRAM | `sdxl-turbo` |
| No GPU (CPU) | `sdxl-turbo` |

These are recommendation thresholds, not hard floors — Qwen-Image runs on a
24 GB card via block-level streaming, just slower per step, and stays
downloadable by key. `krea-2-turbo` is never auto-recommended: it is
downloadable explicitly, but the first-run default is left unchanged so an
existing install's wizard behaviour does not shift.

## The download flow

Nothing is ever fetched silently. When the models directory is empty the
first-run wizard *offers* a download; the player clicks.

1. `c2s/embedded/list_models` → `s2c/embedded/models` populates the dropdown.
2. `c2s/embedded/recommend_model` → `s2c/embedded/recommended_model` carries the
   key, display name, the reason, `has_models`, and `approx_bytes` for the
   "~N GB" label shown before any real Content-Length arrives.
3. `c2s/embedded/download_model { key, models_dir }` streams
   `s2c/embedded/download_progress` (`stage`, `bytes_done`, `bytes_total` — null
   when the server sends no Content-Length).
4. A failure raises `ModelDownloadError`, surfaced as `s2c/error` so the wizard
   can offer a retry or point the player at Civitai to pick their own fine-tune.

Retries are safe by construction, which matters because a dropped WebSocket does
not cancel the download already running:

* calls for the same target file serialise on a per-path lock, so the retry
  waits and then finds the finished file instead of racing the writer;
* each attempt writes to its own uuid-suffixed `.part`, so no attempt can
  truncate another's temp file;
* the atomic rename is gated on the published `sha256` (or, with no digest, on
  a `Content-Length` comparison). A truncated or corrupt stream is deleted. This
  is the check that matters most, because `download_model` returns an existing
  target untouched — anything published into place is never re-fetched.

An interrupted download is resumed by restarting it: delete the partial file
from the models directory and retry. Players who prefer to supply their own
weights can simply drop a `.safetensors` into the models directory — the
dropdown lists whatever is there.

## See also

- [torch-overlay.md](torch-overlay.md) — the runtime that actually executes these checkpoints.
- [operations.md](operations.md) — where the models directory sits on each platform.
- `SAFETY.md` — why no weights ship in the installer.

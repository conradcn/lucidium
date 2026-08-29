# Adding a custom ComfyUI workflow

Lucidium's `comfyui` image backend renders every portrait and background by
POSTing an **API-format** workflow graph to your ComfyUI server. The graphs are
plain JSON files on disk; adding your own is a matter of exporting a graph,
inserting the engine's placeholder tokens, dropping it in the workflow
directory, and pointing Settings at the filename.

This document covers the ComfyUI backend only. The default `embedded` backend
runs diffusers in-process and ignores these files — it reimplements the two
bundled pipelines in Python
(`backend/src/lucidium/providers/embedded_image_client.py`); see
[character-rendering-stack.md](character-rendering-stack.md).

## 1. Where workflows live

`backend/workflows/` in a source checkout, resolved by
`_default_workflow_root()` in `backend/src/lucidium/orchestration/session.py`.
Packaged builds ship the same directory next to the executable as `workflows/`
(`backend/lucidium.spec`), so the same relative path works frozen or not. If
that directory is missing entirely, the engine falls back to the toy graphs in
`backend/src/lucidium/orchestration/prompts/comfy/` — those exist for tests and
are not meant to be run against a real ComfyUI.

Currently shipped:

| File | Role | Output size |
|---|---|---|
| `character.json` | Portrait: SDXL → FaceDetailer → RMBG background removal → `PreviewImage` | 832×1216 (2:3) |
| `background.json` | Environment: SDXL → `PreviewImage` | 1536×1024 (3:2) |

Supporting assets in the same directory (`idle_pose_reference.png` for the
ControlNet slot, `placeholders/`) are referenced by filename from the graphs.

## 2. Export the graph in API format

In ComfyUI: build and test the graph in the web UI, enable
*Settings → Enable Dev mode options*, then **Save (API Format)**. The result is
a flat object keyed by node id:

```json
{
  "4": { "class_type": "EmptyLatentImage",
         "inputs": { "width": 832, "height": 1216, "batch_size": 1 } }
}
```

The engine never reads the editor-format export (the one with `nodes`/`links`
arrays) — it POSTs the file verbatim as `{"prompt": <graph>}`, so it must be the
API-format file.

## 3. Insert the placeholder tokens

`ComfyUIImageClient._instantiate_workflow`
(`backend/src/lucidium/providers/image_client.py:130`) does **textual**
substitution on the raw file before parsing it: every `PLACEHOLDER_<KEY>`
token, uppercased from a param key, is replaced with the JSON-escaped value.
Put the token inside a string field, unquoted-by-itself, exactly as in
`character.json`:

```json
"2": { "class_type": "CLIPTextEncode",
       "inputs": { "text": "PLACEHOLDER_POSITIVE_PROMPT", "clip": ["1", 1] } }
```

Tokens the engine supplies (see `backend/src/lucidium/orchestration/assets.py`):

| Token | Supplied for | Meaning |
|---|---|---|
| `PLACEHOLDER_POSITIVE_PROMPT` | portraits and backgrounds | The main scene/subject prompt. |
| `PLACEHOLDER_FACE_PROMPT` | portraits | Face-only prompt, intended for the FaceDetailer node's positive input. |
| `PLACEHOLDER_NEGATIVE_EXTRAS` | portraits | Per-character extra negatives, appended to your static negative string. |
| `PLACEHOLDER_SUBJECT_KIND` | portraits | `human` / non-human subject kind; the embedded backend routes on it, ComfyUI graphs can ignore it. |

Rules that fall out of the implementation:

- **Substitution is textual, parsing comes after.** Values are escaped with
  `json.dumps`, so prompts containing quotes or commas stay valid JSON — but a
  malformed template is only detected at render time, not at startup.
- **Unused tokens are safe.** Any `PLACEHOLDER_[A-Z_]+` left over after
  substitution is replaced with the empty string
  (`_strip_unfilled_placeholders`), so optional slots never leak the literal
  token into a prompt.
- **Seeds are injected, not templated.** Every node input named `seed` or
  `noise_seed` is overwritten with the engine's deterministic seed. Do not add a
  `PLACEHOLDER_SEED`; just leave the value at `0`. Both names are set to the
  same value so regenerating a character reproduces its image.
- **The last output node wins.** On completion the engine sorts output nodes by
  numeric id descending and takes the first image it finds, so your final
  `PreviewImage` (or `SaveImage`) must have the **highest numeric node id** in
  the graph. This is what makes the post-processed portrait beat the raw
  sampler output in `character.json`.
- **"pony" in the file triggers pony tagging.** If the raw JSON contains the
  substring `pony` anywhere (case-insensitive — normally the checkpoint
  filename), the engine prefixes `score_9, score_8_up, score_7_up, score_6_up,`
  onto the positive and face prompts and prepends the matching `score_4, …`
  negatives. Avoid the substring if you don't want that; include it if your
  checkpoint is a Pony derivative.

## 4. Keep the size contract

The frontend composites portraits as transparent cut-out figures over a
landscape background. Render at the canonical SDXL buckets — 832×1216 for
character workflows, 1536×1024 for backgrounds. Straying from the bucket (e.g.
square 1024×1024) degrades anatomy on SDXL/Pony checkpoints and misaligns the
composite. Portrait workflows should also end in background removal (RMBG or
equivalent) so the figure arrives with an alpha channel.

## 5. Install and select it

1. Drop `my_portrait.json` into `backend/workflows/` (or the `workflows/`
   directory beside the packaged executable).
2. Install any custom nodes and models the graph needs — for `character.json`
   those are ComfyUI-Impact-Pack, ComfyUI-RMBG, `face_yolov8m.pt` and
   `sam_vit_b_01ec64.pth`; see `backend/workflows/README.md`.
3. In the app, **Settings → Image**: set the backend to ComfyUI, set **Base
   URL** to your server (default `http://127.0.0.1:8000`), and type the
   filename — including `.json` — into **Portrait workflow** or **Background
   workflow**. The value is the filename only; it is resolved relative to the
   workflow directory.

Naming matters beyond the file lookup: the embedded backend and the dimension
resolver classify a workflow as a *background* one when its name contains
`background`, `environment`, or `scene`, and treat everything else as a
portrait. Name accordingly so a fallback to the embedded backend still routes
correctly.

Defaults in `backend/src/lucidium/config.py` are `portrait.workflow.json` /
`background.workflow.json`, which match the test fallbacks rather than the
shipped `character.json` / `background.json` — so a fresh ComfyUI setup does
need these two fields filled in explicitly.

## 6. Verify

- **Direct render, no UI:** `backend/scripts/krea_live_render.py` and
  `backend/scripts/regen_main_menu.py` drive workflows straight through the
  client; `backend/scripts/krea_bisect.py` is useful for isolating which node
  changed a result.
- **End to end against a live ComfyUI:**
  `frontend/e2e/image-pipeline-live.spec.ts` submits the production portrait
  workflow and checks the pipeline; `frontend/e2e/live-app.spec.ts` requires
  both shipped workflows to exist.
- **Failure modes to expect:** a missing file raises
  `ProviderValidationError: missing ComfyUI workflow …`; an unreachable or
  erroring server retries (`IMAGE_RETRY_BUDGET`) then raises
  `ProviderUnreachableError`; a graph that produces no image within the
  completion timeout (600 s default) times out. Every attempt is logged as
  `image-call workflow=… seed=… …` with the full prompt set logged *before* the
  request, so a hung render still leaves its prompts in the journal.

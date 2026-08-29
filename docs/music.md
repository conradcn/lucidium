# Background music (ACE-Step)

Lucidium generates instrumental, looping background music with
[ACE-Step](https://github.com/ace-step/ACE-Step). Unlike image generation, the
model is **not** loaded in-process: ACE-Step's pipeline is ~10 GB and serving it
locally is well-trodden by upstream tooling, so the engine talks to a separate
ACE-Step inference server over HTTP.

Code: `backend/src/lucidium/providers/music_client.py` (`AceStepClient`),
settings in `domain/settings.py::MusicSettings`.

## Turning it on

`music.enabled` defaults to **false** — generating a track adds ~30–90 s to the
new-game flow and not every player wants scored play. When enabled, the engine
needs a reachable ACE-Step server; `music.base_url` defaults to
`http://127.0.0.1:8001` (deliberately not 7860, which the player may already be
using for a Gradio app).

Other settings: `model_name` (the DiT checkpoint the server loads, default
`acestep-v15-turbo`), `clip_seconds` (15–240, default 60), `inference_steps`
(10–100, default 27), `guidance_scale` (1–30, default 15), and
`local_gpu_coordination`.

## When tracks are generated

- **Every new game** — any path (New Game, Surprise Me, or a save load that
  produces a fresh track) generates one track from a prompt the `world_init`
  LLM call emits alongside the world.
- **Mid-game changes** — the storyteller's beat schema carries a `music_change`
  field; when the LLM sets it, the engine re-renders.
- **On demand** — `c2s/music/regenerate` with an explicit prompt (empty reuses
  the scene-derived caption). The finished track arrives as `s2c/music/ready`
  with the on-disk path.

`c2s/music/inventory` probes a server for reachability and its model list
(`GET /v1/model_inventory`); the reply populates the settings dropdown, and a
failed probe comes back as `ok: false` with the error text rather than an
exception.

## Wire shape

The client speaks the real upstream ACE-Step FastAPI server's OpenAI-compatible
surface:

1. `POST {base_url}/v1/init` **once per process** to load the selected model
   into the server's slot 1. Already-initialised servers answer 200 with the
   existing slot info; a server missing the endpoint downshifts this to a no-op.
2. `POST {base_url}/v1/chat/completions` with the prompt wrapped as
   `messages: [{role: "user", content: ...}]`. The response's
   `choices[0].message.audio[0].audio_url.url` is a
   `data:audio/mpeg;base64,...` URL; the engine decodes it and writes the bytes
   straight to disk.

## LM bypass

ACE-Step's chat-completions handler loads its bundled 5 Hz LM whenever any of
`thinking`, `sample_mode`, `use_format`, `use_cot_caption`, or
`use_cot_language` is true — and all five default to true on the server's
request model. Lucidium sets **all five to false** and keeps `init_llm` false.
Our captions already come from a scene-aware LLM call, so ACE-Step's LM would
only rewrite them. Skipping it saves ~3.5 GB of VRAM, removes an LM round trip
from every render, and keeps the music grounded in the actual scene.

## GPU coordination

With `local_gpu_coordination` on (the default), the engine assumes ACE-Step is
competing for the *same* local GPU as the embedded SDXL pipeline: music and
image inference are serialised through a shared lock, and the SDXL pipelines are
evicted to CPU before each music call (the next render brings them back). Turn
it off when ACE-Step runs on another machine or another GPU — consecutive music
calls are still serialised by the audio client's own lock, but the engine stops
shuttling SDXL weights off the GPU on every swap.

## See also

- [model-catalog.md](model-catalog.md) — the image-side checkpoint download flow.
- [torch-overlay.md](torch-overlay.md) — the runtime torch installation.

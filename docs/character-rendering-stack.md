# Character Rendering Stack — Research Notes (May 2026)

Captured during the autoresearch loop on character consistency. The
current SDXL + DreamShaper + OpenPose ControlNet stack hit a ceiling
on three of the four target axes (framing/pose/render-to-render
consistency). This document records the alternatives surveyed and
the recommended migration path.

## Current Stack (May 2026)

- **Base:** SDXL via `dreamshaperXL10_alpha2Xl10.safetensors`
- **Pose conditioning:** `OpenPoseXL2.safetensors` ControlNet,
  strength 0.1, end_percent 0.6, fed `idle_pose_reference.png`
- **Face refinement:** FaceDetailer (`face_yolov8m.pt` + SAM)
- **Background removal:** RMBG-2.0
- **Identity preservation:** seed-only (per-character `Character.seed`)

## Top Recommendation

**FLUX.1 [dev] FP8 + PuLID-Flux + ControlNet-Union-Pro 2.0 (DWPose)**
as the primary stack, with **SDXL + IPAdapter-FaceID-PlusV2 +
DWPose** as a fallback when the player picks anime/painterly
visual styles.

### Why FLUX over SDXL

| Axis | SDXL today | FLUX.1 [dev] |
|------|-----------|--------------|
| Pose adherence | Collapses non-standing poses | Renders kneeling/seated/running with correct anatomy |
| Prompt following | Tag-bag, weak on long natural-language scenes | T5 encoder, genuinely understands sentence-form prompts |
| Hands/anatomy | Routinely deformed | Workable, hands still the weak spot |
| Style flexibility | Strong out of the box | Photoreal bias, needs style LoRAs at 1.0–1.4 |
| 4090 fit | Trivial | FP8 ~13 GB, room for adapters; BF16 too tight |

### Identity preservation: PuLID-Flux + ReActor (hybrid)

Single-seed identity is fragile across many renders. The recommended
pattern:

1. **Once per character:** generate a canonical headshot with
   FLUX-dev + PuLID + a reference image. Lock the seed and prompt;
   this is the identity source of truth.
2. **Per scene:** render with the player-chosen style (FLUX or SDXL
   + style LoRA), driven by a DWPose skeleton sized for full-body
   framing.
3. **Final pass:** ReActor swaps the canonical face from step 1
   onto the rendered output.

This decouples identity (locked in step 3) from style (free in
step 2) from pose/framing (locked by ControlNet in step 2). Side
benefit: PuLID's known style-bleed problem disappears because PuLID
only runs during the one-time character creation.

### Pose conditioning: DWPose, not OpenPose

DWPose is strictly better than OpenPose at hands and full-body
landmarks. The current workflow's `OpenPoseXL2.safetensors` should
be replaced.

For full-body framing specifically, the reliable trick is to feed a
DWPose skeleton **sized for full-body composition** (head near top
15%, feet near bottom 10% of canvas). Prompt-only framing tags hit a
ceiling — the skeleton's spatial extent is what actually forces the
framing. This is consistent with what we observed in iters 1–4.

ControlNet checkpoints to use:
- **FLUX:** `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`
  (single 4 GB checkpoint covers pose, depth, canny, soft-edge).
  Settings: `controlnet_conditioning_scale=0.9`,
  `control_guidance_end=0.65`. Past 0.65 over-constrains.
- **SDXL fallback:** `xinsir/controlnet-union-sdxl-1.0` driven by
  the DWPose preprocessor.

## Models to skip

| Model | Why not |
|-------|---------|
| SD 3.5 Large | ID-adapter support immature; pose adherence below FLUX |
| PixArt-Σ, Hunyuan-DiT, Kolors | No production-grade ID-preservation adapters |
| HiDream-I1 | FP8/Q8 quants barely fit a 4090; adapter ecosystem nascent. Re-evaluate in 6 months. |
| InstantID on FLUX | No clean port; SDXL-only |
| PhotoMaker v2 | Superseded by PuLID for VN-style use |

## Concrete 4090 Build

ComfyUI custom nodes:
- `balazik/ComfyUI-PuLID-Flux` (or `sipie800` enhanced fork)
- `Gourieff/ComfyUI-ReActor`
- `comfyui_controlnet_aux` (DWPose preprocessor)
- `ComfyUI-Advanced-ControlNet`

Weights (~30 GB total):
- `flux1-dev-fp8.safetensors` (~13 GB)
- `t5xxl_fp8_e4m3fn.safetensors` + `clip_l.safetensors` (~5 GB)
- `ae.safetensors` (FLUX VAE, ~330 MB)
- `pulid_flux_v0.9.1.safetensors` (~1.1 GB) + EVA-CLIP +
  antelopev2 face encoder
- `FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors` (~4 GB)
- 3–4 style LoRAs per supported visual style — ~200 MB each
- ReActor: `inswapper_128.onnx` + `GFPGANv1.4.pth` (~700 MB)

Expected per-render: ~8–12 s for FLUX-dev FP8 at 1024×1024, plus
~1–2 s for the ReActor pass. Acceptable for VN scene generation;
cache aggressively keyed on `(character_id, pose, style, prompt_hash)`.

## Migration Sequence (suggested)

1. Add `ComfyUI-ReActor` to the existing SDXL workflow first —
   strongest single win, doesn't replace any current weights.
   Generate one canonical headshot per character at session start
   (already aligned with the new "PC portrait render after
   Character Description" flow); ReActor it onto every subsequent
   in-game render. This alone should solve render-to-render
   consistency without touching the base model.
2. Swap OpenPose → DWPose for the existing SDXL workflow. Should
   immediately help pose adherence on iter-4-class problems.
3. Then evaluate FLUX migration on a separate workflow file
   (`character_flux.json`) so the SDXL path remains as a fallback
   for stylized renders.

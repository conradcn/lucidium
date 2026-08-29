# ComfyUI Workflows

Each JSON in this directory is an API-format ComfyUI workflow. The engine
loads them by filename and injects the prompt / seed at generation time.

To add your own, see
[docs/comfyui-workflows.md](../../docs/comfyui-workflows.md).

## `character.json`

Pipeline: base generation → **face detail refinement** → **background
removal** → preview.

### Required ComfyUI custom nodes

Install via ComfyUI-Manager or git clone into `ComfyUI/custom_nodes/`:

1. **ComfyUI-Impact-Pack** — provides `FaceDetailer`,
   `UltralyticsDetectorProvider`, `SAMLoader`.
   <https://github.com/ltdrdata/ComfyUI-Impact-Pack>

2. **ComfyUI-RMBG** (by 1038lab) — provides the single `RMBG` node that
   loads the model internally (no separate loader node needed).
   <https://github.com/1038lab/ComfyUI-RMBG>

### Required models

- `ComfyUI/models/ultralytics/bbox/face_yolov8m.pt` (face detector).
- `ComfyUI/models/sams/sam_vit_b_01ec64.pth` (segmentation).
- The RMBG model (`RMBG-2.0` by default) downloads automatically on
  first run of the `RMBG` node into `ComfyUI/models/RMBG/`.

### Swapping extensions

If your install uses a different background-remover or face-detailer
(e.g. `rembg` node pack, or `LayerMask: RemBgUltra`), edit the relevant
nodes in `character.json`. The engine only cares that the final
`PreviewImage` node exists and wires to the last image in the chain.

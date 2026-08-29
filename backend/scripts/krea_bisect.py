"""Bisect the Krea 2 embedded path: render the same prompt with the
transformer placed two different ways.

    python scripts/krea_bisect.py <checkpoint> <mode> [<out.png>]

``mode`` is ``fp8`` (torchao, the production placement) or ``bf16``
(sequential CPU offload, no quantization). If ``bf16`` renders cleanly
and ``fp8`` doesn't, the fault is the quantization step, not the
ComfyUI->diffusers conversion.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from lucidium.providers import embedded_image_client as eic

PROMPT = (
    "a full body portrait of a young woman with short auburn hair, wearing a "
    "green wool coat, standing in a plain grey studio, soft lighting"
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    path = Path(sys.argv[1])
    mode = sys.argv[2]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"krea_renders/_bisect_{mode}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    pipeline = eic._load_krea_pipeline(path, device="cpu")  # no placement yet
    print(f"loaded in {time.monotonic() - started:.0f}s", flush=True)

    if mode == "fp8":
        eic._apply_krea_offload(pipeline, "cuda", torch.device("cuda"))
    elif mode == "bf16":
        pipeline.enable_sequential_cpu_offload(device=torch.device("cuda"))
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    steps, guidance = eic._krea_recipe(pipeline)
    print(f"mode={mode} steps={steps} guidance={guidance}", flush=True)
    started = time.monotonic()
    kwargs: dict[str, object] = {
        "prompt": PROMPT,
        "height": 512,
        "width": 512,
        "num_inference_steps": steps,
        "generator": torch.Generator("cpu").manual_seed(12345),
    }
    kwargs["guidance_scale"] = guidance
    negative = "blurry, low quality"
    # CPU-encode-resident placement parks the text encoder on the CPU, so
    # the prompt must be encoded there and handed in as embeds — exactly
    # what _run_pipeline does for the real render.
    cpu_encode = getattr(pipeline, eic._KREA_CPU_ENCODE_ATTR, False)
    if cpu_encode:
        kwargs.update(
            eic._cpu_encode_prompts(
                pipeline,
                PROMPT,
                negative,
                encode_negative=guidance > 0,
            )
        )
        kwargs.pop("prompt")
    else:
        kwargs["negative_prompt"] = negative
    with torch.no_grad():
        result = eic._run_qwen_call(pipeline, kwargs) if cpu_encode else pipeline(**kwargs)
    image = result.images[0]
    image.save(out)
    print(f"{mode}: {time.monotonic() - started:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

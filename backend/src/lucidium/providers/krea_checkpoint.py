"""ComfyUI → diffusers conversion for Krea 2 single-file checkpoints.

diffusers ships ``Krea2Pipeline`` / ``Krea2Transformer2DModel`` (0.39+)
but — unlike SDXL and Z-Image — has NO single-file loader for them. The
checkpoints players actually download (the Comfy-Org repackage, every
Civitai finetune built on it) are ComfyUI-format single files carrying
the transformer ONLY, under the reference implementation's key names
rather than the diffusers module names. This module bridges that gap:

  * :func:`is_krea_state_dict` — sniff a safetensors header so the
    loader can tell a Krea 2 transformer from any other single file
    without trusting the filename.
  * :func:`infer_krea_config` — derive the ``Krea2Transformer2DModel``
    config (layer counts, widths, head counts) from tensor shapes, so a
    finetune that changed a dimension still loads instead of blowing up
    on a shape mismatch against hardcoded defaults.
  * :func:`load_krea_transformer_state_dict` — stream the file, rename
    every key to the diffusers layout, and dequantize the quantized
    variants back to the compute dtype.

Quantized variants
------------------
Comfy-Org publishes Krea 2 as bf16, ``fp8_scaled`` and ``int8_convrot``
(Civitai finetunes mirror those three). Both quantized layouts store the
weight next to a companion ``<name>_scale`` tensor and are recovered the
same way — ``w.to(dtype) * scale``:

  * ``fp8_scaled`` — float8_e4m3fn weights with a per-TENSOR scalar
    scale.
  * ``int8`` — int8 weights with a per-OUTPUT-CHANNEL scale of shape
    ``(out_features, 1)``, which broadcasts over the input axis.

The ``convrot`` ("rotated") int8 variant is NOT supported, and
:func:`krea_rotation_is_folded` detects it so the loader can say so.
Those builds fold a Hadamard rotation ``R`` into the input axis of every
quantized weight (``W' = W R``) and expect the runtime to rotate the
activations to match. ``R`` is generated deterministically by the
inference engine rather than stored, so it cannot be recovered from the
file, and dequantizing ``W'`` on its own yields a network that decodes
to pure noise. Verified live: the rotation preserves per-tensor and
per-row statistics exactly, so only the column-norm flattening it causes
distinguishes a rotated checkpoint from a plain one.

Dequantizing to bf16 costs ~24 GiB of HOST RAM for the 12B transformer.
That's deliberate: the caller (``embedded_image_client._load_krea_
pipeline``) immediately re-quantizes to native fp8 on the GPU via
torchao, which needs bf16 weights as its input. The host copy is freed
straight after.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ComfyUI wraps some repackages in the full-model namespace (the
# ``int8_convrot`` build does; ``fp8_scaled`` does not). Stripped before
# any key matching so both layouts share one mapping table.
_COMFY_PREFIX = "model.diffusion_model."

# Suffix marking a quantization scale companion tensor. Never emitted;
# consumed by the dequantizer that rebuilds the weight it belongs to.
_SCALE_SUFFIX = "_scale"

# Keys that identify a Krea 2 transformer beyond reasonable doubt. The
# ``txtfusion`` stack (layerwise blocks + layer-axis projector) is unique
# to Krea 2 — no other single-file checkpoint family carries it.
_SIGNATURE_KEYS: tuple[str, ...] = (
    "txtfusion.projector.weight",
    "first.weight",
    "last.linear.weight",
)

# Per-block leaf renames, shared by the image blocks and both text-fusion
# block stacks (they're the same ``Krea2TransformerBlock`` shape minus
# the time modulation, so the leaves line up exactly).
_BLOCK_LEAVES: dict[str, str] = {
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.gate.weight": "attn.to_gate.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "mlp.down.weight": "ff.down.weight",
    # Image blocks only — the per-block additive modulation table. Stored
    # flat as (6 * hidden_size,) and reshaped to (6, hidden_size) by
    # :func:`_reshape_for_diffusers`.
    "mod.lin": "scale_shift_table",
}

# Top-level (non-block) renames.
_TOP_LEVEL: dict[str, str] = {
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    # Timestep MLP: index 1 in the reference Sequential is the GELU, so
    # the two Linears are at 0 and 2.
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    # Shared 6-way modulation projection (index 0 is the GELU).
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias": "time_mod_proj.bias",
    # Text projection: RMSNorm, Linear, GELU, Linear.
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
    # Final adaptive-norm layer. ``modulation.lin`` is already (2, hidden).
    "last.modulation.lin": "final_layer.scale_shift_table",
    "last.norm.scale": "final_layer.norm.weight",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
}

# ComfyUI block-stack prefix -> diffusers block-stack prefix.
_BLOCK_STACKS: tuple[tuple[str, str], ...] = (
    # Longest first: ``txtfusion.layerwise_blocks.`` would otherwise be
    # shadowed by a shorter prefix match.
    ("txtfusion.layerwise_blocks.", "text_fusion.layerwise_blocks."),
    ("txtfusion.refiner_blocks.", "text_fusion.refiner_blocks."),
    ("blocks.", "transformer_blocks."),
)


class KreaCheckpointError(RuntimeError):
    """Raised when a checkpoint claims to be Krea 2 but can't be mapped
    onto ``Krea2Transformer2DModel`` — an unknown key, a missing tensor
    the config is derived from, or a quantization layout we don't
    recognise. Surfaced to the player as a load failure naming the file
    rather than a downstream shape-mismatch stack trace."""


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Return the safetensors header (tensor name -> {dtype, shape, ...})
    of ``path`` without mapping any tensor data.

    Reads only the 8-byte length prefix plus the JSON header, so this is
    cheap enough to run on every checkpoint in the models directory
    during a family sniff. The ``__metadata__`` entry is stripped.
    """
    import json
    import struct

    with open(path, "rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) < 8:
            raise KreaCheckpointError(f"{path.name} is not a safetensors file")
        (length,) = struct.unpack("<Q", raw_length)
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise KreaCheckpointError(f"{path.name} has a malformed safetensors header")
    return {k: v for k, v in header.items() if k != "__metadata__"}


def _strip_prefix(key: str) -> str:
    return key[len(_COMFY_PREFIX) :] if key.startswith(_COMFY_PREFIX) else key


def is_krea_state_dict(keys: Iterator[str] | list[str]) -> bool:
    """True when ``keys`` (a safetensors header's tensor names) belong to
    a Krea 2 transformer. Content-based, so a Civitai finetune under any
    filename is still recognised."""
    stripped = {_strip_prefix(k) for k in keys}
    return all(sig in stripped for sig in _SIGNATURE_KEYS)


def _require(header: dict[str, Any], key: str, path_name: str) -> list[int]:
    entry = header.get(key)
    if entry is None:
        raise KreaCheckpointError(
            f"{path_name} looks like a Krea 2 checkpoint but is missing the "
            f"{key!r} tensor, so its architecture can't be determined"
        )
    return list(entry["shape"])


def infer_krea_config(header: dict[str, Any], *, path_name: str = "checkpoint") -> dict[str, Any]:
    """Derive the ``Krea2Transformer2DModel`` config from tensor shapes.

    Everything except the rotary axis split is recoverable from the
    weights, so a finetune that widened a dimension loads correctly
    instead of mismatching hardcoded defaults. ``axes_dims_rope`` is not
    encoded in any tensor; the Krea 2 split ``(32, 48, 48)`` is used and
    validated against the head dim we DID infer.
    """
    stripped = {_strip_prefix(k): v for k, v in header.items()}

    hidden_size, in_channels = _require(stripped, "first.weight", path_name)
    num_layers = _count_stack(stripped, "blocks.")
    num_layerwise = _count_stack(stripped, "txtfusion.layerwise_blocks.")
    num_refiner = _count_stack(stripped, "txtfusion.refiner_blocks.")

    # Head dim comes from the q/k RMSNorm, which is sized per head.
    (head_dim,) = _require(stripped, "blocks.0.attn.qknorm.qnorm.scale", path_name)
    kv_out, _ = _require(stripped, "blocks.0.attn.wk.weight", path_name)
    intermediate_size, _ = _require(stripped, "blocks.0.mlp.gate.weight", path_name)
    _, timestep_embed_dim = _require(stripped, "tmlp.0.weight", path_name)

    text_hidden_dim = _require(stripped, "txtmlp.1.weight", path_name)[1]
    _, num_text_layers = _require(stripped, "txtfusion.projector.weight", path_name)
    (text_head_dim,) = _require(
        stripped, "txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale", path_name
    )
    text_kv_out, _ = _require(stripped, "txtfusion.layerwise_blocks.0.attn.wk.weight", path_name)
    text_intermediate_size, _ = _require(
        stripped, "txtfusion.layerwise_blocks.0.mlp.gate.weight", path_name
    )

    if hidden_size % head_dim:
        raise KreaCheckpointError(
            f"{path_name}: hidden size {hidden_size} is not divisible by the "
            f"head dim {head_dim} inferred from the q/k norm"
        )
    axes_dims_rope = (32, 48, 48)
    if sum(axes_dims_rope) != head_dim:
        raise KreaCheckpointError(
            f"{path_name}: attention head dim is {head_dim}, but the Krea 2 "
            f"rotary axis split {axes_dims_rope} sums to {sum(axes_dims_rope)}. "
            "This checkpoint uses an architecture variant Lucidium can't "
            "configure."
        )

    return {
        "in_channels": in_channels,
        "num_layers": num_layers,
        "attention_head_dim": head_dim,
        "num_attention_heads": hidden_size // head_dim,
        "num_key_value_heads": kv_out // head_dim,
        "intermediate_size": intermediate_size,
        "timestep_embed_dim": timestep_embed_dim,
        "text_hidden_dim": text_hidden_dim,
        "num_text_layers": num_text_layers,
        "text_num_attention_heads": text_hidden_dim // text_head_dim,
        "text_num_key_value_heads": text_kv_out // text_head_dim,
        "text_intermediate_size": text_intermediate_size,
        "num_layerwise_text_blocks": num_layerwise,
        "num_refiner_text_blocks": num_refiner,
        "axes_dims_rope": axes_dims_rope,
        "rope_theta": 1000.0,
        "norm_eps": 1e-5,
    }


def _count_stack(stripped: dict[str, Any], prefix: str) -> int:
    """Number of blocks in the ``prefix`` stack, from the highest index
    present. Ignores the deeper ``txtfusion.*`` stacks when counting the
    top-level ``blocks.`` one (their keys don't start with ``blocks.``)."""
    highest = -1
    for key in stripped:
        if not key.startswith(prefix):
            continue
        index = key[len(prefix) :].split(".", 1)[0]
        if index.isdigit():
            highest = max(highest, int(index))
    if highest < 0:
        raise KreaCheckpointError(
            f"Krea 2 checkpoint has no {prefix!r} blocks; it may be a LoRA or "
            "a partial file rather than a full transformer"
        )
    return highest + 1


def _map_key(key: str, *, path_name: str) -> str:
    """Rename one ComfyUI key to its diffusers module path."""
    key = _strip_prefix(key)
    mapped = _TOP_LEVEL.get(key)
    if mapped is not None:
        return mapped
    for comfy_prefix, diffusers_prefix in _BLOCK_STACKS:
        if not key.startswith(comfy_prefix):
            continue
        remainder = key[len(comfy_prefix) :]
        index, _, leaf = remainder.partition(".")
        target = _BLOCK_LEAVES.get(leaf)
        if target is None:
            break
        return f"{diffusers_prefix}{index}.{target}"
    raise KreaCheckpointError(
        f"{path_name}: unrecognised Krea 2 tensor {key!r}. The checkpoint may "
        "use an architecture variant Lucidium doesn't support yet."
    )


def _reshape_for_diffusers(target_key: str, tensor: Any, hidden_size: int) -> Any:
    """Reshape the flat per-block modulation table to the (6, hidden)
    parameter diffusers declares. Every other tensor passes through."""
    if target_key.endswith("scale_shift_table") and tensor.ndim == 1:
        return tensor.reshape(-1, hidden_size)
    return tensor


def load_krea_transformer_state_dict(
    path: Path,
    *,
    dtype: Any,
    hidden_size: int,
) -> dict[str, Any]:
    """Read ``path`` and return a diffusers-keyed, dequantized state dict
    in ``dtype``.

    Tensors are pulled one at a time from the memory-mapped file so peak
    host RAM is the OUTPUT dict (~24 GiB in bf16 for the 12B transformer)
    rather than twice that.
    """
    import torch
    from safetensors import safe_open

    state_dict: dict[str, Any] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        scale_keys = {k for k in keys if k.endswith(_SCALE_SUFFIX)}
        quantized = 0
        for key in keys:
            if key in scale_keys:
                continue  # Consumed alongside the weight it scales.
            tensor = handle.get_tensor(key)
            scale_key = key + _SCALE_SUFFIX
            if scale_key in scale_keys:
                # fp8 (per-tensor scalar) or int8 (per-output-channel,
                # shape (out, 1)) — both recover as w * scale, and the
                # int8 scale broadcasts over the input axis as-is.
                scale = handle.get_tensor(scale_key).to(torch.float32)
                tensor = (tensor.to(torch.float32) * scale).to(dtype)
                quantized += 1
            else:
                tensor = tensor.to(dtype)
            target = _map_key(key, path_name=path.name)
            state_dict[target] = _reshape_for_diffusers(target, tensor, hidden_size)
    if quantized:
        _log.info(
            "krea checkpoint: dequantized %d/%d tensors from %s to %s",
            quantized,
            len(state_dict),
            path.name,
            dtype,
        )
    return state_dict


# Filename tags for the rotated int8 quantization. Explicit in every
# build that ships one, and checking the name costs nothing — but the
# statistical probe below is the authority, so an untagged rotated
# finetune is still caught.
_ROTATED_TAGS: tuple[str, ...] = ("convrot", "conv_rot", "hadamard", "quarot")

# Quantized tensors to probe for folded rotation, in preference order.
# Probe choice is not free: measured across the real pair, the first
# block's MLP projections separate by 10x (0.03 rotated vs 0.32-0.57
# plain) while the attention projections separate by barely 2x and deep
# blocks by less still. Only the wide-margin tensors are probed.
_ROTATION_PROBE_KEYS: tuple[str, ...] = (
    "blocks.0.mlp.down.weight",
    "blocks.0.mlp.up.weight",
)

# Coefficient of variation of the per-input-column norms, below which the
# input axis is taken to be rotated. Measured on real checkpoints: a
# plain Krea 2 tensor sits at 0.32-0.53 (weight energy is very unevenly
# distributed across input features), while folding a Hadamard spreads
# that energy almost perfectly evenly and collapses it to 0.03. Two
# orders of magnitude of headroom, so the exact threshold isn't delicate.
_ROTATION_CV_THRESHOLD = 0.12


def krea_rotation_is_folded(path: Path, header: dict[str, Any]) -> bool:
    """Whether ``path`` is a rotated ("convrot") int8 checkpoint, which
    Lucidium cannot run.

    Cheap paths first: only int8 checkpoints are ever rotated, and the
    ones in circulation say so in the filename. Otherwise one quantized
    tensor is read from the memory-mapped file and its input-axis column
    norms are tested for the flatness a folded Hadamard produces.
    """
    import torch
    from safetensors import safe_open

    if any(tag in path.name.lower() for tag in _ROTATED_TAGS):
        return True

    stripped = {_strip_prefix(k): (k, v) for k, v in header.items()}
    for probe in _ROTATION_PROBE_KEYS:
        entry = stripped.get(probe)
        if entry is None or entry[1].get("dtype") != "I8":
            continue
        raw_key, _ = entry
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(raw_key).to(torch.float32)
            scale = handle.get_tensor(raw_key + _SCALE_SUFFIX).to(torch.float32)
        column_norms = (weight * scale).norm(dim=0)
        cv: float = (column_norms.std() / column_norms.mean()).item()
        _log.debug(
            "krea checkpoint: %s column-norm CV %.3f (rotated below %.2f)",
            path.name,
            cv,
            _ROTATION_CV_THRESHOLD,
        )
        return cv < _ROTATION_CV_THRESHOLD
    return False


# Filename tags marking the few-step distilled (TDM / "turbo")
# checkpoint. It samples at 8 steps with CFG OFF and a fixed timestep
# shift; the undistilled "raw" base needs 28 steps with CFG 4.5. Getting
# this wrong doesn't crash — it renders mush — and the distinction is not
# recorded anywhere in the weights, so the filename is the only signal.
_DISTILLED_TAGS: tuple[str, ...] = ("turbo", "distill", "tdm", "lightning")
# Tags marking the undistilled base. Checked FIRST so a merge named
# e.g. "krea2_raw_turbocharged" isn't misread as few-step.
_RAW_TAGS: tuple[str, ...] = ("raw", "base", "midtrain")


def krea_is_distilled(path: Path) -> bool:
    """Whether ``path`` is the few-step distilled Krea 2 checkpoint.

    Defaults to ``True`` for unmarked names: the overwhelming majority of
    Krea 2 checkpoints in circulation (and every Civitai finetune) are
    built on Turbo, and the failure is asymmetric — running the distilled
    recipe on a raw checkpoint gives a soft-but-legible image, whereas
    running 28 CFG steps on a distilled one burns the render out.
    """
    name = path.name.lower()
    if any(tag in name for tag in _RAW_TAGS):
        return False
    if any(tag in name for tag in _DISTILLED_TAGS):
        return True
    return True

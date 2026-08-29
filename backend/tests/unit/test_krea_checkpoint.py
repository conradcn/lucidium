"""Unit tests for the ComfyUI -> diffusers Krea 2 conversion.

No checkpoint on disk: the safetensors header is just JSON, so a tiny
synthetic header exercises the sniffing and config inference, and the
key mapper is a pure string function.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest

from lucidium.providers.krea_checkpoint import (
    KreaCheckpointError,
    _map_key,
    infer_krea_config,
    is_krea_state_dict,
    krea_is_distilled,
    krea_rotation_is_folded,
    read_safetensors_header,
)

# Shapes of the real Comfy-Org Krea 2 release, shrunk only where a count
# is derived from the number of keys (2 blocks per stack instead of 48).
_HIDDEN = 3584
_HEAD_DIM = 128  # == sum((32, 48, 48))
_TEXT_HIDDEN = 2560
_TEXT_HEAD_DIM = 128


def _block_keys(prefix: str, count: int, hidden: int, head_dim: int) -> dict:
    keys = {}
    for i in range(count):
        keys.update(
            {
                f"{prefix}{i}.prenorm.scale": [hidden],
                f"{prefix}{i}.postnorm.scale": [hidden],
                f"{prefix}{i}.attn.wq.weight": [hidden, hidden],
                f"{prefix}{i}.attn.wk.weight": [head_dim * 4, hidden],
                f"{prefix}{i}.attn.wv.weight": [head_dim * 4, hidden],
                f"{prefix}{i}.attn.wo.weight": [hidden, hidden],
                f"{prefix}{i}.attn.qknorm.qnorm.scale": [head_dim],
                f"{prefix}{i}.attn.qknorm.knorm.scale": [head_dim],
                f"{prefix}{i}.mlp.gate.weight": [hidden * 4, hidden],
                f"{prefix}{i}.mlp.up.weight": [hidden * 4, hidden],
                f"{prefix}{i}.mlp.down.weight": [hidden, hidden * 4],
            }
        )
    return keys


def _header(prefix: str = "") -> dict:
    """A minimal but complete Krea 2 safetensors header."""
    shapes: dict = {
        "first.weight": [_HIDDEN, 64],
        "first.bias": [_HIDDEN],
        "tmlp.0.weight": [_HIDDEN, 256],
        "tmlp.2.weight": [_HIDDEN, _HIDDEN],
        "tproj.1.weight": [_HIDDEN * 6, _HIDDEN],
        "txtmlp.0.scale": [_TEXT_HIDDEN],
        "txtmlp.1.weight": [_HIDDEN, _TEXT_HIDDEN],
        "txtmlp.3.weight": [_HIDDEN, _HIDDEN],
        "txtfusion.projector.weight": [1, 36],
        "last.modulation.lin": [2, _HIDDEN],
        "last.norm.scale": [_HIDDEN],
        "last.linear.weight": [64, _HIDDEN],
    }
    shapes.update(_block_keys("blocks.", 2, _HIDDEN, _HEAD_DIM))
    shapes.update(_block_keys("txtfusion.layerwise_blocks.", 3, _TEXT_HIDDEN, _TEXT_HEAD_DIM))
    shapes.update(_block_keys("txtfusion.refiner_blocks.", 4, _TEXT_HIDDEN, _TEXT_HEAD_DIM))
    return {
        f"{prefix}{k}": {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]}
        for k, v in shapes.items()
    }


def _write_safetensors(path: Path, header: dict) -> None:
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


class TestSniffing:
    def test_recognises_a_krea_header(self):
        assert is_krea_state_dict(_header())

    def test_recognises_the_comfy_prefixed_layout(self):
        # The int8_convrot repackage wraps every key in the full-model
        # namespace; both layouts must sniff the same.
        assert is_krea_state_dict(_header("model.diffusion_model."))

    def test_rejects_a_non_krea_checkpoint(self):
        assert not is_krea_state_dict(
            ["model.diffusion_model.input_blocks.0.0.weight", "first.weight"]
        )

    def test_reads_a_header_off_disk(self, tmp_path):
        path = tmp_path / "krea2_turbo_fp8_scaled.safetensors"
        header = _header()
        _write_safetensors(path, {**header, "__metadata__": {"format": "pt"}})
        loaded = read_safetensors_header(path)
        assert "__metadata__" not in loaded
        assert loaded == header

    def test_rejects_a_truncated_file(self, tmp_path):
        path = tmp_path / "stub.safetensors"
        path.write_bytes(b"abc")
        with pytest.raises(KreaCheckpointError):
            read_safetensors_header(path)


class TestConfigInference:
    def test_infers_every_dimension_from_shapes(self):
        config = infer_krea_config(_header())
        assert config == {
            "in_channels": 64,
            "num_layers": 2,
            "attention_head_dim": _HEAD_DIM,
            "num_attention_heads": _HIDDEN // _HEAD_DIM,
            "num_key_value_heads": 4,
            "intermediate_size": _HIDDEN * 4,
            "timestep_embed_dim": 256,
            "text_hidden_dim": _TEXT_HIDDEN,
            "num_text_layers": 36,
            "text_num_attention_heads": _TEXT_HIDDEN // _TEXT_HEAD_DIM,
            "text_num_key_value_heads": 4,
            "text_intermediate_size": _TEXT_HIDDEN * 4,
            "num_layerwise_text_blocks": 3,
            "num_refiner_text_blocks": 4,
            "axes_dims_rope": (32, 48, 48),
            "rope_theta": 1000.0,
            "norm_eps": 1e-5,
        }

    def test_prefixed_layout_infers_identically(self):
        assert infer_krea_config(_header("model.diffusion_model.")) == infer_krea_config(_header())

    def test_block_count_comes_from_the_highest_index(self):
        # ``blocks.`` must not swallow the ``txtfusion.*`` stacks.
        assert infer_krea_config(_header())["num_layers"] == 2

    def test_missing_tensor_names_the_file(self):
        header = {k: v for k, v in _header().items() if k != "first.weight"}
        with pytest.raises(KreaCheckpointError, match=re.escape("k2.safetensors")):
            infer_krea_config(header, path_name="k2.safetensors")

    def test_rejects_an_incompatible_head_dim(self):
        header = _header()
        header["blocks.0.attn.qknorm.qnorm.scale"]["shape"] = [64]
        with pytest.raises(KreaCheckpointError, match="rotary axis split"):
            infer_krea_config(header)

    def test_rejects_a_lora(self):
        header = {k: v for k, v in _header().items() if not k.startswith("blocks.")}
        with pytest.raises(KreaCheckpointError, match="LoRA"):
            infer_krea_config(header)


class TestKeyMapping:
    @pytest.mark.parametrize(
        ("comfy", "diffusers"),
        [
            ("first.weight", "img_in.weight"),
            ("tmlp.0.bias", "time_embed.linear_1.bias"),
            ("tmlp.2.weight", "time_embed.linear_2.weight"),
            ("tproj.1.weight", "time_mod_proj.weight"),
            ("txtmlp.0.scale", "txt_in.norm.weight"),
            ("txtmlp.3.bias", "txt_in.linear_2.bias"),
            ("txtfusion.projector.weight", "text_fusion.projector.weight"),
            ("last.modulation.lin", "final_layer.scale_shift_table"),
            ("last.norm.scale", "final_layer.norm.weight"),
            ("last.linear.bias", "final_layer.linear.bias"),
            ("blocks.7.attn.wq.weight", "transformer_blocks.7.attn.to_q.weight"),
            ("blocks.7.attn.wo.weight", "transformer_blocks.7.attn.to_out.0.weight"),
            ("blocks.7.attn.gate.weight", "transformer_blocks.7.attn.to_gate.weight"),
            ("blocks.0.attn.qknorm.knorm.scale", "transformer_blocks.0.attn.norm_k.weight"),
            ("blocks.3.prenorm.scale", "transformer_blocks.3.norm1.weight"),
            ("blocks.3.postnorm.scale", "transformer_blocks.3.norm2.weight"),
            ("blocks.3.mlp.down.weight", "transformer_blocks.3.ff.down.weight"),
            ("blocks.3.mod.lin", "transformer_blocks.3.scale_shift_table"),
            (
                "txtfusion.layerwise_blocks.2.mlp.up.weight",
                "text_fusion.layerwise_blocks.2.ff.up.weight",
            ),
            (
                "txtfusion.refiner_blocks.11.attn.wv.weight",
                "text_fusion.refiner_blocks.11.attn.to_v.weight",
            ),
        ],
    )
    def test_maps_key(self, comfy, diffusers):
        assert _map_key(comfy, path_name="k2.safetensors") == diffusers

    def test_strips_the_comfy_namespace(self):
        assert (
            _map_key("model.diffusion_model.blocks.1.mlp.gate.weight", path_name="k2")
            == "transformer_blocks.1.ff.gate.weight"
        )

    def test_layerwise_prefix_is_not_shadowed_by_the_refiner_one(self):
        # Both start with ``txtfusion.``; the longest prefix must win.
        assert _map_key("txtfusion.layerwise_blocks.0.prenorm.scale", path_name="k2").startswith(
            "text_fusion.layerwise_blocks."
        )

    def test_unknown_key_raises_naming_the_file(self):
        with pytest.raises(KreaCheckpointError, match=re.escape("k2.safetensors")):
            _map_key("blocks.0.attn.something_new.weight", path_name="k2.safetensors")


class TestRotationDetection:
    """The rotated int8 quantization renders noise, so it has to be
    caught at load time rather than at the player's expense."""

    def _int8_file(self, tmp_path: Path, name: str, weight, scale):
        """Write a one-tensor safetensors file the probe can read."""
        from safetensors.torch import save_file

        path = tmp_path / name
        key = "model.diffusion_model.blocks.0.mlp.down.weight"
        save_file({key: weight, key + "_scale": scale}, str(path))
        return path, read_safetensors_header(path)

    def test_filename_tag_short_circuits(self, tmp_path):
        # No probe needed, and none possible — the header is empty.
        path = tmp_path / "krea2_gptINT4INT8Convrot.safetensors"
        assert krea_rotation_is_folded(path, {})

    def test_flat_column_norms_read_as_rotated(self, tmp_path):
        import torch

        # A Hadamard fold spreads energy evenly across the input axis.
        torch.manual_seed(0)
        weight = torch.randint(-127, 128, (64, 256), dtype=torch.int8)
        scale = torch.ones(64, 1)
        path, header = self._int8_file(tmp_path, "k2_int8.safetensors", weight, scale)
        assert krea_rotation_is_folded(path, header)

    def test_uneven_column_norms_read_as_plain(self, tmp_path):
        import torch

        torch.manual_seed(0)
        weight = torch.randint(-127, 128, (64, 256), dtype=torch.int8)
        # Per-input-feature energy varying over two orders of magnitude,
        # as in a real unrotated checkpoint.
        weight = (weight.float() * torch.logspace(-2, 0, 256)).to(torch.int8)
        scale = torch.ones(64, 1)
        path, header = self._int8_file(tmp_path, "k2_int8.safetensors", weight, scale)
        assert not krea_rotation_is_folded(path, header)

    def test_non_quantized_checkpoint_is_never_rotated(self, tmp_path):
        # bf16/fp8 builds carry no int8 tensor to probe, and rotation is
        # an int8-only scheme.
        path = tmp_path / "krea2_turbo_fp8_scaled.safetensors"
        assert not krea_rotation_is_folded(path, _header())


_MODELS_DIR = Path("C:/Users/conra/sd-new/ComfyUI/models/diffusion_models")


@pytest.mark.parametrize(
    ("name", "rotated"),
    [
        ("krea2_turbo_fp8_scaled.safetensors", False),
        ("krea2GPTGrandPTruth_gptINT4INT8Convrot.safetensors", True),
    ],
)
def test_rotation_verdict_on_real_checkpoints(name, rotated):
    """Pins the detector against the two checkpoints it was calibrated
    on. Skipped where they aren't installed."""
    path = _MODELS_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} not installed")
    header = read_safetensors_header(path)
    assert is_krea_state_dict(header)
    assert krea_rotation_is_folded(path, header) is rotated


def test_real_convrot_is_caught_without_its_filename_tag(tmp_path):
    """The statistical probe alone must catch the rotated build, so an
    untagged finetune of it doesn't slip through to a noise render."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    path = _MODELS_DIR / "krea2GPTGrandPTruth_gptINT4INT8Convrot.safetensors"
    if not path.is_file():
        pytest.skip("convrot checkpoint not installed")
    key = "model.diffusion_model.blocks.0.mlp.down.weight"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        probe = {key: handle.get_tensor(key), key + "_scale": handle.get_tensor(key + "_scale")}
    untagged = tmp_path / "krea2_someones_finetune.safetensors"
    save_file(probe, str(untagged))
    assert krea_rotation_is_folded(untagged, read_safetensors_header(untagged))


class TestDistilledDetection:
    @pytest.mark.parametrize(
        "name",
        [
            "krea2_turbo_fp8_scaled.safetensors",
            "krea2GPTGrandPTruth_gptINT4INT8Convrot.safetensors",  # turbo-derived
            "krea2_tdm_8step.safetensors",
        ],
    )
    def test_distilled(self, name):
        assert krea_is_distilled(Path(name))

    @pytest.mark.parametrize(
        "name",
        [
            "krea2_raw_bf16.safetensors",
            "krea2_midtrain.safetensors",
            # RAW wins over a turbo-ish tag elsewhere in the name.
            "krea2_raw_turbocharged.safetensors",
        ],
    )
    def test_not_distilled(self, name):
        assert not krea_is_distilled(Path(name))

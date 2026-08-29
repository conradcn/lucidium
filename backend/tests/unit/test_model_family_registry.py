"""Model-family registry: selection, mutual exclusivity, exhaustiveness.

The embedded image backend used to dispatch on four chained
``_is_*_pipeline`` predicates whose evaluation order was load-bearing.
It now dispatches through :func:`resolve_model_family`, and these tests
are the guard rails for adding a fifth family:

* every registered family is selected for a representative pipeline;
* exactly one family's sniffer claims each of those pipelines (so a new
  family can't silently steal an existing one's checkpoints);
* every family implements every call-site hook in its own class body, so
  a new family can't inherit SDXL behaviour at a branch point it never
  considered.
"""

from __future__ import annotations

from typing import Any

import pytest

from lucidium.providers.embedded_image_client import (
    ALL_MODEL_FAMILIES,
    KREA_FAMILY,
    MODEL_FAMILY_HOOKS,
    QWEN_FAMILY,
    SDXL_FAMILY,
    TURBO_FAMILY,
    Z_IMAGE_FAMILY,
    ModelFamily,
    resolve_model_family,
)


class _Config:
    def __init__(self, name: str) -> None:
        self._name_or_path = name


class _FakePipeline:
    """SDXL-shaped stub; family is driven by class name + config path."""

    def __init__(self, name: str = "stabilityai/stable-diffusion-xl-base-1.0") -> None:
        self.config = _Config(name)


class ZImagePipeline(_FakePipeline):
    pass


class Krea2Pipeline(_FakePipeline):
    pass


class QwenImagePipeline(_FakePipeline):
    pass


# One representative pipeline per registered family. Class-name matches
# and ``_name_or_path`` matches are both exercised. These names avoid the
# known sniffer overlaps ("Z-Image-Turbo" is also a turbo match, a
# Qwen-derived Krea distill is also a qwen match) so mutual exclusivity
# is meaningful here; the overlaps get their own precedence tests below.
REPRESENTATIVES: list[tuple[ModelFamily, Any]] = [
    (Z_IMAGE_FAMILY, ZImagePipeline("Tongyi-MAI/Z-Image")),
    (Z_IMAGE_FAMILY, _FakePipeline("models/zimage-base.safetensors")),
    (KREA_FAMILY, Krea2Pipeline("krea/krea-2")),
    (KREA_FAMILY, _FakePipeline("models/Krea-2-midtrain.safetensors")),
    (QWEN_FAMILY, QwenImagePipeline("Qwen/Qwen-Image")),
    (QWEN_FAMILY, _FakePipeline("models/qwen-image-lightning.safetensors")),
    (TURBO_FAMILY, _FakePipeline("stabilityai/sdxl-turbo")),
    (SDXL_FAMILY, _FakePipeline()),
    (SDXL_FAMILY, _FakePipeline("")),
]


@pytest.mark.parametrize(
    ("expected", "pipeline"),
    REPRESENTATIVES,
    ids=[f"{fam.name}-{i}" for i, (fam, _) in enumerate(REPRESENTATIVES)],
)
def test_representative_pipeline_selects_its_family(
    expected: ModelFamily,
    pipeline: Any,
) -> None:
    assert resolve_model_family(pipeline) is expected


def test_every_registered_family_has_a_representative() -> None:
    covered = {family.name for family, _ in REPRESENTATIVES}
    assert covered == {family.name for family in ALL_MODEL_FAMILIES}


@pytest.mark.parametrize(
    ("expected", "pipeline"),
    REPRESENTATIVES,
    ids=[f"{fam.name}-{i}" for i, (fam, _) in enumerate(REPRESENTATIVES)],
)
def test_exactly_one_family_matches_each_representative(
    expected: ModelFamily,
    pipeline: Any,
) -> None:
    """Sniffers must not overlap on a representative checkpoint.

    ``sdxl`` is the fallback and never sniffs, so its representatives
    are expected to match zero families; every other family's
    representative must match exactly one -- itself.
    """
    matching = [f.name for f in ALL_MODEL_FAMILIES if f.matches(pipeline)]
    if expected is SDXL_FAMILY:
        assert matching == []
    else:
        assert matching == [expected.name]


@pytest.mark.parametrize(
    "family",
    ALL_MODEL_FAMILIES,
    ids=[f.name for f in ALL_MODEL_FAMILIES],
)
def test_family_declares_every_call_site_hook(family: ModelFamily) -> None:
    """Every hook must resolve to a concrete family class, not to the
    abstract stub, and a family derived straight from ``ModelFamily``
    must spell out all of them in its own body.

    Inheriting a hook is only allowed by subclassing another *concrete*
    family (``turbo`` does this: it is structurally SDXL), which is an
    explicit, reviewable statement of "same behaviour here". A brand-new
    family bolted onto ``ModelFamily`` cannot quietly pick up SDXL's
    answer at a branch point it never considered.
    """
    cls = type(family)
    inherits_concrete_family = any(
        base is not ModelFamily and issubclass(base, ModelFamily) for base in cls.__bases__
    )
    if inherits_concrete_family:
        # Reuse is deliberate; the ABC already guarantees every hook is
        # implemented somewhere in the MRO.
        assert all(hasattr(family, hook) for hook in MODEL_FAMILY_HOOKS)
        return
    missing = [hook for hook in MODEL_FAMILY_HOOKS if hook not in cls.__dict__]
    assert missing == [], f"{family.name} does not implement: {missing}"


def test_abstract_hooks_match_the_documented_hook_list() -> None:
    """``MODEL_FAMILY_HOOKS`` is what the tests above check against, so
    it must stay in sync with the ABC -- otherwise a hook added to
    ``ModelFamily`` would go unchecked."""
    assert set(ModelFamily.__abstractmethods__) == set(MODEL_FAMILY_HOOKS)


def test_incomplete_family_cannot_be_instantiated() -> None:
    class _Incomplete(ModelFamily):
        name = "incomplete"

        def matches(self, pipeline: Any) -> bool:
            return True

    with pytest.raises(TypeError):
        _Incomplete()


# --- behaviour parity with the pre-registry if/elif chain -----------------


def test_sdxl_face_inpaint_support_per_family() -> None:
    assert not Z_IMAGE_FAMILY.supports_sdxl_face_inpaint(ZImagePipeline())
    assert not KREA_FAMILY.supports_sdxl_face_inpaint(Krea2Pipeline())
    assert QWEN_FAMILY.supports_sdxl_face_inpaint(QwenImagePipeline())
    assert TURBO_FAMILY.supports_sdxl_face_inpaint(_FakePipeline())
    assert SDXL_FAMILY.supports_sdxl_face_inpaint(_FakePipeline())


def test_qwen_img2img_support_is_qwen_only() -> None:
    supporting = [f.name for f in ALL_MODEL_FAMILIES if f.supports_qwen_img2img(_FakePipeline())]
    assert supporting == ["qwen"]


def test_sampling_kwargs_match_the_legacy_recipes() -> None:
    assert Z_IMAGE_FAMILY.sampling_kwargs(ZImagePipeline()) == {
        "num_inference_steps": 9,
        "guidance_scale": 0.0,
    }
    assert TURBO_FAMILY.sampling_kwargs(_FakePipeline()) == {
        "num_inference_steps": 1,
        "guidance_scale": 0.0,
    }
    assert SDXL_FAMILY.sampling_kwargs(_FakePipeline()) == {
        "num_inference_steps": 25,
        "guidance_scale": 7.0,
    }
    krea = KREA_FAMILY.sampling_kwargs(Krea2Pipeline())
    assert krea["max_sequence_length"] == 512
    assert set(krea) == {
        "num_inference_steps",
        "guidance_scale",
        "max_sequence_length",
    }
    qwen = QWEN_FAMILY.sampling_kwargs(QwenImagePipeline())
    assert qwen["guidance_scale"] == 1.0
    assert set(qwen) == {
        "num_inference_steps",
        "true_cfg_scale",
        "guidance_scale",
    }


def test_prompt_strategy_matches_the_legacy_branch() -> None:
    plain = {f.name for f in ALL_MODEL_FAMILIES if f.prompt_strategy(_FakePipeline()) == "plain"}
    assert plain == {"z_image", "krea", "qwen"}


def test_cpu_encode_negative_thresholds() -> None:
    # Krea gates the unconditional branch on guidance_scale > 0 ...
    assert KREA_FAMILY.cpu_encode_negative({"guidance_scale": 4.5})
    assert not KREA_FAMILY.cpu_encode_negative({"guidance_scale": 0.0})
    # ... Qwen on true_cfg_scale > 1.
    assert QWEN_FAMILY.cpu_encode_negative({"true_cfg_scale": 4.0})
    assert not QWEN_FAMILY.cpu_encode_negative({"true_cfg_scale": 1.0})


def test_cpu_encode_attr_only_for_transformer_families() -> None:
    with_attr = {
        f.name for f in ALL_MODEL_FAMILIES if f.cpu_encode_attr(_FakePipeline()) is not None
    }
    assert with_attr == {"krea", "qwen"}


def test_precedence_z_image_turbo_is_not_turbo() -> None:
    """ "Z-Image-Turbo" satisfies the turbo name sniffer too; Z-Image must
    win. This ordering was previously implicit in the call-site chain."""
    assert TURBO_FAMILY.matches(_FakePipeline("Tongyi-MAI/Z-Image-Turbo"))
    assert resolve_model_family(_FakePipeline("Tongyi-MAI/Z-Image-Turbo")) is (Z_IMAGE_FAMILY)


def test_precedence_krea_outranks_qwen() -> None:
    """Krea 2's components are Qwen-derived, so a Krea checkpoint naming
    Qwen also satisfies the Qwen sniffer. Krea must win."""
    pipeline = _FakePipeline("krea-2-qwen-distill.safetensors")
    assert QWEN_FAMILY.matches(pipeline)
    assert resolve_model_family(pipeline) is KREA_FAMILY


def test_inference_context_per_family() -> None:
    class _FakeTorch:
        def no_grad(self) -> str:
            return "no_grad"

        def inference_mode(self) -> str:
            return "inference_mode"

    torch = _FakeTorch()
    assert QWEN_FAMILY.inference_context(torch) == "no_grad"
    assert KREA_FAMILY.inference_context(torch) == "no_grad"
    assert Z_IMAGE_FAMILY.inference_context(torch) == "inference_mode"
    assert TURBO_FAMILY.inference_context(torch) == "inference_mode"
    assert SDXL_FAMILY.inference_context(torch) == "inference_mode"

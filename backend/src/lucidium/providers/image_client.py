"""Image client protocol + ComfyUI HTTP impl + recorded-fixture fake."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Protocol

import httpx

from ..api.errors import ProviderUnreachableError, ProviderValidationError
from ..config import IMAGE_RETRY_BUDGET
from ..domain.settings import ImageSettings

# Shared timing-line logger with the LLM client; both use the
# ``lucidium.api_calls`` channel so a single log filter routes every
# external API call to the renderer's console / a tail-able file.
_call_log = logging.getLogger("lucidium.api_calls")


class ImageClient(Protocol):
    async def generate(self, workflow: str, params: dict[str, object], *, seed: int) -> bytes: ...


def fixture_key(workflow: str, params: dict[str, object], *, seed: int) -> str:
    canonical = json.dumps(
        {"workflow": workflow, "params": params, "seed": seed},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


class ComfyUiImageClient:
    """ComfyUI HTTP client.

    ComfyUI's REST surface is workflow-graph oriented: we POST a workflow
    JSON to ``/prompt``, poll ``/history/{prompt_id}`` for completion,
    and download the produced image from ``/view``.
    """

    def __init__(
        self,
        settings: ImageSettings,
        *,
        workflow_root: Path,
        http: httpx.AsyncClient | None = None,
        completion_timeout_seconds: float = 600.0,
    ) -> None:
        self._settings = settings
        self._workflow_root = workflow_root
        # Long-running ComfyUI workflows (face-detail + RMBG) can take
        # 60–180 s on a typical local GPU; the per-request timeout has
        # to accommodate that or we'll time out before the image is
        # ever produced.
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
        self._http = http or httpx.AsyncClient(timeout=timeout)
        self._completion_timeout_seconds = completion_timeout_seconds

    async def generate(self, workflow: str, params: dict[str, object], *, seed: int) -> bytes:
        template_path = self._workflow_root / workflow
        if not template_path.exists():
            raise ProviderValidationError(f"missing ComfyUI workflow {template_path}")

        graph = self._instantiate_workflow(template_path, params=params, seed=seed)
        url = self._settings.base_url.rstrip("/")

        # Log the full prompt set BEFORE firing so even a hung
        # ComfyUI request leaves the prompts in the journal — that
        # way debugging "why did the render look like THAT" doesn't
        # require waiting for the response. Each field on its own
        # line so a tail-friendly grep can extract just one of them.
        positive = str(params.get("positive_prompt", ""))
        face = str(params.get("face_prompt", ""))
        negative_extras = str(params.get("negative_extras", ""))
        _call_log.info(
            "image-call workflow=%s seed=%d backend=comfyui positive=%r face=%r negative_extras=%r",
            workflow,
            seed,
            positive,
            face,
            negative_extras,
        )

        last_exc: Exception | None = None
        for _attempt in range(IMAGE_RETRY_BUDGET):
            started = time.monotonic()
            try:
                queue_response = await self._http.post(f"{url}/prompt", json={"prompt": graph})
                queue_response.raise_for_status()
                prompt_id = str(queue_response.json()["prompt_id"])
                file_info = await self._await_completion(url, prompt_id)
                image_response = await self._http.get(
                    f"{url}/view",
                    params={
                        "filename": file_info["filename"],
                        "subfolder": file_info.get("subfolder", ""),
                        "type": file_info.get("type", "output"),
                    },
                )
                image_response.raise_for_status()
                elapsed_ms = int((time.monotonic() - started) * 1000)
                _call_log.info(
                    "image-call workflow=%s seed=%d bytes=%d elapsed_ms=%d attempt=%d",
                    workflow,
                    seed,
                    len(image_response.content),
                    elapsed_ms,
                    _attempt,
                )
                return image_response.content
            except httpx.HTTPError as exc:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                _call_log.warning(
                    "image-call workflow=%s seed=%d elapsed_ms=%d attempt=%d FAILED: %s",
                    workflow,
                    seed,
                    elapsed_ms,
                    _attempt,
                    exc,
                )
                last_exc = exc
                await asyncio.sleep(0.5 * (1 + _attempt))
        raise ProviderUnreachableError(f"image provider failed: {last_exc}")

    def _instantiate_workflow(
        self, template: Path, *, params: dict[str, object], seed: int
    ) -> dict[str, object]:
        # The production workflows under ``backend/workflows/`` use
        # ``PLACEHOLDER_<KEY>`` tokens inside string fields. We do
        # textual substitution before parsing so the prompts can
        # contain commas, quotes, or any other JSON-significant
        # characters.
        raw = template.read_text(encoding="utf-8")
        # Pony-based checkpoints expect the canonical scoring tags at
        # the head of the positive prompt (and matching negatives).
        # Detection is per-template: if the workflow's checkpoint name
        # contains "pony" we mutate the params accordingly.
        if _is_pony_workflow(raw):
            params = _apply_pony_tags(params)
        for key, value in params.items():
            token = f"PLACEHOLDER_{key.upper()}"
            # Embed the value as a JSON string fragment, then strip the
            # enclosing quotes so the original quotes around the
            # placeholder remain intact in the template.
            escaped = json.dumps(str(value))[1:-1]
            raw = raw.replace(token, escaped)
        # Any placeholder that wasn't supplied is replaced with the
        # empty string — workflows with optional slots stay valid JSON.
        raw = _strip_unfilled_placeholders(raw)
        graph: dict[str, object] = json.loads(raw)
        # Inject the seed wherever the workflow expects one. Both
        # ``seed`` and ``noise_seed`` are conventional; we set them
        # all to the same value so re-generations of the same
        # character or scene stay visually identical.
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            if "seed" in inputs:
                inputs["seed"] = seed
            if "noise_seed" in inputs:
                inputs["noise_seed"] = seed
        return graph

    async def _await_completion(self, base_url: str, prompt_id: str) -> dict[str, str]:
        deadline = asyncio.get_event_loop().time() + self._completion_timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            history_response = await self._http.get(f"{base_url}/history/{prompt_id}")
            history_response.raise_for_status()
            history = history_response.json()
            entry = history.get(prompt_id)
            if entry and entry.get("outputs"):
                outputs = entry["outputs"]
                # Prefer the latest node in the graph (highest numeric
                # id) so the post-processed PreviewImage from the
                # face-detail + background-removal pipeline wins over
                # the intermediate bare-sampler output.
                ranked = sorted(outputs.items(), key=lambda kv: _node_sort_key(kv[0]), reverse=True)
                for _node_id, node_outputs in ranked:
                    images = node_outputs.get("images") or []
                    if images:
                        return dict(images[0])
            await asyncio.sleep(1.0)
        raise ProviderUnreachableError(f"timed out waiting for ComfyUI prompt {prompt_id}")

    async def aclose(self) -> None:
        await self._http.aclose()


_PONY_POSITIVE_PREFIX = "score_9, score_8_up, score_7_up, score_6_up, "
_PONY_NEGATIVE_TAGS = "score_4, score_3, score_2, score_1, score_low, source_pony"


def _is_pony_workflow(raw: str) -> bool:
    """Detect a pony-based checkpoint by string match on the workflow.

    Pony Diffusion derivatives all carry "pony" in their filename
    (cyberrealisticPony, ponyDiffusionV6, ponyRealism, ...). Matching
    on the raw workflow JSON is fast and avoids a second JSON parse.
    """
    return "pony" in raw.lower()


def _apply_pony_tags(params: dict[str, object]) -> dict[str, object]:
    """Prefix the positive prompt with pony scoring tags and append the
    matching negative tags to ``negative_extras``."""
    out = dict(params)
    # ``str(... or "")`` only normalises the declared ``object`` value type;
    # every caller passes strings, so this is the identity for real params.
    positive = str(out.get("positive_prompt") or "")
    if positive and not positive.startswith("score_"):
        out["positive_prompt"] = _PONY_POSITIVE_PREFIX + positive
    face = str(out.get("face_prompt") or "")
    if face and not face.startswith("score_"):
        out["face_prompt"] = _PONY_POSITIVE_PREFIX + face
    existing_neg = str(out.get("negative_extras") or "")
    out["negative_extras"] = (
        f"{_PONY_NEGATIVE_TAGS}, {existing_neg}" if existing_neg else _PONY_NEGATIVE_TAGS
    )
    return out


def _strip_unfilled_placeholders(raw: str) -> str:
    """Replace any remaining ``PLACEHOLDER_FOO`` tokens with empty strings.

    Workflows can carry optional slots (e.g. ``PLACEHOLDER_NEGATIVE_EXTRAS``)
    that the engine doesn't always have a value for. Leaving the literal
    token in would poison the prompt; replacing with empty string is
    invisible.
    """
    import re

    return re.sub(r"PLACEHOLDER_[A-Z_]+", "", raw)


def _node_sort_key(node_id: str) -> tuple[int, str]:
    try:
        return (int(node_id), "")
    except ValueError:
        return (10**9, node_id)


class RecordedImageClient:
    """Returns a fixture PNG keyed by ``fixture_key``."""

    def __init__(self, fixture_root: Path) -> None:
        self._root = fixture_root

    async def generate(self, workflow: str, params: dict[str, object], *, seed: int) -> bytes:
        key = fixture_key(workflow, params, seed=seed)
        fixture_file = self._root / f"{key}.png"
        if not fixture_file.exists():
            raise ProviderUnreachableError(
                f"missing image fixture {fixture_file.name} for workflow={workflow!r}"
            )
        return fixture_file.read_bytes()

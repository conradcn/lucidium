"""Chunked CLIP prompt encoding for SDXL — past the 77-token window.

*** WHY THIS EXISTS ***

CLIP's text encoder has a hard 77-token context. diffusers' default
path tokenizes with ``truncation=True`` and silently drops everything
past it, so a composed portrait prompt (identity + expression + pose +
outfit + style routinely runs 100+ tokens) loses its tail without any
warning.

This module implements the standard fix: split the prompt into
75-token slices, wrap each slice in its own BOS/EOS pair, run every
slice through the encoder independently, and concatenate the resulting
per-token embeddings along the sequence axis. The UNet's cross-attention
is sequence-length agnostic, so it simply gets to attend over every
token instead of the first 77.

This used to be delegated to the ``compel`` library. compel was dropped
as a dependency because it declares ``notebook`` as a hard requirement,
which dragged ``jupyter-server`` + ``jupyterlab`` into every contributor
and CI environment for what is, for us, ~150 lines of tokenizer
bookkeeping. Only the chunking was ever used — none of compel's prompt
DSL (``(word)1.2`` weights, ``.and()`` conjunctions, blends), which the
engine never emits.

Two side benefits over the compel path:

  * No forward hooks. compel captured SDXL's pooled output by
    registering a forward hook on ``text_encoder_2`` that it never
    removed, which forced an awkward per-pipeline instance cache to
    stop hooks (and the VRAM they pinned) accumulating across renders.
    Here the pooled output is read straight off the encoder's return
    value, so there is no hook and nothing to cache.
  * It actually runs in the shipped build. compel was imported lazily
    and never listed in ``lucidium.spec``'s collected packages, so the
    frozen bundle fell through to the truncating path.

Algorithm (matching what compel did, and what A1111 / ComfyUI do):

  1. Tokenize the prompt with no special tokens and no truncation.
  2. Slice into runs of at most 75 ids, preferring to break at a
     sentence (``.``) then phrase (``,`` ``;`` ``:``) then word
     boundary so a comma-separated tag is not cut in half.
  3. Wrap each slice as ``[BOS] slice [EOS] <pad...>`` to exactly 77.
  4. Encode each chunk, taking the *penultimate* hidden state — SDXL
     is trained on the second-to-last CLIP layer, not the last.
  5. Copy chunk 0's EOS ("CLS") embedding over every later chunk's EOS
     position. That slot carries whole-prompt summary information, and
     a later chunk's version summarises only its own fragment.
  6. Concatenate chunk embeddings along the sequence axis, then
     concatenate the two encoders' outputs along the feature axis
     (768 + 1280 = 2048, SDXL's expected width).
  7. Pooled embeddings come from ``text_encoder_2``'s pooled output on
     the *truncated* 77-token prompt. SDXL's pooled conditioning is a
     single fixed-width vector — there is no chunked equivalent, and
     this is what compel did too.

Positive and negative prompts must come out the same sequence length
(SDXL concatenates them into one batch for classifier-free guidance),
so the shorter prompt is padded with additional *empty* chunks before
encoding — equivalent to compel's
``pad_conditioning_tensors_to_same_length``, which appended the
encoding of the empty string.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

# CLIP's context window. Each chunk spends two slots on BOS/EOS.
_MODEL_MAX_TOKENS = 77
_CHUNK_PAYLOAD = _MODEL_MAX_TOKENS - 2

# CLIP's tokenizer marks word-final subwords with a ``</w>`` suffix, so
# a split point is only mid-word if the token does NOT end in it.
_SENTENCE_PUNCTUATION = (".</w>",)
_PHRASE_PUNCTUATION = (".</w>", ",</w>", ";</w>", ":</w>")

_SDXL_ENCODER_ATTRS = ("tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2")


def supports_long_prompt(pipeline: Any) -> bool:
    """True when the pipeline exposes SDXL's dual CLIP tokenizer /
    encoder pair, which is the only shape this module handles.

    Qwen-Image, Krea 2 and Z-Image use a single long-context text
    encoder with no 77-token cliff, so they never come through here.
    """
    return all(getattr(pipeline, attr, None) is not None for attr in _SDXL_ENCODER_ATTRS)


def _find_split_point(tokenizer: Any, token_ids: list[int]) -> int:
    """Return how many of ``token_ids`` belong in the next chunk.

    Prefers the last sentence end within the window, then the last
    phrase end, then the last word end; falls back to a hard cut at
    ``_CHUNK_PAYLOAD`` when the window contains no boundary at all
    (a single run of 75+ subwords with no punctuation or spaces).
    """
    if len(token_ids) <= _CHUNK_PAYLOAD:
        return len(token_ids)
    window = tokenizer.convert_ids_to_tokens(token_ids[:_CHUNK_PAYLOAD])

    def last_index(predicate: Any) -> int | None:
        for index in range(len(window) - 1, -1, -1):
            if predicate(window[index]):
                # Split AFTER this token, so it stays in this chunk.
                return index + 1
        return None

    for predicate in (
        lambda tok: tok in _SENTENCE_PUNCTUATION,
        lambda tok: tok in _PHRASE_PUNCTUATION,
        lambda tok: isinstance(tok, str) and tok.endswith("</w>"),
    ):
        found = last_index(predicate)
        # A boundary at index 0 would make no progress and loop forever.
        if found:
            return found
    return _CHUNK_PAYLOAD


def _chunk_prompt(tokenizer: Any, text: str) -> list[list[int]]:
    """Tokenize ``text`` and slice it into padded 77-id chunks.

    Always returns at least one chunk — an empty prompt (the common
    case for ``negative``) yields a single ``[BOS][EOS]<pad...>``
    chunk, which is exactly the conditioning SDXL expects for "no
    negative prompt".
    """
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    token_ids: list[int] = list(encoded["input_ids"])
    chunks: list[list[int]] = []
    while True:
        split = _find_split_point(tokenizer, token_ids)
        chunks.append(_pad_chunk(tokenizer, token_ids[:split]))
        token_ids = token_ids[split:]
        if not token_ids:
            return chunks


def _pad_chunk(tokenizer: Any, payload: list[int]) -> list[int]:
    """Wrap one slice as ``[BOS] payload [EOS] <pad...>``, length 77."""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    # SDXL's second tokenizer pads with id 0 rather than EOS; read the
    # value off the tokenizer instead of assuming either convention.
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos
    chunk = [bos, *payload, eos]
    return chunk + [pad] * (_MODEL_MAX_TOKENS - len(chunk))


def _empty_chunk(tokenizer: Any) -> list[int]:
    return _pad_chunk(tokenizer, [])


def _encode_chunks(
    torch: Any,
    tokenizer: Any,
    text_encoder: Any,
    chunks: list[list[int]],
) -> Any:
    """Encode every chunk and concatenate along the sequence axis.

    Returns a ``[1, 77 * len(chunks), dim]`` tensor of penultimate
    hidden states, with chunk 0's EOS embedding replicated into the
    later chunks' EOS slots.
    """
    device = text_encoder.device
    ids = torch.tensor(chunks, dtype=torch.long, device=device)
    output = text_encoder(ids, output_hidden_states=True, return_dict=True)
    # ``hidden_states[-2]``: SDXL conditions on CLIP's penultimate layer.
    # ``[-1]`` is the final layer and produces visibly different (and
    # wrong-for-SDXL) conditioning.
    hidden = output.hidden_states[-2]  # [n_chunks, 77, dim]

    if len(chunks) > 1:
        eos = tokenizer.eos_token_id
        eos_positions = [chunk.index(eos) for chunk in chunks]
        first = hidden[0, eos_positions[0]]
        # ``torch.cat`` of per-chunk slices rather than in-place writes:
        # ``hidden`` may be an inference-mode tensor, which refuses
        # mutation.
        rows = [hidden[0]]
        for index in range(1, len(chunks)):
            row = hidden[index].clone()
            row[eos_positions[index]] = first
            rows.append(row)
        hidden = torch.stack(rows, dim=0)

    # [n_chunks, 77, dim] -> [1, n_chunks * 77, dim]
    return hidden.reshape(1, -1, hidden.shape[-1])


def _pooled_embedding(torch: Any, tokenizer: Any, text_encoder: Any, text: str) -> Any:
    """SDXL's pooled conditioning, from the truncated 77-token prompt."""
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=_MODEL_MAX_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    ids = encoded["input_ids"].to(text_encoder.device)
    output = text_encoder(ids, output_hidden_states=False, return_dict=True)
    # CLIPTextModelWithProjection returns ``text_embeds``; a plain
    # CLIPTextModel returns ``pooler_output``.
    pooled = getattr(output, "text_embeds", None)
    if pooled is None:
        pooled = output.pooler_output
    return pooled


def encode_long_prompt(
    pipeline: Any,
    positive: str,
    negative: str,
) -> dict[str, Any] | None:
    """Encode ``positive`` / ``negative`` past CLIP's 77-token window.

    Returns the keyword-argument dict that replaces ``prompt`` /
    ``negative_prompt`` on an SDXL pipeline call, or ``None`` when the
    pipeline isn't SDXL-shaped or encoding failed — in which case the
    caller falls back to plain string prompts (accepting diffusers'
    silent truncation).
    """
    if not supports_long_prompt(pipeline):
        return None
    try:
        import torch
    except ImportError:
        return None

    try:
        tokenizers = (pipeline.tokenizer, pipeline.tokenizer_2)
        encoders = (pipeline.text_encoder, pipeline.text_encoder_2)

        # Chunk against tokenizer 1 and tokenizer 2 separately — they
        # have different vocabularies, so the same prompt can land on a
        # different chunk count. Pad BOTH prompts, on BOTH tokenizers,
        # up to one shared count so every tensor concatenates cleanly.
        chunked = {
            (which, index): _chunk_prompt(tokenizers[index], text)
            for which, text in (("positive", positive), ("negative", negative))
            for index in (0, 1)
        }
        target = max(len(chunks) for chunks in chunked.values())
        for (_which, index), chunks in chunked.items():
            chunks.extend([_empty_chunk(tokenizers[index])] * (target - len(chunks)))

        with torch.no_grad():
            embeds = {
                which: torch.cat(
                    [
                        _encode_chunks(
                            torch,
                            tokenizers[index],
                            encoders[index],
                            chunked[(which, index)],
                        )
                        for index in (0, 1)
                    ],
                    # Feature axis: SDXL wants 768 (CLIP-L) + 1280
                    # (CLIP-bigG) = 2048 per token.
                    dim=-1,
                )
                for which in ("positive", "negative")
            }
            # Only encoder 2 produces SDXL's pooled conditioning.
            pooled = {
                which: _pooled_embedding(torch, tokenizers[1], encoders[1], text)
                for which, text in (("positive", positive), ("negative", negative))
            }
    except Exception:  # best-effort; fall back on plain prompts
        _log.warning(
            "chunked CLIP encoding failed; falling back to truncated prompts",
            exc_info=True,
        )
        return None

    return {
        "prompt_embeds": embeds["positive"],
        "pooled_prompt_embeds": pooled["positive"],
        "negative_prompt_embeds": embeds["negative"],
        "negative_pooled_prompt_embeds": pooled["negative"],
    }

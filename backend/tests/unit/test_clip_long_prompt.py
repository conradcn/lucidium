"""Pin the chunked CLIP encoder that replaced ``compel``.

The behaviour that matters is not "some tensor comes back" but the
exact shapes and token layout SDXL is fed:

  * a prompt under 75 tokens produces ONE 77-token chunk, i.e. the
    same conditioning length diffusers' own path would produce, so
    short prompts are not silently perturbed by the rewrite;
  * a prompt over 75 tokens produces enough chunks to carry every
    token — this is the whole point, and the failure mode it replaces
    (silent truncation) is invisible at runtime;
  * positive and negative come out the SAME sequence length, because
    SDXL concatenates them into one CFG batch and a mismatch is a
    hard shape error mid-render;
  * the feature width is 768 + 1280 = 2048 (both encoders), and the
    hidden state taken is the PENULTIMATE one — SDXL is trained on
    CLIP's second-to-last layer and using ``[-1]`` produces
    plausible-looking but subtly wrong conditioning that no shape
    assertion would catch.

Fakes rather than real CLIP weights: the algorithm under test is
tokenizer bookkeeping, and a real SDXL text encoder pair is a ~1.5 GB
download that the offline suite must not need.
"""

from __future__ import annotations

from typing import Any

import torch

from lucidium.providers import clip_long_prompt as clp

# ---------- Fakes ------------------------------------------------------------


class _FakeTokenizer:
    """Whitespace tokenizer with CLIP's ``</w>`` word-end convention.

    Ids are ``offset + index-in-vocab`` so the two tokenizers in a
    test disagree about ids for the same word, exactly as CLIP-L and
    CLIP-bigG do.
    """

    def __init__(self, offset: int = 1000) -> None:
        self._offset = offset
        self._vocab: dict[str, int] = {}
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

    def _id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._offset + len(self._vocab) + 10
        return self._vocab[token]

    def _tokens(self, text: str) -> list[str]:
        # Split trailing punctuation into its own token so the
        # sentence/phrase split heuristic has something to find,
        # mirroring how CLIP's BPE emits ``,</w>`` separately.
        out: list[str] = []
        for word in text.split():
            while word and word[-1] in ".,;:":
                punct = word[-1]
                word = word[:-1]
                if word:
                    out.append(word + "</w>")
                    word = ""
                out.append(punct + "</w>")
            if word:
                out.append(word + "</w>")
        return out

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        padding: str | None = None,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> Any:
        ids = [self._id(tok) for tok in self._tokens(text)]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length" and max_length is not None:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        reverse = {value: key for key, value in self._vocab.items()}
        return [reverse.get(i, "<unk>") for i in ids]


class _FakeEncoderOutput:
    def __init__(self, hidden_states: Any, text_embeds: Any) -> None:
        self.hidden_states = hidden_states
        self.text_embeds = text_embeds


class _FakeEncoder:
    """Returns deterministic per-token vectors so tests can tell the
    penultimate hidden state apart from the last one, and tell one
    chunk's EOS embedding apart from another's."""

    def __init__(self, dim: int, *, last_layer_marker: float = 99.0) -> None:
        self.dim = dim
        self.device = torch.device("cpu")
        self._last_layer_marker = last_layer_marker
        self.hooks: list[Any] = []
        self.calls: list[Any] = []

    def register_forward_hook(self, hook: Any) -> Any:  # pragma: no cover
        self.hooks.append(hook)
        raise AssertionError("clip_long_prompt must never register forward hooks")

    def __call__(
        self,
        ids: Any,
        *,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> _FakeEncoderOutput:
        self.calls.append(ids)
        batch, seq = ids.shape
        # Penultimate layer: value == the token id, broadcast across dim.
        penultimate = ids.unsqueeze(-1).expand(batch, seq, self.dim).float().clone()
        # Make each chunk's EOS row distinguishable by chunk index, so
        # the CLS-copy assertion has something to bite on.
        for row in range(batch):
            penultimate[row] += row * 0.5
        last = torch.full((batch, seq, self.dim), self._last_layer_marker)
        pooled = torch.full((batch, self.dim), 7.0)
        return _FakeEncoderOutput([last, penultimate, last], pooled)


class _FakeSdxlPipeline:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer(offset=1000)
        self.tokenizer_2 = _FakeTokenizer(offset=5000)
        self.text_encoder = _FakeEncoder(dim=768)
        self.text_encoder_2 = _FakeEncoder(dim=1280)


def _words(count: int) -> str:
    return ", ".join(f"tag{i}" for i in range(count))


# ---------- Chunking ---------------------------------------------------------


def test_short_prompt_produces_exactly_one_chunk() -> None:
    """A prompt that fits CLIP's window must not be reshaped —
    77 tokens out, same as diffusers' own path."""
    tokenizer = _FakeTokenizer()
    chunks = clp._chunk_prompt(tokenizer, "a quiet room")

    assert len(chunks) == 1
    assert len(chunks[0]) == 77
    assert chunks[0][0] == tokenizer.bos_token_id
    assert tokenizer.eos_token_id in chunks[0]


def test_empty_prompt_still_produces_one_padded_chunk() -> None:
    """The negative prompt is routinely empty; it must still yield
    the ``[BOS][EOS]<pad...>`` conditioning SDXL expects rather than
    an empty list (which would make ``torch.cat`` fail)."""
    tokenizer = _FakeTokenizer()
    chunks = clp._chunk_prompt(tokenizer, "")

    assert len(chunks) == 1
    assert chunks[0][:2] == [tokenizer.bos_token_id, tokenizer.eos_token_id]
    assert len(chunks[0]) == 77


def test_long_prompt_chunks_and_keeps_every_token() -> None:
    """The regression this module exists for: a 100+ token prompt
    must reach the model in full, not get cut at 77."""
    tokenizer = _FakeTokenizer()
    text = _words(120)
    total = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    chunks = clp._chunk_prompt(tokenizer, text)

    assert total > 150  # "tagN" + "," per entry — comfortably past one window
    assert len(chunks) > 1
    assert all(len(chunk) == 77 for chunk in chunks)

    # Every payload token survives, in order, across the chunk seam.
    payload: list[int] = []
    for chunk in chunks:
        body = chunk[1:]
        body = body[: body.index(tokenizer.eos_token_id)]
        payload.extend(body)
    assert payload == tokenizer(text, add_special_tokens=False)["input_ids"]


def test_split_prefers_a_punctuation_boundary() -> None:
    """Prompts are comma-separated tag lists; splitting mid-tag
    hands the model half a concept. The split point must land right
    after a separator, not at a hard 75."""
    tokenizer = _FakeTokenizer()
    chunks = clp._chunk_prompt(tokenizer, _words(120))

    first = chunks[0]
    body = first[1 : first.index(tokenizer.eos_token_id)]
    last_token = tokenizer.convert_ids_to_tokens([body[-1]])[0]
    assert last_token == ",</w>"


def test_split_falls_back_to_a_hard_cut_without_boundaries() -> None:
    """A pathological prompt with no separator inside the window
    must still make progress rather than loop forever."""

    class _NoBoundaryTokenizer(_FakeTokenizer):
        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return ["nope"] * len(ids)

    tokenizer = _NoBoundaryTokenizer()
    chunks = clp._chunk_prompt(tokenizer, _words(120))

    assert len(chunks) > 1
    first = chunks[0]
    # Full 75-token payload: BOS + 75 + EOS == 77, no padding.
    assert first.index(tokenizer.eos_token_id) == 76


# ---------- Encoding ---------------------------------------------------------


def test_encode_uses_penultimate_hidden_state_not_the_last() -> None:
    """SDXL conditions on CLIP's second-to-last layer. Taking
    ``hidden_states[-1]`` yields correctly-shaped, subtly wrong
    conditioning — so this is asserted on values, not shape."""
    tokenizer = _FakeTokenizer()
    encoder = _FakeEncoder(dim=4, last_layer_marker=99.0)
    chunks = clp._chunk_prompt(tokenizer, "a quiet room")

    out = clp._encode_chunks(torch, tokenizer, encoder, chunks)

    assert out.shape == (1, 77, 4)
    assert not torch.any(out == 99.0)


def test_encode_copies_chunk_zero_cls_into_later_chunks() -> None:
    """The EOS slot carries whole-prompt summary information; a
    later chunk's own EOS summarises only its fragment. Chunk 0's
    wins, for every chunk."""
    tokenizer = _FakeTokenizer()
    encoder = _FakeEncoder(dim=4)
    chunks = clp._chunk_prompt(tokenizer, _words(120))
    assert len(chunks) > 1

    out = clp._encode_chunks(torch, tokenizer, encoder, chunks)[0]

    eos_rows = [
        out[index * 77 + chunk.index(tokenizer.eos_token_id)] for index, chunk in enumerate(chunks)
    ]
    for row in eos_rows[1:]:
        assert torch.equal(row, eos_rows[0])


def test_encode_long_prompt_returns_sdxl_shaped_embeddings() -> None:
    pipeline = _FakeSdxlPipeline()

    result = clp.encode_long_prompt(pipeline, _words(120), "blurry, low quality")

    assert result is not None
    prompt_embeds = result["prompt_embeds"]
    negative_embeds = result["negative_prompt_embeds"]

    # 768 (CLIP-L) + 1280 (CLIP-bigG).
    assert prompt_embeds.shape[-1] == 2048
    assert prompt_embeds.shape[0] == 1
    assert prompt_embeds.shape[1] % 77 == 0
    assert prompt_embeds.shape[1] > 77  # actually chunked
    # SDXL batches positive + negative for CFG — lengths MUST match.
    assert negative_embeds.shape == prompt_embeds.shape
    # Pooled conditioning comes from encoder 2 only.
    assert result["pooled_prompt_embeds"].shape == (1, 1280)
    assert result["negative_pooled_prompt_embeds"].shape == (1, 1280)


def test_encode_long_prompt_pads_across_differing_tokenizers() -> None:
    """The two SDXL tokenizers have different vocabularies, so the
    same prompt can chunk differently under each. Every tensor still
    has to line up."""
    pipeline = _FakeSdxlPipeline()
    # Make tokenizer_2 split more aggressively than tokenizer 1.
    pipeline.tokenizer_2._tokens = (  # type: ignore[method-assign]
        lambda text: _FakeTokenizer._tokens(pipeline.tokenizer_2, text) * 2
    )

    result = clp.encode_long_prompt(pipeline, _words(80), "")

    assert result is not None
    assert result["prompt_embeds"].shape == result["negative_prompt_embeds"].shape
    assert result["prompt_embeds"].shape[-1] == 2048


def test_encode_long_prompt_returns_none_for_non_sdxl() -> None:
    """Qwen-Image / Krea 2 / Z-Image use one long-context encoder
    and have no 77-token cliff — they must not come through here."""

    class _SingleEncoder:
        tokenizer = _FakeTokenizer()
        text_encoder = _FakeEncoder(dim=768)

    assert clp.encode_long_prompt(_SingleEncoder(), "a prompt", "") is None


def test_encode_long_prompt_swallows_encoder_failure() -> None:
    """A broken encoder must degrade to plain (truncated) string
    prompts rather than failing the render outright."""

    class _Boom:
        device = torch.device("cpu")

        def __call__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("CUDA OOM")

    pipeline = _FakeSdxlPipeline()
    pipeline.text_encoder = _Boom()  # type: ignore[assignment]

    assert clp.encode_long_prompt(pipeline, "a prompt", "") is None

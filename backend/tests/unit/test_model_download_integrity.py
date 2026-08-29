"""Integrity + concurrency guards on the one-click checkpoint download.

``download_model`` publishes a multi-GB file into the models directory
with an atomic rename, and ``target.exists()`` means whatever lands there
is never re-fetched. So the rename is the last line of defence: anything
that gets past it is permanent, and surfaces months later as an opaque
safetensors parse error at render time with no way to force a
re-download. These tests pin the three ways a bad file could get past it:

  * a stream cut short of the advertised Content-Length,
  * bytes that don't match the published sha256,
  * a retry racing the download already in flight over one ``.part``.
"""

from __future__ import annotations

import hashlib
import io
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from lucidium.providers import embedded_models as em


class _FakeResponse(io.BytesIO):
    """``urlopen`` stand-in whose advertised Content-Length is decoupled
    from the bytes it actually serves, so truncation is expressible."""

    def __init__(self, data: bytes, *, advertised: int | None = None):
        super().__init__(data)
        self.status = 200
        total = len(data) if advertised is None else advertised
        self.headers = {} if total is None else {"Content-Length": str(total)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _spec(**overrides) -> em.ModelSpec:
    return replace(em.MODEL_CATALOG["sdxl-turbo"], **overrides)


# --------------------------------------------------------------------------
# Content-Length: the fallback check for entries with no published digest
# --------------------------------------------------------------------------


def test_truncated_response_leaves_no_file_at_the_target(tmp_path, monkeypatch):
    """A connection cut at 60% must not be renamed into place. Without
    the size check the partial file becomes the checkpoint — and the
    exists() fast path means it stays that way forever."""
    spec = _spec(sha256=None)
    served = b"y" * 600
    monkeypatch.setattr(
        em.urllib.request,
        "urlopen",
        lambda request, timeout=None: _FakeResponse(served, advertised=1000),
    )

    with pytest.raises(em.ModelDownloadError) as excinfo:
        em.download_model(spec, tmp_path)

    assert "incomplete" in str(excinfo.value)
    assert not (tmp_path / spec.local_filename).exists()
    assert list(tmp_path.glob("*.part")) == []


def test_complete_response_without_a_digest_is_accepted(tmp_path, monkeypatch):
    """The size check must not reject the honest case — full bytes, no
    published digest, save it."""
    spec = _spec(sha256=None)
    payload = b"z" * 4096
    monkeypatch.setattr(
        em.urllib.request, "urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )

    out = em.download_model(spec, tmp_path)
    assert out.read_bytes() == payload


def test_missing_content_length_still_saves(tmp_path, monkeypatch):
    """A server that omits Content-Length leaves us nothing to compare
    against; we can't turn that into a hard failure without breaking
    downloads that are fine."""
    spec = _spec(sha256=None)
    payload = b"w" * 128
    monkeypatch.setattr(
        em.urllib.request,
        "urlopen",
        lambda request, timeout=None: _FakeResponse(payload, advertised=None),
    )

    assert em.download_model(spec, tmp_path).read_bytes() == payload


# --------------------------------------------------------------------------
# sha256 — the real check, for entries carrying a published digest
# --------------------------------------------------------------------------


def test_wrong_sha256_is_rejected(tmp_path, monkeypatch):
    """Right length, wrong bytes — exactly what a truncation check can't
    see. ``resolve/main`` is a moving ref, so this is also what catches a
    checkpoint being re-published under the same path."""
    payload = b"q" * 2048
    spec = _spec(sha256=hashlib.sha256(b"the bytes we expected").hexdigest())
    monkeypatch.setattr(
        em.urllib.request, "urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )

    with pytest.raises(em.ModelDownloadError) as excinfo:
        em.download_model(spec, tmp_path)

    assert "integrity" in str(excinfo.value).lower()
    assert not (tmp_path / spec.local_filename).exists()
    assert list(tmp_path.glob("*.part")) == []


def test_matching_sha256_is_published(tmp_path, monkeypatch):
    payload = b"q" * 2048
    spec = _spec(sha256=hashlib.sha256(payload).hexdigest().upper())  # case-insensitive
    monkeypatch.setattr(
        em.urllib.request, "urlopen", lambda request, timeout=None: _FakeResponse(payload)
    )

    assert em.download_model(spec, tmp_path).read_bytes() == payload


def test_catalog_digests_are_well_formed():
    """Every published digest is 64 lowercase hex chars. A typo here would
    fail every download of that model with an integrity error, so it's
    worth one cheap assertion."""
    for key, spec in em.MODEL_CATALOG.items():
        if spec.sha256 is None:
            continue
        assert len(spec.sha256) == 64, key
        assert spec.sha256 == spec.sha256.lower(), key
        int(spec.sha256, 16)  # raises if not hex


# --------------------------------------------------------------------------
# Concurrency — the reconnect-and-retry race
# --------------------------------------------------------------------------


def test_concurrent_downloads_do_not_share_a_part_path(tmp_path, monkeypatch):
    """Two downloads of the SAME model must never write to one ``.part``.

    A dropped WebSocket doesn't cancel the in-flight download, so the
    user's retry arrives while the first is still streaming. With a fixed
    ``<name>.part`` the retry's ``open(..., "wb")`` truncated the running
    download's temp file and the loser published a zero-holed multi-GB
    checkpoint. Here the second caller is released only after the first
    has opened its temp file, and we assert both on the distinctness of
    the paths and on the bytes that end up on disk.
    """
    spec = _spec(sha256=None)
    payload = b"p" * 4096
    opened: list[Path] = []
    first_opened = threading.Event()
    real_open = open  # captured before the patch below, or we'd recurse

    def tracking_open(path, mode="r", *args, **kwargs):
        if str(path).endswith(".part"):
            opened.append(Path(path))
            first_opened.set()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(payload)

    monkeypatch.setattr(em.urllib.request, "urlopen", fake_urlopen)

    errors: list[BaseException] = []

    def run() -> None:
        try:
            em.download_model(spec, tmp_path)
        except BaseException as exc:  # reported below, off the worker thread
            errors.append(exc)

    first = threading.Thread(target=run)
    first.start()
    assert first_opened.wait(10), "the first download never opened its temp file"
    second = threading.Thread(target=run)
    second.start()
    first.join(30)
    second.join(30)

    assert errors == []
    # The lock means the second caller finds the finished file and never
    # opens a temp at all; if it does open one, it must be a *different*
    # path from the first's. Either way, no shared .part.
    assert len(set(opened)) == len(opened)
    assert (tmp_path / spec.local_filename).read_bytes() == payload
    assert list(tmp_path.glob("*.part")) == []


def test_part_paths_are_unique_per_attempt(tmp_path):
    """The uuid suffix is what makes the shared-temp truncation
    impossible even for a caller that bypasses the lock."""
    target = tmp_path / "model.safetensors"
    a = em._partial_path(target)
    b = em._partial_path(target)
    assert a != b
    assert a.name.startswith(target.name) and a.name.endswith(".part")


def test_second_download_reuses_the_first_result(tmp_path, monkeypatch):
    """Serialised retries are cheap: once one attempt has published the
    file, the next returns it instead of re-fetching gigabytes."""
    spec = _spec(sha256=None)
    payload = b"r" * 512
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        return _FakeResponse(payload)

    monkeypatch.setattr(em.urllib.request, "urlopen", fake_urlopen)

    em.download_model(spec, tmp_path)
    em.download_model(spec, tmp_path)
    assert calls["n"] == 1

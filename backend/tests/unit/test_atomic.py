from __future__ import annotations

from pathlib import Path

import pytest

from lucidium.persistence.atomic import atomic_write_bytes, atomic_write_text


def test_atomic_write_text_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "settings.json"
    atomic_write_text(target, '{"hello": "world"}')
    assert target.read_text(encoding="utf-8") == '{"hello": "world"}'


def test_atomic_write_replaces_existing(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_leaves_no_temp_files_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.txt"

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("os.replace", explode)
    with pytest.raises(RuntimeError):
        atomic_write_text(target, "content")
    leftovers = list(tmp_path.glob(".file.txt.*"))
    assert leftovers == []


def test_atomic_write_bytes_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    payload = b"\x00\x01\x02\xff"
    atomic_write_bytes(target, payload)
    assert target.read_bytes() == payload

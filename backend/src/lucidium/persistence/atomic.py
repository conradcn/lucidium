"""Atomic file writes via temp file + ``os.replace``.

Constitution I (Reliability): "Persist player progress after each
accepted turn so a crash, network failure, or provider outage cannot
lose more than the in-flight turn." Every save commit must go through
these helpers.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import IO, TypeVar


def _fsync_dir(directory: Path) -> None:
    """Flush the directory entry created by ``os.replace``.

    ``fsync`` on the file handle only makes the *contents* durable; on
    POSIX the rename that publishes them lives in the parent directory
    and can still be lost on power failure, leaving a save that was
    reported as committed but is missing on the next boot. Windows has
    no directory handle to sync (``O_DIRECTORY`` does not exist and
    opening a directory fails), so there the replace is all we have.
    """
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems (and bind mounts) refuse fsync on directories;
        # the data write already succeeded, so this is not fatal.
        pass
    finally:
        os.close(fd)


# The payload writer is paired with ``mode``: text callers get an
# ``IO[str]`` handle, binary callers an ``IO[bytes]``. A TypeVar keeps
# that pairing checkable at the call sites instead of erasing it.
_HandleT = TypeVar("_HandleT", IO[str], IO[bytes])


def _write_via_temp(
    path: Path,
    write_payload: Callable[[_HandleT], object],
    mode: str,
    *,
    encoding: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        # On Windows, ``os.fdopen(fd, "w")`` without ``encoding=`` falls
        # back to ``locale.getpreferredencoding()`` (typically cp1252),
        # silently mangling UTF-8 game text — em-dash 0x2014 ends up
        # written as the cp1252 byte 0x97 and fails the next utf-8
        # load. The encoding kwarg here is not optional.
        if mode == "wb":
            with os.fdopen(fd, mode) as handle:
                write_payload(handle)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with os.fdopen(fd, mode, encoding=encoding, newline="") as handle:
                write_payload(handle)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write text to ``path`` as UTF-8."""

    def write_payload(handle: IO[str]) -> None:
        handle.write(content)

    _write_via_temp(path, write_payload, "w", encoding="utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to ``path``."""

    def write_payload(handle: IO[bytes]) -> None:
        handle.write(data)

    _write_via_temp(path, write_payload, "wb", encoding=None)

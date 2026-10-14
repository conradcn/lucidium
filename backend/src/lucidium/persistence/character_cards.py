"""Read and write SillyTavern character cards on disk.

Supported containers:

  * ``.png`` — the card JSON is base64-encoded in a ``tEXt`` chunk
    keyed ``chara`` (V1/V2) and/or ``ccv3`` (V3). ``zTXt`` / ``iTXt``
    are accepted too, since some editors re-save with those. When both
    keys are present ``ccv3`` wins, matching SillyTavern.
  * ``.json`` — the bare card object (V1 flat, V2 or V3 envelope; the
    old Pygmalion ``char_name`` / ``char_persona`` aliases are mapped).
  * ``.charx`` — the V3 zip container: ``card.json`` plus an
    ``assets/`` tree whose ``icon`` image is the avatar.

Exports are always PNG carrying BOTH a V2 ``chara`` chunk and a V3
``ccv3`` chunk, which every current frontend reads.

Card files are untrusted input. Nothing here evaluates their content;
sizes are capped and a card's "file" handle on the wire is a bare file
name that must resolve directly inside the library directory.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import logging
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.character_card import CharacterCardData
from .atomic import atomic_write_bytes

_log = logging.getLogger(__name__)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CARD_EXTENSIONS: frozenset[str] = frozenset({".png", ".json", ".charx"})
# Real cards are a few MB at most (a large avatar plus a lorebook);
# anything past this is not a card and we refuse to slurp it.
MAX_CARD_BYTES = 64 * 1024 * 1024
THUMBNAIL_SIZE = (192, 288)

_CARD_TEXT_KEYS = ("ccv3", "chara")
_V1_FIELDS = ("name", "description", "personality", "scenario", "first_mes", "mes_example")
# Pygmalion / old TavernAI aliases for the V1 fields.
_V1_ALIASES = {
    "char_name": "name",
    "char_persona": "personality",
    "world_scenario": "scenario",
    "char_greeting": "first_mes",
    "example_dialogue": "mes_example",
}
_STR_FIELDS = (
    "name",
    "description",
    "personality",
    "scenario",
    "first_mes",
    "mes_example",
    "creator_notes",
    "system_prompt",
    "post_history_instructions",
    "creator",
    "character_version",
    "nickname",
)
_LIST_FIELDS = ("alternate_greetings", "tags")
_MODELLED_DATA_KEYS = frozenset({*_STR_FIELDS, *_LIST_FIELDS, "extensions", "character_book"})
# ``data`` keys that only exist in V3; dropped from the V2 chunk so a
# strict V2 validator doesn't trip over them.
_V3_ONLY_KEYS = frozenset(
    {
        "assets",
        "nickname",
        "creator_notes_multilingual",
        "source",
        "group_only_greetings",
        "creation_date",
        "modification_date",
    }
)
# Characters Windows refuses in a file name, plus control characters.
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


class CardError(ValueError):
    """The file is not a readable character card."""


# ---------------------------------------------------------------------------
# JSON <-> CharacterCardData
# ---------------------------------------------------------------------------


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        # Some editors write tags as one comma-separated string.
        return [t.strip() for t in value.split(",") if t.strip()]
    if not isinstance(value, list):
        return []
    return [s for s in (_as_str(v) for v in value) if s]


def parse_card_json(obj: Any, *, fallback_name: str = "") -> CharacterCardData:
    """Normalise any card generation's JSON into ``CharacterCardData``."""
    if not isinstance(obj, dict):
        raise CardError("card JSON is not an object")
    spec = _as_str(obj.get("spec"))
    nested = obj.get("data")
    if isinstance(nested, dict) and (spec.startswith("chara_card_v") or "name" in nested):
        data: dict[str, Any] = nested
        spec = spec if spec.startswith("chara_card_v") else "chara_card_v2"
        extra = {k: v for k, v in data.items() if k not in _MODELLED_DATA_KEYS}
    else:
        data = dict(obj)
        for alias, field in _V1_ALIASES.items():
            if not _as_str(data.get(field)) and _as_str(data.get(alias)):
                data[field] = data[alias]
        spec = "chara_card_v1"
        # A V1 object is usually SillyTavern's internal record, full of
        # chat bookkeeping (``chat``, ``create_date``, ``avatar``) that
        # is not part of the character — don't carry it.
        extra = {}

    name = _as_str(data.get("name")).strip() or fallback_name.strip()
    if not name:
        raise CardError("card has no name")
    extensions = data.get("extensions")
    book = data.get("character_book")
    fields: dict[str, Any] = {f: _as_str(data.get(f)) for f in _STR_FIELDS if f != "name"}
    fields.update({f: _as_str_list(data.get(f)) for f in _LIST_FIELDS})
    return CharacterCardData(
        name=name,
        **fields,
        extensions=extensions if isinstance(extensions, dict) else {},
        character_book=book if isinstance(book, dict) else None,
        extra_fields=extra,
        spec=spec,
    )


def _data_dict(card: CharacterCardData) -> dict[str, Any]:
    data: dict[str, Any] = dict(card.extra_fields)
    for f in _STR_FIELDS:
        data[f] = getattr(card, f)
    for f in _LIST_FIELDS:
        data[f] = list(getattr(card, f))
    data["extensions"] = dict(card.extensions)
    if card.character_book is not None:
        data["character_book"] = card.character_book
    return data


def card_to_v2(card: CharacterCardData) -> dict[str, Any]:
    data = {k: v for k, v in _data_dict(card).items() if k not in _V3_ONLY_KEYS}
    out: dict[str, Any] = {"spec": "chara_card_v2", "spec_version": "2.0", "data": data}
    # Top-level V1 mirror so pre-V2 readers still get the basics.
    for f in _V1_FIELDS:
        out[f] = getattr(card, f)
    out["tags"] = list(card.tags)
    return out


def card_to_v3(card: CharacterCardData) -> dict[str, Any]:
    data = _data_dict(card)
    data.setdefault("group_only_greetings", [])
    return {"spec": "chara_card_v3", "spec_version": "3.0", "data": data}


def _b64_json(obj: dict[str, Any]) -> str:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_card_text(value: str) -> Any:
    try:
        raw = base64.b64decode(value.strip(), validate=False)
        return json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        pass
    # A handful of tools store the JSON un-encoded.
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise CardError("card text chunk is neither base64 JSON nor JSON") from exc


# ---------------------------------------------------------------------------
# PNG chunks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Chunk:
    ctype: bytes
    body: bytes


def _iter_png_chunks(data: bytes) -> list[_Chunk]:
    if not data.startswith(PNG_SIGNATURE):
        raise CardError("not a PNG file")
    chunks: list[_Chunk] = []
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        end = pos + 12 + length
        if end > len(data):
            raise CardError("truncated PNG chunk")
        chunks.append(_Chunk(ctype, data[pos + 8 : pos + 8 + length]))
        pos = end
        if ctype == b"IEND":
            break
    return chunks


def _text_chunk(chunk: _Chunk) -> tuple[str, str] | None:
    """Decode a ``tEXt`` / ``zTXt`` / ``iTXt`` chunk to (keyword, text)."""
    try:
        if chunk.ctype == b"tEXt":
            key, _, text = chunk.body.partition(b"\x00")
            return key.decode("latin-1"), text.decode("latin-1")
        if chunk.ctype == b"zTXt":
            key, _, rest = chunk.body.partition(b"\x00")
            return key.decode("latin-1"), zlib.decompress(rest[1:]).decode("latin-1")
        if chunk.ctype == b"iTXt":
            key, _, rest = chunk.body.partition(b"\x00")
            compressed = rest[0] == 1
            rest = rest[2:]
            _lang, _, rest = rest.partition(b"\x00")
            _translated, _, text = rest.partition(b"\x00")
            if compressed:
                text = zlib.decompress(text)
            return key.decode("latin-1"), text.decode("utf-8")
    except (IndexError, zlib.error, UnicodeDecodeError):
        return None
    return None


def _encode_chunk(ctype: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(ctype + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", crc)


def read_png_card(data: bytes, *, fallback_name: str = "") -> CharacterCardData:
    texts: dict[str, str] = {}
    for chunk in _iter_png_chunks(data):
        decoded = _text_chunk(chunk)
        if decoded is not None and decoded[0].lower() in _CARD_TEXT_KEYS:
            texts.setdefault(decoded[0].lower(), decoded[1])
    for key in _CARD_TEXT_KEYS:
        if key in texts:
            return parse_card_json(_decode_card_text(texts[key]), fallback_name=fallback_name)
    raise CardError("PNG has no character card data")


def _ensure_png(image: bytes | None) -> bytes:
    """Return ``image`` as PNG bytes, converting or synthesising as needed."""
    if image and image.startswith(PNG_SIGNATURE):
        return image
    from PIL import Image

    if image:
        try:
            with Image.open(io.BytesIO(image)) as im:
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                return buf.getvalue()
        except Exception:
            _log.warning("card export: avatar is not a readable image; using a blank avatar")
    buf = io.BytesIO()
    Image.new("RGB", (512, 768), (34, 30, 44)).save(buf, format="PNG")
    return buf.getvalue()


def embed_card_in_png(image: bytes | None, card: CharacterCardData) -> bytes:
    """Return a PNG of ``image`` carrying ``card`` as ``chara`` + ``ccv3``.

    Any card chunks already in the image are replaced, so re-exporting
    a card's own avatar never leaves two conflicting definitions.
    """
    png = _ensure_png(image)
    out = bytearray(PNG_SIGNATURE)
    new_text = [
        _encode_chunk(b"tEXt", b"chara\x00" + _b64_json(card_to_v2(card)).encode("ascii")),
        _encode_chunk(b"tEXt", b"ccv3\x00" + _b64_json(card_to_v3(card)).encode("ascii")),
    ]
    for chunk in _iter_png_chunks(png):
        decoded = _text_chunk(chunk)
        if decoded is not None and decoded[0].lower() in _CARD_TEXT_KEYS:
            continue
        if chunk.ctype == b"IEND":
            for encoded in new_text:
                out += encoded
        out += _encode_chunk(chunk.ctype, chunk.body)
    return bytes(out)


# ---------------------------------------------------------------------------
# CHARX
# ---------------------------------------------------------------------------


def _read_charx(data: bytes, *, fallback_name: str) -> tuple[CharacterCardData, bytes | None]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            try:
                info = zf.getinfo("card.json")
            except KeyError as exc:
                raise CardError("CHARX archive has no card.json") from exc
            if info.file_size > MAX_CARD_BYTES:
                raise CardError("CHARX card.json is too large")
            obj = json.loads(zf.read(info).decode("utf-8"))
            card = parse_card_json(obj, fallback_name=fallback_name)
            return card, _charx_icon(zf, card)
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CardError(f"unreadable CHARX archive: {type(exc).__name__}") from exc


def _charx_icon(zf: zipfile.ZipFile, card: CharacterCardData) -> bytes | None:
    names = set(zf.namelist())
    candidates: list[str] = []
    assets = card.extra_fields.get("assets")
    if isinstance(assets, list):
        for asset in assets:
            if not isinstance(asset, dict) or asset.get("type") != "icon":
                continue
            uri = _as_str(asset.get("uri"))
            # The spec's own spelling is ``embeded://``; accept both.
            for prefix in ("embeded://", "embedded://"):
                if uri.startswith(prefix):
                    candidates.append(uri[len(prefix) :])
    candidates += sorted(
        n for n in names if n.lower().startswith("assets/icon/") and not n.endswith("/")
    )
    for name in candidates:
        if name in names and zf.getinfo(name).file_size <= MAX_CARD_BYTES:
            return zf.read(name)
    return None


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedCard:
    card: CharacterCardData
    # The avatar image bytes (PNG cards: the file itself; CHARX: the
    # icon asset), or ``None`` for JSON cards / CHARX without an icon.
    avatar: bytes | None


def _read_bounded(path: Path) -> bytes:
    size = path.stat().st_size
    if size > MAX_CARD_BYTES:
        raise CardError(f"file is larger than {MAX_CARD_BYTES // (1024 * 1024)} MB")
    return path.read_bytes()


def load_card_file(path: Path) -> LoadedCard:
    """Parse one card file. Raises ``CardError`` if it isn't a card."""
    suffix = path.suffix.lower()
    if suffix not in CARD_EXTENSIONS:
        raise CardError(f"unsupported card file type {suffix!r}")
    data = _read_bounded(path)
    stem = path.stem
    if suffix == ".png":
        card = read_png_card(data, fallback_name=stem)
        avatar: bytes | None = data
    elif suffix == ".charx":
        card, avatar = _read_charx(data, fallback_name=stem)
    else:
        try:
            obj = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CardError("file is not valid JSON") from exc
        card = parse_card_json(obj, fallback_name=stem)
        avatar = None
    return LoadedCard(card=card.model_copy(update={"source_file": path.name}), avatar=avatar)


# Parsed-card cache keyed on (path, mtime_ns, size) so re-listing a
# large library only re-reads files that changed.
_PARSE_CACHE: dict[tuple[str, int, int], LoadedCard] = {}
_PARSE_CACHE_MAX = 2048


def _cache_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size)


def load_card_cached(path: Path) -> LoadedCard:
    key = _cache_key(path)
    hit = _PARSE_CACHE.get(key)
    if hit is not None:
        return hit
    loaded = load_card_file(path)
    if len(_PARSE_CACHE) >= _PARSE_CACHE_MAX:
        _PARSE_CACHE.clear()
    _PARSE_CACHE[key] = loaded
    return loaded


@dataclass(frozen=True)
class LibraryEntry:
    file: str
    path: Path
    card: CharacterCardData
    thumbnail: Path | None


def list_library(
    directory: Path, *, thumbnail_dir: Path | None = None
) -> tuple[list[LibraryEntry], list[tuple[str, str]]]:
    """Every readable card directly inside ``directory`` (not recursive,
    like SillyTavern's ``characters/`` folder), plus ``(file, reason)``
    for each card-typed file that failed to parse."""
    entries: list[LibraryEntry] = []
    errors: list[tuple[str, str]] = []
    if not directory.is_dir():
        return entries, errors
    for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if path.suffix.lower() not in CARD_EXTENSIONS or not path.is_file():
            continue
        try:
            loaded = load_card_cached(path)
        except CardError as exc:
            errors.append((path.name, str(exc)))
            continue
        except OSError as exc:
            errors.append((path.name, f"could not read file ({type(exc).__name__})"))
            continue
        thumb = (
            _ensure_thumbnail(path, loaded.avatar, thumbnail_dir)
            if thumbnail_dir is not None
            else None
        )
        entries.append(LibraryEntry(file=path.name, path=path, card=loaded.card, thumbnail=thumb))
    return entries, errors


def _ensure_thumbnail(path: Path, avatar: bytes | None, thumbnail_dir: Path) -> Path | None:
    if not avatar:
        return None
    try:
        st = path.stat()
        digest = hashlib.sha1(
            f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()
        ).hexdigest()
        out = thumbnail_dir / f"{digest}.png"
        if out.exists():
            return out
        from PIL import Image

        with Image.open(io.BytesIO(avatar)) as im:
            im.thumbnail(THUMBNAIL_SIZE)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(out, buf.getvalue())
        return out
    except Exception:
        _log.debug("card thumbnail failed for %s", path.name, exc_info=True)
        return None


def resolve_card_file(directory: Path, file: str) -> Path:
    """Map a wire-level card ``file`` name to a path inside ``directory``.

    ``file`` must be a bare name: anything that resolves outside the
    library (separators, ``..``, an absolute path, a symlink pointing
    elsewhere) is rejected with ``CardError``.
    """
    if not file or file in {".", ".."} or "/" in file or "\\" in file or ":" in file:
        raise CardError("card file must be a plain file name")
    if Path(file).suffix.lower() not in CARD_EXTENSIONS:
        raise CardError("not a character card file")
    root = directory.resolve()
    candidate = (root / file).resolve()
    if candidate.parent != root:
        raise CardError("card file is outside the card library")
    if not candidate.is_file():
        raise FileNotFoundError(file)
    return candidate


def safe_card_filename(name: str, directory: Path, *, suffix: str = ".png") -> Path:
    """A non-colliding ``<name>.png`` path inside ``directory``."""
    stem = _UNSAFE_FILENAME_CHARS.sub("", name).strip().strip(".")[:80].strip() or "character"
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"{stem}_"
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem} ({n}){suffix}"
        n += 1
    return candidate


def write_png_card(path: Path, card: CharacterCardData, image: bytes | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, embed_card_in_png(image, card))

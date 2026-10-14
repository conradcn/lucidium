"""SillyTavern card file formats: PNG chunks, bare JSON, CHARX.

Cards are untrusted files from the internet, so alongside the happy
paths these pin the defensive edges: the wire-level ``file`` handle
can't reach outside the library, unreadable files are reported rather
than crashing a listing, and exports don't leave two conflicting card
definitions in one PNG.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from lucidium.domain.character_card import CharacterCardData
from lucidium.persistence import character_cards as cc


def _png(color: tuple[int, int, int] = (200, 40, 40), size: tuple[int, int] = (64, 96)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _png_with_text(chunks: list[tuple[str, str]]) -> bytes:
    """A PNG with raw ``tEXt`` chunks inserted before IEND."""
    base = _png()
    iend = base.rindex(b"IEND") - 4
    extra = b"".join(
        cc._encode_chunk(b"tEXt", key.encode("latin-1") + b"\x00" + text.encode("latin-1"))
        for key, text in chunks
    )
    return base[:iend] + extra + base[iend:]


def _b64(obj: dict) -> str:
    return base64.b64encode(json.dumps(obj).encode("utf-8")).decode("ascii")


V2_CARD = {
    "spec": "chara_card_v2",
    "spec_version": "2.0",
    "data": {
        "name": "Seraphine",
        "description": "{{char}} is a forest guardian who protects {{user}}.",
        "personality": "gentle, wary of strangers",
        "scenario": "a moonlit glade",
        "first_mes": "Who walks my woods?",
        "mes_example": "<START>\n{{char}}: Stay close.",
        "creator_notes": "",
        "system_prompt": "",
        "post_history_instructions": "",
        "alternate_greetings": ["Hello, traveller."],
        "tags": ["fantasy", "elf"],
        "creator": "someone",
        "character_version": "1.2",
        "extensions": {"talkativeness": "0.5", "depth_prompt": {"prompt": "x", "depth": 4}},
        "character_book": {"entries": [{"keys": ["glade"], "content": "The glade is old."}]},
    },
}


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_parse_v2_keeps_every_field() -> None:
    card = cc.parse_card_json(V2_CARD)
    assert card.name == "Seraphine"
    assert card.spec == "chara_card_v2"
    assert card.tags == ["fantasy", "elf"]
    assert card.alternate_greetings == ["Hello, traveller."]
    assert card.extensions["depth_prompt"]["depth"] == 4
    assert card.character_book is not None
    assert card.extra_fields == {}


def test_parse_v1_flat_and_pygmalion_aliases() -> None:
    v1 = cc.parse_card_json({"name": "Bob", "description": "a baker", "chat": "x", "avatar": "y"})
    assert v1.spec == "chara_card_v1"
    assert v1.description == "a baker"
    # SillyTavern bookkeeping in a V1 record is not part of the character.
    assert v1.extra_fields == {}

    pyg = cc.parse_card_json(
        {"char_name": "Ada", "char_persona": "curious", "char_greeting": "Hi!"}
    )
    assert pyg.name == "Ada"
    assert pyg.personality == "curious"
    assert pyg.first_mes == "Hi!"


def test_parse_v3_preserves_unmodelled_fields() -> None:
    card = cc.parse_card_json(
        {
            "spec": "chara_card_v3",
            "spec_version": "3.0",
            "data": {
                "name": "Vex",
                "group_only_greetings": ["Hey all"],
                "assets": [{"type": "icon", "uri": "ccdefault:", "name": "main", "ext": "png"}],
                "creation_date": 1700000000,
            },
        }
    )
    assert card.spec == "chara_card_v3"
    assert card.extra_fields["group_only_greetings"] == ["Hey all"]
    assert card.extra_fields["creation_date"] == 1700000000


def test_parse_tolerates_wrong_types_and_falls_back_to_file_name() -> None:
    card = cc.parse_card_json(
        {"description": 12, "tags": "a, b ,,c", "alternate_greetings": [1, None, "x"]},
        fallback_name="From File",
    )
    assert card.name == "From File"
    assert card.description == "12"
    assert card.tags == ["a", "b", "c"]
    assert card.alternate_greetings == ["1", "x"]


def test_parse_rejects_non_object_and_nameless() -> None:
    with pytest.raises(cc.CardError):
        cc.parse_card_json([1, 2])
    with pytest.raises(cc.CardError):
        cc.parse_card_json({"description": "no name"})


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------


def test_read_png_chara_chunk() -> None:
    card = cc.read_png_card(_png_with_text([("chara", _b64(V2_CARD))]))
    assert card.name == "Seraphine"


def test_ccv3_wins_over_chara_like_sillytavern() -> None:
    v3 = {"spec": "chara_card_v3", "spec_version": "3.0", "data": {"name": "V3 Name"}}
    png = _png_with_text([("chara", _b64(V2_CARD)), ("ccv3", _b64(v3))])
    assert cc.read_png_card(png).name == "V3 Name"


def test_png_without_card_or_not_png() -> None:
    with pytest.raises(cc.CardError, match="no character card"):
        cc.read_png_card(_png())
    with pytest.raises(cc.CardError, match="not a PNG"):
        cc.read_png_card(b"GIF89a....")


def test_embed_round_trips_and_replaces_existing_card_chunks() -> None:
    original = _png_with_text([("chara", _b64({"name": "Stale"}))])
    card = cc.parse_card_json(V2_CARD)
    out = cc.embed_card_in_png(original, card)

    # Still a valid image.
    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (64, 96)
    keys = [
        cc._text_chunk(c)[0]  # type: ignore[index]
        for c in cc._iter_png_chunks(out)
        if cc._text_chunk(c) is not None
    ]
    # Exactly one V2 and one V3 definition — the stale one is gone.
    assert sorted(keys) == ["ccv3", "chara"]
    back = cc.read_png_card(out)
    exclude = {"spec", "extra_fields"}
    assert back.model_dump(exclude=exclude) == card.model_dump(exclude=exclude)
    assert back.spec == "chara_card_v3"
    # V3 requires group_only_greetings, so the export adds an empty one.
    assert back.extra_fields == {"group_only_greetings": []}

    # The V2 chunk is a strict V2 envelope (no V3-only keys) with a V1 mirror.
    v2_text = next(
        t[1]
        for c in cc._iter_png_chunks(out)
        if (t := cc._text_chunk(c)) is not None and t[0] == "chara"
    )
    v2 = json.loads(base64.b64decode(v2_text))
    assert v2["spec"] == "chara_card_v2"
    assert v2["name"] == "Seraphine"
    assert "nickname" not in v2["data"]


def test_embed_without_image_synthesises_avatar() -> None:
    out = cc.embed_card_in_png(None, CharacterCardData(name="Blank"))
    assert out.startswith(cc.PNG_SIGNATURE)
    assert cc.read_png_card(out).name == "Blank"


def test_embed_converts_non_png_avatar() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="JPEG")
    out = cc.embed_card_in_png(buf.getvalue(), CharacterCardData(name="Jay"))
    assert cc.read_png_card(out).name == "Jay"


def test_unicode_survives_png_round_trip() -> None:
    card = CharacterCardData(name="Zoë 雪", description="naïve — “quoted” 🌙")
    back = cc.read_png_card(cc.embed_card_in_png(None, card))
    assert back.name == "Zoë 雪"
    assert back.description == "naïve — “quoted” 🌙"


# ---------------------------------------------------------------------------
# Files: JSON / CHARX / library listing
# ---------------------------------------------------------------------------


def test_load_json_file_with_bom(tmp_path: Path) -> None:
    path = tmp_path / "Bob.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"name": "Bob"}).encode())
    loaded = cc.load_card_file(path)
    assert loaded.card.name == "Bob"
    assert loaded.card.source_file == "Bob.json"
    assert loaded.avatar is None


def test_load_charx_with_icon(tmp_path: Path) -> None:
    icon = _png((10, 200, 10))
    path = tmp_path / "vex.charx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "card.json",
            json.dumps(
                {
                    "spec": "chara_card_v3",
                    "data": {
                        "name": "Vex",
                        "assets": [
                            {"type": "icon", "uri": "embeded://assets/icon/images/main.png"}
                        ],
                    },
                }
            ),
        )
        zf.writestr("assets/icon/images/main.png", icon)
    loaded = cc.load_card_file(path)
    assert loaded.card.name == "Vex"
    assert loaded.avatar == icon


def test_charx_without_card_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.charx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "hi")
    with pytest.raises(cc.CardError, match=r"card.json"):
        cc.load_card_file(path)


def test_list_library_reports_bad_files_and_makes_thumbnails(tmp_path: Path) -> None:
    lib = tmp_path / "cards"
    lib.mkdir()
    (lib / "Seraphine.png").write_bytes(_png_with_text([("chara", _b64(V2_CARD))]))
    (lib / "bob.json").write_text(json.dumps({"name": "Bob"}), encoding="utf-8")
    (lib / "broken.json").write_text("{not json", encoding="utf-8")
    (lib / "plain.png").write_bytes(_png())
    (lib / "notes.txt").write_text("ignored", encoding="utf-8")
    (lib / "sub").mkdir()
    (lib / "sub" / "nested.json").write_text(json.dumps({"name": "Nested"}), encoding="utf-8")

    thumbs = tmp_path / "thumbs"
    entries, errors = cc.list_library(lib, thumbnail_dir=thumbs)

    assert [e.file for e in entries] == ["bob.json", "Seraphine.png"]
    assert sorted(f for f, _ in errors) == ["broken.json", "plain.png"]
    by_file = {e.file: e for e in entries}
    assert by_file["bob.json"].thumbnail is None
    thumb = by_file["Seraphine.png"].thumbnail
    assert thumb is not None and thumb.parent == thumbs and thumb.exists()

    # A second listing reuses the cached thumbnail.
    entries2, _ = cc.list_library(lib, thumbnail_dir=thumbs)
    assert {e.file: e.thumbnail for e in entries2}["Seraphine.png"] == thumb


def test_list_library_missing_dir(tmp_path: Path) -> None:
    assert cc.list_library(tmp_path / "nope") == ([], [])


def test_oversized_file_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "MAX_CARD_BYTES", 10)
    path = tmp_path / "big.json"
    path.write_text(json.dumps({"name": "Big Card"}), encoding="utf-8")
    with pytest.raises(cc.CardError, match="larger"):
        cc.load_card_file(path)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "file",
    ["", ".", "..", "../x.png", "..\\x.png", "sub/x.png", "C:x.png", "C:\\x.png", "x.exe"],
)
def test_resolve_card_file_rejects_non_bare_names(tmp_path: Path, file: str) -> None:
    with pytest.raises(cc.CardError):
        cc.resolve_card_file(tmp_path, file)


def test_resolve_card_file_missing_and_present(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cc.resolve_card_file(tmp_path, "ghost.png")
    (tmp_path / "real.json").write_text("{}", encoding="utf-8")
    assert cc.resolve_card_file(tmp_path, "real.json") == (tmp_path / "real.json").resolve()


def test_safe_card_filename(tmp_path: Path) -> None:
    first = cc.safe_card_filename('Mr. <Bad>/Name?:"', tmp_path)
    assert first.parent == tmp_path
    assert first.name == "Mr. BadName.png"
    first.write_bytes(b"x")
    assert cc.safe_card_filename('Mr. <Bad>/Name?:"', tmp_path).name == "Mr. BadName (2).png"
    assert cc.safe_card_filename("CON", tmp_path).name == "CON_.png"
    assert cc.safe_card_filename("...", tmp_path).name == "character.png"
    assert cc.safe_card_filename("../../escape", tmp_path).parent == tmp_path

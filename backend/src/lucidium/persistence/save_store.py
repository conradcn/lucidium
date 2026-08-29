"""On-disk save layout: ``<saves>/<save-id>/{game.json,meta.json,images/}``.

Each ``commit_save`` writes ``game.json`` and ``meta.json`` atomically;
images live alongside as content-addressed PNGs deduplicated by hash.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

from pydantic import BaseModel, ConfigDict, ValidationError

from ..api.errors import SaveVersionError
from ..config import GAME_SCHEMA_VERSION, saves_dir
from ..domain.dialog import DialogNodeState
from ..domain.game import Game
from ..domain.settings import Settings
from .atomic import atomic_write_text
from .save_migrations import migrate


class SaveSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    last_played_at: datetime
    created_at: datetime
    schema_version: int
    summary: str
    # True when ``meta.json`` could not be read or parsed. The entry is
    # still listed — with placeholder display fields — so the player can
    # delete it from the load screen. Hiding it would make the directory
    # unreachable from inside the app: every recovery path needs a
    # ``save_id`` that only the list can supply.
    corrupt: bool = False


class SaveMeta(BaseModel):
    # Tolerant of unknown keys, unlike the rest of the models: saves
    # written before ``settings_snapshot`` was removed still carry it on
    # disk, and an ``extra="forbid"`` here would make every one of those
    # saves unloadable (``list_saves`` validates every meta it walks).
    # The stale key is dropped on the next ``commit_save``.
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    last_played_at: datetime
    created_at: datetime
    schema_version: int
    summary: str = ""
    # Set once the placeholder-name recovery in ``list_saves`` has run
    # (or once ``commit_save`` has established there is nothing to
    # recover). Without it, a save whose ``world.game_name`` is empty
    # re-parses the whole ``game.json`` on every single listing to
    # derive the same empty string — full cost, zero benefit, forever.
    name_recovered: bool = False


class SaveIdError(ValueError):
    """A ``save_id`` that would resolve outside the saves root.

    The wire schema (``C2SSavesLoad`` / ``Rename`` / ``Delete``) already
    constrains ``save_id`` to a bare directory name, so reaching this is
    a bug in a caller that bypassed the schema — not something a client
    can trigger. It exists so the filesystem layer is safe on its own
    terms rather than by trusting its callers."""


def _require_bare_id(save_id: str) -> None:
    """Refuse any ``save_id`` that is not a single directory name.

    Parsed with the WINDOWS flavour on every platform, deliberately. A
    ``save_id`` arrives over a socket that a Windows client can also
    speak, and the POSIX flavour disagrees about what those strings
    mean: ``"C:/Windows"`` is a *relative* two-component path on Linux
    (so joining it lands harmlessly under the root) while it is an
    absolute path that replaces the root on Windows, and
    a backslash-separated ``C:`` path is one legal filename on
    Linux and a rooted path on Windows. Judging the id by where it
    happens to resolve therefore gives a different verdict per platform
    for the same frame. Judging
    the id itself — one component, no drive, no anchor, not a ``..`` —
    gives the same verdict everywhere, and it is the constraint the wire
    schema already advertises."""
    parsed = PureWindowsPath(save_id)
    if save_id in ("", ".", "..") or parsed.drive or parsed.root or len(parsed.parts) != 1:
        raise SaveIdError(f"save id {save_id!r} is not a bare directory name")


def _save_dir(save_id: str, root: Path | None = None) -> Path:
    """Resolve ``<saves-root>/<save_id>``, refusing anything that escapes.

    ``Path.__truediv__`` DISCARDS the base when the right operand is
    absolute (``Path("saves") / "C:/Windows"`` is ``C:/Windows``), and a
    ``..`` component walks out of the root, so the join alone is not a
    confinement. ``_require_bare_id`` rejects the shapes that could do
    either; the containment check below then still holds the result to
    the root, which also covers a symlink ``resolve()`` follows out —
    the check every destructive caller (``delete_save``) and every write
    (``commit_save``) depends on."""
    _require_bare_id(save_id)
    base = (root or saves_dir()).expanduser()
    candidate = (base / save_id).resolve()
    resolved_base = base.resolve()
    if candidate != resolved_base and not candidate.is_relative_to(resolved_base):
        raise SaveIdError(f"save id {save_id!r} resolves outside the saves root")
    if candidate == resolved_base:
        raise SaveIdError("save id must name a directory inside the saves root")
    return candidate


def _meta_path(save_id: str, root: Path | None = None) -> Path:
    return _save_dir(save_id, root) / "meta.json"


def _game_path(save_id: str, root: Path | None = None) -> Path:
    return _save_dir(save_id, root) / "game.json"


# A meta whose ``name`` matches one of these is treated as a
# placeholder rather than an authoritative player choice: ``commit_save``
# upgrades it as soon as a real ``world.game_name`` is available, and
# ``list_saves`` substitutes the saved game's ``world.game_name``
# when surfacing the list to the renderer. The "Untitled" string was
# the historic placeholder when ``Session.commit`` didn't pass a
# default; a "" empty meta name is what older test fixtures wrote.
_PLACEHOLDER_NAMES: frozenset[str] = frozenset({"", "untitled"})


def _is_placeholder_name(name: str | None) -> bool:
    return (name or "").strip().casefold() in _PLACEHOLDER_NAMES


_CORRUPT_DISPLAY_NAME = "Damaged save"
_RECOVERED_DISPLAY_NAME = "Recovered save"


def _meta_from_game(entry: Path) -> SaveMeta | None:
    """Rebuild a minimal meta from ``game.json`` alone, or ``None``.

    ``commit_save`` writes ``game.json`` and ``meta.json`` as two
    independent atomic writes, so a crash between them on a new game's
    first commit leaves a directory with a complete playthrough and no
    meta at all. Skipping such a directory made it unreachable from the
    UI forever — there is no other source of the ``save_id`` that Load,
    Rename and Delete all need. The game file already carries everything
    a listing needs, so we treat it as the authority and synthesise the
    meta in memory; the next ``commit_save`` writes a real one.

    Parsed as raw JSON rather than through ``Game``: a save written by a
    newer build must still be listed (``load_save`` is the layer that
    reports the version mismatch), and a partially-understood payload is
    plenty for an id, a name and a timestamp.
    """
    game_file = entry / "game.json"
    try:
        data = json.loads(game_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    try:
        mtime = datetime.fromtimestamp(game_file.stat().st_mtime, UTC)
    except OSError:
        mtime = datetime.fromtimestamp(0, UTC)

    world = data.get("world")
    derived = ""
    if isinstance(world, dict):
        derived = str(world.get("game_name") or "").strip()

    created = mtime
    raw_created = data.get("created_at")
    if isinstance(raw_created, str):
        try:
            parsed = datetime.fromisoformat(raw_created)
        except ValueError:
            parsed = None
        if parsed is not None:
            created = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    version = data.get("schema_version")
    return SaveMeta(
        # The directory name, not ``data["id"]``: it is the token every
        # other entry point (``load_save``, ``delete_save``) resolves,
        # and a mismatched id inside the payload would list an entry
        # that cannot be opened.
        id=entry.name,
        name=derived or _RECOVERED_DISPLAY_NAME,
        last_played_at=mtime,
        created_at=created,
        schema_version=version if isinstance(version, int) else GAME_SCHEMA_VERSION,
        summary="",
        # Nothing to recover: we just read the game file ourselves.
        name_recovered=True,
    )


def _read_meta(meta_file: Path) -> SaveMeta | None:
    """Parse ``meta.json``, or ``None`` if it is missing or unreadable.

    A meta that fails to parse is treated as absent rather than fatal:
    one corrupt directory used to take down ``list_saves`` entirely,
    which also killed Continue, Delete and Rename (all of which need a
    ``save_id`` the list is the only source of) and ``commit_save``.
    """
    try:
        return SaveMeta.model_validate_json(meta_file.read_text(encoding="utf-8"))
    except (ValidationError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _summarize(game: Game) -> str:
    if game.current_node_id is None:
        return ""
    node = game.dialog_tree.nodes.get(game.current_node_id)
    if node is None or node.text is None:
        return ""
    text = node.text.strip()
    if len(text) <= 140:
        return text
    return text[:137].rstrip() + "..."


def _recovered_display_name(entry: Path, meta: SaveMeta, meta_file: Path) -> str:
    """Display name for ``meta``, recovering a placeholder at most once.

    Parsing ``game.json`` to dig out ``world.game_name`` costs ~14 ms on
    a 1000-node save, so it must not repeat on every listing. It is
    skipped entirely once ``name_recovered`` is set, and the successful
    result is written back into ``meta.json`` so the next listing reads
    the real name straight out of the (tiny) meta.

    Every failure mode — unreadable game, empty ``world.game_name``, a
    derived name that is itself a placeholder, an unwritable meta — is
    swallowed: the listing is the only route to Delete and Rename, so it
    must never fail on a damaged save.
    """
    if not _is_placeholder_name(meta.name) or meta.name_recovered:
        return meta.name

    game_path = entry / "game.json"
    derived = ""
    if game_path.exists():
        try:
            game = Game.model_validate_json(game_path.read_text(encoding="utf-8"))
            derived = (game.world.game_name or "").strip()
        except Exception:
            derived = ""
    if _is_placeholder_name(derived):
        # Nothing better than the placeholder we already have. Record
        # the dead end so the parse is never repeated for this save.
        derived = ""

    try:
        atomic_write_text(
            meta_file,
            meta.model_copy(
                update={"name": derived or meta.name, "name_recovered": True}
            ).model_dump_json(indent=2)
            + "\n",
        )
    except OSError:
        # A read-only saves directory just means we pay the parse again
        # next time; it is not a reason to hide the save.
        pass
    return derived or meta.name


def list_saves(root: Path | None = None) -> list[SaveSummary]:
    base = root or saves_dir()
    if not base.exists():
        return []
    summaries: list[SaveSummary] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        meta_file = entry / "meta.json"
        meta = _read_meta(meta_file) if meta_file.exists() else None
        if meta is None:
            # A meta that is absent entirely is not a damaged save: it is
            # the window between the two writes in ``commit_save``, and
            # ``game.json`` still holds everything a listing needs, so
            # rebuild the descriptor rather than dropping the save.
            # A meta that exists but does not parse is left on the
            # corrupt path below — something wrote nonsense over it, and
            # the game file may be no more trustworthy.
            recovered = None if meta_file.exists() else _meta_from_game(entry)
            if recovered is not None:
                summaries.append(
                    SaveSummary(
                        id=recovered.id,
                        name=recovered.name,
                        last_played_at=recovered.last_played_at,
                        created_at=recovered.created_at,
                        schema_version=recovered.schema_version,
                        summary=recovered.summary,
                    )
                )
                continue
            if not meta_file.exists():
                # No meta and no usable game: nothing to offer, not even
                # a Delete worth surfacing.
                continue
            # Surface the directory anyway so the load screen can offer
            # Delete. ``id`` is the directory name — the same token
            # ``delete_save``/``rename_save`` take — and the timestamps
            # fall back to the directory's mtime purely so the entry
            # sorts somewhere sane.
            try:
                mtime = datetime.fromtimestamp(entry.stat().st_mtime, UTC)
            except OSError:
                mtime = datetime.fromtimestamp(0, UTC)
            summaries.append(
                SaveSummary(
                    id=entry.name,
                    name=_CORRUPT_DISPLAY_NAME,
                    last_played_at=mtime,
                    created_at=mtime,
                    schema_version=0,
                    summary="",
                    corrupt=True,
                )
            )
            continue
        # Read-time recovery: if the meta is a placeholder ("Untitled"
        # or empty), prefer the saved game's ``world.game_name`` so
        # the load screen shows the real name even before the player
        # has played another turn (which is what would normally
        # rewrite the meta via ``commit_save``). Older saves committed
        # before ``Session.commit`` passed a default landed on
        # "Untitled" forever; this surfaces them correctly.
        display_name = _recovered_display_name(entry, meta, meta_file)
        summaries.append(
            SaveSummary(
                id=meta.id,
                name=display_name,
                last_played_at=meta.last_played_at,
                created_at=meta.created_at,
                schema_version=meta.schema_version,
                summary=meta.summary,
            )
        )
    summaries.sort(key=lambda s: s.last_played_at, reverse=True)
    return summaries


def load_save(save_id: str, root: Path | None = None) -> Game:
    """Read, migrate and validate ``game.json``.

    The version is checked BEFORE validation, because it is the only
    point at which a useful message exists: ``Game`` is ``extra="forbid"``,
    so a save from a newer build that added a field fails validation with
    a Pydantic error that says nothing about versions. Worse, a newer save
    that happens to still validate would load silently and be re-stamped
    at the current version by the next ``commit_save`` — dropping every
    field this build doesn't know about. So a higher version is refused
    outright and nothing is written.

    Older versions go through :mod:`.save_migrations`, which rewrites the
    decoded payload in place up to the current version.
    """
    raw = _game_path(save_id, root).read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, dict):
        version = data.get("schema_version", GAME_SCHEMA_VERSION)
        if isinstance(version, int) and version > GAME_SCHEMA_VERSION:
            raise SaveVersionError(version, GAME_SCHEMA_VERSION)
        if isinstance(version, int):
            data = migrate(data, version)
    return Game.model_validate(data)


def commit_save(
    game: Game,
    settings: Settings,
    *,
    name: str | None = None,
    root: Path | None = None,
) -> SaveMeta:
    """Persist ``game`` and refresh ``meta.json``. Returns the new meta.

    ``settings`` is accepted but NOT written. It used to be snapshotted
    into ``meta.json`` as ``settings_snapshot``, which nothing ever read
    back and which meant every save directory carried a live, billable
    ``llm.api_key`` — so a player zipping a save for a bug report handed
    their key over. The parameter stays on the signature because callers
    pass it positionally and a future migration may want it.

    Precedence for the save name: an existing meta on disk wins over
    the passed-in default. This lets callers (in particular
    ``Session.commit``) supply a sensible first-time default — the
    world's ``game_name`` — without that default clobbering a player
    rename on every subsequent commit. ``rename_save`` is the only
    path that overwrites an existing name.
    """
    directory = _save_dir(game.id, root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "images").mkdir(exist_ok=True)

    atomic_write_text(_game_path(game.id, root), game.model_dump_json(indent=2) + "\n")

    existing_name: str | None = None
    existing_created: datetime | None = None
    meta_file = _meta_path(game.id, root)
    if meta_file.exists():
        # An unparseable meta counts as absent: the commit rewrites it
        # from scratch rather than failing the save.
        existing = _read_meta(meta_file)
        if existing is not None:
            existing_name = existing.name
            existing_created = existing.created_at

    fallback = (name or "").strip() or "Untitled"
    # If the existing meta says "Untitled" (or empty), it was a
    # placeholder from a commit that ran before ``Session.commit``
    # threaded ``world.game_name`` in as the default. Treat it as
    # absent so the fallback wins and the meta upgrades to the
    # real name on the next commit. Player renames (anything other
    # than the placeholders) still take precedence over the fallback.
    chosen_name = fallback if _is_placeholder_name(existing_name) else (existing_name or fallback)
    meta = SaveMeta(
        id=game.id,
        name=chosen_name,
        last_played_at=datetime.now(UTC),
        created_at=existing_created or game.created_at,
        schema_version=GAME_SCHEMA_VERSION,
        summary=_summarize(game),
        # We have the live ``Game`` in hand, so we can answer for free
        # the question ``list_saves`` would otherwise re-parse a 1.4 MB
        # game.json to ask: is there a ``world.game_name`` worth
        # recovering? If the name we settled on is already real, or the
        # world has no name to offer, mark the recovery settled. Note
        # that ``commit_save`` does NOT itself fall back to
        # ``world.game_name`` (only ``Session.commit`` passes it as
        # ``name``), so a placeholder here with a real world name must
        # stay recoverable.
        name_recovered=not _is_placeholder_name(chosen_name)
        or _is_placeholder_name(game.world.game_name),
    )
    atomic_write_text(meta_file, meta.model_dump_json(indent=2) + "\n")
    return meta


def rename_save(save_id: str, new_name: str, root: Path | None = None) -> SaveMeta:
    meta_file = _meta_path(save_id, root)
    meta = SaveMeta.model_validate_json(meta_file.read_text(encoding="utf-8"))
    # A player-chosen name is authoritative, so recovery has nothing
    # left to do — even if they renamed the save to "Untitled".
    updated = meta.model_copy(update={"name": new_name, "name_recovered": True})
    atomic_write_text(meta_file, updated.model_dump_json(indent=2) + "\n")
    return updated


def delete_save(save_id: str, root: Path | None = None) -> None:
    shutil.rmtree(_save_dir(save_id, root), ignore_errors=True)


def most_recent_save_id(root: Path | None = None) -> str | None:
    """The save ``Continue`` should resume, without building the list.

    Called on every ``hello``, where the answer feeds a single boolean
    (``has_save``). Going through ``list_saves`` made that cost a
    display name for every save on disk — including, before the
    ``name_recovered`` gate, a full ``game.json`` parse each. This walks
    ``meta.json`` alone: a few hundred bytes per save, no game parsed,
    no summaries built.

    ``meta.json``'s mtime would be cheaper still, but it is not the same
    ordering: ``rename_save`` rewrites the meta without touching
    ``last_played_at``, so an mtime sort would resume a renamed save
    over a more recently played one. ``last_played_at`` is the field
    that actually means "most recent", so we read it.
    """
    base = root or saves_dir()
    if not base.exists():
        return None
    best: tuple[datetime, str] | None = None
    # Sorted so ties break the same way ``list_saves`` breaks them
    # (its reverse sort is stable over a directory-name ordering).
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        meta_file = entry / "meta.json"
        if not meta_file.exists():
            continue
        meta = _read_meta(meta_file)
        # Corrupt entries exist for the load screen's Delete affordance
        # only; Continue must never land on one.
        if meta is None:
            continue
        if best is None or meta.last_played_at > best[0]:
            best = (meta.last_played_at, meta.id)
    return best[1] if best is not None else None


def is_committed_node_state(state: DialogNodeState) -> bool:
    """A small reuse seam used by the render and edit handlers."""
    return state == DialogNodeState.committed

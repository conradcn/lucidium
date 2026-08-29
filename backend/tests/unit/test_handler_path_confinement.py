"""The local WebSocket is a trust boundary — these are its guardrails.

Three sinks used to be reachable in a single frame over an
unauthenticated socket, and each one is covered here:

  * ``save_id`` reaching the filesystem. ``Path.__truediv__`` DISCARDS
    the base when the right operand is absolute, so an unconstrained
    ``{"save_id": "C:/Users/<u>/Documents"}`` made ``saves_delete``
    rmtree that directory and report success. Both the wire schema and
    ``_save_dir`` now refuse it.
  * ``models_dir`` reaching ``mkdir`` + a multi-GB stream. The wire
    value used to take precedence over the configured directory.
  * ``llm.api_key`` reaching the renderer. It's a write-only secret now:
    masked in every JSON dump except the settings file itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lucidium.api.errors import SchemaError
from lucidium.api.handlers import (
    HandlerContext,
    embedded_download_model_handler,
    saves_delete_handler,
    settings_get_handler,
)
from lucidium.api.messages import (
    C2SEmbeddedDownloadModel,
    C2SSavesDelete,
    C2SSavesLoad,
    C2SSavesRename,
)
from lucidium.domain.settings import ImageSettings, LlmSettings, Settings
from lucidium.persistence import save_store
from lucidium.persistence.save_store import SaveIdError

REAL_KEY = "sk-or-v1-totally-real-billable-key"


class _StubSession:
    """The slice of ``Session`` the handlers under test actually touch."""

    def __init__(self, *, saves_root: Path, settings: Settings) -> None:
        self.saves_root = saves_root
        self.settings = settings
        self.game = None
        self.emit = None


def _ctx(saves_root: Path, settings: Settings | None = None) -> HandlerContext:
    return HandlerContext(
        session=_StubSession(  # type: ignore[arg-type]
            saves_root=saves_root,
            settings=settings or Settings(),
        )
    )


# ---------------------------------------------------------------------------
# (b) save_id confinement
# ---------------------------------------------------------------------------

# Each of these, before the fix, resolved somewhere the player never
# agreed to delete: an absolute path replaces the saves root outright,
# and the ``..`` walks out of it.
ESCAPING_SAVE_IDS = [
    "C:/Windows",
    "../../..",
    "C:\\Users\\someone\\Documents",
    "/etc",
    "sub/dir",
    "",
    "a" * 65,
]


@pytest.mark.parametrize("save_id", ESCAPING_SAVE_IDS)
def test_saves_delete_payload_rejects_path_like_ids(save_id: str) -> None:
    """The wire schema refuses the id before a handler ever runs."""
    with pytest.raises(ValidationError):
        C2SSavesDelete(save_id=save_id)


@pytest.mark.parametrize("save_id", ESCAPING_SAVE_IDS)
def test_saves_load_and_rename_reject_the_same_ids(save_id: str) -> None:
    """Load and rename take the same ``save_id`` into the same
    ``_save_dir`` join, so they carry the same constraint."""
    with pytest.raises(ValidationError):
        C2SSavesLoad(save_id=save_id)
    with pytest.raises(ValidationError):
        C2SSavesRename(save_id=save_id, new_name="whatever")


def test_saves_delete_accepts_a_real_save_id() -> None:
    """The constraint admits the ids the engine actually mints (uuid4
    hex with dashes) — it isn't so tight that it breaks deletion."""
    payload = C2SSavesDelete(save_id="0f3c1b2a-4d5e-6f70-8192-a3b4c5d6e7f8")
    assert payload.save_id.startswith("0f3c")


@pytest.mark.asyncio
@pytest.mark.parametrize("save_id", ["C:/Windows", "../../.."])
async def test_saves_delete_handler_never_touches_the_filesystem(
    tmp_path: Path,
    save_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing the payload raises, so the handler is unreachable —
    and ``delete_save`` is never called with the hostile id."""
    calls: list[str] = []
    monkeypatch.setattr(
        save_store,
        "delete_save",
        lambda sid, root=None: calls.append(sid),
    )

    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "irreplaceable.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ValidationError):
        payload = C2SSavesDelete(save_id=save_id)
        await saves_delete_handler(payload, _ctx(tmp_path / "saves"))

    assert calls == []
    assert (outside / "irreplaceable.txt").exists()


def test_save_dir_refuses_to_escape_the_root_even_off_the_wire(
    tmp_path: Path,
) -> None:
    """Belt and braces: the filesystem layer is safe on its own terms,
    not merely because its callers validate first."""
    root = tmp_path / "saves"
    root.mkdir()
    for save_id in ("C:/Windows", "../../..", "..", ""):
        with pytest.raises(SaveIdError):
            save_store._save_dir(save_id, root)
    assert save_store._save_dir("run-1", root) == (root / "run-1").resolve()


def test_delete_save_refuses_an_escaping_id(tmp_path: Path) -> None:
    root = tmp_path / "saves"
    root.mkdir()
    victim = tmp_path / "documents"
    victim.mkdir()
    (victim / "thesis.txt").write_text("years of work", encoding="utf-8")

    with pytest.raises(SaveIdError):
        save_store.delete_save(str(victim), root)
    assert (victim / "thesis.txt").exists()


# ---------------------------------------------------------------------------
# (c) models_dir confinement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedded_download_rejects_models_dir_outside_the_root(
    tmp_path: Path,
) -> None:
    """A ``models_dir`` the player never configured is refused outright.

    Before this, the wire value WON over the configured setting, so one
    frame could point a multi-GB download at (say) the Startup folder.
    """
    configured = tmp_path / "models"
    configured.mkdir()
    elsewhere = tmp_path / "startup"
    settings = Settings(
        image=ImageSettings(embedded_models_dir=str(configured)),
    )

    payload = C2SEmbeddedDownloadModel(key="sdxl", models_dir=str(elsewhere))
    with pytest.raises(SchemaError):
        await embedded_download_model_handler(
            payload,
            _ctx(tmp_path / "saves", settings),
        )
    # Nothing was created on the way to the rejection.
    assert not elsewhere.exists()


@pytest.mark.asyncio
async def test_embedded_download_allows_a_subdirectory_of_the_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Narrowing WITHIN the configured root still works — that's what
    lets the Settings screen probe a path being edited."""
    from lucidium.providers import embedded_models

    configured = tmp_path / "models"
    configured.mkdir()
    seen: list[Path] = []
    monkeypatch.setattr(
        embedded_models,
        "download_model",
        lambda spec, models_dir, on_progress=None: (
            seen.append(models_dir),
            models_dir / spec.local_filename,
        )[1],
    )
    settings = Settings(image=ImageSettings(embedded_models_dir=str(configured)))
    payload = C2SEmbeddedDownloadModel(
        key="sdxl",
        models_dir=str(configured / "sdxl"),
    )
    result = await embedded_download_model_handler(
        payload,
        _ctx(tmp_path / "saves", settings),
    )
    # Drain the generator so the download thread actually runs.
    async for _ in result:
        pass
    assert seen == [(configured / "sdxl").resolve()]


def test_resolve_models_dir_confinement_is_directly_testable(
    tmp_path: Path,
) -> None:
    from lucidium.providers.embedded_models import (
        ModelsDirOutsideRootError,
        resolve_models_dir,
    )

    root = tmp_path / "models"
    root.mkdir()
    assert resolve_models_dir(str(root), allowed_root=root) == root.resolve()
    assert (
        resolve_models_dir(str(root / "nested"), allowed_root=root) == (root / "nested").resolve()
    )
    with pytest.raises(ModelsDirOutsideRootError):
        resolve_models_dir(str(tmp_path / "elsewhere"), allowed_root=root)
    with pytest.raises(ModelsDirOutsideRootError):
        resolve_models_dir(str(root / ".." / "elsewhere"), allowed_root=root)


# ---------------------------------------------------------------------------
# (d) the API key never leaves the process
# ---------------------------------------------------------------------------


def test_settings_json_dump_never_carries_the_real_key() -> None:
    settings = Settings(llm=LlmSettings(api_key=REAL_KEY))
    dumped = settings.model_dump(mode="json")
    assert dumped["llm"]["api_key"] != REAL_KEY
    assert dumped["llm"]["api_key"] == ""
    # And it isn't hiding anywhere else in the serialized blob.
    assert REAL_KEY not in settings.model_dump_json()
    # The live value is still readable in-process, explicitly.
    assert settings.llm.api_key.get_secret_value() == REAL_KEY


def test_repr_does_not_leak_the_key() -> None:
    settings = Settings(llm=LlmSettings(api_key=REAL_KEY))
    assert REAL_KEY not in repr(settings)
    assert REAL_KEY not in str(settings.llm.api_key)


@pytest.mark.asyncio
async def test_settings_get_patch_op_masks_the_key(tmp_path: Path) -> None:
    """The ``/settings`` PatchOp the renderer receives carries a mask."""
    settings = Settings(llm=LlmSettings(api_key=REAL_KEY))
    result = await settings_get_handler(None, _ctx(tmp_path, settings))
    _message_type, patch = await anext(aiter(result))
    ops = patch.ops
    assert ops[0].path == "/settings"
    assert ops[0].value["llm"]["api_key"] == ""


def test_settings_store_still_writes_the_real_key(tmp_path: Path) -> None:
    """The engine has to authenticate on the next launch, so the ONE
    unmasked sink stays unmasked."""
    from lucidium.persistence.settings_store import load_settings, save_settings

    target = tmp_path / "settings.json"
    save_settings(Settings(llm=LlmSettings(api_key=REAL_KEY)), target)
    assert REAL_KEY in target.read_text(encoding="utf-8")
    assert load_settings(target).llm.api_key.get_secret_value() == REAL_KEY


@pytest.mark.asyncio
async def test_settings_update_does_not_wipe_the_key_on_a_masked_echo(
    tmp_app_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The renderer only ever holds the MASK, so its autosave echoes ""
    back at us. Treating that as "clear the key" would delete the
    player's credentials the first time they nudged a slider."""
    from lucidium.api.handlers import settings_update_handler
    from lucidium.api.messages import C2SSettingsUpdate
    from lucidium.persistence import settings_store

    monkeypatch.setattr(settings_store, "save_settings", lambda s, p=None: None)
    ctx = _ctx(tmp_app_data, Settings(llm=LlmSettings(api_key=REAL_KEY)))

    payload = C2SSettingsUpdate(
        patch={"llm": {"api_key": "", "model": "some/other-model"}},
    )
    result = await settings_update_handler(payload, ctx)
    _message_type, patch = await anext(aiter(result))

    assert ctx.session.settings.llm.api_key.get_secret_value() == REAL_KEY
    assert ctx.session.settings.llm.model == "some/other-model"
    # ...and the echo the renderer gets back is still masked.
    assert patch.ops[0].value["llm"]["api_key"] == ""

    # A genuinely new key still lands.
    payload = C2SSettingsUpdate(patch={"llm": {"api_key": "sk-brand-new"}})
    await anext(aiter(await settings_update_handler(payload, ctx)))
    assert ctx.session.settings.llm.api_key.get_secret_value() == "sk-brand-new"


def test_save_meta_carries_no_settings_snapshot(tmp_app_data: Path) -> None:
    """A player zipping a save for a bug report used to hand over a live
    billable key — ``SaveMeta.settings_snapshot`` is gone."""
    assert "settings_snapshot" not in save_store.SaveMeta.model_fields

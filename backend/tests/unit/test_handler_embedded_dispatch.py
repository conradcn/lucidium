"""``c2s/embedded/*`` through dispatch.

The three embedded handlers all funnel their wire-supplied ``models_dir``
through ``_embedded_models_dir``, which pins the request to the directory
the player configured. That matters most for ``download_model``, which
``mkdir``s whatever it is handed and streams multiple gigabytes into it —
an unconfined value on an unauthenticated socket is a write primitive
pointed at any path the process can reach.

``download_model`` itself is monkeypatched throughout: the real one
fetches from HuggingFace, and the offline gate would (correctly) refuse.
What's under test is the handler's confinement, spec selection, and
progress/terminal frame sequence — not the transfer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lucidium.api.errors import ProviderUnreachableError, SchemaError
from lucidium.api.handlers import HandlerContext
from lucidium.api.messages import MessageType
from lucidium.domain.settings import ImageSettings, Settings
from lucidium.providers import embedded_models

from .handler_harness import dispatch, make_registry, make_session, types_of


def _settings_rooted_at(models_root: Path) -> Settings:
    return Settings(image=ImageSettings(embedded_models_dir=str(models_root)))


@pytest.fixture
def models_root(tmp_app_data: Path) -> Path:
    root = tmp_app_data / "models"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# c2s/embedded/list_models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models_reports_checkpoints_in_the_configured_root(
    tmp_app_data: Path, models_root: Path
) -> None:
    (models_root / "b.safetensors").write_bytes(b"")
    (models_root / "a.ckpt").write_bytes(b"")
    (models_root / "notes.txt").write_text("ignored", encoding="utf-8")

    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_embedded_list_models,
        {},
    )

    assert types_of(messages) == [MessageType.s2c_embedded_models]
    reply = messages[0][1]
    assert Path(reply.models_dir) == models_root.resolve()  # noqa: ASYNC240 - sync fs in test assertion
    # Sorted, and non-checkpoint files filtered out.
    assert reply.models == ["a.ckpt", "b.safetensors"]


@pytest.mark.asyncio
async def test_list_models_allows_narrowing_to_a_subdirectory(
    tmp_app_data: Path, models_root: Path
) -> None:
    nested = models_root / "sdxl"
    nested.mkdir()
    (nested / "inner.safetensors").write_bytes(b"")
    (models_root / "outer.safetensors").write_bytes(b"")

    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_embedded_list_models,
        {"models_dir": str(nested)},
    )

    reply = messages[0][1]
    assert Path(reply.models_dir) == nested.resolve()
    assert reply.models == ["inner.safetensors"]


@pytest.mark.asyncio
async def test_list_models_rejects_a_dir_outside_the_configured_root(
    tmp_app_data: Path, models_root: Path
) -> None:
    outside = tmp_app_data / "elsewhere"
    outside.mkdir()
    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))

    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_embedded_list_models,
            {"models_dir": str(outside)},
        )


# ---------------------------------------------------------------------------
# c2s/embedded/recommend_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recommend_model_reports_the_hardware_pick_and_empty_inventory(
    tmp_app_data: Path, models_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = embedded_models.MODEL_CATALOG["sdxl-turbo"]
    monkeypatch.setattr(embedded_models, "recommend_model", lambda: spec)

    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_embedded_recommend_model,
        {},
    )

    assert types_of(messages) == [MessageType.s2c_embedded_recommended_model]
    reply = messages[0][1]
    assert reply.key == "sdxl-turbo"
    assert reply.display_name == "SDXL Turbo"
    assert reply.has_models is False
    # The TOTAL download, including the components fetched at first
    # render — not the checkpoint alone.
    assert reply.approx_bytes == spec.total_approx_bytes
    assert Path(reply.models_dir) == models_root.resolve()  # noqa: ASYNC240 - sync fs in test assertion
    assert spec.hf_repo in reply.reason


@pytest.mark.asyncio
async def test_recommend_model_sets_has_models_when_the_dir_is_populated(
    tmp_app_data: Path, models_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        embedded_models, "recommend_model", lambda: embedded_models.MODEL_CATALOG["sdxl"]
    )
    (models_root / "already-here.safetensors").write_bytes(b"")

    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_embedded_recommend_model,
        {},
    )

    assert messages[0][1].has_models is True


@pytest.mark.asyncio
async def test_recommend_model_rejects_a_dir_outside_the_configured_root(
    tmp_app_data: Path, models_root: Path
) -> None:
    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))
    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_embedded_recommend_model,
            {"models_dir": str(tmp_app_data / "elsewhere")},
        )


# ---------------------------------------------------------------------------
# c2s/embedded/download_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_model_streams_progress_then_the_refreshed_inventory(
    tmp_app_data: Path, models_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, Path]] = []

    def _fake_download(spec, dest: Path, *, on_progress=None) -> Path:
        calls.append((spec, dest))
        dest.mkdir(parents=True, exist_ok=True)
        if on_progress is not None:
            on_progress(1024, 4096)
            on_progress(4096, 4096)
        target = dest / spec.local_filename
        target.write_bytes(b"")
        return target

    monkeypatch.setattr(embedded_models, "download_model", _fake_download)

    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_embedded_download_model,
        {"key": "sdxl-turbo"},
    )

    kinds = types_of(messages)
    assert kinds[0] == MessageType.s2c_embedded_download_progress
    assert kinds[-1] == MessageType.s2c_embedded_models
    stages = [m[1].stage for m in messages if m[0] == MessageType.s2c_embedded_download_progress]
    assert stages[0] == "resolving"
    assert stages[-1] == "saving"
    assert "downloading" in stages

    # It downloaded the requested spec into the configured root, and the
    # terminal inventory sees the new file.
    assert [c[1] for c in calls] == [models_root.resolve()]  # noqa: ASYNC240 - sync fs in test assertion
    assert calls[0][0].key == "sdxl-turbo"
    assert messages[-1][1].models == ["sdxl-turbo.safetensors"]


@pytest.mark.asyncio
async def test_download_model_falls_back_to_the_recommendation_on_an_unknown_key(
    tmp_app_data: Path, models_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = embedded_models.MODEL_CATALOG["sdxl"]
    monkeypatch.setattr(embedded_models, "recommend_model", lambda: spec)
    seen: list[str] = []

    def _fake_download(s, dest: Path, *, on_progress=None) -> Path:
        seen.append(s.key)
        return dest / s.local_filename

    monkeypatch.setattr(embedded_models, "download_model", _fake_download)

    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))
    for key in ("", "no-such-model"):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_embedded_download_model,
            {"key": key},
        )

    assert seen == ["sdxl", "sdxl"]


@pytest.mark.asyncio
async def test_download_model_surfaces_a_transfer_failure_as_provider_unreachable(
    tmp_app_data: Path, models_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: Any, **_kw: Any) -> Path:
        raise OSError("connection reset mid-transfer")

    monkeypatch.setattr(embedded_models, "download_model", _boom)

    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))
    with pytest.raises(ProviderUnreachableError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_embedded_download_model,
            {"key": "sdxl-turbo"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("outside_name", ["elsewhere", "Startup"])
async def test_download_model_rejects_a_models_dir_outside_the_root(
    tmp_app_data: Path,
    models_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    outside_name: str,
) -> None:
    """The abuse case. ``download_model`` must never be reached, and no
    directory may be created at the requested path — the confinement has
    to reject BEFORE the mkdir, not clean up after it."""
    outside = tmp_app_data / outside_name

    def _must_not_run(*_a: Any, **_kw: Any) -> Path:
        raise AssertionError("download_model reached with an unconfined dir")

    monkeypatch.setattr(embedded_models, "download_model", _must_not_run)

    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))
    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_embedded_download_model,
            {"key": "sdxl-turbo", "models_dir": str(outside)},
        )

    assert not outside.exists()


@pytest.mark.asyncio
async def test_download_model_rejects_a_traversal_out_of_the_root(
    tmp_app_data: Path, models_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``<root>/../escaped`` normalises outside the root even though it is
    written as a relative path underneath it."""
    monkeypatch.setattr(
        embedded_models,
        "download_model",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("download_model reached via traversal")
        ),
    )

    session = make_session(tmp_app_data, settings=_settings_rooted_at(models_root))
    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_embedded_download_model,
            {"key": "sdxl", "models_dir": str(models_root / ".." / "escaped")},
        )

    assert not (tmp_app_data / "escaped").exists()

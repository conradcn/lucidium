"""Degraded-mode reporting for the output-side content filter.

SAFETY.md §3 promises every generated portrait is classified by the
nudity detector + face-age estimator. When that backstop can't load,
the failure must be LOUD — ERROR in the log and a one-time modal in
the renderer — not a debug line nobody reads. These tests pin that
contract so a future refactor can't quietly restore silent
degradation.
"""

from __future__ import annotations

import logging

import pytest

from lucidium.api.messages import MessageType
from lucidium.orchestration import content_filter


@pytest.fixture(autouse=True)
def _reset_latches():
    """The report latches are process-globals by design (one modal per
    run, not one per portrait) — clear them around every test."""
    content_filter.reset_unavailable_reporting_for_tests()
    yield
    content_filter.reset_unavailable_reporting_for_tests()


class _RecordingSession:
    """Minimal stand-in for the orchestration session: all
    ``_output_filter_passes`` touches is ``.emit``."""

    def __init__(self) -> None:
        self.emitted: list[tuple[object, object]] = []

    def emit(self, msg_type, payload) -> None:
        self.emitted.append((msg_type, payload))


def test_load_failure_yields_unavailable_verdict_with_reason():
    flt = content_filter._MlContentFilter()
    flt._load_failed = True
    flt._load_error = "nudenet not importable (boom)"

    result = flt.classify(b"not-really-a-png")

    assert result.verdict is content_filter.Verdict.unavailable
    assert not flt.is_active
    assert "nudenet" in result.detail


def test_report_unavailable_emits_notice_once():
    session = _RecordingSession()

    first = content_filter.report_unavailable(session.emit, detail="no models")
    second = content_filter.report_unavailable(session.emit, detail="no models")

    assert first is True, "first degraded render must notify the renderer"
    assert second is False, "a degraded filter must not pop a modal per render"
    assert len(session.emitted) == 1
    msg_type, payload = session.emitted[0]
    assert msg_type is MessageType.s2c_notice
    assert payload.title == content_filter._UNAVAILABLE_TITLE
    assert "under" in payload.body


def test_report_unavailable_logs_at_error_once(caplog):
    with caplog.at_level(logging.DEBUG, logger=content_filter.__name__):
        content_filter.report_unavailable(None, detail="no models")
        content_filter.report_unavailable(None, detail="no models")

    records = [
        r
        for r in caplog.records
        if r.name == content_filter.__name__ and r.levelno >= logging.WARNING
    ]
    assert len(records) == 1, "degraded-filter log must be once per process"
    assert records[0].levelno == logging.ERROR, (
        "SAFETY.md documents this filter as always-on; a build without it "
        "is a defect, so WARNING is too quiet"
    )


def test_report_unavailable_without_renderer_still_logs(caplog):
    """Headless runs (no connected renderer) must still leave a trace."""
    with caplog.at_level(logging.DEBUG, logger=content_filter.__name__):
        assert content_filter.report_unavailable(None) is False
    assert any(
        r.levelno == logging.ERROR for r in caplog.records if r.name == content_filter.__name__
    )


def test_output_filter_path_surfaces_notice_when_load_failed(monkeypatch):
    """End-to-end through the caller: a filter whose ``_load_failed``
    is set lets the image through (prompt+storage guards still apply)
    but pushes the notice to the renderer."""
    from lucidium.domain.character import Character
    from lucidium.orchestration import assets

    degraded = content_filter._MlContentFilter()
    degraded._load_failed = True
    degraded._load_error = "insightface not importable (boom)"
    monkeypatch.setattr(content_filter, "default_filter", lambda: degraded)

    session = _RecordingSession()
    character = Character(id="c1", name="Ada", description="a person", seed=1)

    allowed = assets._output_filter_passes(session, character, b"png-bytes")

    assert allowed is True, "a degraded backstop must not block every render"
    assert len(session.emitted) == 1
    assert session.emitted[0][0] is MessageType.s2c_notice

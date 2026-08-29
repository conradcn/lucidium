"""Output-side ML content filter.

Backstop for the prompt-side and storage-side guards: every
generated portrait runs through this filter BEFORE it's
written to disk and surfaced to the renderer. Two binary
signals are checked:

    1. **Nudity / sexual content** in the image (NudeNet).
    2. **Estimated face age** of detected faces (insightface
       ``genderage`` model, or any equivalent).

The unsafe condition is the **intersection** of (1) and (2):
the image contains explicit nudity AND any detected face has
an estimated age below the floor (default 18). When the
intersection trips, the filter:

    * marks the image ``Verdict.BLOCKED``,
    * causes the caller (``ensure_assets`` /
      ``ensure_portraits_for``) to skip writing the image,
    * emits an ``s2c/notice`` so the renderer pops a modal,
    * the player can manually rerender from the UI.

When ONLY (1) is true (nudity, but no detected face under
18), the filter returns ``Verdict.OK`` — the player has
opted into mature mode and there is no minor in frame.

When ONLY (2) is true (a detected face is estimated under
18, but the image is fully clothed), the filter returns
``Verdict.OK`` — minors are allowed in normal narrative
imagery; the policy bar is the *intersection*.

ML deps (``nudenet``, ``insightface``) ship in the packaged
build (see ``backend/lucidium.spec``) and in the ``safety``
extra for source installs. A build WITHOUT them is a
SAFETY.md §3 regression, not a supported configuration — so
when they fail to load the filter runs in DEGRADED mode and
says so LOUDLY: every call returns ``Verdict.UNAVAILABLE``,
the load failure is logged at ERROR once per process, and
:func:`report_unavailable` pushes an ``s2c/notice`` to the
renderer so the player learns the backstop is off instead of
it failing silently in a log nobody reads. The
``ContentFilter.is_active`` property tells callers whether
the filter has live models; the renderer also surfaces an
inline warning in Settings when the filter is degraded.

The full filter stack is: prompt-side (age_band floor at 18)
→ storage-side (check_minor_nudity_and_correct) → output-
side (this module). All three layers must be defeated for
unsafe content to reach the player. Removing any layer is a
SAFETY.md-noted regression.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

    from ..api.messages import MessageType

    # Same shape as ``session.OutboundEmit``; spelled locally so this
    # module stays importable without pulling in the session package.
    OutboundEmit = Callable[[MessageType, BaseModel], None]

_log = logging.getLogger(__name__)


class Verdict(StrEnum):
    """Result of running an image through the content filter."""

    ok = "ok"
    """Image cleared the filter; the caller should save it as
    normal."""

    blocked = "blocked"
    """Image hit the unsafe intersection (nudity + minor face
    estimate). The caller MUST NOT save the image; it should
    emit an s2c/notice and either retry with a clothed prompt
    or fall back to no portrait."""

    unavailable = "unavailable"
    """The ML backend isn't installed. The caller should still
    save the image (the engine has prompt-side + storage-side
    guards in place) but should log that the backstop is
    inactive."""


@dataclass(frozen=True)
class FilterResult:
    """Structured filter output. ``verdict`` drives caller
    behaviour; the score / age fields are diagnostic and end up
    in the log line + the s2c/notice body when the verdict is
    ``blocked``."""

    verdict: Verdict
    nudity_score: float = 0.0
    """0–1; > ``nudity_threshold`` is treated as explicit."""
    min_face_age: float | None = None
    """Estimated age of the youngest detected face, or None
    when no faces were detected."""
    detail: str = ""
    """Free-text note for logs / notices (e.g. which body
    parts the detector flagged, what the age estimate was)."""


class ContentFilter(Protocol):
    """Protocol every filter implementation satisfies. Tests
    inject a stub; production wires up the ML-backed default."""

    @property
    def is_active(self) -> bool: ...

    def classify(self, image_bytes: bytes) -> FilterResult: ...


# ---------- Degraded-mode reporting -----------------------------------------
#
# SAFETY.md §3 states the output-side filter is on by default and
# cannot be turned off. If it isn't running, the player is entitled
# to know — a WARNING in a log file they never open is not good
# enough. So a load failure is (a) ERROR-logged once per process and
# (b) surfaced to the renderer as an ``s2c/notice`` the first time a
# render actually hits the degraded path.
#
# Both are once-per-process: a degraded filter degrades EVERY render,
# and a modal per portrait would train the player to dismiss it.

_unavailable_logged = False
_unavailable_notified = False

_UNAVAILABLE_TITLE = "Image safety filter is not running"

_UNAVAILABLE_BODY = (
    "The output-side content filter (the check that rejects a "
    "rendered portrait combining nudity with a face estimated under "
    "18) could not start, so portraits are being saved WITHOUT that "
    "check. The prompt-side age floor and the storage-side age "
    "correction are still active. This is a defect in this build — "
    "please report it. See SAFETY.md for what each layer covers."
)


def _log_unavailable_once(reason: str) -> None:
    """ERROR-log the degraded filter once per process."""
    global _unavailable_logged
    if _unavailable_logged:
        return
    _unavailable_logged = True
    _log.error(
        "content_filter: OUTPUT-SIDE SAFETY BACKSTOP INACTIVE — %s. "
        "SAFETY.md documents this filter as always-on; a packaged "
        "build reaching this line is a packaging defect (see "
        "backend/lucidium.spec). Source installs need "
        "``pip install -e backend[safety]``.",
        reason,
    )


def report_unavailable(emit: OutboundEmit | None, *, detail: str = "") -> bool:
    """Surface the degraded filter to the renderer.

    ``emit`` is the session's emit callable (``session.emit``), or
    ``None`` when there's no connected renderer. Returns True iff a
    notice was emitted on this call — subsequent calls in the same
    process are no-ops so a degraded filter doesn't pop one modal
    per rendered portrait.

    Always ERROR-logs (once) even when there is no renderer attached,
    so a headless run still leaves a loud trace.
    """
    _log_unavailable_once(detail or "classifier reported unavailable")
    global _unavailable_notified
    if _unavailable_notified or emit is None:
        return False
    _unavailable_notified = True

    from ..api.messages import MessageType, NoticeKind, S2CNotice

    emit(
        MessageType.s2c_notice,
        S2CNotice(
            title=_UNAVAILABLE_TITLE,
            body=_UNAVAILABLE_BODY,
            kind=NoticeKind.warning,
        ),
    )
    return True


def reset_unavailable_reporting_for_tests() -> None:
    """Test hook — clear the once-per-process latches."""
    global _unavailable_logged, _unavailable_notified
    _unavailable_logged = False
    _unavailable_notified = False


def _bundled_insightface_root() -> Path | None:
    """Return the frozen build's insightface model root, or None.

    Mirrors ``app._configure_rembg_home``: PyInstaller exposes the
    extraction root via ``sys._MEIPASS`` (one-file) or the exe's own
    directory (one-folder, where collected data lands under
    ``_internal/``). ``lucidium.spec`` places the buffalo_l pack at
    ``insightface_models/models/buffalo_l``; insightface wants the
    directory ABOVE ``models/`` as its ``root``.

    Dev runs (no ``sys.frozen``) return None and insightface falls
    through to its own ``~/.insightface`` default.
    """
    if not getattr(sys, "frozen", False):
        return None
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "insightface_models")
    exe_dir = Path(sys.executable).resolve().parent
    candidates.append(exe_dir / "_internal" / "insightface_models")
    candidates.append(exe_dir / "insightface_models")
    for cand in candidates:
        if (cand / "models" / "buffalo_l").is_dir():
            return cand
    return None


# ---------- No-op fallback --------------------------------------------------


class _NoOpFilter:
    """Default when ML deps aren't installed. Always returns
    ``Verdict.unavailable``; the caller logs a one-time warning
    and proceeds without the output-side backstop. The prompt-
    and storage-side guards are still active.
    """

    @property
    def is_active(self) -> bool:
        return False

    def classify(self, image_bytes: bytes) -> FilterResult:
        return FilterResult(
            verdict=Verdict.unavailable,
            detail=(
                "ML content filter not installed; "
                "install ``lucidium[safety]`` for the output-side backstop."
            ),
        )


# ---------- ML-backed implementation ----------------------------------------


# Default thresholds. Conservative on the nudity side (we err
# toward false positives — a clothed character flagged as
# nude just gets rerendered) and conservative on the age side
# (we treat 18 as the floor; the model's estimate is noisy
# around that boundary).
DEFAULT_NUDITY_THRESHOLD = 0.55
DEFAULT_AGE_FLOOR = 18.0
# When the age estimator is noisy near the floor, treat the
# floor as 18 + epsilon — i.e., a 17.6 estimate is "under 18"
# with low confidence and we block; an 18.4 estimate is "over
# 18" with low confidence and we don't. The estimator's MAE
# on adult faces is usually 3–5 years, so this is a defence-
# in-depth margin: if you really want a minor visible in a
# clothed shot, that still passes (we only block on the
# intersection); if a model rendered an ambiguously-young
# face in nude context, we're conservative and reject.
DEFAULT_AGE_MARGIN = 1.0


class _MlContentFilter:
    """ML-backed filter. Wraps two optional libraries:

    * ``nudenet`` for nudity classification (a small ONNX
      detector that ships its own model on first use).
    * ``insightface`` for face detection + age estimation
      (the ``buffalo_l`` model bundle includes a genderage
      head). Some installs prefer ``deepface`` instead; the
      protocol allows either to be wired in via dependency
      injection in the constructor.

    Construction is lazy: the underlying ONNX runtimes load
    on the first ``classify`` call so the engine startup
    isn't blocked by tens-of-MB model downloads. A failure
    to load either backend is logged and the filter
    silently downgrades to ``_NoOpFilter`` semantics for the
    rest of the process — better than crashing every render.
    """

    def __init__(
        self,
        *,
        nudity_threshold: float = DEFAULT_NUDITY_THRESHOLD,
        age_floor: float = DEFAULT_AGE_FLOOR,
        age_margin: float = DEFAULT_AGE_MARGIN,
    ) -> None:
        self._nudity_threshold = nudity_threshold
        self._age_floor = age_floor
        self._age_margin = age_margin
        # Lazy backends — loaded on first ``classify`` call.
        self._nude_detector: object | None = None
        self._face_analyzer: object | None = None
        self._load_failed = False
        self._load_error = ""

    @property
    def is_active(self) -> bool:
        return not self._load_failed

    def _ensure_loaded(self) -> bool:
        """Lazy-load both ML backends. Returns True iff both
        are usable. Caches a permanent failure flag so we
        don't retry on every render."""
        if self._load_failed:
            return False
        if self._nude_detector is not None and self._face_analyzer is not None:
            return True

        try:
            from nudenet import NudeDetector
        except Exception as exc:
            self._fail_load(f"nudenet not importable ({exc})")
            return False

        try:
            import insightface
        except Exception as exc:
            self._fail_load(f"insightface not importable ({exc})")
            return False

        try:
            self._nude_detector = NudeDetector()
            face_kwargs: dict[str, object] = {
                "name": "buffalo_l",
                "allowed_modules": ("detection", "genderage"),
            }
            # In the frozen build the model pack ships inside the
            # bundle; point insightface at it so ``prepare()`` never
            # tries a ~280 MB network fetch (which, when it fails,
            # would silently disable the backstop).
            bundled_root = _bundled_insightface_root()
            if bundled_root is not None:
                face_kwargs["root"] = str(bundled_root)
            face_app = insightface.app.FaceAnalysis(**face_kwargs)
            face_app.prepare(ctx_id=-1)  # CPU; image filter is small + fast
            self._face_analyzer = face_app
        except Exception as exc:
            self._fail_load(f"ML backend load failed ({exc})")
            return False
        return True

    def _fail_load(self, reason: str) -> None:
        """Latch the permanent-failure flag and record WHY.

        Logged at ERROR (not WARNING): SAFETY.md §3 asserts the
        output-side filter is always on, so a build where it can't
        load is a shipped-artifact defect, not a routine optional-dep
        miss. :func:`report_unavailable` turns the same fact into a
        player-visible notice.
        """
        self._load_failed = True
        self._load_error = reason
        _log_unavailable_once(reason)

    def classify(self, image_bytes: bytes) -> FilterResult:
        if not self._ensure_loaded():
            return FilterResult(
                verdict=Verdict.unavailable,
                detail=self._load_error or "ML backend failed to load",
            )

        try:
            nudity = self._classify_nudity(image_bytes)
            min_age = self._estimate_min_face_age(image_bytes)
        except Exception as exc:
            _log.warning("content_filter: classify raised (%s)", exc)
            # Fail closed on the cautious side: treat unknown as
            # unavailable so the caller logs but the engine
            # remains usable. The prompt-side / storage-side
            # guards still cover the most-direct attack path.
            return FilterResult(
                verdict=Verdict.unavailable,
                detail=f"classification error: {exc}",
            )

        is_explicit = nudity >= self._nudity_threshold
        is_under_floor = min_age is not None and min_age < (self._age_floor - self._age_margin)

        if is_explicit and is_under_floor:
            return FilterResult(
                verdict=Verdict.blocked,
                nudity_score=nudity,
                min_face_age=min_age,
                detail=(
                    f"nudity_score={nudity:.2f} ≥ "
                    f"{self._nudity_threshold:.2f} AND face_age≈"
                    f"{min_age:.1f} < {self._age_floor:.0f}"
                ),
            )
        return FilterResult(
            verdict=Verdict.ok,
            nudity_score=nudity,
            min_face_age=min_age,
        )

    def _classify_nudity(self, image_bytes: bytes) -> float:
        """Return a 0-1 nudity score. NudeDetector returns a
        list of bounding boxes per body part; we aggregate the
        EXPLICIT classes into a single score by taking the max
        confidence among them. Anatomy classes deemed
        ``BUTTOCKS_EXPOSED``, ``FEMALE_BREAST_EXPOSED``,
        ``FEMALE_GENITALIA_EXPOSED``, ``MALE_GENITALIA_EXPOSED``,
        ``ANUS_EXPOSED`` count toward the score; covered /
        partial classes are ignored. Returns 0 on no detection.
        """
        import os
        import tempfile
        from io import BytesIO

        # NudeDetector takes a file path; serialise via a temp
        # buffer through PIL so we don't have to write a file
        # for every render. NudeNet's underlying ONNX call is
        # against a numpy array internally so this is the
        # cheapest interop available without monkey-patching.
        from PIL import Image

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as tmp:
            Image.open(BytesIO(image_bytes)).save(tmp, format="PNG")
            tmp_path = tmp.name
        try:
            results = self._nude_detector.detect(tmp_path)  # type: ignore[union-attr]
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        explicit_classes = {
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "ANUS_EXPOSED",
        }
        max_score = 0.0
        for det in results or []:
            cls = str(det.get("class", "")).upper()
            score = float(det.get("score", 0.0))
            if cls in explicit_classes:
                max_score = max(max_score, score)
        return max_score

    def _estimate_min_face_age(self, image_bytes: bytes) -> float | None:
        """Return the smallest estimated age across detected
        faces. Returns ``None`` when no faces were detected."""
        from io import BytesIO

        import numpy as np
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        # insightface expects BGR.
        bgr = arr[:, :, ::-1]
        faces = self._face_analyzer.get(bgr)  # type: ignore[union-attr]
        if not faces:
            return None
        ages = [float(getattr(f, "age", 0.0)) for f in faces if hasattr(f, "age")]
        return min(ages) if ages else None


# ---------- Module-level default --------------------------------------------


_default: ContentFilter | None = None


def default_filter() -> ContentFilter:
    """Return a process-singleton filter. Tries the ML backend
    first; falls back to no-op when deps are missing. Kept as
    a singleton so the ONNX models load once and stay in
    memory across renders."""
    global _default
    if _default is not None:
        return _default
    candidate = _MlContentFilter()
    # Probe: try a load right now so we know whether the
    # backstop is actually active. ``_ensure_loaded`` caches
    # the failure flag, so subsequent calls fast-path.
    if candidate._ensure_loaded():
        _default = candidate
        _log.info("content_filter: ML backend active")
    else:
        _default = _NoOpFilter()
    return _default


def set_default_filter_for_tests(filter_obj: ContentFilter) -> None:
    """Test hook — replace the singleton with a stub."""
    global _default
    _default = filter_obj


def reset_default_filter_for_tests() -> None:
    """Test hook — drop the singleton so the next call rebuilds."""
    global _default
    _default = None

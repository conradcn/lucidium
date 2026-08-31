"""Runtime torch-overlay manager pins.

Pure-Python, NO real torch / network / hashing / gigabyte downloads.
Every wheel "download" is a mock that writes a tiny zip; the simple-
index HTML is canned. We cover:

  * ``recommend_flavor`` for each (platform, vendor) combination.
  * simple-index HTML parsing + interpreter-aware wheel selection.
  * download → unpack → atomic swap → pointer read/write on a tmp dir.
  * idempotent re-install (already-present flavor is cheap / skipped).
  * the entry-shim path resolution agreeing with the module.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import ClassVar

import pytest

from lucidium.api.errors import ProviderValidationError
from lucidium.providers import torch_overlay as ov

# ---------------------------------------------------------------------------
# recommend_flavor — every (platform, vendor) combo
# ---------------------------------------------------------------------------


def _probes(*, nvidia=False, amd=False, arc=False):
    return {
        "has_nvidia": lambda: nvidia,
        "has_amd": lambda: amd,
        "has_intel_arc": lambda: arc,
    }


@pytest.mark.parametrize(
    ("platform", "probes", "expected"),
    [
        # NVIDIA wins on every OS.
        ("linux", _probes(nvidia=True), "cuda"),
        ("win32", _probes(nvidia=True), "cuda"),
        ("darwin", _probes(nvidia=True), "cuda"),
        # NVIDIA precedence even when another vendor also reports present.
        ("win32", _probes(nvidia=True, amd=True, arc=True), "cuda"),
        # AMD splits by OS.
        ("linux", _probes(amd=True), "rocm"),
        ("win32", _probes(amd=True), "directml"),
        ("darwin", _probes(amd=True), "directml"),  # non-linux AMD -> directml
        # Intel Arc.
        ("linux", _probes(arc=True), "xpu"),
        ("win32", _probes(arc=True), "xpu"),
        # Nothing detected -> cpu (incl. macOS with no discrete GPU).
        ("linux", _probes(), "cpu"),
        ("win32", _probes(), "cpu"),
        ("darwin", _probes(), "cpu"),
    ],
)
def test_recommend_flavor_matrix(platform, probes, expected) -> None:
    assert ov.recommend_flavor(platform=platform, **probes) == expected


def test_recommend_flavor_uses_live_probes_by_default(monkeypatch) -> None:
    """With no injected probes, it pulls the embedded_image_client vendor
    helpers — patch those and confirm they drive the result."""
    from lucidium.providers import embedded_image_client as eic

    monkeypatch.setattr(eic, "_has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(eic, "_has_amd_gpu", lambda: True)
    monkeypatch.setattr(eic, "_has_intel_arc_gpu", lambda: False)
    assert ov.recommend_flavor(platform="linux") == "rocm"


# ---------------------------------------------------------------------------
# Index HTML parsing + wheel selection
# ---------------------------------------------------------------------------

# A canned PyTorch-style simple-index page. Mixes Python versions and
# platforms so selection has to discriminate. ``cp314 / win_amd64`` is
# the wheel this (3.14 / win32) interpreter should pick when run here;
# the tests pass explicit tags so they're deterministic on any host.
_CANNED_INDEX = """
<!DOCTYPE html><html><body>
<a href="/whl/cu128/torch-2.9.0%2Bcu128-cp313-cp313-win_amd64.whl#sha256=aaa">torch-2.9.0+cu128-cp313-cp313-win_amd64.whl</a><br/>
<a href="/whl/cu128/torch-2.9.0%2Bcu128-cp314-cp314-win_amd64.whl#sha256=bbb">torch-2.9.0+cu128-cp314-cp314-win_amd64.whl</a><br/>
<a href="/whl/cu128/torch-2.9.0%2Bcu128-cp314-cp314-linux_x86_64.whl#sha256=ccc">torch-2.9.0+cu128-cp314-cp314-linux_x86_64.whl</a><br/>
<a href="/whl/cu128/torch-2.8.0%2Bcu128-cp314-cp314-win_amd64.whl#sha256=ddd">torch-2.8.0+cu128-cp314-cp314-win_amd64.whl</a><br/>
<a href="some-junk-readme.txt">readme</a>
</body></html>
"""


def test_parse_index_extracts_wheels_and_hashes() -> None:
    cands = ov.parse_index(_CANNED_INDEX, "https://download.pytorch.org/whl/cu128/torch/")
    # Only the four .whl anchors; the .txt is ignored.
    assert len(cands) == 4
    by_name = {c.filename: c for c in cands}
    win314 = by_name["torch-2.9.0+cu128-cp314-cp314-win_amd64.whl"]
    assert win314.expected_hash == ("sha256", "bbb")
    # Relative href resolved against the base.
    assert win314.url.startswith("https://download.pytorch.org/whl/cu128/")
    assert win314.url.endswith("win_amd64.whl")


def test_select_wheel_picks_interpreter_and_platform() -> None:
    cands = ov.parse_index(_CANNED_INDEX, "https://x/")
    picked = ov.select_wheel(cands, py_tag="cp314", abi_tag="cp314", platform_tag="win_amd64")
    # Highest matching version for cp314 / win_amd64 -> 2.9.0, not 2.8.0,
    # and not the cp313 or linux wheels.
    assert picked.filename == "torch-2.9.0+cu128-cp314-cp314-win_amd64.whl"


def test_select_wheel_picks_linux_when_asked() -> None:
    cands = ov.parse_index(_CANNED_INDEX, "https://x/")
    picked = ov.select_wheel(cands, py_tag="cp314", abi_tag="cp314", platform_tag="linux")
    assert picked.filename == "torch-2.9.0+cu128-cp314-cp314-linux_x86_64.whl"


# Regression: a bare "linux" platform match also matches linux_s390x and
# manylinux_*_aarch64, and the version tie-break once selected the s390x
# (IBM mainframe) wheel on an x86_64 host -- a torch that can't load. The
# default platform_tag now carries the host ARCH ("linux x86_64").
_MULTIARCH_LINUX_INDEX = (
    '<a href="torch-2.12.0+cpu-cp312-cp312-linux_s390x.whl">a</a>'
    '<a href="torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl">b</a>'
    '<a href="torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl">c</a>'
)


def test_select_wheel_rejects_wrong_linux_arch() -> None:
    cands = ov.parse_index(_MULTIARCH_LINUX_INDEX, "https://x/")
    picked = ov.select_wheel(cands, py_tag="cp312", abi_tag="cp312", platform_tag="linux x86_64")
    assert picked.filename == "torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl"


def test_select_wheel_linux_aarch64_picks_aarch64() -> None:
    cands = ov.parse_index(_MULTIARCH_LINUX_INDEX, "https://x/")
    picked = ov.select_wheel(cands, py_tag="cp312", abi_tag="cp312", platform_tag="linux aarch64")
    assert picked.filename == "torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl"


def test_interpreter_tags_includes_arch(monkeypatch) -> None:
    """_interpreter_tags must emit an OS token AND an arch token so the
    selector can't grab the wrong-arch linux wheel."""
    import platform as _p

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_p, "machine", lambda: "x86_64")
    _, _, plat = ov._interpreter_tags()
    assert "linux" in plat and "x86_64" in plat

    monkeypatch.setattr(_p, "machine", lambda: "aarch64")
    _, _, plat = ov._interpreter_tags()
    assert "linux" in plat and "aarch64" in plat

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(_p, "machine", lambda: "AMD64")
    _, _, plat = ov._interpreter_tags()
    assert "win" in plat and "amd64" in plat


def test_select_wheel_no_match_raises() -> None:
    cands = ov.parse_index(_CANNED_INDEX, "https://x/")
    with pytest.raises(ProviderValidationError) as ei:
        ov.select_wheel(cands, py_tag="cp399", abi_tag="cp399", platform_tag="win_amd64")
    assert "no compatible wheel" in str(ei.value)


def test_select_wheel_abi3_forward_compat() -> None:
    """An abi3 wheel built for an OLDER CPython must match a newer one."""
    html = '<a href="torch-2.0.0-cp39-abi3-win_amd64.whl">x</a>'
    cands = ov.parse_index(html, "https://x/")
    picked = ov.select_wheel(cands, py_tag="cp314", abi_tag="cp314", platform_tag="win_amd64")
    assert picked.filename == "torch-2.0.0-cp39-abi3-win_amd64.whl"


# ---------------------------------------------------------------------------
# install_flavor — download (mocked) -> unpack -> atomic swap -> pointer
# ---------------------------------------------------------------------------


def _fake_torch_wheel_bytes() -> bytes:
    """A minimal wheel (zip) with the top-level dirs the overlay should
    expose: ``torch/``, ``torchgen/`` (ships inside torch), a dist-info,
    and a ``.data/scripts`` entry that must be SKIPPED on unpack."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("torch/__init__.py", "# torch\n")
        zf.writestr("torchgen/__init__.py", "# torchgen ships in torch\n")
        zf.writestr("torch-2.9.0.dist-info/METADATA", "Name: torch\n")
        zf.writestr("torch-2.9.0.data/scripts/torchrun", "#!/bin/sh\n")
    return buf.getvalue()


def _fake_torchvision_wheel_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("torchvision/__init__.py", "# torchvision\n")
        zf.writestr("torchvision-0.24.0.dist-info/METADATA", "Name: torchvision\n")
    return buf.getvalue()


def _install_fixture(tmp_path: Path, monkeypatch):
    """Wire the runtime dir to tmp_path and pin the interpreter tags so
    the canned index always has a match regardless of the host Python.
    Returns a fake downloader that serves index pages + tiny wheels."""
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    # Make selection deterministic across hosts.
    monkeypatch.setattr(ov, "_interpreter_tags", lambda: ("cp314", "cp314", "win_amd64"))

    torch_index = '<a href="torch-2.9.0+cu128-cp314-cp314-win_amd64.whl#sha256=HASHTORCH">x</a>'
    tv_index = '<a href="torchvision-0.24.0+cu128-cp314-cp314-win_amd64.whl#sha256=HASHTV">x</a>'

    calls: list[str] = []

    def fake_downloader(url: str, dest: Path, on_progress) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.rstrip("/").endswith("/torch"):
            dest.write_text(torch_index, encoding="utf-8")
        elif url.rstrip("/").endswith("/torchvision"):
            dest.write_text(tv_index, encoding="utf-8")
        elif "torchvision-" in url:
            data = _fake_torchvision_wheel_bytes()
            dest.write_bytes(data)
            if on_progress:
                on_progress(len(data), len(data))
        elif "torch-" in url:
            data = _fake_torch_wheel_bytes()
            dest.write_bytes(data)
            if on_progress:
                on_progress(len(data), len(data))
        else:
            raise AssertionError(f"unexpected url {url}")

    # Every install requires an advertised hash; our index supplies one,
    # so patch the verifier to accept (we don't want to hand-compute the
    # zip's sha here — the verifier itself is tested separately).
    monkeypatch.setattr(ov, "_verify_hash", lambda path, expected: None)
    return fake_downloader, calls


def test_install_unpacks_and_swaps_and_writes_pointer(tmp_path, monkeypatch) -> None:
    downloader, _calls = _install_fixture(tmp_path, monkeypatch)

    progress: list[tuple[int, int | None]] = []
    result = ov.install_flavor(
        "cuda",
        downloader=downloader,
        on_progress=lambda d, t: progress.append((d, t)),
    )

    assert result.skipped is False
    assert result.activated is True
    overlay = ov.overlay_dir_for("cuda")
    # Both wheels' top-level dirs landed in the overlay.
    assert (overlay / "torch" / "__init__.py").is_file()
    assert (overlay / "torchgen" / "__init__.py").is_file()
    assert (overlay / "torchvision" / "__init__.py").is_file()
    # The .data/scripts payload was SKIPPED.
    assert not (overlay / "torch-2.9.0.data").exists()
    # Pointer written + readable.
    assert ov.active_flavor() == "cuda"
    assert ov.active_overlay_path() == overlay
    assert ov.installed_flavors() == ["cuda"]
    # Progress callback fired with cumulative bytes.
    assert progress and progress[-1][0] > 0


def test_install_is_idempotent(tmp_path, monkeypatch) -> None:
    downloader, calls = _install_fixture(tmp_path, monkeypatch)
    ov.install_flavor("cuda", downloader=downloader)
    n_first = len(calls)
    # Second install of the same flavor: no new downloads, marked skipped.
    result = ov.install_flavor("cuda", downloader=downloader)
    assert result.skipped is True
    assert len(calls) == n_first, "idempotent re-install must not re-download"


def test_install_unknown_flavor_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    with pytest.raises(ProviderValidationError) as ei:
        ov.install_flavor("quantum", downloader=lambda *a: None)
    assert "unknown torch overlay flavor" in str(ei.value)


def test_install_refuses_index_without_hash_fragments(tmp_path, monkeypatch) -> None:
    """An index page that advertises no ``#sha256=`` fragment must abort
    the install before anything is downloaded or written — the wheel ends
    up on sys.path, so verification fails closed, not open."""
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(ov, "_interpreter_tags", lambda: ("cp314", "cp314", "win_amd64"))

    calls: list[str] = []

    def fake_downloader(url: str, dest: Path, on_progress) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.rstrip("/").endswith("/torch"):
            # Same wheel as the happy-path fixture, minus the fragment.
            dest.write_text(
                '<a href="torch-2.9.0+cu128-cp314-cp314-win_amd64.whl">x</a>',
                encoding="utf-8",
            )
        else:
            raise AssertionError(f"must not fetch {url} after a hashless index")

    with pytest.raises(ProviderValidationError) as ei:
        ov.install_flavor("cuda", downloader=fake_downloader)

    msg = str(ei.value)
    assert "no #sha256=" in msg
    # The offending index URL is named so the failure is actionable.
    assert any(url in msg for url in calls)
    assert not ov.overlay_dir_for("cuda").exists()
    assert ov.installed_flavors() == []
    assert ov.active_flavor() is None


def test_install_switch_replaces_active_pointer(tmp_path, monkeypatch) -> None:
    """Installing a second flavor flips the active pointer and keeps
    both overlay dirs on disk (atomic swap of distinct dirs)."""
    downloader, _ = _install_fixture(tmp_path, monkeypatch)
    ov.install_flavor("cuda", downloader=downloader)
    ov.install_flavor("cpu", downloader=downloader)
    assert ov.active_flavor() == "cpu"
    assert sorted(ov.installed_flavors()) == ["cpu", "cuda"]


def test_atomic_swap_overwrites_existing(tmp_path) -> None:
    """A reinstall (force) must replace an existing non-empty overlay dir
    without leaving the old contents or a backup behind."""
    final = tmp_path / "overlays" / "cuda"
    final.mkdir(parents=True)
    (final / "old.txt").write_text("stale")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("fresh")

    ov._atomic_swap_dir(staging, final)

    assert (final / "new.txt").is_file()
    assert not (final / "old.txt").exists()
    # No leftover backup dirs.
    leftovers = [p for p in (tmp_path / "overlays").iterdir() if p.name != "cuda"]
    assert leftovers == []


def test_verify_hash_mismatch_raises(tmp_path) -> None:
    f = tmp_path / "wheel.whl"
    f.write_bytes(b"the bytes")
    with pytest.raises(ProviderValidationError) as ei:
        ov._verify_hash(f, ("sha256", "deadbeef"))
    assert "hash mismatch" in str(ei.value)


def test_verify_hash_accepts_correct(tmp_path) -> None:
    import hashlib

    data = b"the bytes"
    f = tmp_path / "wheel.whl"
    f.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    # Should not raise.
    ov._verify_hash(f, ("sha256", digest))


# ---------------------------------------------------------------------------
# Pointer / status helpers
# ---------------------------------------------------------------------------


def test_active_overlay_none_when_pointer_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    assert ov.active_flavor() is None
    assert ov.active_overlay_path() is None


def test_active_overlay_none_when_dir_missing(tmp_path, monkeypatch) -> None:
    """Pointer names a flavor whose dir was deleted -> resolve to None so
    the entry shim never prepends a stale path."""
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    ov.write_active_flavor("cuda")
    assert ov.active_flavor() == "cuda"
    assert ov.active_overlay_path() is None  # dir never created


def test_runtime_dir_platform_layout(monkeypatch) -> None:
    monkeypatch.delenv("LUCIDIUM_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(ov.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\x\AppData\Local")
    assert ov.runtime_dir() == Path(r"C:\Users\x\AppData\Local") / "Lucidium" / "runtime"

    monkeypatch.setattr(ov.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/home/x/.local/share")
    assert ov.runtime_dir() == Path("/home/x/.local/share") / "lucidium" / "runtime"


# ---------------------------------------------------------------------------
# seed_bundled_overlay — first-run offline CPU seed
# ---------------------------------------------------------------------------


def _make_bundled_cpu_source(tmp_path: Path) -> Path:
    """Build a fake pre-unpacked CPU overlay source dir (what the spec
    bakes into the frozen bundle). Contents stand in for an unpacked
    torch+torchvision wheel tree."""
    src = tmp_path / "bundled" / "cpu"
    (src / "torch").mkdir(parents=True)
    (src / "torch" / "__init__.py").write_text("# torch\n", encoding="utf-8")
    (src / "torchvision").mkdir()
    (src / "torchvision" / "__init__.py").write_text("# tv\n", encoding="utf-8")
    (src / "torch-2.9.0+cpu.dist-info").mkdir()
    (src / "torch-2.9.0+cpu.dist-info" / "METADATA").write_text(
        "Name: torch\nVersion: 2.9.0+cpu\n", encoding="utf-8"
    )
    return src


def test_seed_copies_and_activates_cpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    src = _make_bundled_cpu_source(tmp_path)

    activated = ov.seed_bundled_overlay(source=src)

    assert activated == "cpu"
    overlay = ov.overlay_dir_for("cpu")
    assert (overlay / "torch" / "__init__.py").is_file()
    assert (overlay / "torchvision" / "__init__.py").is_file()
    assert ov.active_flavor() == "cpu"
    assert ov.active_overlay_path() == overlay


def test_seed_is_noop_when_overlay_already_active(tmp_path, monkeypatch) -> None:
    """Never clobber an already-active flavor (e.g. a GPU torch the user
    downloaded). Seeding must be a no-op when there's a live overlay."""
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    # Pretend the user already installed + activated cuda.
    ov.overlay_dir_for("cuda").mkdir(parents=True)
    (ov.overlay_dir_for("cuda") / "torch").mkdir()
    ov.write_active_flavor("cuda")

    src = _make_bundled_cpu_source(tmp_path)
    activated = ov.seed_bundled_overlay(source=src)

    assert activated is None
    assert ov.active_flavor() == "cuda"
    # No cpu overlay was seeded.
    assert not ov.overlay_dir_for("cpu").exists()


def test_seed_reactivates_existing_cpu_without_recopy(tmp_path, monkeypatch) -> None:
    """If overlays/cpu already exists but no pointer is set, seeding just
    re-activates it (no re-copy)."""
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    cpu = ov.overlay_dir_for("cpu")
    cpu.mkdir(parents=True)
    marker = cpu / "already-here.txt"
    marker.write_text("existing", encoding="utf-8")

    src = _make_bundled_cpu_source(tmp_path)
    activated = ov.seed_bundled_overlay(source=src)

    assert activated == "cpu"
    assert ov.active_flavor() == "cpu"
    # Existing contents were NOT overwritten by the bundled source.
    assert marker.read_text(encoding="utf-8") == "existing"
    assert not (cpu / "torch").exists()


def test_seed_noop_when_no_bundled_source(tmp_path, monkeypatch) -> None:
    """Unfrozen / no-source: seeding does nothing and returns None."""
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    # bundled_overlay_source() returns None when not frozen.
    monkeypatch.setattr(ov.sys, "frozen", False, raising=False)
    assert ov.seed_bundled_overlay() is None
    assert ov.active_flavor() is None


def test_bundled_overlay_source_none_when_not_frozen(monkeypatch) -> None:
    monkeypatch.setattr(ov.sys, "frozen", False, raising=False)
    assert ov.bundled_overlay_source() is None


def test_bundled_overlay_source_finds_beside_exe(tmp_path, monkeypatch) -> None:
    """When frozen, the source is found next to the exe under
    ``_internal/bundled-overlay/cpu`` (PyInstaller one-folder layout)."""
    exe = tmp_path / "lucidium-backend.exe"
    exe.write_text("", encoding="utf-8")
    overlay = tmp_path / "_internal" / "bundled-overlay" / "cpu"
    overlay.mkdir(parents=True)
    (overlay / "torch").mkdir()

    monkeypatch.setattr(ov.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ov.sys, "executable", str(exe))
    # Ensure no _MEIPASS shadows the exe-dir probe.
    monkeypatch.delattr(ov.sys, "_MEIPASS", raising=False)

    found = ov.bundled_overlay_source()
    assert found == overlay


def test_bundled_overlay_source_none_when_empty(tmp_path, monkeypatch) -> None:
    """An empty bundled dir must NOT be treated as a usable source —
    seeding from it would write an empty overlay and break torch import."""
    exe = tmp_path / "lucidium-backend.exe"
    exe.write_text("", encoding="utf-8")
    (tmp_path / "_internal" / "bundled-overlay" / "cpu").mkdir(parents=True)

    monkeypatch.setattr(ov.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ov.sys, "executable", str(exe))
    monkeypatch.delattr(ov.sys, "_MEIPASS", raising=False)

    assert ov.bundled_overlay_source() is None


# ---------------------------------------------------------------------------
# Entry-shim path resolution must agree with the module
# ---------------------------------------------------------------------------


def _load_entry_shim():
    """Import the dependency-free helpers out of lucidium_pyi_entry
    WITHOUT triggering its ``from lucidium.app import main`` (which pulls
    in the whole app). We exec the file up to that import line."""

    backend_root = Path(ov.__file__).resolve().parents[3]
    entry = backend_root / "lucidium_pyi_entry.py"
    src = entry.read_text(encoding="utf-8")
    # Chop at the heavy import so we only run the path helpers. Match the
    # import at the START of a line (the phrase also appears inside the
    # comment block above it).
    cut = src.index("\nfrom lucidium.app import main") + 1
    ns: dict = {}
    exec(compile(src[:cut], str(entry), "exec"), ns)
    return ns


def test_entry_shim_runtime_dir_matches_module(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    ns = _load_entry_shim()
    assert Path(ns["_lucidium_runtime_dir"]()) == ov.runtime_dir()


def test_entry_shim_active_overlay_matches_module(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    # No pointer yet -> both None.
    ns = _load_entry_shim()
    assert ns["_lucidium_active_overlay"]() is None
    assert ov.active_overlay_path() is None

    # Create an active flavor with a real dir.
    ov.write_active_flavor("cpu")
    ov.overlay_dir_for("cpu").mkdir(parents=True)
    (ov.overlay_dir_for("cpu") / "torch").mkdir()
    ns2 = _load_entry_shim()
    shim_path = ns2["_lucidium_active_overlay"]()
    assert shim_path is not None
    assert Path(shim_path) == ov.active_overlay_path()


def test_entry_shim_runtime_dir_matches_module_all_platforms(monkeypatch) -> None:
    """Cross-check the inline duplication against the module for each OS
    branch — this is the guard that keeps the shim from drifting."""
    monkeypatch.delenv("LUCIDIUM_RUNTIME_DIR", raising=False)
    for platform, env in (
        ("win32", ("LOCALAPPDATA", r"C:\AppData\Local")),
        ("linux", ("XDG_DATA_HOME", "/x/.local/share")),
        ("darwin", None),
    ):
        monkeypatch.setattr(ov.sys, "platform", platform)
        ns = _load_entry_shim()
        monkeypatch.setattr(ns["_sys"], "platform", platform)
        if env:
            monkeypatch.setenv(*env)
        assert Path(ns["_lucidium_runtime_dir"]()) == ov.runtime_dir(), platform


# ---------------------------------------------------------------------------
# Download timeouts + stall watchdog
# ---------------------------------------------------------------------------


class _BlockingResponse:
    """A response whose ``read`` never returns — the black-holed-TCP case
    (captive portal / dead CDN edge) that used to hang the install
    forever. ``close`` releases the blocked reader, as a real socket
    close does."""

    headers: ClassVar[dict[str, str]] = {"Content-Length": "999999999"}

    def __init__(self) -> None:
        import threading

        self._released = threading.Event()
        self.closed = False
        self.reads_started = 0

    def read(self, _n: int = 0) -> bytes:
        self.reads_started += 1
        self._released.wait()  # unblocked only by close()
        return b""

    def close(self) -> None:
        self.closed = True
        self._released.set()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def test_stalled_download_aborts_and_lowers_install_gate(tmp_path, monkeypatch) -> None:
    """A read that never returns must not pin ``is_install_in_flight``.

    Before the watchdog, ``urlopen`` had no timeout, so this install
    thread parked forever, ``install_flavor``'s ``finally: _exit_install``
    never ran, and image generation refused every render with
    ``torch_installing`` for the process lifetime.
    """
    import time

    from lucidium.api.errors import ProviderUnreachableError

    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(ov, "_interpreter_tags", lambda: ("cp314", "cp314", "win_amd64"))
    # Short window so the test is fast; the mechanism is identical at 120s.
    # Restore the module globals after the test (configure_* is process-wide).
    monkeypatch.setattr(ov, "_connect_timeout_s", None)
    monkeypatch.setattr(ov, "_stall_timeout_s", None)
    ov.configure_download_timeouts(connect_timeout_s=5.0, stall_timeout_s=0.5)

    response = _BlockingResponse()
    opened: list[str] = []

    def fake_urlopen(req, timeout=None):
        opened.append(getattr(req, "full_url", str(req)))
        # The connect timeout must be passed — never the no-timeout default.
        assert timeout == 5.0
        return response

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    started = time.monotonic()
    # downloader=None => the REAL network path, with urlopen stubbed.
    with pytest.raises(ProviderUnreachableError) as ei:
        ov.install_flavor("cuda", downloader=None)
    elapsed = time.monotonic() - started

    assert "stalled" in str(ei.value)
    assert opened, "the real downloader must have been exercised"
    # Aborted within the watchdog window (plus slack), not hung.
    assert elapsed < 15.0
    assert response.closed, "the stalled response must be closed on abort"
    # The gate is DOWN: the whole point of the fix.
    assert ov.is_install_in_flight() is False


def test_render_not_refused_after_a_stalled_install(tmp_path, monkeypatch) -> None:
    """The renderer's ``torch_installing`` refusal must clear once a
    stalled install has been aborted."""
    import asyncio

    from lucidium.api.errors import ProviderUnreachableError

    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(ov, "_interpreter_tags", lambda: ("cp314", "cp314", "win_amd64"))
    # Restore the module globals after the test (configure_* is process-wide).
    monkeypatch.setattr(ov, "_connect_timeout_s", None)
    monkeypatch.setattr(ov, "_stall_timeout_s", None)
    ov.configure_download_timeouts(connect_timeout_s=5.0, stall_timeout_s=0.5)

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _BlockingResponse())
    with pytest.raises(ProviderUnreachableError):
        ov.install_flavor("cuda", downloader=None)

    from lucidium.providers import embedded_image_client as eic

    client = eic.EmbeddedImageClient.__new__(eic.EmbeddedImageClient)
    with pytest.raises(Exception) as ei:
        asyncio.run(client.generate("character.workflow.json", {}, seed=1))
    # It may still fail (no model/torch in the test env) — but NEVER with
    # the install gate, which the abort released.
    assert "torch_installing" not in str(ei.value)


def test_connect_failure_surfaces_as_unreachable(tmp_path, monkeypatch) -> None:
    """A connect-phase timeout is reported (not swallowed) and names the
    budget, so the failure is diagnosable from the log."""
    from lucidium.api.errors import ProviderUnreachableError

    monkeypatch.setenv("LUCIDIUM_RUNTIME_DIR", str(tmp_path / "runtime"))
    # Restore the module globals after the test (configure_* is process-wide).
    monkeypatch.setattr(ov, "_connect_timeout_s", None)
    monkeypatch.setattr(ov, "_stall_timeout_s", None)
    ov.configure_download_timeouts(connect_timeout_s=3.0, stall_timeout_s=0.5)

    import urllib.request

    def boom(req, timeout=None):
        # What a real connect timeout raises (``socket.timeout`` is an
        # alias for the builtin, and both are ``OSError`` subclasses).
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ProviderUnreachableError) as ei:
        ov._default_downloader("https://example.invalid/x.whl", tmp_path / "x.whl", None)
    assert "3s" in str(ei.value)
    assert ov.is_install_in_flight() is False


def test_download_timeouts_defaults_and_env_override(monkeypatch) -> None:
    """Env overrides apply; junk and non-positive values fall back to the
    defaults — "no timeout" is never reachable."""
    monkeypatch.setattr(ov, "_connect_timeout_s", None)
    monkeypatch.setattr(ov, "_stall_timeout_s", None)
    monkeypatch.delenv("LUCIDIUM_TORCH_CONNECT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("LUCIDIUM_TORCH_STALL_TIMEOUT_S", raising=False)
    assert ov.download_timeouts() == (
        ov.DEFAULT_CONNECT_TIMEOUT_S,
        ov.DEFAULT_STALL_TIMEOUT_S,
    )

    monkeypatch.setenv("LUCIDIUM_TORCH_STALL_TIMEOUT_S", "45")
    assert ov.download_timeouts()[1] == 45.0
    for bad in ("nonsense", "0", "-1"):
        monkeypatch.setenv("LUCIDIUM_TORCH_STALL_TIMEOUT_S", bad)
        assert ov.download_timeouts()[1] == ov.DEFAULT_STALL_TIMEOUT_S

    with pytest.raises(ValueError):
        ov.configure_download_timeouts(stall_timeout_s=0)


def test_progress_reports_total_when_known(tmp_path, monkeypatch) -> None:
    """Byte totals are plumbed through so the UI can render a determinate
    bar — with ``total=None`` a stalled download looked exactly like a
    slow one."""
    downloader, _calls = _install_fixture(tmp_path, monkeypatch)
    progress: list[tuple[int, int | None]] = []
    ov.install_flavor(
        "cuda", downloader=downloader, on_progress=lambda d, t: progress.append((d, t))
    )
    assert progress
    assert all(t is not None and t >= d for d, t in progress)


# ---------------------------------------------------------------------------
# ``-m <module>`` dispatch in the frozen entry shim.
#
# The PyInstaller bootloader hands ``-m mod`` through as plain argv rather
# than acting on it, so without this dispatch the frozen exe answers
# ``lucidium-backend.exe -m lucidium.providers.gpu_selftest`` by starting
# the server -- which spawns that exact command again, forever. These pin
# the dispatch and its allowlist.
# ---------------------------------------------------------------------------


def _run_entry_shim(monkeypatch, argv: list[str]) -> list[tuple[str, str]]:
    """Exec the shim's prefix with ``argv`` in place, recording every
    ``runpy.run_module`` call instead of really running one."""
    import runpy

    calls: list[tuple[str, str]] = []

    def _fake_run_module(mod, run_name=None, alter_sys=False):
        calls.append((mod, run_name))
        return {}

    monkeypatch.setattr(runpy, "run_module", _fake_run_module)
    monkeypatch.setattr(sys, "argv", list(argv))
    _load_entry_shim()
    return calls


def test_entry_shim_dispatches_dash_m_to_runpy(monkeypatch) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_entry_shim(
            monkeypatch, ["lucidium-backend.exe", "-m", "lucidium.providers.gpu_selftest"]
        )
    assert excinfo.value.code == 0


def test_entry_shim_dash_m_rewrites_argv_for_the_target(monkeypatch) -> None:
    """The dispatched module must see the argv CPython would have given
    it: the module name in slot 0, its own arguments after."""
    import runpy

    seen: list[list[str]] = []
    monkeypatch.setattr(runpy, "run_module", lambda *a, **k: seen.append(list(sys.argv)) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["lucidium-backend.exe", "-m", "lucidium.providers.gpu_selftest", "--flavor", "cpu"],
    )
    with pytest.raises(SystemExit):
        _load_entry_shim()
    assert seen == [["lucidium.providers.gpu_selftest", "--flavor", "cpu"]]


def test_entry_shim_does_not_dispatch_foreign_modules(monkeypatch) -> None:
    """A shipped build is not a general-purpose interpreter: ``-m`` on
    anything outside the ``lucidium.`` namespace falls through to the
    normal launch rather than running the module."""
    calls = _run_entry_shim(monkeypatch, ["lucidium-backend.exe", "-m", "http.server"])
    assert calls == []


def test_entry_shim_normal_launch_is_untouched(monkeypatch) -> None:
    calls = _run_entry_shim(monkeypatch, ["lucidium-backend.exe"])
    assert calls == []

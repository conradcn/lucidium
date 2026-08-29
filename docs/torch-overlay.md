# The torch overlay

torch and torchvision are **excluded** from the PyInstaller freeze. Instead they
are loaded at runtime from a writable "overlay" directory that
`lucidium_pyi_entry.py` prepends to `sys.path` before any import that could
transitively pull torch in.

Code: `backend/src/lucidium/providers/torch_overlay.py`; the exclusion lives in
`backend/lucidium.spec`.

## Why

- The correct torch build depends on the player's GPU (CUDA / ROCm / DirectML /
  Intel XPU / CPU). That is unknowable at build time, and bundling every flavour
  would push the installer past 10 GiB.
- The frozen exe has no pip, so it cannot `pip install` the right flavour after
  the fact. The overlay manager instead fetches wheels (a wheel is a zip) from
  PyTorch's PEP 503 simple index and unpacks them itself.

Only torch and torchvision go in the overlay. torch's pure-Python dependencies
(numpy, sympy, typing_extensions, …) stay frozen in the bundle. `torchgen` ships
*inside* the torch wheel, so unpacking torch provides it — it is not fetched
separately. On Windows, torch's own `__init__` calls
`os.add_dll_directory(torch/lib)`, so no manual DLL handling is needed.

## Layout

The overlay tree is deliberately separate from the app-data tree (saves,
settings, models): these are large, regenerable, machine-specific binaries, so a
settings reset or a save-folder sync must never touch them, and a torch
reinstall must never risk a save. On Windows it uses `LOCALAPPDATA`, not the
roaming `APPDATA` app-data uses, so multi-GiB binaries never roam.

| Platform | Runtime root |
|---|---|
| Windows | `%LOCALAPPDATA%\Lucidium\runtime\` |
| macOS | `~/Library/Application Support/Lucidium/runtime/` |
| Linux | `$XDG_DATA_HOME/lucidium/runtime` (or `~/.local/share/lucidium/runtime`) |

```
overlays/
  cpu/  cuda/  rocm/  directml/  xpu/    # unpacked torch + torchvision wheels
active_overlay                           # one line: the active flavor name
```

`LUCIDIUM_RUNTIME_DIR` overrides the root (used by tests and by the packaging
script's staging step). `LUCIDIUM_TORCH_OVERLAY` names a fully resolved overlay
directory directly, bypassing the pointer file.

The path-resolution helpers at the top of the module are dependency-free stdlib
so the tiny PyInstaller entry shim can import them; the shim duplicates the
layout logic inline and a unit test asserts the two stay in agreement. Change
one, change the other.

## Flavors

| Flavor | Index | Notes |
|---|---|---|
| `cuda` | `download.pytorch.org/whl/cu130` | NVIDIA CUDA 13.0. Not `cu132`: same torch build, higher driver floor. |
| `rocm` | `.../rocm7.2` | AMD ROCm 7.2, Linux only |
| `xpu` | `.../xpu` | Intel Arc / oneAPI |
| `cpu` | `.../cpu` | No GPU acceleration |
| `directml` | `.../cpu` + PyPI | Plain CPU torch plus the separate `torch-directml` wheel, which registers the DML `privateuseone` device. Windows AMD / any DX12 GPU. |

`recommend_flavor` picks by vendor precedence NVIDIA → AMD → Intel Arc → CPU:
NVIDIA anywhere gives `cuda`; AMD gives `rocm` on Linux and `directml` on
Windows; Intel Arc gives `xpu`; otherwise `cpu`. The vendor probes are shared
with the live device detection in `embedded_image_client`, so the overlay and
the renderer agree about the hardware, and they are injectable so tests can
sweep every (platform, vendor) pair without real hardware. There is no
Apple-Silicon flavour — a Mac recommends `cpu` here, with MPS torch expected via
the platform's normal install path.

## The bundled CPU overlay

`package.ps1`'s `Build-CpuOverlay` step downloads and unpacks the **CPU** wheels
for the build interpreter into a staging directory and hands it to the spec via
`LUCIDIUM_BUNDLED_OVERLAY_DIR`, which bakes it into the frozen tree. On first
launch the engine seeds that into `overlays/cpu` and writes `active_overlay`, so
image generation works offline immediately, with no download. Every installer is
identical; players with a GPU fetch a faster flavour from Settings afterwards.

## Installing at runtime

`c2s/torch_overlay/status` returns `{ recommended, installed, active,
runtime_dir, activated }`. `c2s/torch_overlay/install { flavor, activate }`
downloads and unpacks the wheels, streaming `s2c/torch_overlay/progress`
(`stage`, `bytes_done`, `bytes_total` — null when there is no Content-Length).

**A newly installed overlay only takes effect on the next backend start**: the
active flavour is chosen by the entry shim before torch is imported, so an
already-running process keeps the torch it loaded.

While an install is in flight, `is_install_in_flight()` is true and image
generation **refuses** with a `ProviderUnreachableError` tagged
`torch_installing`. That is a deliberate product call: without the gate the
renderer would quietly fall back to the currently-active CPU overlay and hand
the player minutes-per-image CPU SDXL during exactly the window when the GPU
wheel they asked for is downloading. It is a counter, not a boolean, so
concurrent or nested installs are tolerated and the gate stays raised until the
last one unwinds.

A stalled or failed install is recovered by deleting
`runtime/overlays/<flavor>/` and retrying; `<app-data>/session.log` has the
detail.

## Dev note

`start.ps1` / `start.sh` runs the backend from `backend/.venv`, which uses the
venv's own torch — the overlay machinery is a packaged-build concern. To
exercise a specific overlay in dev, point `LUCIDIUM_TORCH_OVERLAY` at a resolved
overlay directory.

## See also

- [packaging.md](packaging.md) — the staging step that produces the bundled CPU overlay.
- [model-catalog.md](model-catalog.md) — the checkpoints this runtime executes.
- [operations.md](operations.md) — the on-disk trees.

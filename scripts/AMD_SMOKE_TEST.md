# Lucidium GPU acceleration — smoke test (for testers)

Thanks for helping test GPU image-generation support! This checks that
Lucidium can download the right PyTorch build for your GPU and that your
GPU actually computes correctly. It takes ~5–15 minutes (most of it a
one-time download) and produces **one report file** to send back.

You need a machine with a GPU you want to test:
- **AMD** (e.g. Radeon RX 6700 XT or similar) on **Windows** or **Linux**
- NVIDIA or Intel Arc also work

---

## 1. Get the project set up

Clone/unzip the repo, then set up the backend once:

- **Windows (PowerShell):** `./start.ps1 -Setup`
- **Linux:** create the backend venv as per the project README (or run
  `scripts/package-linux.sh` once, which builds it).

This only needs to be done once.

## 2. Run the smoke test

From the repo root:

- **Windows (PowerShell):**
  ```powershell
  scripts\amd-smoke.ps1
  ```
- **Linux:**
  ```bash
  bash scripts/amd-smoke.sh
  ```

That's it. It will:
1. Detect your GPU and pick the right "flavor" (AMD+Windows → DirectML,
   AMD+Linux → ROCm, NVIDIA → CUDA, Intel Arc → XPU).
2. **Download** the matching PyTorch build (a few hundred MB for DirectML,
   ~2–3 GB for ROCm/CUDA — this is the slow part, and it's cached so a
   second run is instant).
3. Load that PyTorch and run real GPU math (a matrix multiply and a
   convolution — the same operations image generation uses) and check the
   GPU results match a CPU reference.

When it finishes you'll see **`OVERALL: PASS`** or **`FAIL`** and a line like:
```
report written: .../scripts/amd-smoke-report-YYYYMMDD-HHMMSS.txt
```

## 3. Send the report back

**Email/attach that `amd-smoke-report-*.txt` file to the developer**, whether
it passed or failed. It contains your OS, GPU, the detected device, timings,
and any error details — no personal data.

---

## Optional: test a full image render

If you have an SDXL `.safetensors` checkpoint, you can also test a real
(tiny, 2-step) image render:

- **Windows:** `scripts\amd-smoke.ps1 -Checkpoint C:\path\to\model.safetensors`
- **Linux:** `bash scripts/amd-smoke.sh --checkpoint /path/to/model.safetensors`

## Troubleshooting / notes

- **It says `flavor: cpu` / "no GPU detected"** but you have a GPU: force it
  with `--flavor`:
  - Windows AMD: `scripts\amd-smoke.ps1 -Flavor directml`
  - Linux AMD: `bash scripts/amd-smoke.sh --flavor rocm`
  - NVIDIA: `--flavor cuda`  · Intel Arc: `--flavor xpu`
  Then send the report — "GPU present but detected as CPU" is exactly the
  kind of result we need to see.
- **AMD on Linux (RX 6700 XT and similar):** the test automatically sets
  `HSA_OVERRIDE_GFX_VERSION=10.3.0`, which ROCm needs for these cards. If
  your card needs a different value, set it before running:
  `HSA_OVERRIDE_GFX_VERSION=<value> bash scripts/amd-smoke.sh`.
- **The download is large.** It goes into `scripts/amd-smoke-out/runtime/`
  by default and won't touch your actual app install. You can delete that
  folder afterward. Add `--use-app-runtime` if you want the app itself to
  use the downloaded GPU build afterward.
- **It failed during download:** that itself is useful — send the report;
  it captures the exact URL and error.
```

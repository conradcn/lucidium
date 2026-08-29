#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build the Lucidium release artefacts (Windows installer + Linux AppImage).

.DESCRIPTION
    By default produces BOTH the Windows installer .exe AND the Linux
    AppImage (cross-built via WSL2). Pass ``-WindowsOnly`` to skip the
    WSL leg, or ``-Linux`` to skip the Windows leg. If WSL2 isn't
    present, the Linux build is skipped with a warning rather than
    aborting the Windows build.

    Orchestrates the full packaging pipeline:

      1. Verify the backend venv exists with PyInstaller installed.
      2. Build the bundled CPU torch overlay: download + unpack the
         CPU torch/torchvision wheels (matched to the venv's Python)
         into a build staging dir. torch is NOT frozen into the
         bundle anymore; instead this pre-unpacked CPU overlay is
         baked into the frozen tree so image generation works
         OFFLINE on first launch with NO download. A user with a GPU
         can later download a faster flavor (CUDA/ROCm/DirectML/XPU)
         from the app's settings.
      3. Run PyInstaller against ``backend/lucidium.spec`` to bundle
         the engine + Python interpreter + heavy ML deps (minus
         torch) into ``backend/dist/lucidium-backend/``. The spec
         picks up the CPU overlay from step 2 via the
         ``LUCIDIUM_BUNDLED_OVERLAY_DIR`` env var.
      5. Run the frontend build (``tsc -b && vite build``) so the
         renderer + Electron main+preload TypeScript compile to
         ``frontend/dist/`` and ``frontend/dist-electron/``.
      6. Run electron-builder to assemble a self-extracting 7z
         archive (target=``7z``, output is a single .exe) at
         ``frontend/release/Lucidium-Setup.exe``. We use
         the 7z target rather than NSIS because the bundled
         backend (now ~1-2 GB without frozen torch, plus a CPU
         overlay) is larger than NSIS's 32-bit makensis
         can memory-map; 7z self-extractors don't have that
         limit and still produce a single double-clickable .exe.
         The ``extraResources`` block in package.json copies the
         PyInstaller output into the packaged app's resources so
         the runtime ``resolveBackendCommand`` (in
         ``frontend/electron/main.ts``) can spawn the bundled
         engine without the dev venv on PATH.

    Idempotent: each step short-circuits if its output is already
    fresh and ``-Force`` is not passed.

.PARAMETER SkipBackend
    Skip steps 2 + 3 (CPU overlay build + PyInstaller). Useful when
    iterating on the frontend after a successful backend build.

.PARAMETER SkipFrontend
    Skip steps 5 + 6. Useful when iterating on the backend bundle.

.PARAMETER Force
    Rebuild every step from scratch -- clears
    ``backend/build/``, ``backend/dist/``, ``frontend/dist/``,
    ``frontend/dist-electron/``, and ``frontend/release/``.

.PARAMETER Clean
    Same as ``-Force`` but exits after cleaning. Use to wipe
    build artefacts without rebuilding.

.PARAMETER Linux
    Produce ONLY the Linux AppImage (skip the Windows installer).
    Shells out to WSL2 (``wsl.exe``) and runs
    ``scripts/package-linux.sh`` inside it -- PyInstaller bundles the
    host Python interpreter, so the Linux backend must be produced
    from a Linux environment. Requires WSL2 + a distro with
    Python >= 3.11 installed. Output:
    ``frontend/release/Lucidium-*.AppImage``. The ``-SkipBackend``,
    ``-SkipFrontend``, ``-Force``, and ``-Clean`` switches forward
    through to the bash script. Mutually exclusive with
    ``-WindowsOnly``.

.PARAMETER WindowsOnly
    Produce ONLY the Windows installer (skip the Linux AppImage step
    that the default flow would otherwise run via WSL). Useful on
    machines without WSL2, or when iterating on a Windows-only
    change. Mutually exclusive with ``-Linux``.

.PARAMETER WslDistro
    Override the WSL distribution used for ``-Linux`` builds. Defaults
    to whatever ``wsl.exe`` resolves to without an explicit ``-d``
    flag (i.e. the user's default distro). Ignored unless ``-Linux``
    is also set.

.EXAMPLE
    pwsh package.ps1
    Full build of BOTH the Windows installer and the Linux AppImage
    (Linux first via WSL, then Windows on the host).

.EXAMPLE
    pwsh package.ps1 -WindowsOnly
    Build only the Windows installer. Use on machines without WSL2.

.EXAMPLE
    pwsh package.ps1 -SkipBackend
    Quick rebuild of both installers after a renderer-only change.

.EXAMPLE
    pwsh package.ps1 -Force
    Rebuild everything (both targets) from scratch.

.EXAMPLE
    pwsh package.ps1 -Linux
    Cross-build only the Linux AppImage from this Windows host (via WSL2).

.NOTES
    Caveats -- read before shipping:

      * PyInstaller bundles diffusers, transformers, and accelerate
        but NOT torch — torch is loaded at runtime from a per-user
        overlay dir (see ``backend/src/lucidium/providers/
        torch_overlay.py`` and ``lucidium_pyi_entry.py``). UPX
        compression is DISABLED in lucidium.spec because it corrupts
        several of the bundled .pyd / .dll files.
      * The installer ships a pre-unpacked CPU torch overlay so the
        app works offline on first launch with no download. The
        flavour of that overlay is ALWAYS cpu regardless of what
        torch the venv has installed (Build-CpuOverlay downloads the
        CPU wheels explicitly). Users with a GPU download a faster
        flavour (CUDA/ROCm/DirectML/XPU) from the app afterwards, so
        there's no longer a "rebuild the venv with the CUDA wheel"
        step to produce a GPU installer — every installer is the
        same and GPU torch is fetched on the user's machine.
      * The first ``pyinstaller`` run is slow (~5-10 min) -- it
        walks every transitive import of the ML stack. Subsequent
        runs reuse ``backend/build/`` and complete much faster
        unless ``-Force`` is passed.
      * Code signing is not configured. The installer will trip
        SmartScreen warnings until signed.
#>

[CmdletBinding()]
param(
    [switch]$SkipBackend,
    [switch]$SkipFrontend,
    [switch]$Force,
    [switch]$Clean,
    [switch]$Linux,
    [switch]$WindowsOnly,
    [string]$WslDistro
)

if ($Linux -and $WindowsOnly) {
    throw "-Linux and -WindowsOnly are mutually exclusive. Pass at most one."
}

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $RepoRoot "backend"
$FrontendDir = Join-Path $RepoRoot "frontend"
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"

function Write-Section($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Assert-Tool($name, $hint) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Required tool '$name' not found on PATH. $hint"
    }
}

function Invoke-Clean {
    Write-Section "Cleaning build artefacts"
    # Vite owns ``frontend/dist`` (renderer bundle); electron-
    # builder owns ``frontend/release`` (the packaged app + the
    # SFX-wrapped .exe). Listed separately because they're
    # written by different tools -- wiping the wrong one breaks
    # an in-progress build.
    $paths = @(
        (Join-Path $BackendDir "build"),
        (Join-Path $BackendDir "dist"),
        (Join-Path $FrontendDir "dist"),
        (Join-Path $FrontendDir "dist-electron"),
        (Join-Path $FrontendDir "release")
    )
    foreach ($p in $paths) {
        if (Test-Path $p) {
            Write-Host "  removing $p"
            Remove-Item -Recurse -Force $p
        }
    }
}

function Stop-StaleBuildProcesses {
    # Previous failed runs (or a manual test-launch of the
    # packaged app) can leave processes holding open file
    # handles inside frontend/release/ or frontend/dist/. The
    # next vite/electron-builder run then EPERMs trying to
    # clear those directories. Kill anything that looks like
    # ours before we start:
    #
    #   * 7za / esbuild -- child processes from a prior failed
    #     electron-builder run.
    #   * Lucidium / lucidium-backend -- instances of the
    #     packaged app the user launched to smoke-test the
    #     last build (Electron spawns a tree of helper
    #     processes: main, GPU, renderer, network, utility,
    #     each holding handles to the .exe / .dll files).
    $stale = Get-Process |
        Where-Object {
            try {
                if ($_.Name -in @("Lucidium", "lucidium-backend")) {
                    return $true
                }
                ($_.Path -like "*frontend\node_modules*") -and
                ($_.Name -in @("7za", "esbuild"))
            } catch { $false }
        }
    if ($stale) {
        Write-Host "==> Killing stale build/launch processes: $($stale.Name -join ', ')"
        $stale | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

function Test-VenvSetup {
    if (-not (Test-Path $VenvPython)) {
        throw @"
backend/.venv missing. Set it up first:
    .\start.ps1 -Setup
Then re-run this script.
"@
    }
}

function Confirm-U2NetCached {
    # Warm rembg's U2NET cache before PyInstaller runs, so the
    # spec can copy the .onnx into the bundle. Skipped when the
    # file is already in place (the common case).
    Write-Section "Ensuring U2NET ONNX is cached for the bundle"
    $u2netHome = $env:U2NET_HOME
    if (-not $u2netHome) { $u2netHome = (Join-Path $env:USERPROFILE ".u2net") }
    $target = Join-Path $u2netHome "u2net.onnx"
    if (Test-Path $target) {
        $sizeMB = [Math]::Round((Get-Item $target).Length / 1MB, 1)
        Write-Host "  cached: $target ($sizeMB MB)"
        return
    }
    Write-Host "  fetching u2net.onnx (~170 MB) into $u2netHome..."
    $script = @"
import os
from rembg.sessions.u2net import U2netSession
print('downloading to', U2netSession.u2net_home())
U2netSession.download_models()
print('done')
"@
    $tmp = New-TemporaryFile
    Rename-Item $tmp.FullName ($tmp.FullName + ".py") -Force
    $tmpPy = $tmp.FullName + ".py"
    Set-Content -Path $tmpPy -Value $script -Encoding utf8
    try {
        & $VenvPython $tmpPy
        if ($LASTEXITCODE -ne 0) {
            throw "u2net.onnx download failed."
        }
    } finally {
        Remove-Item $tmpPy -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path $target)) {
        throw "u2net.onnx still missing after download attempt at $target"
    }
}

function Install-SafetyExtras {
    # The ``safety`` extra carries the output-side content filter
    # (nudenet + insightface). SAFETY.md §3 documents that filter as
    # always-on and not disableable, so the frozen bundle MUST ship
    # it — the venv start.ps1 built may predate this requirement, so
    # install it here rather than assuming.
    #
    # Also warms insightface's buffalo_l model pack into
    # ``~/.insightface`` so lucidium.spec can copy the detection +
    # genderage ONNX files into the bundle; without them the packaged
    # filter would try a ~280 MB download on first render and
    # silently degrade when it failed.
    Write-Section "Ensuring content filter (safety extras) is in the backend venv"
    & $VenvPython -c "import nudenet, insightface" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  installing backend[safety]..."
        & $VenvPython -m pip install --quiet -e "$BackendDir[safety]"
        if ($LASTEXITCODE -ne 0) { throw "backend[safety] install failed." }
    } else {
        Write-Host "  nudenet + insightface already present"
    }

    $ifHome = $env:INSIGHTFACE_HOME
    if (-not $ifHome) { $ifHome = (Join-Path $env:USERPROFILE ".insightface") }
    $pack = Join-Path $ifHome "models\buffalo_l"
    if ((Test-Path (Join-Path $pack "det_10g.onnx")) -and
        (Test-Path (Join-Path $pack "genderage.onnx"))) {
        Write-Host "  cached: $pack"
    } else {
        Write-Host "  fetching insightface buffalo_l (~280 MB) into $ifHome..."
    }
    $script = @"
from insightface.app import FaceAnalysis
from nudenet import NudeDetector

NudeDetector()
app = FaceAnalysis(name='buffalo_l', allowed_modules=('detection', 'genderage'))
app.prepare(ctx_id=-1)
print('content filter models ready')
"@
    $tmp = New-TemporaryFile
    Rename-Item $tmp.FullName ($tmp.FullName + ".py") -Force
    $tmpPy = $tmp.FullName + ".py"
    Set-Content -Path $tmpPy -Value $script -Encoding utf8
    try {
        & $VenvPython $tmpPy
        if ($LASTEXITCODE -ne 0) { throw "content filter model warm-up failed." }
    } finally {
        Remove-Item $tmpPy -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path (Join-Path $pack "genderage.onnx"))) {
        throw "insightface buffalo_l models still missing under $pack"
    }
}

function Install-PyInstaller {
    # Install the exact pin from backend/pyproject.toml's ``packaging``
    # extra rather than probing for "any pyinstaller". lucidium.spec
    # depends on PyInstaller internals, so a venv that happens to carry
    # a different release would produce a materially different bundle
    # from the same commit. pip is a no-op when the pinned version is
    # already installed, so this stays cheap on repeat runs.
    Write-Section "Ensuring pinned PyInstaller is in the backend venv"
    & $VenvPython -m pip install --quiet -e "$BackendDir[packaging]"
    if ($LASTEXITCODE -ne 0) { throw "pyinstaller (pinned) install failed." }
    & $VenvPython -m pip show pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "pyinstaller missing after install." }
}

function Build-CpuOverlay {
    # Build the pre-unpacked CPU torch+torchvision overlay that ships
    # INSIDE the frozen bundle so the app works offline on first launch
    # with NO download. torch is excluded from the freeze (loaded from a
    # runtime overlay), so without a baked CPU overlay the very first
    # render would have no torch at all.
    #
    # We reuse the production install path: point torch_overlay's runtime
    # dir at a build staging dir and call ``install_flavor("cpu",
    # activate=False)``. That downloads + unpacks the CPU wheels for THIS
    # interpreter (Python/platform-matched, exactly what the frozen exe
    # will load) into ``<staging>/overlays/cpu``. The spec then bundles
    # that dir via ``LUCIDIUM_BUNDLED_OVERLAY_DIR``. Idempotent: a
    # non-empty staging overlay is reused (no re-download) unless -Force.
    Write-Section "Building bundled CPU torch overlay (downloads CPU wheels once)"
    # OS-SPECIFIC staging dir (``-win``). The Linux build
    # (scripts/package-linux.sh) stages under ``bundled-overlay-cpu-linux``.
    # A shared path would let whichever build ran first bake its platform's
    # torch into the other's bundle (the reuse check only tests for a
    # non-empty dir, not which OS's wheels it holds) -- that shipped Windows
    # .dll/.lib torch inside a Linux AppImage once.
    $stagingRoot = Join-Path $BackendDir "build\bundled-overlay-cpu-win"
    $overlayDir = Join-Path $stagingRoot "overlays\cpu"
    if ($Force -and (Test-Path $stagingRoot)) {
        Remove-Item -Recurse -Force $stagingRoot
    }
    if ((Test-Path $overlayDir) -and (Get-ChildItem $overlayDir -ErrorAction SilentlyContinue)) {
        Write-Host "  CPU overlay already staged at $overlayDir"
    } else {
        $prevRt = $env:LUCIDIUM_RUNTIME_DIR
        $env:LUCIDIUM_RUNTIME_DIR = $stagingRoot
        try {
            & $VenvPython -c "from lucidium.providers.torch_overlay import install_flavor; r = install_flavor('cpu', activate=False); print('staged CPU overlay at', r.overlay_dir)"
            if ($LASTEXITCODE -ne 0) { throw "CPU overlay build failed." }
        } finally {
            if ($null -eq $prevRt) {
                Remove-Item Env:LUCIDIUM_RUNTIME_DIR -ErrorAction SilentlyContinue
            } else {
                $env:LUCIDIUM_RUNTIME_DIR = $prevRt
            }
        }
    }
    if (-not (Test-Path $overlayDir)) {
        throw "CPU overlay still missing after build at $overlayDir"
    }
    # Hand the resolved overlay dir to the spec so it bundles it.
    $script:BundledOverlayDir = (Resolve-Path $overlayDir).Path
    Write-Host "  bundled CPU overlay: $script:BundledOverlayDir"
}

function Invoke-BackendBuild {
    Write-Section "Bundling backend with PyInstaller (slow on first run)"
    Push-Location $BackendDir
    try {
        # The spec reads LUCIDIUM_BUNDLED_OVERLAY_DIR to bake the CPU
        # overlay into the frozen tree (see Build-CpuOverlay + the spec).
        if ($script:BundledOverlayDir) {
            $env:LUCIDIUM_BUNDLED_OVERLAY_DIR = $script:BundledOverlayDir
        }
        $pyiArgs = @(
            "-m", "PyInstaller",
            "--noconfirm",
            "lucidium.spec"
        )
        & $VenvPython @pyiArgs
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

        $expected = Join-Path $BackendDir "dist\lucidium-backend\lucidium-backend.exe"
        if (-not (Test-Path $expected)) {
            throw "PyInstaller finished but $expected is missing -- spec output paths drifted?"
        }
        Write-Host "  bundled backend at $expected"
    } finally {
        Remove-Item Env:LUCIDIUM_BUNDLED_OVERLAY_DIR -ErrorAction SilentlyContinue
        Pop-Location
    }
}

function Invoke-FrontendBuild {
    Write-Section "Building frontend (vite + tsc-electron)"
    & npm --prefix $FrontendDir install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed." }

    & npm --prefix $FrontendDir run codegen
    if ($LASTEXITCODE -ne 0) { throw "codegen failed." }

    & npm --prefix $FrontendDir run build
    if ($LASTEXITCODE -ne 0) { throw "frontend build failed." }

    # Electron main + preload need to be compiled to plain JS.
    # The ``main`` field in package.json points at
    # ``dist-electron/main.js``; tsconfig.electron emits it.
    $tscConfig = Join-Path $FrontendDir "tsconfig.electron.json"
    if (Test-Path $tscConfig) {
        Write-Host "  compiling electron main+preload..."
        & npx --prefix $FrontendDir tsc -p $tscConfig
        if ($LASTEXITCODE -ne 0) { throw "electron tsc failed." }
    } else {
        # Fall back: the main.js / preload.js are kept in lockstep
        # with the .ts files manually. ``npm run build`` already
        # ran ``tsc -b`` which should have emitted them; verify.
        $mainJs = Join-Path $FrontendDir "electron\main.js"
        if (-not (Test-Path $mainJs)) {
            Write-Warning @"
electron/main.js not found and no tsconfig.electron.json exists.
The packaged app may fail to launch. Compile electron/main.ts manually,
or add a tsconfig.electron.json.
"@
        }
    }
}

function Invoke-ElectronBuilder {
    Write-Section "Running electron-builder (7z self-extracting archive)"
    # electron-builder reads ``package.json`` from cwd as its
    # project root; ``--config <path>`` only points at the
    # config file, not the project. Run with cwd = frontend so
    # both the config and the resolved ``files``/``extraResources``
    # globs anchor at the directory the package.json lives in.
    Push-Location $FrontendDir
    try {
        # ``signAndEditExecutable: false`` is set in
        # frontend/package.json's ``build.win`` block so
        # electron-builder skips both ``signtool`` and the
        # ``rcedit`` metadata pass -- both pull the winCodeSign
        # cache archive, which contains macOS dylib SYMLINKS
        # that Windows can only extract under admin or with
        # Developer Mode enabled (otherwise: "A required
        # privilege is not held by the client"). Skipping the
        # cache is the cleanest fix that doesn't require
        # elevation. ``CSC_IDENTITY_AUTO_DISCOVERY=false``
        # backstops by disabling certificate-discovery in case
        # the user's machine has a stray code-signing cert
        # registered system-wide. Output: an unsigned
        # ``Lucidium Setup x.y.z.exe`` that SmartScreen will
        # warn on but installs and runs fine.
        $env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
        try {
            & npx electron-builder --win 7z
            if ($LASTEXITCODE -ne 0) { throw "electron-builder failed." }
        } finally {
            Remove-Item Env:CSC_IDENTITY_AUTO_DISCOVERY -ErrorAction SilentlyContinue
        }
    } finally {
        Pop-Location
    }

    # Wrap the .7z archive in a 7-Zip GUI SFX stub so the
    # output is a single double-clickable .exe. The 7z target
    # is the only one electron-builder offers that actually
    # works at our payload size (~4.6 GB backend); NSIS's
    # 32-bit makensis can't memory-map an archive that large
    # and ``portable`` extracts to %TEMP% on EVERY launch
    # which is awful for a multi-GB app. The SFX wrapper here
    # is the standard 7-Zip approach: ``stub + config + 7z``
    # concatenated into one file. The stub is part of any
    # 7-Zip install; we error out clearly if it isn't found
    # so the user knows what to install.
    $sfxStub = "C:\Program Files\7-Zip\7zSD.sfx"
    if (-not (Test-Path $sfxStub)) {
        $sfxStub = "C:\Program Files\7-Zip\7z.sfx"
    }
    $distDir = Join-Path $FrontendDir "release"
    $archive = Get-ChildItem $distDir -Filter "Lucidium-*-win.7z" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $archive) {
        Write-Warning "electron-builder finished but no Lucidium-*-win.7z found under $distDir"
        return
    }
    if (-not (Test-Path $sfxStub)) {
        Write-Warning @"
7-Zip SFX stub not found at C:\Program Files\7-Zip\.
Cannot build a self-extracting .exe. The plain archive is at:
  $($archive.FullName)
Install 7-Zip from https://www.7-zip.org/ and re-run with -SkipBackend
to produce a single-file installer.
"@
        return
    }

    Write-Section "Wrapping 7z payload in a self-extracting .exe"
    $sfxConfig = Join-Path $distDir "Lucidium-sfx.txt"
    $exeOut = Join-Path $distDir "Lucidium-Setup.exe"

    # The SFX config block tells 7zSD what to do at extract
    # time. ``Title`` and ``BeginPrompt`` show a confirmation
    # before extraction; ``RunProgram`` launches the app
    # immediately after extracting so the user gets a one-
    # double-click experience. The config file MUST be UTF-8
    # WITH BOM and use ``;!@Install@!UTF-8!`` / ``;!@InstallEnd@!``
    # delimiters -- the SFX stub parses these markers verbatim.
    $sfxConfigText = @"
;!@Install@!UTF-8!
Title="Lucidium"
BeginPrompt="Install Lucidium? Extracts to a folder you choose."
RunProgram="Lucidium.exe"
GUIMode="0"
;!@InstallEnd@!
"@
    # PowerShell's default file write produces UTF-8 without
    # BOM; 7-Zip SFX requires a BOM. Use .NET to write it
    # explicitly.
    $utf8Bom = New-Object System.Text.UTF8Encoding($true)
    [System.IO.File]::WriteAllText($sfxConfig, $sfxConfigText, $utf8Bom)

    # Concatenate stub + config + archive → final .exe.
    # ``cmd /c copy /b`` is the canonical way to do binary-
    # concatenation on Windows; PowerShell's ``Get-Content``
    # is text-mode by default and will mangle binary bytes.
    $copyCmd = "copy /b `"$sfxStub`" + `"$sfxConfig`" + `"$($archive.FullName)`" `"$exeOut`""
    cmd /c $copyCmd | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $exeOut)) {
        throw "SFX concatenation failed."
    }
    Remove-Item $sfxConfig -ErrorAction SilentlyContinue

    $size = (Get-Item $exeOut).Length / 1GB
    Write-Host ""
    Write-Host "Installer ready at:" -ForegroundColor Green
    Write-Host "  $exeOut ($([Math]::Round($size, 2)) GB)" -ForegroundColor Green
}

function Invoke-LinuxBuildViaWsl {
    # Cross-build a Linux AppImage by shelling out to WSL2 and running
    # ``scripts/package-linux.sh`` against a Linux-side venv. PyInstaller
    # cannot cross-target Linux from a Windows interpreter -- the
    # bundled python.exe and .dll/.pyd binaries are platform-native --
    # so the actual heavy lifting MUST happen inside a Linux
    # environment, which on Windows means WSL.
    Write-Section "Cross-building Linux AppImage via WSL"

    Assert-Tool "wsl.exe" "Install WSL2 (https://learn.microsoft.com/windows/wsl/install)."

    # Translate the repo root to a WSL path. ``wsl.exe wslpath -a`` is
    # the canonical converter and handles UNC + drive-letter forms.
    # Pin the translation to the TARGET distro so a default that's a
    # minimal helper (e.g. Docker Desktop's ``docker-desktop`` distro,
    # which ships no ``wslpath`` binary) doesn't hijack the lookup.
    # NOTE: backslashes in the path are eaten somewhere on the
    # PowerShell -> wsl.exe argv boundary (``C:\Users\...`` arrives at
    # wslpath as ``C:Users...``). Forward slashes survive the trip
    # intact and Linux tools accept them natively, so normalise here.
    $repoForward = $RepoRoot -replace '\\', '/'
    $translateArgs = @()
    if ($WslDistro) { $translateArgs += @("-d", $WslDistro) }
    $translateArgs += @("--", "wslpath", "-a", $repoForward)
    $wslRepoRoot = & wsl.exe @translateArgs 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslRepoRoot)) {
        $tip = if ($WslDistro) {
            "Distro '$WslDistro' is missing wslpath -- install coreutils, or pick a different distro."
        } else {
            "No -WslDistro passed and the default distro can't translate paths. Try -WslDistro Ubuntu-24.04."
        }
        throw "Could not translate $RepoRoot to a WSL path. $tip"
    }
    $wslRepoRoot = $wslRepoRoot.Trim()

    # Forward the user's flags to the bash script. We deliberately don't
    # forward ``-Linux`` itself -- inside WSL the script is already
    # Linux-targeting by definition.
    $bashArgs = @()
    if ($SkipBackend)  { $bashArgs += "--skip-backend" }
    if ($SkipFrontend) { $bashArgs += "--skip-frontend" }
    if ($Force)        { $bashArgs += "--force" }
    if ($Clean)        { $bashArgs += "--clean" }

    # Quote argv values so a path with spaces / a flag carrying a
    # value can't be re-split by bash's word splitting on the
    # invocation line.
    $argString = ($bashArgs | ForEach-Object { "'" + ($_ -replace "'", "'\\''") + "'" }) -join " "
    $scriptInWsl = "$wslRepoRoot/scripts/package-linux.sh"
    $bashCmd = "cd '$wslRepoRoot' && bash '$scriptInWsl' $argString"

    $wslArgs = @()
    if ($WslDistro) { $wslArgs += @("-d", $WslDistro) }
    $wslArgs += @("--", "bash", "-lc", $bashCmd)

    Write-Host "  wsl.exe $($wslArgs -join ' ')" -ForegroundColor DarkGray
    & wsl.exe @wslArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Linux build via WSL failed (exit $LASTEXITCODE)."
    }
}

# ---- main ----

# ``-Clean`` is a wipe-only action that targets shared paths under
# backend/ + frontend/. Handle it once up-front (regardless of
# -Linux / -WindowsOnly) so we don't double-clean and so a
# combination like ``-Linux -Clean`` actually cleans.
if ($Clean) {
    Stop-StaleBuildProcesses
    Invoke-Clean
    Write-Host "Clean complete. Run without -Clean to rebuild."
    exit 0
}

# Cross-build the Linux AppImage first (slow WSL leg). We do this
# BEFORE the Windows build so that the Windows leg's npm install
# repopulates frontend/node_modules with Windows-native binaries,
# leaving the checkout in a Windows-dev-ready state at the end.
# When wsl.exe is missing we skip with a warning rather than
# aborting -- unless the user explicitly asked for -Linux, in
# which case there's no Windows fallback and we must fail loudly.
$buildLinux = -not $WindowsOnly
$buildWindows = -not $Linux
$linuxBuilt = $false

if ($buildLinux) {
    $wslCmd = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if ($wslCmd) {
        Invoke-LinuxBuildViaWsl
        $linuxBuilt = $true
        Write-Host ""
        Write-Host "Linux package complete." -ForegroundColor Green
        if (-not $buildWindows) { exit 0 }
    } elseif ($Linux) {
        throw "wsl.exe not found. Install WSL2 (https://learn.microsoft.com/windows/wsl/install)."
    } else {
        Write-Warning "wsl.exe not found -- skipping Linux AppImage. Pass -WindowsOnly to silence this warning."
    }
}

Assert-Tool "npm" "Install Node.js (https://nodejs.org)."
Assert-Tool "npx" "Install Node.js (https://nodejs.org)."

Test-VenvSetup
Stop-StaleBuildProcesses

# Skip the Windows-side clean if the Linux leg already ran -- it
# already cleaned (via --force, forwarded by Invoke-LinuxBuildViaWsl)
# AND produced the AppImage in frontend/release/, which Invoke-Clean
# would wipe. PyInstaller --noconfirm overwrites backend/dist anyway,
# so we don't lose the from-scratch semantics of -Force.
if ($Force -and -not $linuxBuilt) {
    Invoke-Clean
}

if (-not $SkipBackend) {
    Install-PyInstaller
    Install-SafetyExtras
    Confirm-U2NetCached
    Build-CpuOverlay
    Invoke-BackendBuild
} else {
    Write-Host "Skipping backend bundle (-SkipBackend)" -ForegroundColor Yellow
}

if (-not $SkipFrontend) {
    Invoke-FrontendBuild
    Invoke-ElectronBuilder
} else {
    Write-Host "Skipping frontend bundle + installer (-SkipFrontend)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Package complete." -ForegroundColor Green

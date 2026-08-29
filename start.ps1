#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Launch the Lucidium desktop app in dev mode.

.DESCRIPTION
    Idempotent first-run setup (venv, npm install, schema codegen,
    Electron main+preload compile), then starts the Vite dev server in
    the background and launches Electron pointing at it. Electron's main
    process spawns the Python backend automatically (see
    frontend/electron/main.ts).

.PARAMETER Setup
    Run setup steps and exit without launching the app.

.PARAMETER Backend
    Launch only the Python backend (no Electron, no Vite). Useful for
    end-to-end testing against a hand-driven WebSocket client.

.PARAMETER Renderer
    Launch only the Vite dev server (no Electron, no backend). The
    renderer will fail to connect — handy for UI iteration.

.PARAMETER SkipCodegen
    Skip the Pydantic -> JSON Schema -> TypeScript step. Use only when
    you know the schemas are current.

.PARAMETER NoSetup
    Skip the dependency-presence checks. Use after the first successful
    run to start faster.

.EXAMPLE
    .\start.ps1
    First-time setup if needed, then launch.

.EXAMPLE
    .\start.ps1 -Setup
    Install everything but do not launch.

.EXAMPLE
    .\start.ps1 -Backend
    Run the backend alone; speak to it from a tool that can drive a
    WebSocket (the integration tests are one such tool).
#>

[CmdletBinding()]
param(
    [switch]$Setup,
    [switch]$Backend,
    [switch]$Renderer,
    [switch]$SkipCodegen,
    [switch]$NoSetup
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$BackendDir   = Join-Path $RepoRoot "backend"
$FrontendDir  = Join-Path $RepoRoot "frontend"
$VenvDir      = Join-Path $BackendDir ".venv"
$PythonExe    = Join-Path $VenvDir   "Scripts\python.exe"
$NodeModules  = Join-Path $FrontendDir "node_modules"
$DistElectron = Join-Path $FrontendDir "dist-electron"
$ElectronBin  = Join-Path $NodeModules ".bin\electron.cmd"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Note([string]$Message) {
    Write-Host "    $Message" -ForegroundColor DarkGray
}

function Test-Required {
    foreach ($Tool in @("python", "node", "npm")) {
        if (-not (Get-Command $Tool -ErrorAction SilentlyContinue)) {
            throw "Required tool not found on PATH: $Tool"
        }
    }
}

function Get-PythonVersion([string]$Exe) {
    if (-not (Test-Path $Exe)) { return $null }
    $previousEAP = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        $output = & $Exe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
        if ($LASTEXITCODE -ne 0) { return $null }
        return [string]$output
    } finally {
        $ErrorActionPreference = $previousEAP
    }
}

function Test-VenvPythonAcceptable {
    $venvVersion = Get-PythonVersion $PythonExe
    if (-not $venvVersion) { return $false }
    $parts = $venvVersion -split "\."
    if ($parts.Length -lt 2) { return $false }
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    # Constitution: Python >= 3.11.
    return ($major -gt 3) -or ($major -eq 3 -and $minor -ge 11)
}

function Test-BackendInstalled {
    if (-not (Test-Path $PythonExe)) { return $false }
    $previousEAP = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        $null = & $PythonExe -c "import lucidium, pydantic" 2>&1
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $previousEAP
    }
}

function Initialize-Backend {
    $needsVenv = -not (Test-Path $PythonExe)

    if (-not $needsVenv -and -not (Test-VenvPythonAcceptable)) {
        $stale = Get-PythonVersion $PythonExe
        $current = Get-PythonVersion (Get-Command python).Source
        Write-Step "Backend venv uses Python $stale; the project requires >= 3.11."
        Write-Note "Rebuilding the venv with the current Python ($current)."
        Remove-Item -Recurse -Force $VenvDir
        $needsVenv = $true
    }

    if ($needsVenv) {
        $hostVersion = Get-PythonVersion (Get-Command python).Source
        if (-not $hostVersion) { throw "could not determine the python on PATH" }
        $hostParts = $hostVersion -split "\."
        $hostMajor = [int]$hostParts[0]
        $hostMinor = [int]$hostParts[1]
        if (($hostMajor -lt 3) -or ($hostMajor -eq 3 -and $hostMinor -lt 11)) {
            throw "Python $hostVersion on PATH is too old; install Python >= 3.11."
        }
        Write-Step "Creating backend virtualenv at $VenvDir (Python $hostVersion)"
        & python -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed" }
    } else {
        Write-Note ("backend venv present (Python " + (Get-PythonVersion $PythonExe) + ")")
    }

    if (Test-BackendInstalled) {
        Write-Note "lucidium + pydantic already installed in venv"
        return
    }

    # ``safety`` alongside ``dev``: the output-side content filter
    # (SAFETY.md §3) is documented as always-on, so a dev venv without
    # it silently exercises a code path the shipped build never takes.
    Write-Step "Installing backend (with dev + safety extras)"
    & $PythonExe -m pip install --upgrade pip wheel | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & $PythonExe -m pip install -e "$BackendDir[dev,safety]" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "pip install -e backend[dev,safety] failed" }

    if (-not (Test-BackendInstalled)) {
        throw "lucidium did not import after install; check pip output above"
    }
}

function Initialize-Frontend {
    if (Test-Path $NodeModules) {
        Write-Note "frontend node_modules already present"
        return
    }
    Write-Step "Installing frontend dependencies"
    & npm --prefix $FrontendDir install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
}

function Invoke-Codegen {
    if ($SkipCodegen) {
        Write-Note "skipping codegen"
        return
    }
    Write-Step "Exporting JSON Schemas (Pydantic -> shared-schemas/)"
    $exportScript = Join-Path $RepoRoot "scripts\codegen\export-schemas.py"
    & $PythonExe $exportScript
    if ($LASTEXITCODE -ne 0) { throw "schema export failed" }

    Write-Step "Generating TypeScript types (json-schema-to-typescript)"
    & npm --prefix $FrontendDir run codegen
    if ($LASTEXITCODE -ne 0) { throw "TS codegen failed" }
}

function Build-ElectronMain {
    Write-Step "Compiling Electron main + preload to dist-electron/"
    Push-Location $FrontendDir
    try {
        & npx tsc -p tsconfig.electron.json
        if ($LASTEXITCODE -ne 0) { throw "tsc -p tsconfig.electron.json failed" }
    } finally {
        Pop-Location
    }
}

function Wait-For-ViteReady([int]$Port = 5173, [int]$TimeoutSeconds = 60) {
    Write-Note "waiting for Vite at http://localhost:$Port ..."
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        foreach ($candidate in @("localhost", "127.0.0.1", "[::1]")) {
            try {
                $reply = Invoke-WebRequest -Uri "http://${candidate}:${Port}" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
                if ($reply.StatusCode -ge 200 -and $reply.StatusCode -lt 500) { return }
            } catch {
                # try the next host candidate
            }
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Vite did not become ready within $TimeoutSeconds seconds"
}

function Start-BackendOnly {
    Write-Step "Starting Python backend (lucidium.app)"
    Write-Note "The backend prints LUCIDIUM_WS_PORT=<port> on stdout when ready."
    & $PythonExe -m lucidium.app
}

function Start-RendererOnly {
    Stop-StaleViteOnPort
    Write-Step "Starting Vite dev server"
    & npm --prefix $FrontendDir run dev
}

function Stop-ProcessTree([int]$ProcessId) {
    try {
        & taskkill /PID $ProcessId /T /F 2>$null | Out-Null
    } catch {
        # the process may already be gone; ignore
    }
}

function Get-ProcessCommandLine([int]$ProcessId) {
    try {
        return (Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop).CommandLine
    } catch {
        return $null
    }
}

# True when the process (or a near ancestor -- npm/cmd wrappers sit
# between the shell and the node process) names this repository on its
# command line.
function Test-OurVite([int]$ProcessId) {
    $current = $ProcessId
    for ($depth = 0; $depth -lt 3 -and $current -gt 0; $depth++) {
        $proc = $null
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $current" -ErrorAction Stop
        } catch {
            return $false
        }
        if (-not $proc) { return $false }
        if ($proc.CommandLine -and $proc.CommandLine -like "*$RepoRoot*") { return $true }
        $current = [int]$proc.ParentProcessId
    }
    return $false
}

function Stop-StaleViteOnPort([int]$Port = 5173) {
    $previousEAP = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        # Get-NetTCPConnection raises a CIM "no instances found" error
        # when the port is free; we wrap it so the absence of listeners
        # is silent rather than spamming stderr.
        $listeners = @(
            Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue 2>$null |
                Where-Object { $_.State -eq 'Listen' }
        )
        if ($listeners.Count -eq 0) { return }
        $killedAny = $false
        foreach ($listener in $listeners) {
            $pidToKill = $listener.OwningProcess
            if (-not $pidToKill -or ($pidToKill -eq $PID)) { continue }
            if (Test-OurVite -ProcessId $pidToKill) {
                Write-Note "killing stale process $pidToKill listening on $Port"
                Stop-ProcessTree -ProcessId $pidToKill
                $killedAny = $true
            } else {
                $cmdLine = Get-ProcessCommandLine -ProcessId $pidToKill
                if (-not $cmdLine) { $cmdLine = "<unavailable>" }
                Write-Host ""
                Write-Host "Port $Port is in use by a process that does not belong to this repository:" -ForegroundColor Yellow
                Write-Host "    PID     : $pidToKill"
                Write-Host "    Command : $cmdLine"
                Write-Host ""
                Write-Host "Refusing to kill it. Either stop that process yourself"
                Write-Host "(Stop-Process -Id $pidToKill), or free port $Port another way,"
                Write-Host "then re-run start.ps1."
                exit 1
            }
        }
        # Give Windows a moment to release the socket before we re-bind.
        if ($killedAny) { Start-Sleep -Milliseconds 500 }
    } finally {
        $ErrorActionPreference = $previousEAP
    }
}

function Start-App {
    Build-ElectronMain
    Stop-StaleViteOnPort
    Write-Step "Starting Vite dev server in the background"
    $viteLog = Join-Path $env:TEMP "lucidium-vite.log"
    # The cmd redirect below uses "> path" which truncates, so we don't
    # need to pre-delete (and pre-deleting fights a file lock when a
    # previous run's Vite is still alive).
    $vite = Start-Process `
        -FilePath "cmd.exe" `
        -ArgumentList "/c", "npm --prefix `"$FrontendDir`" run dev > `"$viteLog`" 2>&1" `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -PassThru
    try {
        Wait-For-ViteReady
        $env:VITE_DEV_SERVER_URL = "http://localhost:5173"
        $env:LUCIDIUM_PYTHON = $PythonExe
        Write-Step "Launching Electron"
        Write-Note "Close the window or press Ctrl+C to stop."
        & $ElectronBin (Join-Path $DistElectron "main.js")
    } catch {
        if (Test-Path $viteLog) {
            Write-Host ""
            Write-Host "--- Vite output (last 40 lines of $viteLog) ---" -ForegroundColor Yellow
            Get-Content $viteLog -Tail 40 | ForEach-Object { Write-Host $_ }
            Write-Host "----------------------------------------------" -ForegroundColor Yellow
        }
        throw
    } finally {
        Write-Step "Stopping Vite"
        if ($vite -and -not $vite.HasExited) {
            Stop-ProcessTree -ProcessId $vite.Id
        }
    }
}

# --- Main ---------------------------------------------------------------

Test-Required

if (-not $NoSetup) {
    Initialize-Backend
    Initialize-Frontend
    Invoke-Codegen
}

if ($Setup) {
    Write-Step "Setup complete."
    return
}

if ($Backend) {
    Start-BackendOnly
    return
}

if ($Renderer) {
    Start-RendererOnly
    return
}

Start-App

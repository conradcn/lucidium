#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run the Lucidium GPU smoke test on Windows.

.DESCRIPTION
    Finds the project's backend venv Python and runs amd_smoke_test.py.
    Downloads the GPU torch build your machine needs, loads it, and
    proves the GPU computes. Writes a report file under scripts\ that
    you send back to the developer.

.PARAMETER Checkpoint
    Optional path to an SDXL .safetensors to also do a real 2-step render.

.PARAMETER Flavor
    Override the auto-detected flavor (cuda / rocm / directml / xpu / cpu).

.EXAMPLE
    scripts\amd-smoke.ps1
.EXAMPLE
    scripts\amd-smoke.ps1 -Checkpoint C:\models\sdxl.safetensors
#>
[CmdletBinding()]
param(
    [string]$Checkpoint,
    [string]$Flavor,
    [switch]$UseAppRuntime
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $RepoRoot "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    # Fall back to any python on PATH (the smoke test only needs the
    # project's pure-python deps, which a `start.ps1 -Setup` venv has).
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $Py = $cmd.Source }
}
if (-not (Test-Path $Py)) {
    Write-Error @"
Could not find the project's Python. Set the backend up first:
    .\start.ps1 -Setup
then re-run this script.
"@
}

$script = Join-Path $PSScriptRoot "amd_smoke_test.py"
$pyArgs = @($script)
if ($Flavor)        { $pyArgs += @("--flavor", $Flavor) }
if ($Checkpoint)    { $pyArgs += @("--checkpoint", $Checkpoint) }
if ($UseAppRuntime) { $pyArgs += @("--use-app-runtime") }

Write-Host "Running: $Py $($pyArgs -join ' ')" -ForegroundColor Cyan
& $Py @pyArgs
exit $LASTEXITCODE

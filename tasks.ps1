#!/usr/bin/env pwsh
# Lucidium dev tasks. Run a target with `pwsh tasks.ps1 <target>`.
param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $RepoRoot "backend\.venv\Scripts\python.exe"

function Invoke-Codegen {
    Write-Host "==> Exporting JSON Schemas from Pydantic..."
    & $VenvPython (Join-Path $RepoRoot "scripts/codegen/export-schemas.py")
    if ($LASTEXITCODE -ne 0) { throw "Schema export failed." }

    Write-Host "==> Generating TypeScript types..."
    & npm --prefix (Join-Path $RepoRoot "frontend") run codegen
    if ($LASTEXITCODE -ne 0) { throw "TS codegen failed." }
}

function Invoke-TestBackend {
    Write-Host "==> Running backend offline tests..."
    Push-Location (Join-Path $RepoRoot "backend")
    try {
        & $VenvPython -m pytest
        if ($LASTEXITCODE -ne 0) { throw "Backend tests failed." }
    } finally {
        Pop-Location
    }
}

function Invoke-TestFrontend {
    Write-Host "==> Running frontend offline tests..."
    & npm --prefix (Join-Path $RepoRoot "frontend") test
    if ($LASTEXITCODE -ne 0) { throw "Frontend tests failed." }
}

function Invoke-TestE2E {
    Write-Host "==> Running Playwright e2e tests (mocked WS)..."
    & npm --prefix (Join-Path $RepoRoot "frontend") run "test:e2e"
    if ($LASTEXITCODE -ne 0) { throw "E2E tests failed." }
}

function Invoke-Smoke {
    Write-Host "==> Running full-pipeline smoke tests (start.ps1 -Setup, -Backend, -Renderer)..."
    if (-not (Test-Path $VenvPython)) {
        throw "backend/.venv missing. Run `.\start.ps1 -Setup` first."
    }
    Push-Location $RepoRoot
    try {
        & $VenvPython -m pytest tests\smoke -v
        if ($LASTEXITCODE -ne 0) { throw "Smoke tests failed." }
    } finally {
        Pop-Location
    }
}

switch ($Target) {
    "codegen"        { Invoke-Codegen }
    "test-backend"   { Invoke-TestBackend }
    "test-frontend"  { Invoke-TestFrontend }
    "test-e2e"       { Invoke-TestE2E }
    "test"           { Invoke-TestBackend; Invoke-TestFrontend }
    "smoke"          { Invoke-Smoke }
    "all"            { Invoke-TestBackend; Invoke-TestFrontend; Invoke-TestE2E; Invoke-Smoke }
    default {
        Write-Host @"
Lucidium dev tasks
  codegen         Pydantic -> JSON Schema -> TypeScript
  test-backend    pytest with offline gate
  test-frontend   vitest with offline gate
  test-e2e        Playwright e2e against mocked WebSocket
  test            Run both unit suites (backend + frontend)
  smoke           Full-pipeline smoke tests via start.ps1
  all             test + test-e2e + smoke
"@
    }
}

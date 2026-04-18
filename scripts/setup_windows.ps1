param(
    [switch]$SkipDocs
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
}

if (-not (Test-Path "venv")) {
    python -m venv venv
}

$PythonExe = Join-Path $RepoRoot "venv\Scripts\python.exe"

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r requirements.txt
& $PythonExe -m pip install -e ".[dev]"

if (-not $SkipDocs) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Set-Location (Join-Path $RepoRoot "docs-site")
        cmd /c npm install
        Set-Location $RepoRoot
    } else {
        Write-Warning "npm is not available. Install Node.js >= 20, then run npm install in docs-site."
    }
}

Write-Host "Setup complete."

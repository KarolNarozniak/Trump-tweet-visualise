param(
    [ValidateSet("all", "app", "docs")]
    [string]$Mode = "all"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = Join-Path $RepoRoot "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Missing venv Python at $PythonExe. Run .\scripts\setup_windows.ps1 first."
}

& $PythonExe "scripts/run_services.py" --mode $Mode

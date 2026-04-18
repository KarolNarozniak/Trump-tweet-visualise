param(
    [switch]$SkipTests,
    [switch]$SkipArtifactBuild,
    [switch]$SkipDocsBuild
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = Join-Path $RepoRoot "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Missing venv Python at $PythonExe. Run .\scripts\setup_windows.ps1 first."
}

$Args = @("scripts/deploy.py")
if ($SkipTests) { $Args += "--skip-tests" }
if ($SkipArtifactBuild) { $Args += "--skip-artifact-build" }
if ($SkipDocsBuild) { $Args += "--skip-docs-build" }

& $PythonExe @Args

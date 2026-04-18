param(
    [switch]$SkipDocs,
    [switch]$InstallEditable,
    [switch]$UpgradePip
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-Host "==> $Description"
    cmd /c $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed ($LASTEXITCODE): $Description"
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
}

if (-not (Test-Path "venv")) {
    Invoke-Step -Command "python -m venv venv" -Description "Creating Python virtual environment"
}

$PythonExe = Join-Path $RepoRoot "venv\Scripts\python.exe"
$SitePackages = Join-Path $RepoRoot "venv\Lib\site-packages"

# Use local temp directory to avoid locked system temp locations.
$LocalTemp = Join-Path $RepoRoot ".tmp"
New-Item -Path $LocalTemp -ItemType Directory -Force | Out-Null
$env:TEMP = $LocalTemp
$env:TMP = $LocalTemp
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

# Clean up known broken pip artifacts (e.g., "~ip") that cause repeated warnings.
if (Test-Path $SitePackages) {
    Get-ChildItem -Path $SitePackages -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "~ip*" -or $_.Name -like "*~ip*" } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
}

if ($UpgradePip) {
    Invoke-Step -Command "`"$PythonExe`" -m pip --disable-pip-version-check install --upgrade pip" -Description "Upgrading pip"
} else {
    Write-Host "==> Skipping pip upgrade (use -UpgradePip to enable)"
}
Invoke-Step -Command "`"$PythonExe`" -m pip --disable-pip-version-check install -r requirements.txt" -Description "Installing Python requirements"
if ($InstallEditable) {
    Invoke-Step -Command "`"$PythonExe`" -m pip --disable-pip-version-check install --no-build-isolation --no-deps -e ." -Description "Installing project in editable mode"
} else {
    Write-Host "==> Skipping editable install (use -InstallEditable to enable)"
}

if (-not $SkipDocs) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Set-Location (Join-Path $RepoRoot "docs-site")
        Invoke-Step -Command "npm install" -Description "Installing Docusaurus dependencies"
        Set-Location $RepoRoot
    } else {
        Write-Warning "npm is not available. Install Node.js >= 20 (LTS recommended), then run npm install in docs-site."
    }
}

Write-Host "Setup complete."

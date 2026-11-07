# Turfolio installer for Windows (PowerShell):
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/SolvAI/turfolio/main/install.ps1 | iex"
# (or .\install.ps1 inside a clone). Installs uv if needed, Python 3.12 + dependencies in .\.venv, then runs the one-time setup.
$ErrorActionPreference = "Stop"
$repo = "https://github.com/SolvAI/turfolio"
function Say($m) { Write-Host "> $m" -ForegroundColor Cyan }

# 1. the code (skip when run from inside a clone)
if (-not (Test-Path "turfolio.py")) {
    if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "turfolio.py"))) {
        Set-Location $PSScriptRoot
    } else {
        Say "Downloading Turfolio into .\turfolio"
        if (Get-Command git -ErrorAction SilentlyContinue) {
            git clone --depth 1 "$repo.git" turfolio
        } else {
            Invoke-WebRequest "$repo/archive/refs/heads/main.zip" -OutFile turfolio.zip
            Expand-Archive turfolio.zip -DestinationPath .
            Rename-Item turfolio-main turfolio
            Remove-Item turfolio.zip
        }
        Set-Location turfolio
    }
}

# 2. uv (fast Python package manager; brings its own Python)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# 3. Python 3.12 + dependencies
Say "Installing Python 3.12 and the dependencies into .\.venv"
uv venv --quiet --allow-existing --python 3.12 .venv
if ($LASTEXITCODE) { throw "uv venv failed" }
uv pip install --quiet --python .venv\Scripts\python.exe -r requirements.txt
if ($LASTEXITCODE) { throw "dependency install failed" }

# 4. one-time data setup
.\turfolio.cmd setup
Write-Host ""
Write-Host "Installed. In $(Get-Location), run:  .\turfolio bets --watch" -ForegroundColor Green

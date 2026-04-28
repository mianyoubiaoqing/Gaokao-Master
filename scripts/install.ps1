param(
    [switch]$Optional,
    [string]$IndexUrl = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = Join-Path $ProjectRoot ".venv"

function Find-Python {
    if ($env:PYTHON -and (Test-Path $env:PYTHON)) {
        return $env:PYTHON
    }

    $commands = @("py", "python", "python3")
    foreach ($command in $commands) {
        $resolved = Get-Command $command -ErrorAction SilentlyContinue
        if ($resolved) {
            return $command
        }
    }

    $codexPython = Join-Path $HOME ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path $codexPython) {
        return $codexPython
    }

    throw "Python was not found. Install Python 3.11+ or set PYTHON to python.exe."
}

$Python = Find-Python
Write-Host "Using Python: $Python"

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment: $VenvPath"
    & $Python -m venv $VenvPath
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python was not found: $VenvPython"
}

Write-Host "Upgrading pip tools..."
& $VenvPython -m pip install --upgrade pip setuptools wheel --default-timeout 180 --retries 10 -i $IndexUrl

Write-Host "Installing core dependencies..."
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt") --default-timeout 180 --retries 10 --prefer-binary -i $IndexUrl

if ($Optional) {
    Write-Host "Installing optional heavy dependencies..."
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-optional.txt") --default-timeout 180 --retries 10 --prefer-binary -i $IndexUrl
}

Write-Host ""
Write-Host "Install finished. Start with:"
Write-Host "  .\scripts\start_agent.ps1"

param(
    [int]$Port = 8501,
    [switch]$Cli
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment was not found. Run first: .\scripts\install.ps1"
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
Set-Location $ProjectRoot

if ($Cli) {
    & $VenvPython ".\agent_start.py" cli
} else {
    & $VenvPython ".\agent_start.py" web --port $Port
}

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectDir

Write-Host 'Starting Frame Mirror Studio on http://127.0.0.1:8000' -ForegroundColor Cyan
python -m uvicorn app:app --host 127.0.0.1 --port 8000


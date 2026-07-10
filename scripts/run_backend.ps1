param(
  [int]$Port = 8011
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:ARTIFACTS_DIR = Join-Path $Repo "artifacts"
Set-Location $Repo
python -m uvicorn app.main:app --host 127.0.0.1 --port $Port

param(
  [int]$Port = 5174
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location (Join-Path $Repo "local_demo_frontend")
python -m http.server $Port --bind 127.0.0.1

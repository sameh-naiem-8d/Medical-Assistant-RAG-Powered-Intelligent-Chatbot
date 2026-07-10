param(
  [string]$Archive = "D:\Project Graduation\MEDBRIDGE_FINAL_AI_DELIVERY"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
python "$Repo\scripts\setup_from_local_archive.py" --archive "$Archive" --repo "$Repo"

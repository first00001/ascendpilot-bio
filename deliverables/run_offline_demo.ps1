$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "offline_demo.py"
python $script

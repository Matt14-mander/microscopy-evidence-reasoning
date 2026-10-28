# Project-local Python, model/data caches, and writable temporary directories.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:HF_HOME = Join-Path $projectRoot '.cache/huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'
$env:HF_HUB_ETAG_TIMEOUT = '30'
$env:PYTHONUTF8 = '1'
# Do not inherit machine-specific TLS key logging into project subprocesses.
$env:SSLKEYLOGFILE = ''
$env:TEMP = Join-Path $projectRoot 'outputs/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP, $env:HF_HOME | Out-Null
& (Join-Path $projectRoot '.venv/Scripts/python.exe') @args
exit $LASTEXITCODE

param([string]$PythonExe = 'python', [switch]$AllowSourceCandidate, [string]$ReleaseDir = '')
$ErrorActionPreference = 'Stop'
# Legacy parameters remain; verify current and protected historical manifests.
& $PythonExe (Join-Path $PSScriptRoot 'check_governance.py')
if ($LASTEXITCODE -ne 0) { throw 'Registered delivery verification failed' }

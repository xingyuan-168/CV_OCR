param([Parameter(Mandatory=$true)][string]$PackageDir,
      [Parameter(Mandatory=$true)][string]$PythonWheel,
      [string]$PythonExe = 'python', [switch]$SelectCurrent)
$ErrorActionPreference = 'Stop'
$arguments = @((Join-Path $PSScriptRoot 'register_release.py'), '--package-dir', $PackageDir, '--wheel', $PythonWheel)
if ($SelectCurrent) { $arguments += '--select-current' }
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) { throw 'Immutable delivery registration failed' }

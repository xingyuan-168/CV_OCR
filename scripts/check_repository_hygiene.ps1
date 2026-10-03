param([switch]$CheckWorkingTree, [switch]$RequireClean, [string]$PythonExe = 'python')
$ErrorActionPreference = 'Stop'
$arguments = @((Join-Path $PSScriptRoot 'check_governance.py'))
if ($CheckWorkingTree) { $arguments += '--working-tree' }
if ($RequireClean) { $arguments += '--require-clean' }
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) { throw 'Repository / manifest / document checks failed' }

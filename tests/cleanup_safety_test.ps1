param([string]$Root)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot '../scripts/CleanupSafety.psm1') -Force
$allowed = @('outpush/scratch', 'output', '../outside', 'outpush/junction')
New-Item -ItemType Directory -Path (Join-Path $Root 'outpush/scratch') | Out-Null
$file = Join-Path $Root 'outpush/scratch/data.txt'
Set-Content -LiteralPath $file -Value 'scratch'
$expectedBytes = (Get-Item -LiteralPath $file).Length
$preview = Remove-RegisteredTarget -Root $Root -Relative 'outpush/scratch' -Allowed $allowed
if ($preview.state -ne 'preview' -or !(Test-Path -LiteralPath $file)) { throw 'Preview mutated the target' }
foreach ($bad in @('output','../outside','outpush/unregistered')) {
    $rejected = $false
    try { $null = Resolve-CleanupTarget -Root $Root -Relative $bad -Allowed $allowed } catch { $rejected = $true }
    if (!$rejected) { throw "Unsafe target accepted: $bad" }
}
$lock = [IO.File]::Open($file, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
try {
    $result = Remove-RegisteredTarget -Root $Root -Relative 'outpush/scratch' -Allowed $allowed -Apply
    if ($result.state -ne 'occupied' -or !(Test-Path -LiteralPath $file)) { throw 'Occupied file was removed' }
} finally { $lock.Dispose() }
$link = Join-Path $Root 'outpush/junction'
New-Item -ItemType Junction -Path $link -Target (Join-Path $Root 'outpush/scratch') | Out-Null
$rejected = $false
try { $null = Resolve-CleanupTarget -Root $Root -Relative 'outpush/junction' -Allowed $allowed } catch { $rejected = $true }
[IO.Directory]::Delete($link)
if (!$rejected -or !(Test-Path -LiteralPath $file)) { throw 'Junction was followed' }
Set-ItemProperty -LiteralPath $file -Name IsReadOnly -Value $true
$result = Remove-RegisteredTarget -Root $Root -Relative 'outpush/scratch' -Allowed $allowed -Apply
if ($result.bytes -ne $expectedBytes) { throw 'Deleted byte count differs' }
if ($result.state -ne 'deleted' -or (Test-Path -LiteralPath $file)) { throw 'Registered deletion failed' }
$again = Remove-RegisteredTarget -Root $Root -Relative 'outpush/scratch' -Allowed $allowed -Apply
if ($again.state -ne 'already_absent' -or $again.bytes -ne 0) { throw 'Repeated cleanup is not idempotent' }
@{passed=$true; cases=7} | ConvertTo-Json -Compress

param(
    [string]$Plan = 'outpush/governance-v23.6/cleanup-plan.json',
    [string]$PythonExe = 'python',
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Import-Module (Join-Path $PSScriptRoot 'CleanupSafety.psm1') -Force
$planPath = [IO.Path]::GetFullPath((Join-Path $project $Plan))
if (!$planPath.StartsWith((Join-Path $project 'outpush') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Cleanup plans must be below this workspace outpush directory'
}
$registry = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($registry.schema -ne 1 -or [IO.Path]::GetFullPath($registry.root) -ne $project) { throw 'Wrong cleanup workspace' }
& $PythonExe (Join-Path $PSScriptRoot 'repository_storage.py') verify-plan --plan $planPath
if ($LASTEXITCODE -ne 0) { throw 'Protected files / retention evidence changed; cleanup refused' }
$allowed = @($registry.targets | ForEach-Object { $_.path })
foreach ($target in $allowed) { $null = Resolve-CleanupTarget -Root $project -Relative $target -Allowed $allowed }
$results = @()
foreach ($target in $registry.targets) {
    $result = Remove-RegisteredTarget -Root $project -Relative $target.path -Allowed $allowed -Apply:$Apply
    $results += $result
    Write-Host "$($result.state): $($target.path)"
}
$report = @{schema=1; applied=[bool]$Apply; root=$project; results=$results}
$stem = [IO.Path]::GetFileNameWithoutExtension($planPath)
$name = if ($Apply) { "$stem-applied.json" } else { "$stem-preview.json" }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path (Split-Path -Parent $planPath) $name) -Encoding UTF8
if (@($results | Where-Object { $_.state -in @('failed','occupied') }).Count) { throw 'Some targets could not be removed; see cleanup report' }

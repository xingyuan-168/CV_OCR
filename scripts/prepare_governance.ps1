param([string]$UvExe = 'uv')
$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$lock = Get-Content -LiteralPath (Join-Path $project 'configs/governance-toolchain.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$source = Join-Path $project '.cache/aios-source'
function Invoke-Checked([string]$Command, [string[]]$Arguments) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Toolchain command failed: $Command" }
}
function Get-TextDigest([string]$Path) {
    $bytes = [Text.Encoding]::UTF8.GetBytes([IO.File]::ReadAllText($Path).Replace("`r`n", "`n"))
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hasher.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $hasher.Dispose() }
}
$patch = Join-Path $project $lock.patch
if ((Get-TextDigest $patch) -ne $lock.patch_sha256) { throw 'AIOS patch hash differs' }
if (!(Test-Path -LiteralPath $source)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $source) -Force | Out-Null
    Invoke-Checked git @('clone', '--filter=blob:none', '--no-checkout', $lock.repository, $source)
    Invoke-Checked git @('-C', $source, 'checkout', '--detach', $lock.revision)
    # Normalize the registered patch to Git text bytes before applying it.
    $normalizedPatch = Join-Path (Split-Path -Parent $source) 'aios-compatibility.patch'
    [IO.File]::WriteAllText($normalizedPatch, [IO.File]::ReadAllText($patch).Replace("`r`n", "`n"), (New-Object Text.UTF8Encoding($false)))
    try { Invoke-Checked git @('-C', $source, 'apply', '--whitespace=nowarn', $normalizedPatch) }
    finally { Remove-Item -LiteralPath $normalizedPatch -Force }
}
if ((Get-TextDigest (Join-Path $source 'uv.lock')) -ne $lock.uv_lock_sha256) { throw 'AIOS dependency lock differs' }
foreach ($item in $lock.source_files.PSObject.Properties) {
    if ((Get-TextDigest (Join-Path $source $item.Name)) -ne $item.Value) { throw "AIOS runtime source differs: $($item.Name)" }
}
$uvVersion = & $UvExe --version
if ($LASTEXITCODE -ne 0 -or $uvVersion -notlike "uv $($lock.uv_version) *") { throw "Use locked uv $($lock.uv_version)" }
Push-Location $source
try { Invoke-Checked $UvExe @('sync', '--frozen', '--no-dev', '--python', $lock.python) }
finally { Pop-Location }
$entry = Join-Path $source '.venv/Scripts/codex-os.exe'
Write-Output $entry

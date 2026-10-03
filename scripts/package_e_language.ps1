param(
    [string]$PythonExe = 'python',
    [string]$X86Dll = 'build-cv-candidate-x86/Release/CQ_X86.dll',
    [string]$WorkerExe = 'build-cv-candidate-worker/Release/CQ_AI_worker.exe',
    [string]$X64Dir = 'build-cv-candidate-x64/Release',
    [string]$ApiHtml = 'docs/易语言_DLL_API_说明.html',
    [string]$OutputDir = 'output',
    [string]$StageDir = 'outpush/v23.6-delivery',
    [string]$ValidationDir = 'outpush/v23.6-validation',
    [switch]$SkipNvidiaPacking,
    [switch]$NoZip
)

$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outpushRoot = [IO.Path]::GetFullPath((Join-Path $project 'outpush'))
function Resolve-ProjectPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $project $Path))
}
function Assert-OutpushPath([string]$Path) {
    if (!$Path.StartsWith($outpushRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Path must be below outpush: $Path" }
}
$finalOutput = Resolve-ProjectPath $OutputDir
$officialOutput = Join-Path $project 'output'
if (!$finalOutput.Equals($officialOutput, [StringComparison]::OrdinalIgnoreCase)) { Assert-OutpushPath $finalOutput }
$stageRoot = Resolve-ProjectPath $StageDir
Assert-OutpushPath $stageRoot
$validation = Resolve-ProjectPath $ValidationDir
Assert-OutpushPath $validation
$x86Path = Resolve-ProjectPath $X86Dll
$workerPath = Resolve-ProjectPath $WorkerExe
$x64Path = Resolve-ProjectPath $X64Dir
$htmlPath = Resolve-ProjectPath $ApiHtml
if (!$htmlPath.Equals((Resolve-ProjectPath 'docs/易语言_DLL_API_说明.html'),[StringComparison]::OrdinalIgnoreCase)) { throw 'Use the generated current API HTML; custom copies are not accepted' }
$python = (Get-Command $PythonExe -ErrorAction Stop).Source
$msvcRoot = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC'
$dumpbin = Get-ChildItem -LiteralPath $msvcRoot -Directory | Sort-Object Name -Descending | ForEach-Object { Join-Path $_.FullName 'bin/Hostx64/x64/dumpbin.exe' } | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (!$dumpbin) { throw 'dumpbin.exe was not found' }
$arguments = @((Join-Path $PSScriptRoot 'package_delivery.py'), '--x86', (Split-Path -Parent $x86Path), '--worker', $workerPath, '--x64', $x64Path, '--out', $stageRoot, '--validation', $validation, '--dumpbin', $dumpbin)
if ($SkipNvidiaPacking) { $arguments += '--skip-nvidia' }
if ($NoZip) { $arguments += '--no-zip' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Delivery staging/validation failed; output was preserved' }
$manifestPath = Join-Path $stageRoot 'manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$coreStage = [IO.Path]::GetFullPath($manifest.core_directory)
Assert-OutpushPath $coreStage
$expected = @('CQ_X86.dll', 'CQ_AI_worker.exe', '易语言_DLL_API_说明.html') | Sort-Object
function Assert-Cohort([string]$Path) {
    $item = Get-Item -LiteralPath $Path
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse directories are not accepted: $Path" }
    $items = @(Get-ChildItem -LiteralPath $Path -Force)
    if ($items.Count -ne 3 -or @($items | Where-Object { $_.PSIsContainer -or ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }).Count -gt 0 -or ((@($items.Name | Sort-Object) -join '|') -cne ($expected -join '|'))) { throw "Preserve unexpected output contents: $Path" }
}
Assert-Cohort $coreStage
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$rollbackRoot = Join-Path $outpushRoot 'rollback'
$backup = Join-Path $rollbackRoot ("before-$($manifest.delivery_version)-$stamp")
Assert-OutpushPath $backup
$hadOutput = Test-Path -LiteralPath $finalOutput
if ($hadOutput) {
    Assert-Cohort $finalOutput
    # Exclusive writable access fails when a business process has mapped a DLL/EXE.
    foreach ($name in $expected) {
        $path = Join-Path $finalOutput $name
        try { $stream = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None); $stream.Dispose() }
        catch { throw "Output is occupied or not writable; original files preserved: $path. $($_.Exception.Message)" }
    }
    $oldHashes = @($expected | ForEach-Object { $path = Join-Path $finalOutput $_; [ordered]@{ name=$_; size=(Get-Item -LiteralPath $path).Length; sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash } })
    New-Item -ItemType Directory -Path $rollbackRoot -Force | Out-Null
    $oldHashes | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath ($backup + '.manifest.json') -Encoding UTF8
    Move-Item -LiteralPath $finalOutput -Destination $backup
}
try {
    Move-Item -LiteralPath $coreStage -Destination $finalOutput
    Assert-Cohort $finalOutput
    foreach ($name in $expected) {
        if ((Get-FileHash -LiteralPath (Join-Path $finalOutput $name) -Algorithm SHA256).Hash -ne $manifest.files.$name.sha256) { throw "Promoted file hash differs: $name" }
    }
    & (Join-Path (Split-Path -Parent $x86Path) 'ai_engine_delivery_verify.exe') --dll (Join-Path $finalOutput 'CQ_X86.dll') --header (Join-Path $project 'include/ai_engine.h') --model (Join-Path $validation '中文路径/best.onnx') --image (Join-Path $project 'input/yolo测试图.bmp') --expected-json (Join-Path $validation 'frozen-detections.json') --report (Join-Path $stageRoot 'delivery-output.json')
    if ($LASTEXITCODE -ne 0) { throw 'Actual output delivery loader failed' }
    $manifest.output_directory = $finalOutput
    $manifest.output_promotion = 'verified'
    $manifest | Add-Member -NotePropertyName rollback_directory -NotePropertyValue $(if ($hadOutput) { $backup } else { $null }) -Force
    $manifest | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
}
catch {
    $errorText = $_.Exception.Message
    $rejected = Join-Path $outpushRoot ("rejected-delivery-$stamp")
    Assert-OutpushPath $rejected
    if (Test-Path -LiteralPath $finalOutput) { Move-Item -LiteralPath $finalOutput -Destination $rejected }
    if ($hadOutput) { Move-Item -LiteralPath $backup -Destination $finalOutput }
    throw "Delivery failed; original cohort restored. $errorText"
}
Get-ChildItem -LiteralPath $finalOutput -File | Select-Object Name,Length,@{Name='SHA256';Expression={(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}} | Format-Table -AutoSize
Write-Host "Verified output: $finalOutput"
Write-Host "Version: $($manifest.project_version) / $($manifest.delivery_version) / protocol$($manifest.worker_protocol)"
Write-Host "Manifest and packages: $stageRoot"
Write-Host 'Functional validation passed; target E5/RTX2070 performance remains pending.'

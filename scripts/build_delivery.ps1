param([string]$PythonExe = 'python', [switch]$SkipDependencyPreparation, [switch]$Nvidia)
$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmakeCommand) { $cmake = $cmakeCommand.Source }
else { $cmake = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe' }
if (!(Test-Path -LiteralPath $cmake)) { throw 'CMake / Visual Studio 2022 Build Tools are required' }
if (!$SkipDependencyPreparation) {
    & (Join-Path $PSScriptRoot 'prepare_dependencies.ps1') -PythonExe $PythonExe
    if ($LASTEXITCODE -ne 0) { throw 'Dependency preparation failed' }
}
$profiles = @('release-x86', 'release-worker', 'release-x64')
if ($Nvidia) {
    & $PythonExe (Join-Path $PSScriptRoot 'prepare_nvidia_headers.py')
    if ($LASTEXITCODE -ne 0) { throw 'Optional NVIDIA headers failed' }
    $profiles += 'nvidia-x64'
}
Push-Location $project
try {
    foreach ($profile in $profiles) {
        & $cmake --preset $profile
        if ($LASTEXITCODE -ne 0) { throw "Configure failed: $profile" }
        if ($profile -eq 'nvidia-x64') { & $cmake --build --preset $profile --target cq_yolo_tensorrt }
        else { & $cmake --build --preset $profile }
        if ($LASTEXITCODE -ne 0) { throw "Build failed: $profile" }
    }
    Copy-Item -LiteralPath 'build/release-worker/Release/CQ_AI_worker.exe' -Destination 'build/release-x86/Release/CQ_AI_worker.exe'
} finally { Pop-Location }
Write-Host 'Built candidates under build/. Complete binary-bound validation before packaging.'

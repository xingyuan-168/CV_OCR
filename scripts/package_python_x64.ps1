param(
    [string]$PythonExe = 'python',
    [string]$BuildDir = 'build-release-python-x64',
    [string]$OpenCVDir = 'third_party/opencv-5.0.0-static-mt/x64',
    [string]$RuntimeDir = 'third_party/runtime/ort-directml-1.24.4',
    [string]$OutputDir = '',
    [switch]$SkipDependencyPreparation,
    [switch]$SkipNativeTests,
    [switch]$SkipNativeBuild,
    [switch]$BuildTensorRTModule
)
$ErrorActionPreference='Stop'
$project=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
function Resolve-ProjectPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $project $Path))
}
$buildRoot=Resolve-ProjectPath $BuildDir
if (!$buildRoot.StartsWith($project + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'BuildDir must be a workspace child directory' }
$opencvPath=Resolve-ProjectPath $OpenCVDir
$runtimePath=Resolve-ProjectPath $RuntimeDir
$python=(Get-Command $PythonExe -ErrorAction Stop).Source
$cmake=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$ctest=Join-Path (Split-Path -Parent $cmake) 'ctest.exe'
$meta=Get-Content -LiteralPath (Join-Path $project 'docs/YOLO_CANDIDATE_METADATA.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
if (!$OutputDir) { $OutputDir="outpush/python-$($meta.delivery_version)-$stamp" }
$output=Resolve-ProjectPath $OutputDir
$outpush=Join-Path $project 'outpush'
if (!$output.StartsWith($outpush + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'OutputDir must be a child of outpush' }
if (Test-Path -LiteralPath $output) { throw 'Choose a fresh OutputDir; previous packages are preserved' }
if (!$SkipNativeBuild) {
    if (!$SkipDependencyPreparation) { & (Join-Path $PSScriptRoot 'prepare_dependencies.ps1') -OpenCVArchitecture x64 }
    $arguments=@('-S',$project,'-B',$buildRoot,'-G','Visual Studio 17 2022','-A','x64','-DAIENGINE_BUILD_X64_DLL=ON','-DAIENGINE_BUILD_WORKER=OFF','-DAIENGINE_WITH_OPENCV=ON',"-DAIENGINE_OPENCV_DIR=$opencvPath",'-DAIENGINE_WITH_ONNXRUNTIME=ON',"-DAIENGINE_ONNXRUNTIME_DIR=$runtimePath",'-DAIENGINE_EMBED_OCR_ASSETS=ON','-DAIENGINE_BUILD_TESTS=ON')
    if ($BuildTensorRTModule) { $arguments += '-DAIENGINE_BUILD_TENSORRT_MODULE=ON' }
    & $cmake @arguments
    if ($LASTEXITCODE -ne 0) { throw 'x64 configure failed' }
    & $cmake --build $buildRoot --config Release --parallel 4
    if ($LASTEXITCODE -ne 0) { throw 'x64 build failed' }
}
if (!$SkipNativeTests) {
    & $ctest --test-dir $buildRoot -C Release --output-on-failure -R 'ai_engine_(smoke|e_api_smoke|yolo_.*|ocr_business_test|ocr_reading_order_business_test|compact_result_test|origin_zip_test|acp_compat_test|runtime_probe)$'
    if ($LASTEXITCODE -ne 0) { throw 'x64 native regression failed' }
}
$msvcRoot=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC'
$dumpbin=Get-ChildItem -LiteralPath $msvcRoot -Directory | Sort-Object Name -Descending | ForEach-Object { Join-Path $_.FullName 'bin/Hostx64/x64/dumpbin.exe' } | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (!$dumpbin) { throw 'dumpbin.exe was not found' }
& $python (Join-Path $PSScriptRoot 'package_delivery.py') --python-only --x64 (Join-Path $buildRoot 'Release') --out $output --dumpbin $dumpbin
if ($LASTEXITCODE -ne 0) { throw 'Wheel staging or content verification failed' }
$wheel=Join-Path $output ("cq_ai_engine-$($meta.project_version)-py3-none-win_amd64.whl")
$venv=Join-Path $output 'installed-venv'
& $python -m venv $venv
if ($LASTEXITCODE -ne 0) { throw 'Fresh virtual environment failed' }
$venvPython=Join-Path $venv 'Scripts/python.exe'
& $venvPython -m pip install --no-index --no-deps $wheel
if ($LASTEXITCODE -ne 0) { throw 'Offline Wheel installation failed' }
& $venvPython -I (Join-Path $project 'tests/python_x64_wheel_acceptance.py') --json-output (Join-Path $output 'installed-acceptance.json')
if ($LASTEXITCODE -ne 0) { throw 'Installed Wheel CV/OCR/YOLO acceptance failed' }
Get-FileHash -LiteralPath $wheel -Algorithm SHA256 | Format-List
Write-Host "Verified Wheel: $wheel"
Write-Host 'Target NVIDIA performance is pending; installing this Wheel does not constitute GPU acceptance.'

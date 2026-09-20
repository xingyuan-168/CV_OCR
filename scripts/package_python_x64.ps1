param(
    [string]$PythonExe = "python",
    [string]$BuildDir = "build-release-python-x64",
    [string]$OpenCVDir = "third_party\opencv-5.0.0-static-mt\x64",
    [string]$RuntimeDir = "third_party\runtime\ort-directml-1.24.4",
    [string]$OutputDir = "outpush\CQ_AI_python_x64_v23.5",
    [switch]$SkipDependencyPreparation,
    [switch]$SkipNativeTests
)

$ErrorActionPreference = "Stop"
$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$outpushRoot = [System.IO.Path]::GetFullPath((Join-Path $project "outpush"))
$buildRoot = [System.IO.Path]::GetFullPath((Join-Path $project $BuildDir))
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $project $OutputDir))
$opencvPath = [System.IO.Path]::GetFullPath((Join-Path $project $OpenCVDir))
$runtimePath = [System.IO.Path]::GetFullPath((Join-Path $project $RuntimeDir))
if (!$buildRoot.StartsWith($project, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "BuildDir must remain under $project"
}
if (!$outputPath.StartsWith($outpushRoot, [System.StringComparison]::OrdinalIgnoreCase) -or $outputPath -eq $outpushRoot) {
    throw "OutputDir must be a child of $outpushRoot"
}

$python = (Get-Command $PythonExe -ErrorAction Stop).Source
$pythonBits = (& $python -c "import struct; print(struct.calcsize('P') * 8)").Trim()
if ($LASTEXITCODE -ne 0 -or $pythonBits -ne "64") {
    throw "Python Wheel must be built with 64-bit Python; detected $pythonBits-bit: $python"
}

$cmake = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = Join-Path (Split-Path -Parent $cmake) "ctest.exe"
foreach ($tool in @($cmake, $ctest)) {
    if (!(Test-Path -LiteralPath $tool -PathType Leaf)) { throw "required build tool is missing: $tool" }
}
$dumpbin = Get-ChildItem -LiteralPath (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC") -Filter dumpbin.exe -Recurse |
    Where-Object { $_.FullName -match 'Hostx64\\x64\\dumpbin\.exe$' } |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
if (!$dumpbin) { throw "dumpbin.exe was not found" }

if (!$SkipDependencyPreparation) {
    & (Join-Path $PSScriptRoot "prepare_dependencies.ps1") -OpenCVArchitecture x64
}
$opencvConfig = Join-Path $opencvPath "OpenCVConfig.cmake"
if (!(Test-Path -LiteralPath $opencvConfig -PathType Leaf)) {
    throw "x64 OpenCV package configuration is missing: $opencvConfig"
}

$runtimeNames = @(
    "onnxruntime.dll",
    "DirectML.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll"
)
$licenseNames = @(
    "onnxruntime-LICENSE.txt",
    "onnxruntime-ThirdPartyNotices.txt",
    "directml-LICENSE.txt",
    "directml-LICENSE-CODE.txt",
    "directml-ThirdPartyNotices.txt",
    "visual-cpp-runtime-license.txt"
)
foreach ($name in $runtimeNames) {
    $path = Join-Path $runtimePath "bin\$name"
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "runtime DLL is missing: $path" }
}
foreach ($name in $licenseNames) {
    $path = Join-Path $runtimePath "licenses\$name"
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "runtime license is missing: $path" }
}

$configure = @(
    "-S", $project,
    "-B", $buildRoot,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DAIENGINE_BUILD_TESTS=ON",
    "-DAIENGINE_BUILD_X64_DLL=ON",
    "-DAIENGINE_BUILD_WORKER=OFF",
    "-DAIENGINE_WITH_OPENCV=ON",
    "-DAIENGINE_OPENCV_DIR=$opencvPath",
    "-DAIENGINE_WITH_ONNXRUNTIME=ON",
    "-DAIENGINE_ONNXRUNTIME_DIR=$runtimePath",
    "-DAIENGINE_EMBED_ASSETS=OFF",
    "-DAIENGINE_EMBED_OCR_ASSETS=ON"
)
& $cmake @configure
if ($LASTEXITCODE -ne 0) { throw "CQ_AI Python x64 configure failed" }
& $cmake --build $buildRoot --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "CQ_AI Python x64 build failed" }

$releaseDir = Join-Path $buildRoot "Release"
$coreDll = Join-Path $releaseDir "CQ_AI_x64.dll"
if (!(Test-Path -LiteralPath $coreDll -PathType Leaf)) { throw "CQ_AI_x64.dll was not built" }
if (Test-Path -LiteralPath (Join-Path $releaseDir "CQ_AI_worker.exe") -PathType Leaf) {
    throw "worker was unexpectedly built for the Python x64 target"
}

function Get-Dependencies([string]$Path) {
    $output = & $dumpbin /DEPENDENTS $Path
    if ($LASTEXITCODE -ne 0) { throw "dumpbin /DEPENDENTS failed for $Path" }
    return @($output | ForEach-Object {
        if ($_ -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') { $Matches[1] }
    })
}

$headers = (& $dumpbin /HEADERS $coreDll) -join "`n"
if ($LASTEXITCODE -ne 0 -or $headers -notmatch '(?i)machine \(x64\)') {
    throw "CQ_AI_x64.dll is not an x64 PE image"
}
$dependencies = @(Get-Dependencies -Path $coreDll)
$allowedDependencies = @("onnxruntime.dll", "d3d12.dll", "dxgi.dll", "gdi32.dll", "KERNEL32.dll")
$unexpectedDependencies = @($dependencies | Where-Object { $_ -notin $allowedDependencies })
if ($unexpectedDependencies.Count -gt 0 -or "onnxruntime.dll" -notin $dependencies) {
    throw "unexpected CQ_AI_x64.dll dependencies: $($dependencies -join ', ')"
}

$header = Get-Content -LiteralPath (Join-Path $project "include\ai_engine.h") -Raw -Encoding UTF8
$headerExports = @([regex]::Matches($header, 'AIENGINE_EXPORT[\s\S]*?AIENGINE_CALL\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
if ($headerExports.Count -ne 60) { throw "public header must contain exactly 60 exports; found $($headerExports.Count)" }
$binaryExports = @((& $dumpbin /EXPORTS $coreDll) | ForEach-Object {
    if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+([A-Za-z][A-Za-z0-9_]*)\s*$') { $Matches[1] }
} | Sort-Object -Unique)
if ($LASTEXITCODE -ne 0) { throw "dumpbin /EXPORTS failed for CQ_AI_x64.dll" }
$missingExports = @($headerExports | Where-Object { $_ -notin $binaryExports })
$extraExports = @($binaryExports | Where-Object { $_ -notin $headerExports })
if ($binaryExports.Count -ne 60 -or $missingExports.Count -gt 0 -or $extraExports.Count -gt 0) {
    throw "x64 ABI export mismatch. Missing: $($missingExports -join ', '); extra: $($extraExports -join ', ')"
}

if (!$SkipNativeTests) {
    $testPattern = '^ai_engine_(smoke|e_api_smoke|ocr_color_filter_test|ocr_reading_order_test|compact_result_test|origin_zip_test|acp_compat_test|ocr_business_test|ocr_reading_order_business_test|runtime_probe)$'
    & $ctest --test-dir $buildRoot -C Release --output-on-failure -R $testPattern
    if ($LASTEXITCODE -ne 0) { throw "CQ_AI Python x64 native regression tests failed" }
}

$stageRoot = Join-Path $buildRoot "wheel-staging"
$stagePackage = Join-Path $stageRoot "cq_ai_engine"
$stageNative = Join-Path $stagePackage "_native"
$stageLicenses = Join-Path $stagePackage "_licenses"
if (Test-Path -LiteralPath $stageRoot) {
    $resolvedStage = [System.IO.Path]::GetFullPath($stageRoot)
    if (!$resolvedStage.StartsWith($buildRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw "invalid staging path" }
    Remove-Item -LiteralPath $resolvedStage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageNative, $stageLicenses | Out-Null
foreach ($name in @("pyproject.toml", "setup.py", "README.md", "ai_engine.py")) {
    Copy-Item -LiteralPath (Join-Path $project "python\$name") -Destination (Join-Path $stageRoot $name)
}
Copy-Item -LiteralPath (Join-Path $project "python\cq_ai_engine\__init__.py") -Destination (Join-Path $stagePackage "__init__.py")
Copy-Item -LiteralPath (Join-Path $project "python\cq_ai_engine\_native\__init__.py") -Destination (Join-Path $stageNative "__init__.py")
Copy-Item -LiteralPath (Join-Path $project "python\cq_ai_engine\_licenses\__init__.py") -Destination (Join-Path $stageLicenses "__init__.py")
Copy-Item -LiteralPath $coreDll -Destination (Join-Path $stageNative "CQ_AI_x64.dll")
foreach ($name in $runtimeNames) {
    Copy-Item -LiteralPath (Join-Path $runtimePath "bin\$name") -Destination (Join-Path $stageNative $name)
}
foreach ($name in $licenseNames) {
    Copy-Item -LiteralPath (Join-Path $runtimePath "licenses\$name") -Destination (Join-Path $stageLicenses $name)
}

if (Test-Path -LiteralPath $outputPath) { Remove-Item -LiteralPath $outputPath -Recurse -Force }
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
Push-Location $stageRoot
try {
    & $python -m build --wheel --no-isolation --outdir $outputPath
    if ($LASTEXITCODE -ne 0) { throw "Python Wheel build failed" }
} finally {
    Pop-Location
}

$deliverables = @(Get-ChildItem -LiteralPath $outputPath -File)
if ($deliverables.Count -ne 1 -or $deliverables[0].Name -ne "cq_ai_engine-0.14.5-py3-none-win_amd64.whl") {
    throw "Python output must contain only cq_ai_engine-0.14.5-py3-none-win_amd64.whl; found: $($deliverables.Name -join ', ')"
}
$wheelPath = $deliverables[0].FullName

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($wheelPath)
try {
    $entryNames = @($archive.Entries | ForEach-Object FullName)
    $requiredEntries = @("cq_ai_engine/_native/CQ_AI_x64.dll")
    $requiredEntries += $runtimeNames | ForEach-Object { "cq_ai_engine/_native/$_" }
    $requiredEntries += $licenseNames | ForEach-Object { "cq_ai_engine/_licenses/$_" }
    $missingEntries = @($requiredEntries | Where-Object { $_ -notin $entryNames })
    if ($missingEntries.Count -gt 0) { throw "Wheel is missing required entries: $($missingEntries -join ', ')" }
    $forbiddenEntries = @($entryNames | Where-Object {
        $_ -match '(?i)(^|/)CQ_AI_worker\.exe$' -or
        $_ -match '(?i)(^|/)CQ_X86\.dll$' -or
        $_ -match '(?i)\.exe$' -or
        $_ -match '(?i)\.(onnx|ini)$'
    })
    if ($forbiddenEntries.Count -gt 0) { throw "Wheel contains forbidden files: $($forbiddenEntries -join ', ')" }
    $wheelMetadataEntry = $archive.Entries | Where-Object FullName -like '*.dist-info/WHEEL' | Select-Object -First 1
    if (!$wheelMetadataEntry) { throw "Wheel metadata entry is missing" }
    $reader = [System.IO.StreamReader]::new($wheelMetadataEntry.Open())
    try { $wheelMetadata = $reader.ReadToEnd() } finally { $reader.Dispose() }
    if ($wheelMetadata -notmatch 'Root-Is-Purelib:\s*false' -or $wheelMetadata -notmatch 'Tag:\s*py3-none-win_amd64') {
        throw "Wheel metadata does not declare py3-none-win_amd64 platform payload"
    }
} finally {
    $archive.Dispose()
}

$workerBefore = @(Get-Process -Name "CQ_AI_worker" -ErrorAction SilentlyContinue | ForEach-Object Id)
$venvRoot = Join-Path $buildRoot "wheel-smoke-venv"
if (Test-Path -LiteralPath $venvRoot) {
    $resolvedVenv = [System.IO.Path]::GetFullPath($venvRoot)
    if (!$resolvedVenv.StartsWith($buildRoot, [System.StringComparison]::OrdinalIgnoreCase)) { throw "invalid venv path" }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}
& $python -m venv $venvRoot
if ($LASTEXITCODE -ne 0) { throw "clean Wheel smoke-test venv creation failed" }
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
& $venvPython -m pip install --no-index --no-deps $wheelPath
if ($LASTEXITCODE -ne 0) { throw "offline Wheel installation failed" }
$fixture = Join-Path $project "examples\ocr_test_abc123.bmp"
$smokeCode = @'
import pathlib
import struct
import sys

import cq_ai_engine
from ai_engine import AI_DEVICE_CPU, Engine

assert struct.calcsize('P') * 8 == 64
assert cq_ai_engine.__version__ == '0.14.5'
assert cq_ai_engine.Engine is Engine
bmp = pathlib.Path(sys.argv[1]).read_bytes()
with Engine() as engine:
    assert engine.dll_path.name == 'CQ_AI_x64.dll', engine.dll_path
    assert 'cq_ai_engine' in str(engine.dll_path.parent.parent)
    assert engine.version() == 'CQ_AI_x64/0.14.5'
    engine.shutdown_worker()
    engine.ocr_load_embedded_models(runtime_device=AI_DEVICE_CPU)
    text = engine.ocr_recognize_bmp(bmp, min_confidence=0.0)
    assert isinstance(text, str)
    status = engine.runtime_status()
    assert status.get('active') == 'cpu', status

with Engine() as engine:
    assert engine.version() == 'CQ_AI_x64/0.14.5'
'@
& $venvPython -I -c $smokeCode $fixture
if ($LASTEXITCODE -ne 0) { throw "installed Wheel smoke test failed" }
$acceptanceScript = Join-Path $project "tests\python_x64_wheel_acceptance.py"
& $venvPython -I $acceptanceScript
if ($LASTEXITCODE -ne 0) { throw "installed Wheel CV/OCR/YOLO acceptance failed" }
$workerAfter = @(Get-Process -Name "CQ_AI_worker" -ErrorAction SilentlyContinue | ForEach-Object Id)
$newWorkers = @($workerAfter | Where-Object { $_ -notin $workerBefore })
if ($newWorkers.Count -gt 0) { throw "Python x64 Wheel unexpectedly started worker PIDs: $($newWorkers -join ', ')" }

$wheelHash = Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256
Write-Host "Python x64 offline Wheel ready: $wheelPath"
Write-Host "SHA256: $($wheelHash.Hash)"
Write-Host "Native exports: $($binaryExports.Count)"
Write-Host "Native dependencies: $($dependencies -join ', ')"

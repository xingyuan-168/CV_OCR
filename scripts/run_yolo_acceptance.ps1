param(
    [string]$RuntimeDir = (Join-Path $PSScriptRoot '../../output'),
    [string]$ImagesDir = (Join-Path $PSScriptRoot '../validation'),
    [string]$Model = (Join-Path $PSScriptRoot '../validation/best.onnx'),
    [string]$PythonExe = 'python',
    [string]$ResultsDir = '',
    [switch]$BusinessWindowsActive,
    [switch]$CalibrateOnly,
    [switch]$SmokeOnly
)
$ErrorActionPreference = 'Stop'
$toolsRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtimeSource = [IO.Path]::GetFullPath($RuntimeDir)
$Model = [IO.Path]::GetFullPath($Model)
$ImagesDir = [IO.Path]::GetFullPath($ImagesDir)
$benchmarkSource = Join-Path $toolsRoot 'yolo_benchmark_x86.exe'
$verifier = Join-Path $toolsRoot 'delivery_verify_x86.exe'
$header = Join-Path $toolsRoot 'include/ai_engine.h'
foreach ($path in @($benchmarkSource,$verifier,$header,$Model,(Join-Path $runtimeSource 'CQ_X86.dll'),(Join-Path $runtimeSource 'CQ_AI_worker.exe'))) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing delivery/tool file: $path. Extract the separate tools ZIP first." }
}
if (!(Test-Path -LiteralPath $ImagesDir -PathType Container) -or @(Get-ChildItem -LiteralPath $ImagesDir -Filter '*.bmp' -File).Count -eq 0) { throw 'ImagesDir must contain real BMP frames' }
$output = if ($ResultsDir) { [IO.Path]::GetFullPath($ResultsDir) } else { Join-Path $toolsRoot ('acceptance-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
if (Test-Path -LiteralPath $output) { throw 'Choose a new ResultsDir; previous reports are preserved' }
if ($output.Equals($runtimeSource,[StringComparison]::OrdinalIgnoreCase) -or $output.StartsWith($runtimeSource + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'ResultsDir must be outside RuntimeDir' }
New-Item -ItemType Directory -Path $output | Out-Null
$python = (Get-Command $PythonExe -ErrorAction Stop).Source
$env:CQ_AI_YOLO_TRACE_PREFIX = ''
$env:CQ_AI_ORT_PROFILE_PREFIX = ''
$env:CQ_AI_YOLO_ENGINE_CACHE = Join-Path $output 'engine-cache'
$nvidiaDirectory = Join-Path $runtimeSource 'nvidia'
if (Test-Path -LiteralPath $nvidiaDirectory) { $env:CQ_AI_NVIDIA_DIR = $nvidiaDirectory }
$inventory = [ordered]@{
    time = (Get-Date).ToString('o')
    cpu = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
    system = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
    gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion)
    baseline_only = [IO.Path]::GetFullPath((Join-Path $toolsRoot 'validation')).Equals($ImagesDir,[StringComparison]::OrdinalIgnoreCase)
    business_windows_active = $BusinessWindowsActive.IsPresent
    runtime_directory = $runtimeSource
    files = @('CQ_X86.dll','CQ_AI_worker.exe' | ForEach-Object { $path=Join-Path $runtimeSource $_; [ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash} })
}
$inventory | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'hardware.json') -Encoding UTF8
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) { & $smi.Source -q | Set-Content -LiteralPath (Join-Path $output 'nvidia-smi.txt') -Encoding UTF8 }
& $verifier --dll (Join-Path $runtimeSource 'CQ_X86.dll') --header $header --model $Model --image (Get-ChildItem -LiteralPath $ImagesDir -Filter '*.bmp' -File | Sort-Object Name | Select-Object -First 1 -ExpandProperty FullName) --report (Join-Path $output 'delivery-loader.json') --preserve-worker 1
if ($LASTEXITCODE -ne 0) { throw 'The requested DLL and its actual paired Worker did not pass delivery verification' }
# The benchmark imports the DLL from its directory. Hash-equal copies keep output at three files.
$stage = Join-Path $output 'benchmark-runtime'
New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -LiteralPath $benchmarkSource -Destination (Join-Path $stage 'yolo_benchmark_x86.exe')
foreach ($name in @('CQ_X86.dll','CQ_AI_worker.exe','CQ_YOLO_TensorRT.dll')) {
    $source = Join-Path $runtimeSource $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $stage $name)
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath (Join-Path $stage $name) -Algorithm SHA256).Hash) { throw "Runtime copy differs: $name" }
    }
}
$benchmark = Join-Path $stage 'yolo_benchmark_x86.exe'
if ($SmokeOnly) {
    & $benchmark --model $Model --images-dir $ImagesDir --device 2 --sessions 5 --threads 1 --warmup 2 --samples 5 --rounds 1 --mode continuous --shutdown-worker 0 --verify-output (Join-Path $output 'smoke-detections.json') --output (Join-Path $output 'smoke.json')
    if ($LASTEXITCODE -ne 0) { throw 'Tools smoke test failed' }
    Write-Host "Tools smoke passed. Performance acceptance was not run. Reports: $output"
    exit 0
}
$firstImage = Get-ChildItem -LiteralPath $ImagesDir -Filter '*.bmp' -File | Sort-Object Name | Select-Object -First 1 -ExpandProperty FullName
& $python (Join-Path $PSScriptRoot 'verify_yolo_optional_dependencies.py') --runtime (Join-Path $toolsRoot 'x64') --model $Model --image $firstImage --output (Join-Path $output 'optional-dependency-checks')
if ($LASTEXITCODE -ne 0) { throw 'Optional dependency absence checks failed' }
& $python (Join-Path $PSScriptRoot 'calibrate_yolo.py') --benchmark $benchmark --model $Model --images-dir $ImagesDir --sessions 1,2,3,5 --cpu-threads 1,2,4,6,8 --devices 2,1,3 --fp16 --graphs --preserve-worker --warmup 100 --samples 1000 --rounds 3 --output $output
if ($LASTEXITCODE -ne 0) { throw 'Business calibration failed; no qualified configuration was published' }
$calibration = Get-Content -LiteralPath (Join-Path $output 'calibration-report.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$profile = Join-Path $output 'business-calibration.tsv'
$gpuChecks = [ordered]@{state='pending'; cache_hit=$false; corruption_rebuilt=$false; gpu_soak=$false}
$gpuCandidates = @($calibration.candidates | Where-Object { $_.device -eq 3 -and $_.precision -eq 'fp32' -and $_.graph -eq 0 } | Sort-Object p95,p50)
if ($gpuCandidates.Count -gt 0) {
    # Use independent x64 processes so active business engines cannot hide disk-cache behavior.
    $cacheRuntime = Join-Path $stage 'x64'
    Copy-Item -LiteralPath (Join-Path $toolsRoot 'x64') -Destination $cacheRuntime -Recurse
    $cacheBenchmark = Join-Path $cacheRuntime 'yolo_benchmark_x64.exe'
    if (!(Test-Path -LiteralPath $cacheBenchmark)) { throw 'The x64 cache-verification runtime is missing from the tools package' }
    if (Test-Path -LiteralPath (Join-Path $runtimeSource 'CQ_YOLO_TensorRT.dll')) { Copy-Item -LiteralPath (Join-Path $runtimeSource 'CQ_YOLO_TensorRT.dll') -Destination (Join-Path $cacheRuntime 'CQ_YOLO_TensorRT.dll') -Force }
    $gpuBest = $gpuCandidates[0]
    $cacheTest = Join-Path $output 'cache-verification'
    New-Item -ItemType Directory -Path $cacheTest | Out-Null
    $env:CQ_AI_YOLO_ENGINE_CACHE = $cacheTest
    $cachePaths = @()
    foreach ($run in @('first-build','cache-hit','corrupt-rebuild')) {
        if ($run -eq 'corrupt-rebuild') {
            $plan = $cachePaths[0]
            $safePlan = [IO.Path]::GetFullPath($plan)
            if (!$safePlan.StartsWith([IO.Path]::GetFullPath($cacheTest) + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe cache mutation path' }
            Copy-Item -LiteralPath $safePlan -Destination ($safePlan + '.saved')
            [IO.File]::WriteAllBytes($safePlan,[byte[]](0,1,2,3))
        }
        $report = Join-Path $output ("cache-$run.json")
        & $cacheBenchmark --model $Model --images-dir $ImagesDir --device 3 --sessions $gpuBest.sessions --precision fp32 --graph 0 --threads 1 --warmup 2 --samples 10 --rounds 1 --mode continuous --shutdown-worker 0 --verify-output (Join-Path $output ("cache-$run.detections.json")) --output $report
        if ($LASTEXITCODE -ne 0) { throw "TensorRT cache test failed: $run" }
        $cacheResult = Get-Content -LiteralPath $report -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($run -eq 'first-build') {
            $cachePaths = @(Get-ChildItem -LiteralPath $cacheTest -Filter '*.plan' -File | Select-Object -ExpandProperty FullName)
            if ($cachePaths.Count -eq 0 -or $cacheResult.runtime.engine_cache_hit) { throw 'A dedicated new engine cache was not built' }
        }
        if ($run -eq 'cache-hit') { $gpuChecks.cache_hit=$cacheResult.runtime.engine_cache_hit; if (!$gpuChecks.cache_hit) { throw 'Engine cache did not hit' } }
        if ($run -eq 'corrupt-rebuild') { $gpuChecks.corruption_rebuilt=(!$cacheResult.runtime.engine_cache_hit -and $cacheResult.runtime.engine_build_us -gt 0); if (!$gpuChecks.corruption_rebuilt) { throw 'Corrupt engine cache was not rebuilt' } }
    }
    $gpuChecks.state='cache_passed'
}
$env:CQ_AI_YOLO_ENGINE_CACHE = Join-Path $output 'engine-cache'
if (!$CalibrateOnly) {
    & $python (Join-Path $PSScriptRoot 'monitor_yolo_benchmark.py') --gpu-log (Join-Path $output 'soak.gpu.csv') -- $benchmark --model $Model --images-dir $ImagesDir --device 0 --sessions $calibration.best.sessions --threads 0 --profile $profile --workload $calibration.workload --warmup 100 --seconds 1800 --rounds 1 --mode continuous --shutdown-worker 0 --output (Join-Path $output 'soak.json')
    if ($LASTEXITCODE -ne 0) { throw 'Five-caller 30-minute AUTO stability test failed' }
    if ($gpuCandidates.Count -gt 0) {
        if ($calibration.best.device -eq 3 -and $calibration.best.precision -eq 'fp32' -and $calibration.best.graph -eq 0) { $gpuChecks.gpu_soak=$true }
        else {
            & $python (Join-Path $PSScriptRoot 'monitor_yolo_benchmark.py') --gpu-log (Join-Path $output 'soak-tensorrt-fp32.gpu.csv') -- $benchmark --model $Model --images-dir $ImagesDir --device 3 --sessions $gpuCandidates[0].sessions --precision fp32 --graph 0 --threads 1 --warmup 100 --seconds 1800 --rounds 1 --mode continuous --shutdown-worker 0 --output (Join-Path $output 'soak-tensorrt-fp32.json')
            if ($LASTEXITCODE -ne 0) { throw 'TensorRT FP32 five-caller 30-minute stability test failed' }
            $gpuChecks.gpu_soak=$true
        }
    }
}
# Trace is a separate short diagnostic; formal calibration above disables it.
$tracePrefix = Join-Path $output 'stages'
$env:CQ_AI_YOLO_TRACE_PREFIX = $tracePrefix
& $benchmark --model $Model --images-dir $ImagesDir --device 0 --sessions $calibration.best.sessions --threads 0 --profile $profile --workload $calibration.workload --warmup 10 --samples 100 --rounds 1 --mode continuous --shutdown-worker 0 --output (Join-Path $output 'stage-diagnostic.json')
$env:CQ_AI_YOLO_TRACE_PREFIX = ''
if ($LASTEXITCODE -ne 0) { throw 'Stage diagnostic failed' }
& $python (Join-Path $PSScriptRoot 'summarize_yolo_trace.py') $tracePrefix --output (Join-Path $output 'stages.json')
if ($LASTEXITCODE -ne 0) { throw 'Stage trace summary failed' }
$stages = Get-Content -LiteralPath (Join-Path $output 'stages.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$targetHost = (@($inventory.cpu | Where-Object Name -Match '2696.*v4').Count -gt 0 -and @($inventory.gpu | Where-Object Name -Match 'RTX\s*2070').Count -gt 0)
$accepted = $targetHost -and $BusinessWindowsActive.IsPresent -and !$inventory.baseline_only -and !$CalibrateOnly -and $calibration.host_sample_threshold_met -and $gpuChecks.gpu_soak
$summary = [ordered]@{schema=1; delivery='0.14.6/v23.6/protocol26'; target_host=$targetHost; business_windows_active=$BusinessWindowsActive.IsPresent; baseline_only=$inventory.baseline_only; target_performance_accepted=$accepted; measured_worst_lane_p50_ms=$calibration.best.p50; measured_worst_lane_p95_ms=$calibration.best.p95; gpu=$gpuChecks; ipc_p95_ms=$stages.ipc_p95_ms; shared_memory_followup=$stages.shared_memory_measurement_gate_exceeded; limitation='Only this host, model, frames and actual business window load are covered. Functional success does not imply the 20/30ms target.'}
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'acceptance-summary.json') -Encoding UTF8
$summaryArguments = @((Join-Path $PSScriptRoot 'summarize_yolo_acceptance.py'), $output)
if ($CalibrateOnly) { $summaryArguments += '--calibrate-only' }
& $python @summaryArguments
if ($LASTEXITCODE -ne 0) { throw 'Acceptance consistency/memory summary failed' }
$summary = Get-Content -LiteralPath (Join-Path $output 'acceptance-summary.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$accepted = $summary.target_performance_accepted
Write-Host "Reports to return: $output"
if ($accepted) { Write-Host 'Target performance accepted for the measured business environment.' }
else { Write-Host 'Functional calibration completed; target performance remains unaccepted. Read acceptance-summary.json for the measured gaps.' }

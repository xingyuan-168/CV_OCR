param(
    [string]$X86Dll = "build-release-x86\Release\CQ_X86.dll",
    [string]$WorkerExe = "build-release-worker-x64\Release\CQ_AI_worker.exe",
    [string]$SmokeExe = "build-release-x86\Release\ai_engine_smoke.exe",
    [string]$ApiHtml = "",
    [string]$OutputDir = "outpush\CQ_AI_e_language_v23.5",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$outpushRoot = [System.IO.Path]::GetFullPath((Join-Path $project "outpush"))
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $project $OutputDir))
$finalOutput = Join-Path $project "output"
$isFinalOutput = $outputPath.Equals($finalOutput, [System.StringComparison]::OrdinalIgnoreCase)
if (!$isFinalOutput -and !$outputPath.StartsWith($outpushRoot + [IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must be $finalOutput or a directory below $outpushRoot"
}
if ($isFinalOutput -and !$NoZip) { throw "The final output directory requires -NoZip" }
$x86Path = [System.IO.Path]::GetFullPath((Join-Path $project $X86Dll))
$workerPath = [System.IO.Path]::GetFullPath((Join-Path $project $WorkerExe))
$smokePath = [System.IO.Path]::GetFullPath((Join-Path $project $SmokeExe))
$htmlPath = if ($ApiHtml) {
    [System.IO.Path]::GetFullPath((Join-Path $project $ApiHtml))
} else {
    $candidate = Get-ChildItem -LiteralPath (Join-Path $project "docs") -Filter "*DLL_API*.html" -File |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($candidate) { $candidate.FullName } else { "" }
}
foreach ($required in @($x86Path, $workerPath, $smokePath, $htmlPath)) {
    if (!(Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required artifact is missing: $required" }
}
$python = (Get-Command python -ErrorAction Stop).Source
& $python (Join-Path $project "scripts\generate_e_language_api_doc.py") `
    --header (Join-Path $project "include\ai_engine.h") `
    --protocol (Join-Path $project "src\worker_protocol.h") `
    --manifest (Join-Path $project "release\v23.5\manifest.json") `
    --output $htmlPath `
    --check
if ($LASTEXITCODE -ne 0) {
    throw "API HTML differs from the public header, Worker protocol or release metadata"
}
$htmlText = Get-Content -LiteralPath $htmlPath -Raw -Encoding UTF8
if ($htmlText -notmatch '0\.14\.5' -or $htmlText -notmatch 'v23\.5' -or
    $htmlText -notmatch 'ID,x,y\|ID,x,y' -or $htmlText -notmatch 'ID,cx,cy\|ID,cx,cy' -or
    $htmlText -notmatch 'origin_x' -or $htmlText -notmatch 'CV_LoadTemplateZipFromMemory') {
    throw "API HTML is not the current v23.5 document or is missing compact/origin/ZIP declarations"
}

$dumpbin = Get-ChildItem -LiteralPath (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC") -Filter dumpbin.exe -Recurse |
    Where-Object { $_.FullName -match 'Hostx64\\x64\\dumpbin\.exe$' } |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
if (!$dumpbin) { throw "dumpbin.exe was not found" }

function Get-Dependencies {
    param([string]$Path)
    $output = & $dumpbin /DEPENDENTS $Path
    if ($LASTEXITCODE -ne 0) { throw "dumpbin /DEPENDENTS failed for $Path" }
    return @($output | ForEach-Object {
        if ($_ -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') { $Matches[1] }
    })
}

$x86Headers = (& $dumpbin /HEADERS $x86Path) -join "`n"
if ($LASTEXITCODE -ne 0 -or $x86Headers -notmatch '(?i)machine \(x86\)') {
    throw "CQ_X86.dll is not an x86 PE image"
}
$workerHeaders = (& $dumpbin /HEADERS $workerPath) -join "`n"
if ($LASTEXITCODE -ne 0 -or $workerHeaders -notmatch '(?i)machine \(x64\)') {
    throw "CQ_AI_worker.exe is not an x64 PE image"
}
$x86Dependencies = @(Get-Dependencies -Path $x86Path)
$workerDependencies = @(Get-Dependencies -Path $workerPath)
$forbiddenDependencies = @(
    "onnxruntime.dll",
    "DirectML.dll",
    "MSVCP140.dll",
    "MSVCP140_1.dll",
    "VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll"
)
foreach ($dependency in @($x86Dependencies + $workerDependencies)) {
    if ($dependency -in $forbiddenDependencies -or
        $dependency -match '(?i)(cuda|cudnn|cublas|nvrtc)') {
        throw "Forbidden ordinary PE dependency detected: $dependency"
    }
}
if ($x86Dependencies.Count -ne 1 -or $x86Dependencies[0] -ne "KERNEL32.dll") {
    throw "CQ_X86.dll must depend only on KERNEL32.dll; found: $($x86Dependencies -join ', ')"
}

# Inspect the build that produced the supplied DLL, including isolated candidates.
$cache = Join-Path (Split-Path -Parent (Split-Path -Parent $x86Path)) "CMakeCache.txt"
$cacheText = Get-Content -LiteralPath $cache -Raw -Encoding UTF8
if ($cacheText -notmatch 'AIENGINE_WITH_OPENCV:BOOL=ON' -or $cacheText -notmatch 'opencv-5\.0\.0-static-mt[/\\]x86') {
    throw "CQ_X86.dll was not configured against the required x86 /MT static OpenCV build"
}
$header = Get-Content -LiteralPath (Join-Path $project "include\ai_engine.h") -Raw -Encoding UTF8
$headerExports = @([regex]::Matches($header, 'AIENGINE_EXPORT[\s\S]*?AIENGINE_CALL\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
if ($headerExports.Count -ne 60) { throw "Public header must contain exactly 60 exports; found $($headerExports.Count)" }
if ($header -notmatch '#define\s+AIENGINE_VERSION_MINOR\s+14' -or $header -notmatch '#define\s+AIENGINE_VERSION_PATCH\s+5') {
    throw "Public header version must be 0.14.5 for v23.5"
}

$exports = & $dumpbin /EXPORTS $x86Path
if ($LASTEXITCODE -ne 0) { throw "dumpbin /EXPORTS failed" }
$exportText = $exports -join "`n"
$binaryExports = @($exports | ForEach-Object {
    if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+([A-Za-z][A-Za-z0-9_]*)\s*$') { $Matches[1] }
} | Sort-Object -Unique)
$missing = @($headerExports | Where-Object { $_ -notin $binaryExports })
$extra = @($binaryExports | Where-Object { $_ -notin $headerExports })
if ($binaryExports.Count -ne 60 -or $missing.Count -gt 0 -or $extra.Count -gt 0) {
    throw "ABI export mismatch. Missing: $($missing -join ', '); extra: $($extra -join ', ')"
}
foreach ($decorated in @(
    '_AI_GetLastError@0',
    '_CV_LoadTemplateZipFromMemory@12',
    '_CV_FindOne@36',
    '_CV_FindTransparentOne@40',
    '_CV_FindMultiText@36',
    '_CV_FindTransparentMultiText@36',
    '_OCR_Recognize@28',
    '_OCR_FindOneText@32',
    '_OCR_FindMultiText@28',
    '_OCR_FindOneCoord@32',
    '_YOLO_InferJson@24')) {
    if ($exportText -notmatch [regex]::Escape($decorated)) { throw "Required decorated export is missing: $decorated" }
}
foreach ($old in @(
    '_AI_GetLastError@8',
    '_CV_FindOne@28',
    '_CV_FindTransparentOne@32',
    '_CV_FindMultiText@28',
    '_CV_FindTransparentMultiText@28',
    '_OCR_Recognize@20',
    '_OCR_FindOneText@24',
    '_OCR_FindMultiText@20',
    '_OCR_FindOneCoord@24',
    '_YOLO_InferJson@16')) {
    if ($exportText -match [regex]::Escape($old)) { throw "A forbidden legacy ABI export is present: $old" }
}

& $smokePath
if ($LASTEXITCODE -ne 0) {
    throw "x86 runtime smoke test failed; AI_GetVersion or the public ABI is not the expected v23.5 build"
}

$verifyOutput = & $workerPath --verify-embedded-runtime
if ($LASTEXITCODE -ne 0) { throw "Worker embedded runtime verification failed" }
$verify = $verifyOutput | ConvertFrom-Json
if (!$verify.valid -or $verify.file_count -lt 13 -or
    $verify.manifest.project_version -ne "0.14.5" -or
    $verify.manifest.delivery_version -ne "v23.5" -or
    $verify.manifest.worker_protocol -ne 25 -or
    $verify.manifest.ort_version -ne "1.24.4" -or
    $verify.manifest.directml_version -ne "1.15.4") {
    throw "Embedded runtime metadata is invalid or differs from the current baseline"
}
$probeOutput = & $workerPath --runtime-probe
if ($LASTEXITCODE -ne 0) { throw "Worker runtime probe failed" }
$probe = $probeOutput | ConvertFrom-Json
if ($probe.runtime_flavor -ne "core" -or $probe.ort_version -ne "1.24.4" -or $probe.worker_protocol -ne 25 -or
    "DmlExecutionProvider" -notin $probe.available_providers -or "CPUExecutionProvider" -notin $probe.available_providers) {
    throw "Worker probe does not match the v23.5 CPU/DirectML requirements"
}
$notices = & $workerPath --third-party-notices
if ($LASTEXITCODE -ne 0 -or ($notices -join "`n") -notmatch "onnxruntime-LICENSE" -or ($notices -join "`n") -notmatch "directml-LICENSE") {
    throw "Embedded third-party notices are incomplete"
}

$temporary = "$outputPath.tmp"
foreach ($path in @($temporary, $outputPath)) {
    if (Test-Path -LiteralPath $path) {
        $resolved = [System.IO.Path]::GetFullPath($path)
        if ($isFinalOutput) {
            if ($resolved -notin @($finalOutput, "$finalOutput.tmp")) { throw "Unsafe cleanup path: $resolved" }
            # Never recursively delete unrelated files in the user's output.
            $existing = @(Get-ChildItem -LiteralPath $resolved)
            $allowed = @('CQ_X86.dll', 'CQ_AI_worker.exe', (Split-Path -Leaf $htmlPath))
            if (@($existing | Where-Object { $_.PSIsContainer -or $_.Name -notin $allowed }).Count) {
                throw "Output contains unrelated files; preserve them and choose a clean delivery directory: $resolved"
            }
        } elseif (!$resolved.StartsWith($outpushRoot + [IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe cleanup path: $resolved" }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}
New-Item -ItemType Directory -Path $temporary | Out-Null
Copy-Item -LiteralPath $x86Path -Destination (Join-Path $temporary "CQ_X86.dll")
Copy-Item -LiteralPath $workerPath -Destination (Join-Path $temporary "CQ_AI_worker.exe")
Copy-Item -LiteralPath $htmlPath -Destination (Join-Path $temporary (Split-Path -Leaf $htmlPath))
$names = @(Get-ChildItem -LiteralPath $temporary -File | Select-Object -ExpandProperty Name | Sort-Object)
$htmlName = Split-Path -Leaf $htmlPath
$expected = @("CQ_AI_worker.exe", "CQ_X86.dll", $htmlName)
if (($names -join "|") -ne (($expected | Sort-Object) -join "|")) { throw "The output must contain exactly CQ_X86.dll, CQ_AI_worker.exe and the current API HTML" }
Move-Item -LiteralPath $temporary -Destination $outputPath

$zipPath = "$outputPath.zip"
if (!$NoZip) {
    Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
    Compress-Archive -LiteralPath (Join-Path $outputPath "CQ_X86.dll"), (Join-Path $outputPath "CQ_AI_worker.exe"), (Join-Path $outputPath $htmlName) -DestinationPath $zipPath -CompressionLevel Optimal
}
Get-ChildItem -LiteralPath $outputPath -File | Sort-Object Name |
    Select-Object Name,Length,@{Name="sha256";Expression={(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}} | Format-Table -AutoSize
Write-Host "ABI exports: 60/60"
Write-Host "Worker protocol: $($probe.worker_protocol)"
Write-Host "v23.5 output: $outputPath"
if (!$NoZip) { Write-Host "ZIP: $zipPath" }

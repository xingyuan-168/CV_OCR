param(
    [string]$EasyZip = "outpush\CQ_AI_e_language_v23.5.zip",
    [string]$PythonWheel = "outpush\CQ_AI_python_x64_v23.5\cq_ai_engine-0.14.5-py3-none-win_amd64.whl"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$outpushRoot = [System.IO.Path]::GetFullPath((Join-Path $project "outpush"))
$releaseRoot = [System.IO.Path]::GetFullPath((Join-Path $project "release"))
$releasePath = Join-Path $releaseRoot "v23.5"
$stagePath = Join-Path $releaseRoot ".v23.5-stage-$PID"
$backupPath = Join-Path $releaseRoot ".v23.5-backup-$PID"

function Resolve-OutpushFile {
    param([string]$Path, [string]$ExpectedName)
    $resolved = [System.IO.Path]::GetFullPath((Join-Path $project $Path))
    if (!$resolved.StartsWith(
            $outpushRoot + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Artifact must remain below $outpushRoot`: $resolved"
    }
    if ([System.IO.Path]::GetFileName($resolved) -cne $ExpectedName) {
        throw "Unexpected artifact name: $resolved"
    }
    if (!(Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "Artifact is missing: $resolved"
    }
    return $resolved
}

function Remove-ReleaseWorkDirectory {
    param([string]$Path)
    $resolved = [System.IO.Path]::GetFullPath($Path)
    if (!$resolved.StartsWith(
            $releaseRoot + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase) -or
        [System.IO.Path]::GetFileName($resolved) -notmatch '^\.v23\.5-(stage|backup)-\d+$') {
        throw "Unsafe release work directory: $resolved"
    }
    if (Test-Path -LiteralPath $resolved) {
        [System.IO.Directory]::Delete($resolved, $true)
    }
}

function Get-StreamSha256 {
    param([System.IO.Stream]$Stream)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($Stream))).Replace("-", "")
    } finally {
        $sha.Dispose()
    }
}

function Get-ArchiveText {
    param([System.IO.Compression.ZipArchiveEntry]$Entry)
    $reader = [System.IO.StreamReader]::new($Entry.Open(), [System.Text.Encoding]::UTF8)
    try {
        return $reader.ReadToEnd()
    } finally {
        $reader.Dispose()
    }
}

$easyPath = Resolve-OutpushFile -Path $EasyZip -ExpectedName "CQ_AI_e_language_v23.5.zip"
$wheelPath = Resolve-OutpushFile -Path $PythonWheel -ExpectedName "cq_ai_engine-0.14.5-py3-none-win_amd64.whl"

$cmakeText = Get-Content -LiteralPath (Join-Path $project "CMakeLists.txt") -Raw -Encoding UTF8
$headerText = Get-Content -LiteralPath (Join-Path $project "include\ai_engine.h") -Raw -Encoding UTF8
$protocolText = Get-Content -LiteralPath (Join-Path $project "src\worker_protocol.h") -Raw -Encoding UTF8
$pythonProjectText = Get-Content -LiteralPath (Join-Path $project "python\pyproject.toml") -Raw -Encoding UTF8
$pythonInitText = Get-Content -LiteralPath (Join-Path $project "python\cq_ai_engine\__init__.py") -Raw -Encoding UTF8
if ($cmakeText -notmatch 'project\(ai_engine_dll VERSION ([0-9]+\.[0-9]+\.[0-9]+)') {
    throw "Cannot read project version from CMakeLists.txt"
}
$projectVersion = $Matches[1]
if ($protocolText -notmatch 'kVersion\s*=\s*(\d+)') {
    throw "Cannot read Worker protocol version"
}
$workerProtocol = [int]$Matches[1]
$headerExports = @([regex]::Matches(
        $headerText,
        'AIENGINE_EXPORT[\s\S]*?AIENGINE_CALL\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
if ($projectVersion -ne "0.14.5" -or $workerProtocol -ne 25 -or
    $cmakeText -notmatch 'set\(AIENGINE_DELIVERY_VERSION\s+"v23\.5"\)' -or
    $cmakeText -notmatch 'set\(AIENGINE_WORKER_PROTOCOL_VERSION\s+"25"\)' -or
    $headerText -notmatch '#define\s+AIENGINE_VERSION_MAJOR\s+0' -or
    $headerText -notmatch '#define\s+AIENGINE_VERSION_MINOR\s+14' -or
    $headerText -notmatch '#define\s+AIENGINE_VERSION_PATCH\s+5' -or
    $pythonProjectText -notmatch 'version\s*=\s*"0\.14\.5"' -or
    $pythonInitText -notmatch '__version__\s*=\s*"0\.14\.5"' -or
    $headerExports.Count -ne 60) {
    throw "Source metadata is not the 0.14.5 / v23.5 / protocol 25 / 60-export baseline"
}

$expectedEasyMembers = @(
    "CQ_X86.dll",
    "CQ_AI_worker.exe",
    "易语言_DLL_API_说明.html"
)
$easyMembers = @()
$easyArchive = [System.IO.Compression.ZipFile]::OpenRead($easyPath)
try {
    $entries = @($easyArchive.Entries | Where-Object { $_.Name })
    if ($entries.Count -ne 3 -or
        (@($entries.FullName | Sort-Object) -join "|") -cne
        (@($expectedEasyMembers | Sort-Object) -join "|")) {
        throw "Easy Language ZIP must contain exactly the three current files"
    }
    foreach ($name in $expectedEasyMembers) {
        $entry = $entries | Where-Object FullName -CEQ $name
        $stream = $entry.Open()
        try {
            $hash = Get-StreamSha256 -Stream $stream
        } finally {
            $stream.Dispose()
        }
        $easyMembers += [ordered]@{
            file = $name
            size = [int64]$entry.Length
            sha256 = $hash
        }
    }
    $apiEntry = $entries | Where-Object FullName -CEQ "易语言_DLL_API_说明.html"
    $apiText = Get-ArchiveText -Entry $apiEntry
    if ($apiText -notmatch 'CQ_AI 0\.14\.5（v23\.5）' -or
        $apiText -notmatch '公开导出为60个' -or
        $apiText -notmatch 'Worker协议为25') {
        throw "Easy Language ZIP contains a stale API document"
    }
} finally {
    $easyArchive.Dispose()
}

$wheelArchive = [System.IO.Compression.ZipFile]::OpenRead($wheelPath)
try {
    $wheelNames = @($wheelArchive.Entries.FullName)
    foreach ($required in @(
        "ai_engine.py",
        "cq_ai_engine/__init__.py",
        "cq_ai_engine/_native/CQ_AI_x64.dll",
        "cq_ai_engine/_native/onnxruntime.dll",
        "cq_ai_engine/_native/DirectML.dll")) {
        if ($required -notin $wheelNames) {
            throw "Wheel member is missing: $required"
        }
    }
    if (@($wheelNames | Where-Object { $_ -match '(?i)\.exe$|CQ_X86\.dll$' }).Count -gt 0) {
        throw "Python Wheel contains a forbidden executable or x86 DLL"
    }
    $wheelInfo = $wheelArchive.Entries | Where-Object FullName -Like "*.dist-info/WHEEL" | Select-Object -First 1
    $wheelMetadata = $wheelArchive.Entries | Where-Object FullName -Like "*.dist-info/METADATA" | Select-Object -First 1
    if ($null -eq $wheelInfo -or $null -eq $wheelMetadata) {
        throw "Wheel metadata is incomplete"
    }
    $wheelInfoText = Get-ArchiveText -Entry $wheelInfo
    $wheelMetadataText = Get-ArchiveText -Entry $wheelMetadata
    if ($wheelInfoText -notmatch 'Root-Is-Purelib:\s*false' -or
        $wheelInfoText -notmatch 'Tag:\s*py3-none-win_amd64' -or
        $wheelMetadataText -notmatch '(?m)^Version:\s*0\.14\.5\s*$') {
        throw "Wheel metadata does not match the current platform and version"
    }
} finally {
    $wheelArchive.Dispose()
}

Remove-ReleaseWorkDirectory -Path $stagePath
Remove-ReleaseWorkDirectory -Path $backupPath
New-Item -ItemType Directory -Path $stagePath | Out-Null
Copy-Item -LiteralPath $easyPath -Destination (Join-Path $stagePath (Split-Path -Leaf $easyPath))
Copy-Item -LiteralPath $wheelPath -Destination (Join-Path $stagePath (Split-Path -Leaf $wheelPath))

$easyItem = Get-Item -LiteralPath $easyPath
$wheelItem = Get-Item -LiteralPath $wheelPath
$manifest = [ordered]@{
    schema = 1
    project_version = $projectVersion
    delivery_version = "v23.5"
    worker_protocol = $workerProtocol
    artifacts = @(
        [ordered]@{
            kind = "easy-language-x86"
            file = $easyItem.Name
            size = [int64]$easyItem.Length
            sha256 = (Get-FileHash -LiteralPath $easyPath -Algorithm SHA256).Hash
            members = $easyMembers
        },
        [ordered]@{
            kind = "python-x64"
            file = $wheelItem.Name
            size = [int64]$wheelItem.Length
            sha256 = (Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash
            platform_tag = "py3-none-win_amd64"
        }
    )
}
$manifestJson = ($manifest | ConvertTo-Json -Depth 8) + "`n"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText(
    (Join-Path $stagePath "manifest.json"),
    $manifestJson,
    $utf8NoBom)

$stageRelative = Join-Path "release" ([System.IO.Path]::GetFileName($stagePath))
& (Join-Path $PSScriptRoot "verify_current_release.ps1") -ReleaseDir $stageRelative | Out-Host

$backupCreated = $false
try {
    if (Test-Path -LiteralPath $releasePath) {
        Move-Item -LiteralPath $releasePath -Destination $backupPath
        $backupCreated = $true
    }
    Move-Item -LiteralPath $stagePath -Destination $releasePath
    try {
        & (Join-Path $PSScriptRoot "verify_current_release.ps1") | Out-Host
    } catch {
        if (Test-Path -LiteralPath $releasePath) {
            [System.IO.Directory]::Delete($releasePath, $true)
        }
        if ($backupCreated -and (Test-Path -LiteralPath $backupPath)) {
            Move-Item -LiteralPath $backupPath -Destination $releasePath
            $backupCreated = $false
        }
        throw
    }
    if ($backupCreated) {
        Remove-ReleaseWorkDirectory -Path $backupPath
        $backupCreated = $false
    }
} catch {
    if (!(Test-Path -LiteralPath $releasePath) -and
        $backupCreated -and (Test-Path -LiteralPath $backupPath)) {
        Move-Item -LiteralPath $backupPath -Destination $releasePath
        $backupCreated = $false
    }
    throw
} finally {
    if (Test-Path -LiteralPath $stagePath) {
        Remove-ReleaseWorkDirectory -Path $stagePath
    }
}

Write-Host "v23.5 release promoted from verified outpush artifacts"
Get-Content -LiteralPath (Join-Path $releasePath "manifest.json") -Raw -Encoding UTF8

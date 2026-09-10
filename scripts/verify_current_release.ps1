param(
    [string]$ReleaseDir = "release\v23.5"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$releaseRoot = [System.IO.Path]::GetFullPath((Join-Path $project "release"))
$releasePath = [System.IO.Path]::GetFullPath((Join-Path $project $ReleaseDir))
if (!$releasePath.StartsWith($releaseRoot + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "ReleaseDir must be a child of $releaseRoot"
}

$manifestPath = Join-Path $releasePath "manifest.json"
if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Current release manifest is missing: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.schema -ne 1 -or $manifest.project_version -ne "0.14.5" -or
    $manifest.delivery_version -ne "v23.5" -or $manifest.worker_protocol -ne 25) {
    throw "Release manifest version metadata is not the v23.5 baseline"
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

$artifacts = @($manifest.artifacts)
if ($artifacts.Count -ne 2) {
    throw "The current release must contain exactly two artifacts; found $($artifacts.Count)"
}
$expectedKinds = @("easy-language-x86", "python-x64")
if ((@($artifacts.kind | Sort-Object) -join "|") -ne (($expectedKinds | Sort-Object) -join "|")) {
    throw "Unexpected artifact kinds: $($artifacts.kind -join ', ')"
}

foreach ($artifact in $artifacts) {
    $artifactPath = [System.IO.Path]::GetFullPath((Join-Path $releasePath $artifact.file))
    if (!$artifactPath.StartsWith($releasePath + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe artifact path in manifest: $($artifact.file)"
    }
    if (!(Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
        throw "Release artifact is missing: $artifactPath"
    }
    $item = Get-Item -LiteralPath $artifactPath
    if ($item.Length -ne [int64]$artifact.size) {
        throw "Artifact size mismatch: $($artifact.file)"
    }
    $actualHash = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash
    if ($actualHash -ne $artifact.sha256) {
        throw "Artifact SHA-256 mismatch: $($artifact.file)"
    }
}

$easy = $artifacts | Where-Object kind -eq "easy-language-x86"
$easyPath = Join-Path $releasePath $easy.file
$easyArchive = [System.IO.Compression.ZipFile]::OpenRead($easyPath)
try {
    $fileEntries = @($easyArchive.Entries | Where-Object { $_.Name })
    $expectedMembers = @($easy.members)
    if ($fileEntries.Count -ne 3 -or $expectedMembers.Count -ne 3) {
        throw "Easy Language ZIP must contain exactly three files"
    }
    if ((@($fileEntries.FullName | Sort-Object) -join "|") -ne
        (@($expectedMembers.file | Sort-Object) -join "|")) {
        throw "Easy Language ZIP member list differs from the manifest"
    }
    foreach ($member in $expectedMembers) {
        $entry = $fileEntries | Where-Object FullName -eq $member.file
        if ($null -eq $entry -or $entry.Length -ne [int64]$member.size) {
            throw "ZIP member size mismatch: $($member.file)"
        }
        $stream = $entry.Open()
        try {
            $actualHash = Get-StreamSha256 -Stream $stream
        } finally {
            $stream.Dispose()
        }
        if ($actualHash -ne $member.sha256) {
            throw "ZIP member SHA-256 mismatch: $($member.file)"
        }
    }
} finally {
    $easyArchive.Dispose()
}

$pythonArtifact = $artifacts | Where-Object kind -eq "python-x64"
$wheelPath = Join-Path $releasePath $pythonArtifact.file
$wheel = [System.IO.Compression.ZipFile]::OpenRead($wheelPath)
try {
    $wheelNames = @($wheel.Entries.FullName)
    foreach ($requiredPattern in @(
        "ai_engine.py",
        "cq_ai_engine/__init__.py",
        "cq_ai_engine/_native/CQ_AI_x64.dll",
        "cq_ai_engine/_native/onnxruntime.dll",
        "cq_ai_engine/_native/DirectML.dll"
    )) {
        if ($requiredPattern -notin $wheelNames) {
            throw "Wheel member is missing: $requiredPattern"
        }
    }
    if (@($wheelNames | Where-Object { $_ -like "*.exe" }).Count -gt 0) {
        throw "Python Wheel must not contain an executable"
    }
    $wheelMetadataEntry = $wheel.Entries | Where-Object FullName -like "*.dist-info/WHEEL" |
        Select-Object -First 1
    if ($null -eq $wheelMetadataEntry) { throw "Wheel metadata is missing" }
    $reader = [System.IO.StreamReader]::new($wheelMetadataEntry.Open())
    try { $wheelMetadata = $reader.ReadToEnd() } finally { $reader.Dispose() }
    if ($wheelMetadata -notmatch "Root-Is-Purelib:\s*false" -or
        $wheelMetadata -notmatch "Tag:\s*py3-none-win_amd64") {
        throw "Wheel metadata does not declare py3-none-win_amd64"
    }
} finally {
    $wheel.Dispose()
}

$cmake = Get-Content -LiteralPath (Join-Path $project "CMakeLists.txt") -Raw -Encoding UTF8
$header = Get-Content -LiteralPath (Join-Path $project "include\ai_engine.h") -Raw -Encoding UTF8
$protocol = Get-Content -LiteralPath (Join-Path $project "src\worker_protocol.h") -Raw -Encoding UTF8
$pythonProject = Get-Content -LiteralPath (Join-Path $project "python\pyproject.toml") -Raw -Encoding UTF8
$pythonInit = Get-Content -LiteralPath (Join-Path $project "python\cq_ai_engine\__init__.py") -Raw -Encoding UTF8
if ($cmake -notmatch "project\(ai_engine_dll VERSION 0\.14\.5" -or
    $header -notmatch "AIENGINE_VERSION_MINOR\s+14" -or
    $header -notmatch "AIENGINE_VERSION_PATCH\s+5" -or
    $protocol -notmatch "kVersion\s*=\s*25" -or
    $pythonProject -notmatch 'version\s*=\s*"0\.14\.5"' -or
    $pythonInit -notmatch '__version__\s*=\s*"0\.14\.5"') {
    throw "Source, protocol, Python package and release versions are inconsistent"
}

Write-Host "v23.5 release verified: 2 artifacts, Easy Language ZIP members, Wheel metadata and source versions"
$artifacts | Select-Object kind,file,size,sha256 | Format-Table -AutoSize

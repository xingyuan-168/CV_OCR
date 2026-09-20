param(
    [switch]$CheckWorkingTree,
    [switch]$RequireClean
)

$ErrorActionPreference = "Stop"
$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

$allowedRoots = @(
    ".gitattributes", ".github", ".gitignore", "CMakeLists.txt", "README.md",
    "configs", "docs", "examples", "include", "models", "python", "release",
    "scripts", "src", "tests", "tools"
)
$tracked = @(& git -c core.quotePath=false -C $project ls-files)
if ($LASTEXITCODE -ne 0) { throw "git ls-files failed" }
foreach ($path in $tracked) {
    $root = ($path -split "/")[0]
    if ($root -notin $allowedRoots) {
        throw "Tracked path is outside the repository allowlist: $path"
    }
    if ($path -match '(^|/)(archive|outpush|test-results|release-manifests|moudels|0810问题|测试图)(/|$)') {
        throw "Legacy or generated tracked path is forbidden: $path"
    }
    if ($path -match '(?i)\.(csv|tsv)$') {
        throw "Generated result file must not be tracked: $path"
    }
    if ($path -match '(?i)\.(dll|exe|zip|whl)$' -and $path -notlike "release/v23.5/*") {
        throw "Compiled artifact is outside release/v23.5: $path"
    }
    if ($path -match '(?i)\.onnx$' -and $path -notlike "models/*") {
        throw "ONNX model is outside models/: $path"
    }
    $diskPath = Join-Path $project $path
    if ((Test-Path -LiteralPath $diskPath -PathType Leaf) -and
        (Get-Item -LiteralPath $diskPath).Length -ge 100MB) {
        throw "Tracked file reaches the GitHub 100 MB limit: $path"
    }
    if ($path -match '(?i)(^|/)(\.env($|\.)|id_rsa|credentials)(/|$)|\.(pem|pfx|key)$') {
        throw "Potential secret file must not be tracked: $path"
    }
}

$releaseDirectories = @(Get-ChildItem -LiteralPath (Join-Path $project "release") -Directory |
    Select-Object -ExpandProperty Name)
if ($releaseDirectories.Count -ne 1 -or $releaseDirectories[0] -ne "v23.5") {
    throw "release/ must contain only v23.5"
}

$grepArguments = @(
    "-C", $project, "grep", "-n", "-I", "-E",
    'docs/archive|release-manifests|runtime-v22|moudels/|moudels\\|V23[._]?[1-4]_|[vV](20|21|22)([._][0-9]+)?|V23_5_COMPACT_MULTI_RESULT_REPORT_CN|opencv-yolo-ocr|build-codex|opencv-5\.0\.0-build-mt-x86',
    "--", ":!release/**", ":!scripts/check_repository_hygiene.ps1"
)
$legacyReferences = @(& git @grepArguments)
if ($LASTEXITCODE -notin @(0, 1)) { throw "git grep failed while checking legacy references" }
if ($legacyReferences.Count -gt 0) {
    throw "Legacy references remain:`n$($legacyReferences -join "`n")"
}

$secretArguments = @(
    "-C", $project, "grep", "-n", "-I", "-E", "-e",
    '-----BEGIN ([A-Z ]+ )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}',
    "--", ":!release/**", ":!scripts/check_repository_hygiene.ps1"
)
$secretReferences = @(& git @secretArguments)
if ($LASTEXITCODE -notin @(0, 1)) { throw "git grep failed while checking secrets" }
if ($secretReferences.Count -gt 0) {
    throw "Potential secret material is tracked:`n$($secretReferences -join "`n")"
}

$requiredDocuments = @(
    "README.md",
    "docs\README.md",
    "docs\PROJECT_GUIDE_CN.md",
    "docs\MINIMAL_RUNTIME_README_CN.md",
    "docs\V23_5_DELIVERY_BASELINE_CN.md",
    "docs\易语言_DLL_API_说明.html",
    "python\README.md",
    "examples\e_language.md"
)
foreach ($relative in $requiredDocuments) {
    if (!(Test-Path -LiteralPath (Join-Path $project $relative) -PathType Leaf)) {
        throw "Required current document is missing: $relative"
    }
}
if (Test-Path -LiteralPath (Join-Path $project "docs\V23_5_COMPACT_MULTI_RESULT_REPORT_CN.md")) {
    throw "Retrospective delivery report must not remain in the current baseline"
}

$markdownPaths = @($tracked | Where-Object { $_ -match '(?i)\.md$' -and (Test-Path -LiteralPath (Join-Path $project $_) -PathType Leaf) })
$linkPattern = [regex]'!?\[[^\]]*\]\((?<target>[^)\s]+)(?:\s+"[^"]*")?\)'
foreach ($relative in $markdownPaths) {
    $documentPath = Join-Path $project $relative
    $documentDirectory = Split-Path -Parent $documentPath
    $content = Get-Content -LiteralPath $documentPath -Raw -Encoding UTF8
    foreach ($match in $linkPattern.Matches($content)) {
        $target = $match.Groups["target"].Value.Trim('<', '>')
        if ($target -match '^(?i:https?|mailto):' -or $target.StartsWith('#')) { continue }
        $targetPath = ($target -split '#', 2)[0]
        if ([string]::IsNullOrWhiteSpace($targetPath)) { continue }
        $resolvedTarget = [System.IO.Path]::GetFullPath((Join-Path $documentDirectory $targetPath))
        if (!$resolvedTarget.StartsWith($project, [System.StringComparison]::OrdinalIgnoreCase) -or
            !(Test-Path -LiteralPath $resolvedTarget)) {
            throw "Broken local Markdown link in $relative`: $target"
        }
    }
}

$python = (Get-Command python -ErrorAction Stop).Source
& $python (Join-Path $project "scripts\generate_e_language_api_doc.py") `
    --header (Join-Path $project "include\ai_engine.h") `
    --protocol (Join-Path $project "src\worker_protocol.h") `
    --manifest (Join-Path $project "release\v23.5\manifest.json") `
    --output (Join-Path $project "docs\易语言_DLL_API_说明.html") `
    --check
if ($LASTEXITCODE -ne 0) {
    throw "Generated Easy Language API document differs from source facts"
}

if ($CheckWorkingTree) {
    $allowedDiskRoots = @($allowedRoots + ".git")
    foreach ($item in Get-ChildItem -LiteralPath $project -Force) {
        if ($item.Name -notin $allowedDiskRoots) {
            throw "Unexpected top-level working-tree item: $($item.Name)"
        }
    }
}

& (Join-Path $PSScriptRoot "verify_current_release.ps1") | Out-Host

if ($RequireClean) {
    $status = @(& git -C $project status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0) { throw "git status failed" }
    if ($status.Count -gt 0) {
        throw "Working tree is not clean:`n$($status -join "`n")"
    }
}

Write-Host "Repository hygiene verified: allowlist, current-only release/docs, links, generated API, size and secret checks"

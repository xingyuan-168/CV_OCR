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
}

$releaseDirectories = @(Get-ChildItem -LiteralPath (Join-Path $project "release") -Directory |
    Select-Object -ExpandProperty Name)
if ($releaseDirectories.Count -ne 1 -or $releaseDirectories[0] -ne "v23.5") {
    throw "release/ must contain only v23.5"
}

$grepArguments = @(
    "-C", $project, "grep", "-n", "-I", "-E",
    'docs/archive|release-manifests|runtime-v22|moudels/|moudels\\|V22_|V23[._]?[1-4]_',
    "--", ":!release/**", ":!scripts/check_repository_hygiene.ps1"
)
$legacyReferences = @(& git @grepArguments)
if ($LASTEXITCODE -notin @(0, 1)) { throw "git grep failed while checking legacy references" }
if ($legacyReferences.Count -gt 0) {
    throw "Legacy references remain:`n$($legacyReferences -join "`n")"
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

Write-Host "Repository hygiene verified: allowlist, current-only release, no legacy paths"

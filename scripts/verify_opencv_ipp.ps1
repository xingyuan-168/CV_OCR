param([Parameter(Mandatory = $true)][string]$ConfigHeader, [string]$LibraryRoot = "")
$ErrorActionPreference = "Stop"
if (!(Test-Path -LiteralPath $ConfigHeader -PathType Leaf)) { throw "Missing OpenCV feature header: $ConfigHeader" }
$config = Get-Content -LiteralPath $ConfigHeader -Raw
foreach ($feature in @("HAVE_IPP", "HAVE_IPP_ICV", "HAVE_IPP_IW")) {
    if ($config -notmatch "(?m)^\s*#\s*define\s+$feature(?:\s|$)") {
        throw "Required OpenCV acceleration is disabled: $feature ($ConfigHeader)"
    }
}
if ($LibraryRoot) {
    foreach ($library in @("ippicvmt.lib", "ippiw.lib")) {
        $path = Join-Path $LibraryRoot $library
        if (!(Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) {
            throw "Missing OpenCV IPP library: $path"
        }
    }
}

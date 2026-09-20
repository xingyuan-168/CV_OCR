$ErrorActionPreference = "Stop"
$check = Join-Path $PSScriptRoot "../scripts/verify_opencv_ipp.ps1"
$root = Join-Path ([IO.Path]::GetTempPath()) ("cq-ipp-check-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root | Out-Null
$header = Join-Path $root 'cvconfig.h'
$valid = "#define HAVE_IPP`n#define HAVE_IPP_ICV`n#define HAVE_IPP_IW`n"
function Expect-Failure([scriptblock]$Action) {
    $failed = $false
    try { & $Action } catch { $failed = $true }
    if (!$failed) { throw 'Expected IPP guard failure was not reported' }
}
try {
    Expect-Failure { & $check -ConfigHeader $header }
    foreach ($bad in @('/* #undef HAVE_IPP */', '#define HAVE_IPP_FAKE', '#define HAVE_IPP')) {
        Set-Content -LiteralPath $header -Value $bad
        Expect-Failure { & $check -ConfigHeader $header }
    }
    Set-Content -LiteralPath $header -Value $valid
    & $check -ConfigHeader $header
    Expect-Failure { & $check -ConfigHeader $header -LibraryRoot $root }
    [IO.File]::WriteAllBytes((Join-Path $root 'ippicvmt.lib'), [byte[]]@(1))
    [IO.File]::WriteAllBytes((Join-Path $root 'ippiw.lib'), [byte[]]@())
    Expect-Failure { & $check -ConfigHeader $header -LibraryRoot $root }
    [IO.File]::WriteAllBytes((Join-Path $root 'ippiw.lib'), [byte[]]@(1))
    & $check -ConfigHeader $header -LibraryRoot $root
    Write-Output 'IPP dependency guard: positive and negative cases PASS'
} finally {
    foreach ($name in @('cvconfig.h','ippicvmt.lib','ippiw.lib')) {
        $file = Join-Path $root $name
        if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force }
    }
    Remove-Item -LiteralPath $root -Force
}

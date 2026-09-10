param(
    [ValidateSet("All", "x86", "x64", "None")]
    [string]$OpenCVArchitecture = "All",
    [switch]$SkipRuntime,
    [switch]$ForceDownload
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$project = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$thirdPartyRoot = [System.IO.Path]::GetFullPath((Join-Path $project "third_party"))
$downloadRoot = Join-Path $thirdPartyRoot "downloads"
$extractRoot = Join-Path $thirdPartyRoot "extract"

function Assert-UnderRoot {
    param([string]$Path, [string]$Root, [string]$Label)
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root)
    if (!$resolvedPath.StartsWith($resolvedRoot + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must remain below $resolvedRoot`: $resolvedPath"
    }
    return $resolvedPath
}

function Remove-SafeDirectory {
    param([string]$Path)
    $safePath = Assert-UnderRoot -Path $Path -Root $thirdPartyRoot -Label "Cleanup path"
    if (Test-Path -LiteralPath $safePath) {
        [System.IO.Directory]::Delete($safePath, $true)
    }
}

function Get-VerifiedDownload {
    param([string]$Uri, [string]$Path, [string]$ExpectedSha256)
    $safePath = Assert-UnderRoot -Path $Path -Root $thirdPartyRoot -Label "Download path"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $safePath) | Out-Null
    if ($ForceDownload -or !(Test-Path -LiteralPath $safePath -PathType Leaf)) {
        $temporary = "$safePath.download"
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $temporary
        Move-Item -LiteralPath $temporary -Destination $safePath -Force
    }
    $actual = (Get-FileHash -LiteralPath $safePath -Algorithm SHA256).Hash
    if ($actual -ne $ExpectedSha256) {
        throw "Dependency SHA-256 mismatch for $safePath. Expected $ExpectedSha256, got $actual"
    }
    return $safePath
}

function Find-CMake {
    $command = Get-Command cmake -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidate = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    throw "CMake from Visual Studio 2022 Build Tools was not found"
}

function Invoke-GitWithRetry {
    param(
        [string[]]$Arguments,
        [string]$Description,
        [int]$Attempts = 3
    )
    for ($attempt = 1; $attempt -le $Attempts; ++$attempt) {
        $output = @(& git @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
        $output | ForEach-Object { Write-Host $_ }
        if ($exitCode -eq 0) { return }
        if ($attempt -eq $Attempts) {
            throw "$Description failed after $Attempts attempts"
        }
        Write-Warning "$Description failed (attempt $attempt/$Attempts); retrying"
        Start-Sleep -Seconds (5 * $attempt)
    }
}

function Prepare-Runtime {
    $runtimePath = Assert-UnderRoot `
        -Path (Join-Path $thirdPartyRoot "runtime\ort-directml-1.24.4") `
        -Root $thirdPartyRoot -Label "Runtime path"
    $runtimeExtract = Join-Path $extractRoot "runtime"
    $ortPackage = Get-VerifiedDownload `
        -Uri "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.24.4/microsoft.ml.onnxruntime.directml.1.24.4.nupkg" `
        -Path (Join-Path $downloadRoot "microsoft.ml.onnxruntime.directml.1.24.4.nupkg") `
        -ExpectedSha256 "57E9F11B73437BEF7A309496135D4C1F96B1A8E9DDBA60013FA27BFC1D788681"
    $dmlPackage = Get-VerifiedDownload `
        -Uri "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg" `
        -Path (Join-Path $downloadRoot "microsoft.ai.directml.1.15.4.nupkg") `
        -ExpectedSha256 "4E7CB7DDCE8CF837A7A75DC029209B520CA0101470FCDF275C1F49736A3615B9"

    Remove-SafeDirectory -Path $runtimeExtract
    New-Item -ItemType Directory -Force -Path $runtimeExtract | Out-Null
    $ortExtract = Join-Path $runtimeExtract "onnxruntime"
    $dmlExtract = Join-Path $runtimeExtract "directml"
    [System.IO.Compression.ZipFile]::ExtractToDirectory($ortPackage, $ortExtract)
    [System.IO.Compression.ZipFile]::ExtractToDirectory($dmlPackage, $dmlExtract)

    Remove-SafeDirectory -Path $runtimePath
    $includeDir = Join-Path $runtimePath "include"
    $binDir = Join-Path $runtimePath "bin"
    $libDir = Join-Path $runtimePath "lib"
    $licenseDir = Join-Path $runtimePath "licenses"
    New-Item -ItemType Directory -Force -Path $includeDir,$binDir,$libDir,$licenseDir | Out-Null

    Copy-Item -Path (Join-Path $ortExtract "build\native\include\*") -Destination $includeDir -Force
    Copy-Item -LiteralPath (Join-Path $dmlExtract "include\DirectML.h") -Destination $includeDir
    Copy-Item -LiteralPath (Join-Path $dmlExtract "include\DirectMLConfig.h") -Destination $includeDir
    Copy-Item -LiteralPath (Join-Path $ortExtract "runtimes\win-x64\native\onnxruntime.dll") -Destination $binDir
    Copy-Item -LiteralPath (Join-Path $ortExtract "runtimes\win-x64\native\onnxruntime.lib") -Destination $libDir
    Copy-Item -LiteralPath (Join-Path $dmlExtract "bin\x64-win\DirectML.dll") -Destination $binDir

    $redistRoot = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC"
    $crtDir = Get-ChildItem -LiteralPath $redistRoot -Directory |
        Where-Object Name -match '^\d+\.\d+\.\d+$' |
        Sort-Object { [version]$_.Name } -Descending |
        ForEach-Object { Join-Path $_.FullName "x64\Microsoft.VC143.CRT" } |
        Where-Object { Test-Path -LiteralPath (Join-Path $_ "msvcp140.dll") } |
        Select-Object -First 1
    if (!$crtDir) { throw "Visual C++ x64 redistributable files were not found" }
    foreach ($name in @("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll")) {
        Copy-Item -LiteralPath (Join-Path $crtDir $name) -Destination $binDir
    }

    Copy-Item -LiteralPath (Join-Path $ortExtract "LICENSE") -Destination (Join-Path $licenseDir "onnxruntime-LICENSE.txt")
    Copy-Item -LiteralPath (Join-Path $ortExtract "ThirdPartyNotices.txt") -Destination (Join-Path $licenseDir "onnxruntime-ThirdPartyNotices.txt")
    Copy-Item -LiteralPath (Join-Path $dmlExtract "LICENSE.txt") -Destination (Join-Path $licenseDir "directml-LICENSE.txt")
    Copy-Item -LiteralPath (Join-Path $dmlExtract "LICENSE-CODE.txt") -Destination (Join-Path $licenseDir "directml-LICENSE-CODE.txt")
    Copy-Item -LiteralPath (Join-Path $dmlExtract "ThirdPartyNotices.txt") -Destination (Join-Path $licenseDir "directml-ThirdPartyNotices.txt")
    Invoke-WebRequest -UseBasicParsing -Uri "https://aka.ms/vs/17/redist.txt" `
        -OutFile (Join-Path $licenseDir "visual-cpp-runtime-license.txt")
    Set-Content -LiteralPath (Join-Path $runtimePath "VERSION_NUMBER") -Value "1.24.4" -Encoding Ascii

    foreach ($required in @(
        "include\onnxruntime_cxx_api.h", "include\DirectML.h", "bin\onnxruntime.dll",
        "bin\DirectML.dll", "lib\onnxruntime.lib", "licenses\onnxruntime-LICENSE.txt",
        "licenses\directml-LICENSE.txt", "VERSION_NUMBER"
    )) {
        if (!(Test-Path -LiteralPath (Join-Path $runtimePath $required) -PathType Leaf)) {
            throw "Prepared runtime is incomplete: $required"
        }
    }
    Write-Host "Runtime ready: $runtimePath"
}

function Prepare-OpenCVSource {
    $sourcePath = Assert-UnderRoot `
        -Path (Join-Path $thirdPartyRoot "src\opencv-5.0.0") `
        -Root $thirdPartyRoot -Label "OpenCV source path"
    $commit = "40738fb16ceddb5fb3fea747585f7ce6abb0605b"
    if (!(Test-Path -LiteralPath (Join-Path $sourcePath ".git") -PathType Container)) {
        Remove-SafeDirectory -Path $sourcePath
        New-Item -ItemType Directory -Force -Path $sourcePath | Out-Null
        Invoke-GitWithRetry -Description "OpenCV repository initialization" -Arguments @(
            "-C", $sourcePath, "init"
        )
        Invoke-GitWithRetry -Description "OpenCV remote configuration" -Arguments @(
            "-C", $sourcePath, "remote", "add", "origin", "https://github.com/opencv/opencv.git"
        )
    }
    $actualCommit = @(& git -C $sourcePath rev-parse HEAD 2>$null) | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0 -or $actualCommit.Trim() -ne $commit) {
        $fetched = $false
        foreach ($remoteUrl in @(
            "https://github.com/opencv/opencv.git",
            "ssh://git@ssh.github.com:443/opencv/opencv.git"
        )) {
            & git -C $sourcePath remote set-url origin $remoteUrl
            if ($LASTEXITCODE -ne 0) { throw "OpenCV remote update failed" }
            try {
                Invoke-GitWithRetry -Attempts 2 `
                    -Description "OpenCV pinned commit fetch from $remoteUrl" `
                    -Arguments @("-C", $sourcePath, "fetch", "--depth", "1", "origin", $commit)
                $fetched = $true
                break
            } catch {
                Write-Warning $_.Exception.Message
            }
        }
        if (!$fetched) { throw "OpenCV pinned commit could not be fetched from any configured transport" }
        Invoke-GitWithRetry -Description "OpenCV pinned commit checkout" -Arguments @(
            "-C", $sourcePath, "checkout", "--force", "--detach", "FETCH_HEAD"
        )
    }
    $actualCommit = (& git -C $sourcePath rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $commit) {
        throw "OpenCV source is not the pinned 5.0.0 commit: $actualCommit"
    }

    $patchPath = Join-Path $PSScriptRoot "patches\opencv-5.0.0-msvc-x86-avx2.patch"
    if (!(Test-Path -LiteralPath $patchPath -PathType Leaf)) { throw "OpenCV x86 patch is missing" }
    & git -C $sourcePath apply --reverse --check $patchPath 2>$null
    if ($LASTEXITCODE -ne 0) {
        & git -C $sourcePath apply --check $patchPath
        if ($LASTEXITCODE -ne 0) { throw "OpenCV x86 patch cannot be applied" }
        & git -C $sourcePath apply $patchPath
        if ($LASTEXITCODE -ne 0) { throw "OpenCV x86 patch failed" }
    }
    return $sourcePath
}

function Build-OpenCV {
    param([string]$Architecture, [string]$SourcePath)
    $platform = if ($Architecture -eq "x86") { "Win32" } else { "x64" }
    $buildPath = Assert-UnderRoot `
        -Path (Join-Path $thirdPartyRoot "build\opencv-5.0.0-$Architecture") `
        -Root $thirdPartyRoot -Label "OpenCV build path"
    $installPath = Assert-UnderRoot `
        -Path (Join-Path $thirdPartyRoot "opencv-5.0.0-static-mt\$Architecture") `
        -Root $thirdPartyRoot -Label "OpenCV install path"
    $cmake = Find-CMake
    $configure = @(
        "-S", $SourcePath, "-B", $buildPath,
        "-G", "Visual Studio 17 2022", "-A", $platform,
        "-DCMAKE_INSTALL_PREFIX=$installPath",
        "-DBUILD_LIST=core,imgproc", "-DBUILD_SHARED_LIBS=OFF",
        "-DBUILD_WITH_STATIC_CRT=ON", "-DBUILD_TESTS=OFF", "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF", "-DBUILD_DOCS=OFF", "-DBUILD_opencv_apps=OFF",
        "-DBUILD_JAVA=OFF", "-DBUILD_opencv_python2=OFF", "-DBUILD_opencv_python3=OFF",
        "-DWITH_OPENCL=OFF", "-DWITH_OPENMP=OFF", "-DWITH_TBB=OFF",
        "-DWITH_FFMPEG=OFF", "-DWITH_MSMF=OFF", "-DWITH_IPP=ON",
        "-DCPU_BASELINE=SSE2", "-DCPU_DISPATCH=SSE4_1;AVX;AVX2"
    )
    & $cmake @configure
    if ($LASTEXITCODE -ne 0) { throw "OpenCV $Architecture configure failed" }
    # OpenCV 5 exports these bundled support targets from OpenCVModules.cmake
    # even when BUILD_LIST is limited to core/imgproc. Build them explicitly so
    # a clean install never contains dangling imported-library references.
    $supportBuild = @(
        "--build", $buildPath, "--config", "Release", "--target",
        "libtiff", "libwebp", "libpng", "libprotobuf", "--parallel"
    )
    & $cmake @supportBuild
    if ($LASTEXITCODE -ne 0) { throw "OpenCV $Architecture support-library build failed" }
    & $cmake --build $buildPath --config Release --target INSTALL --parallel
    if ($LASTEXITCODE -ne 0) { throw "OpenCV $Architecture build failed" }
    $configPath = Join-Path $installPath "OpenCVConfig.cmake"
    if (!(Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "OpenCV $Architecture installation is incomplete: $configPath"
    }
    $libraryRoot = Join-Path $installPath "$Architecture\vc17\staticlib"
    foreach ($library in @("libtiff.lib", "libwebp.lib", "libpng.lib", "libprotobuf.lib")) {
        if (!(Test-Path -LiteralPath (Join-Path $libraryRoot $library) -PathType Leaf)) {
            throw "OpenCV $Architecture installation has a dangling export: $library"
        }
    }
    Write-Host "OpenCV $Architecture /MT ready: $installPath"
}

New-Item -ItemType Directory -Force -Path $thirdPartyRoot | Out-Null
if (!$SkipRuntime) { Prepare-Runtime }
if ($OpenCVArchitecture -ne "None") {
    $sourcePath = Prepare-OpenCVSource | Select-Object -Last 1
    if ($OpenCVArchitecture -in @("All", "x86")) { Build-OpenCV -Architecture "x86" -SourcePath $sourcePath }
    if ($OpenCVArchitecture -in @("All", "x64")) { Build-OpenCV -Architecture "x64" -SourcePath $sourcePath }
}

Write-Host "Pinned v23.5 dependencies are ready below $thirdPartyRoot"

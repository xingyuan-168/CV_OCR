param([ValidateSet('x86','x64')][string]$Architecture='x64')
$ErrorActionPreference='Stop'
$repo='D:/codex-project/OpenCV'
$vc='C:/cq-ai-msvc-14.44/VC/Tools/MSVC/14.44.35207'
$sdk='C:/cq-ai-msvc-14.44/Windows Kits/10'
$cv="$repo/third_party/opencv-5.0.0-static-mt/$Architecture"
$retained="D:/codex-delivery/CV_OCR/cv-transparent-mask-fix-20261023/build/cv-mask-$Architecture-production"
$dest='D:/codex-diagnostics/CV_OCR/20261023-transparent-performance'
New-Item -ItemType Directory -Path $dest -Force | Out-Null
$env:INCLUDE="$vc/include;$sdk/Include/10.0.26100.0/ucrt;$sdk/Include/10.0.26100.0/shared;$sdk/Include/10.0.26100.0/um"
$env:LIB="$vc/lib/$Architecture;$sdk/Lib/10.0.26100.0/ucrt/$Architecture;$sdk/Lib/10.0.26100.0/um/$Architecture"
$libs=@('opencv_core500','opencv_imgproc500','opencv_geometry500','opencv_flann500','zlib','libclapack','ittnotify','ipphal','ippiw','ippicvmt') | ForEach-Object {"$cv/$Architecture/vc17/staticlib/$_.lib"}
$objs=@('cv_correlation','cv_normalize_avx2','cv_masked_score_avx2') | ForEach-Object {"$retained/CMakeFiles/ai_engine.dir/src/$_.cpp.obj"}
& "$vc/bin/Hostx64/$Architecture/cl.exe" /nologo /O2 /MT /EHsc /std:c++17 /utf-8 "/I$repo/src" "/I$cv/include" "$PSScriptRoot/phase_probe.cpp" "/Fo$dest/phase-$Architecture.obj" "/Fe$dest/phase-$Architecture.exe" @objs @libs /link kernel32.lib user32.lib gdi32.lib advapi32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib shell32.lib
if($LASTEXITCODE -ne 0){throw 'Probe compilation failed'}

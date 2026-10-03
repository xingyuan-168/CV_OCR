Set-StrictMode -Version Latest

function Resolve-CleanupTarget {
    param([string]$Root, [string]$Relative, [string[]]$Allowed)
    $rootPath = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    if ([IO.Path]::IsPathRooted($Relative) -or $Relative -match '(^|[\\/])\.\.([\\/]|$)' -or $Relative.Contains(':')) {
        throw "Cleanup path must be a registered relative path: $Relative"
    }
    $normal = $Relative.Replace('\', '/').TrimEnd('/')
    if ($normal -notin $Allowed) { throw "Unregistered cleanup target: $normal" }
    if ($normal -match '^(\.git|input|output|release|src|include|models|tests)(/|$)' -or $normal -match '^outpush/rollback(/|$)') {
        throw "Protected cleanup target: $normal"
    }
    $path = [IO.Path]::GetFullPath((Join-Path $rootPath $normal))
    if (!$path.StartsWith($rootPath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target escapes workspace: $path"
    }
    $probe = $path
    while ($probe -and $probe.Length -ge $rootPath.Length) {
        if (Test-Path -LiteralPath $probe) {
            $item = Get-Item -LiteralPath $probe -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point rejected: $probe" }
        }
        if ($probe -eq $rootPath) { break }
        $probe = Split-Path -Parent $probe
    }
    if (Test-Path -LiteralPath $path -PathType Container) {
        foreach ($item in @(Get-ChildItem -LiteralPath $path -Recurse -Force)) {
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point rejected: $($item.FullName)" }
        }
    }
    return $path
}

function Remove-RegisteredTarget {
    param([string]$Root, [string]$Relative, [string[]]$Allowed, [switch]$Apply)
    $path = Resolve-CleanupTarget -Root $Root -Relative $Relative -Allowed $Allowed
    if (!(Test-Path -LiteralPath $path)) { return @{path=$Relative; state='already_absent'; bytes=0} }
    $item = Get-Item -LiteralPath $path -Force
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $path -File -Force -Recurse) } else { @($item) }
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    if (!$Apply) { return @{path=$Relative; state='preview'; bytes=[long]$bytes} }
    $handles = [Collections.Generic.List[IDisposable]]::new()
    try {
        foreach ($file in $files) {
            $handles.Add([IO.File]::Open($file.FullName, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None))
        }
    } catch {
        return @{path=$Relative; state='occupied'; bytes=0; error=$_.Exception.Message}
    } finally { foreach ($handle in $handles) { $handle.Dispose() } }
    $removed = 0L
    try {
        # Delete verified files individually, then only empty directories.
        foreach ($file in $files) {
            if ($file.Attributes -band [IO.FileAttributes]::ReadOnly) { Set-ItemProperty -LiteralPath $file.FullName -Name IsReadOnly -Value $false }
            $file.Attributes = $file.Attributes -band (-bnot ([IO.FileAttributes]::Hidden -bor [IO.FileAttributes]::ReadOnly))
            Remove-Item -LiteralPath $file.FullName -ErrorAction Stop
            $removed += $file.Length
        }
        if ($item.PSIsContainer) {
            foreach ($directory in @(Get-ChildItem -LiteralPath $path -Directory -Force -Recurse | Sort-Object { $_.FullName.Length } -Descending)) {
                if (@(Get-ChildItem -LiteralPath $directory.FullName -Force).Count) { throw 'Directory changed during cleanup' }
                $directory.Attributes = $directory.Attributes -band (-bnot ([IO.FileAttributes]::Hidden -bor [IO.FileAttributes]::ReadOnly))
                Remove-Item -LiteralPath $directory.FullName -ErrorAction Stop
            }
            if (@(Get-ChildItem -LiteralPath $path -Force).Count) { throw 'Target changed during cleanup' }
            $item.Attributes = $item.Attributes -band (-bnot ([IO.FileAttributes]::Hidden -bor [IO.FileAttributes]::ReadOnly))
            Remove-Item -LiteralPath $path -ErrorAction Stop
        }
        return @{path=$Relative; state='deleted'; bytes=$removed}
    } catch { return @{path=$Relative; state='failed'; bytes=$removed; error=$_.Exception.Message} }
}
Export-ModuleMember -Function Resolve-CleanupTarget, Remove-RegisteredTarget

param([string]$Root = "$PSScriptRoot/../.cache/tools/nicad-core")
$ErrorActionPreference = 'Stop'
$Commit = '7a90d11795a7fe585282e25b7fa8d9964f202965'
$Base = "https://raw.githubusercontent.com/CordyJ/Open-NiCad/$Commit"
$Root = [IO.Path]::GetFullPath($Root)
New-Item -ItemType Directory -Force -Path "$Root/UNIX", "$Root/UNIX64" | Out-Null
$Files = @('main.c', 'TLI.c', 'TLS.c', 'TLglob.h', 'UNIX/cinterface.h', 'crossclones.c', 'crossclones.t')
foreach ($File in $Files) {
    Invoke-WebRequest -Uri "$Base/src/tools/$File" -OutFile "$Root/$File" -TimeoutSec 60
}
Invoke-WebRequest -Uri "$Base/LICENSE.txt" -OutFile "$Root/LICENSE.txt" -TimeoutSec 60
# Upstream C references UNIX64 while its checked-in portable runtime is in UNIX.
Copy-Item -LiteralPath "$PSScriptRoot/nicad_go_support/cinterface.h" -Destination "$Root/UNIX64/cinterface.h"
$Vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$Vs = & $Vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $Vs) { throw 'MSVC C compiler was not found.' }
$Vcvars = Join-Path $Vs 'VC/Auxiliary/Build/vcvars64.bat'
Push-Location -LiteralPath $Root
try {
    & cmd /c "`"$Vcvars`" && cl /nologo /w /TC /O2 /I. main.c TLI.c TLS.c crossclones.c /Fecrossclones.exe /link /STACK:536870912"
    if ($LASTEXITCODE -ne 0) { throw 'NiCad core build failed.' }
    $Hashes = @{}
    foreach ($File in ($Files + @('UNIX64/cinterface.h', 'crossclones.exe'))) {
        $Hashes[$File] = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash
    }
    @{upstream_commit=$Commit; files=$Hashes; compiler='MSVC cl /TC /O2'; adaptation='header include path only'} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath 'build_manifest.json' -Encoding utf8
} finally {
    Pop-Location
}

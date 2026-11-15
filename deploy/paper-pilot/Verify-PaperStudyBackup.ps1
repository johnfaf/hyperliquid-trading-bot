param(
    [string]$StateDir = "$env:LOCALAPPDATA\HyperliquidStudy",
    [string]$Archive
)

$ErrorActionPreference = 'Stop'
if (-not $Archive) {
    $latest = Get-ChildItem -LiteralPath (Join-Path $StateDir 'encrypted') -Filter 'study-*.tar.gz.gpg' -File |
        Sort-Object Name -Descending | Select-Object -First 1
    if (-not $latest) { throw 'No encrypted study backup has been downloaded' }
    $Archive = $latest.FullName
}
$Archive = (Resolve-Path -LiteralPath $Archive).Path
$git = Join-Path $env:ProgramFiles 'Git/usr/bin'
$gpg = Join-Path $git 'gpg.exe'
$cygpath = Join-Path $git 'cygpath.exe'
$python = Join-Path $env:LOCALAPPDATA 'Python/pythoncore-3.14-64/python.exe'
$check = Join-Path $StateDir 'release/scripts/paper_study_recovery.py'
$scratch = Join-Path $StateDir ('restore-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $scratch
$payload = Join-Path $scratch 'payload.tar.gz'
try {
    $secure = (Get-Content -LiteralPath (Join-Path $StateDir 'passphrase.dpapi') -Raw) | ConvertTo-SecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $gpgHome = (& $cygpath -u (Join-Path $StateDir 'gpg')).Trim()
        $encrypted = (& $cygpath -u $Archive).Trim()
        $plain = (& $cygpath -u $payload).Trim()
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) |
            & $gpg --homedir $gpgHome --batch --no-tty --pinentry-mode loopback --passphrase-fd 0 --output $plain --decrypt $encrypted
        if ($LASTEXITCODE -ne 0) { throw 'Backup decryption failed' }
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
    & $python -I -B $check verify $payload
    if ($LASTEXITCODE -ne 0) { throw 'Restored scorecard could not be verified' }
} finally {
    if (Test-Path -LiteralPath $payload) { Remove-Item -LiteralPath $payload }
    Remove-Item -LiteralPath $scratch
}

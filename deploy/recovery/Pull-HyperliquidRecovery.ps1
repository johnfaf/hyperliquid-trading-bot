param(
    [Parameter(Mandatory)][string]$Server,
    [Parameter(Mandatory)][string]$SshKey,
    [string]$Directory = "$env:LOCALAPPDATA\HyperliquidRecovery",
    [string]$Python = "python"
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $Directory 'recovery_bundle.py'
$age = Join-Path $Directory 'age/age.exe'
$pull = & $Python $runner pull --directory $Directory --host $Server --key $SshKey
if ($LASTEXITCODE -ne 0) { throw 'Encrypted recovery pull failed.' }
$state = $pull | ConvertFrom-Json
$revision = $state.revision
if ($revision -notmatch '^[a-f0-9]{40}$') { throw 'Unexpected recovery revision.' }
$archive = Join-Path $Directory "paper-recovery-$revision.tar.gz.age"
$receipt = Join-Path $Directory "verified-$revision.json"
if (Test-Path -LiteralPath $receipt) {
    $proof = Get-Content -Raw -LiteralPath $receipt | ConvertFrom-Json
    if ($proof.verified -ne $true -or $proof.revision -ne $revision) { throw 'Invalid existing recovery receipt.' }
    if ($proof.PSObject.Properties.Name -contains 'snapshot_at' -and $proof.snapshot_at) {
        return
    }
} else {
    $locked = [IO.File]::ReadAllBytes((Join-Path $Directory 'restore-identity.dpapi'))
    $plain = [Security.Cryptography.ProtectedData]::Unprotect(
        $locked, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    try {
        $result = [Text.Encoding]::ASCII.GetString($plain).TrimEnd() |
            & $Python $runner verify --archive $archive --age $age --scratch $Directory --revision $revision
        if ($LASTEXITCODE -ne 0) { throw 'Isolated restore failed.' }
        $proof = $result | ConvertFrom-Json
        if ($proof.verified -ne $true -or $proof.revision -ne $revision) { throw 'Invalid restore result.' }
        $proof | Add-Member -NotePropertyName verified_at -NotePropertyValue ([DateTimeOffset]::UtcNow.ToString('o'))
        $proof | Add-Member -NotePropertyName encrypted_sha256 -NotePropertyValue ((Get-FileHash -LiteralPath $archive).Hash.ToLowerInvariant())
    } finally {
        [Array]::Clear($plain, 0, $plain.Length)
    }
}
$proof | Add-Member -NotePropertyName snapshot_at -NotePropertyValue $state.snapshot_at -Force
$sshArgs = @('-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes','-o','IdentitiesOnly=yes','-i',$SshKey)
$temp = Join-Path $Directory "receipt-$([guid]::NewGuid()).json"
try {
    $proof | ConvertTo-Json -Compress | Set-Content -LiteralPath $temp -Encoding ascii
    $remote = "/home/ubuntu/hyperliquid-vps-releases/offhost-$revision.json"
    scp.exe @sshArgs $temp "${Server}:$remote"
    if ($LASTEXITCODE -ne 0) { throw 'Could not transfer the restore receipt.' }
    ssh.exe @sshArgs $Server "sudo -n install -m 0600 $remote /var/lib/hyperliquid-dashboard/offhost-restore.json"
    if ($LASTEXITCODE -ne 0) { throw 'Could not publish the restore receipt.' }
    Move-Item -LiteralPath $temp -Destination $receipt -Force
    $proof | ConvertTo-Json -Compress
} finally {
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp }
}

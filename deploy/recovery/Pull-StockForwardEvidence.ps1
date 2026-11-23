param(
    [Parameter(Mandatory)][string]$Server,
    [Parameter(Mandatory)][string]$SshKey,
    [Parameter(Mandatory)][string]$Directory,
    [string]$Python = "python"
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $Directory 'stock_forward_evidence.py'
$age = Join-Path $Directory 'age/age.exe'
$archive = & $Python $runner fetch --host $Server --key $SshKey --directory $Directory
if ($LASTEXITCODE -ne 0 -or !$archive) { throw 'Stock forward checkpoint transfer failed.' }
$temp = Join-Path $Directory "receipt-stock-$([guid]::NewGuid().ToString('N')).json"
try {
    if ($PSVersionTable.PSEdition -eq 'Desktop') { Add-Type -AssemblyName System.Security }
    $locked = [IO.File]::ReadAllBytes((Join-Path $Directory 'restore-identity.dpapi'))
    $plain = [Security.Cryptography.ProtectedData]::Unprotect(
        $locked, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    try {
        $result = [Text.Encoding]::ASCII.GetString($plain).TrimEnd() |
            & $Python $runner verify --archive $archive --age $age --scratch $Directory
        if ($LASTEXITCODE -ne 0) { throw 'Stock forward isolated restore failed.' }
        $proof = $result | ConvertFrom-Json
    } finally {
        [Array]::Clear($plain, 0, $plain.Length)
    }
    if ($proof.verified -ne $true -or $proof.protocol_sha256 -notmatch '^[a-f0-9]{64}$') { throw 'Invalid restore proof.' }
    $sha = (Get-FileHash -LiteralPath $archive).Hash.ToLowerInvariant()
    $proof | Add-Member -NotePropertyName encrypted_sha256 -NotePropertyValue $sha
    $proof | Add-Member -NotePropertyName verified_at -NotePropertyValue ([DateTimeOffset]::UtcNow.ToString('o'))
    Move-Item -LiteralPath $archive -Destination (Join-Path $Directory "stock-forward-$sha.zip.age")
    $proof | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $temp -Encoding ascii
    $sshArgs = @('-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes','-o','IdentitiesOnly=yes','-i',$SshKey)
    $remote = "/home/ubuntu/hyperliquid-vps-releases/stock-forward-proof-$($sha.Substring(0,16)).json"
    scp.exe @sshArgs $temp "${Server}:$remote"
    if ($LASTEXITCODE -ne 0) { throw 'Could not transfer stock restore proof.' }
    ssh.exe @sshArgs $Server "sudo -n install -m 0600 $remote /var/lib/hyperliquid-dashboard/stock-forward-restore.json && rm -- $remote"
    if ($LASTEXITCODE -ne 0) { throw 'Could not publish stock restore proof.' }
    Move-Item -LiteralPath $temp -Destination (Join-Path $Directory 'stock-forward-latest.json') -Force
    $proof | ConvertTo-Json -Depth 5 -Compress
} finally {
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive }
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp }
}

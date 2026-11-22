param(
    [string]$LogDirectory = "$env:LOCALAPPDATA\HyperliquidDashboard"
)

$ErrorActionPreference = 'Stop'
$ssh = Join-Path $env:SystemRoot 'System32\OpenSSH\ssh.exe'
$null = New-Item -ItemType Directory -Force -Path $LogDirectory
$log = Join-Path $LogDirectory 'tunnel.log'

function Write-TunnelLog([string]$Message) {
    if ((Test-Path -LiteralPath $log) -and (Get-Item -LiteralPath $log).Length -gt 1048576) {
        Get-Content -LiteralPath $log -Tail 100 | Set-Content -LiteralPath ($log + '.previous')
        Remove-Item -LiteralPath $log
    }
    Add-Content -LiteralPath $log -Value ((Get-Date).ToUniversalTime().ToString('o') + ' ' + $Message)
}

Write-TunnelLog 'Watcher started; using the existing restricted hl-dashboard SSH profile.'
while ($true) {
    # Leave an existing listener alone, including one the user started manually.
    $listener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 18080 -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        Start-Sleep -Seconds 10
        continue
    }
    Write-TunnelLog 'Starting private dashboard connection.'
    try {
        & $ssh -N -T -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes `
            -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 hl-dashboard 2>> $log
        Write-TunnelLog "Connection ended with exit code $LASTEXITCODE; retrying in 15 seconds."
    } catch {
        Write-TunnelLog 'SSH could not connect; check tunnel.log and the existing SSH profile.'
    }
    Start-Sleep -Seconds 15
}

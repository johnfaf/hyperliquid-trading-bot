$ErrorActionPreference = 'Stop'
$taskName = 'Hyperliquid Private Dashboard Tunnel'
$directory = Join-Path $env:LOCALAPPDATA 'HyperliquidDashboard'
$null = New-Item -ItemType Directory -Force -Path $directory
$watcher = Join-Path $directory 'Watch-PrivateDashboard.ps1'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Watch-PrivateDashboard.ps1') -Destination $watcher -Force
$account = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$arguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File "' + $watcher + '"'
$action = New-ScheduledTaskAction -Execute $powershell -Argument $arguments
$login = New-ScheduledTaskTrigger -AtLogOn -User $account
# An abnormal termination does not always qualify for Task Scheduler's restart-on-failure.
$retry = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 2)
$retry.Repetition.StopAtDurationEnd = $false
$principal = New-ScheduledTaskPrincipal -UserId $account -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$task = New-ScheduledTask -Action $action -Trigger @($login, $retry) -Principal $principal -Settings $settings `
    -Description 'Keep the existing restricted localhost-only dashboard SSH connection available while signed in.'
$null = Register-ScheduledTask -TaskName $taskName -InputObject $task -Force
Start-ScheduledTask -TaskName $taskName
Write-Output 'Private tunnel watcher installed for this Windows login. Dashboard: http://127.0.0.1:18080/'

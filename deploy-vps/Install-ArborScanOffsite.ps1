param([string]$Root = 'D:\ArborScanBackups')
$ErrorActionPreference = 'Stop'
$taskName = 'ArborScan Offsite Backup'
$python = (Get-Command pythonw.exe -ErrorAction Stop).Source
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.Description -ne 'Pull completed ArborScan backups over verified SSH; SHA256; no deletion') {
    throw 'Existing unrelated task preserved; inspect its configuration.'
}
if (-not (Test-Path -LiteralPath $Root)) { throw 'Create and secure the backup directory first.' }
$acl = Get-Acl -LiteralPath $Root
if (-not $acl.AreAccessRulesProtected) { throw 'Backup root must have private protected ACLs.' }
foreach ($rule in $acl.Access) {
    $sid = $rule.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
    if ($rule.AccessControlType -eq 'Allow' -and $sid -notin @($identity.User.Value, 'S-1-5-18')) {
        throw 'Unexpected account has access to backup directory.'
    }
}
$tools = Join-Path $Root 'tools'
New-Item -ItemType Directory -Force -Path $tools | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'ops_pull_backups.py'),(Join-Path $PSScriptRoot 'ops_verify_offsite.py'),(Join-Path $PSScriptRoot 'ops_windows_job.py'),(Join-Path $PSScriptRoot 'ops_delta_copy.py') -Destination $tools
$action = New-ScheduledTaskAction -Execute $python -Argument ('-B "' + (Join-Path $tools 'ops_pull_backups.py') + '" --root "' + $Root + '"') -WorkingDirectory $tools
$trigger = @((New-ScheduledTaskTrigger -Daily -At '08:00'), (New-ScheduledTaskTrigger -AtLogOn -User $identity.Name))
$principal = New-ScheduledTaskPrincipal -UserId $identity.Name -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 3) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Pull completed ArborScan backups over verified SSH; SHA256; no deletion' -Force | Select-Object TaskName,State
# Interactive token: works in Task Scheduler while the user is logged on (also
# when locked). No password is stored. Logged-off/offline periods are unprotected.

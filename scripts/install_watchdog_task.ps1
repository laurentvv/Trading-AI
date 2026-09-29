<#
.SYNOPSIS
  Installe (ou retire) la tâche planifiée Windows du watchdog Trading-AI.

.DESCRIPTION
  Lance `watchdog.py` toutes les N minutes (défaut 15) avec le Python du projet. Le watchdog détecte un
  scheduler mort ou bloqué, une position sans stop broker ou une rafale d'erreurs, alerte (ntfy/Telegram,
  voir .env) et, sans -NoRestart, relance `start_scheduler.bat`.

  La tâche s'exécute sous le compte courant, session ouverte uniquement (la relance ouvre la fenêtre
  « Trading AI - Scheduler » sur le bureau). NE PAS l'installer si tu arrêtes volontairement le scheduler :
  fais d'abord `python watchdog.py --pause` (et `--resume` ensuite).

.EXAMPLE
  .\scripts\install_watchdog_task.ps1                 # installe, relance automatique activée
  .\scripts\install_watchdog_task.ps1 -NoRestart      # alertes seulement
  .\scripts\install_watchdog_task.ps1 -Remove         # retire la tâche
#>
param(
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [int]$EveryMinutes = 15,
    [switch]$NoRestart,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$taskName = "TradingAI-Watchdog"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Tâche '$taskName' retirée."
    return
}

$python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python du projet introuvable : $python (lance d'abord 'uv sync')." }
if (-not (Test-Path (Join-Path $Repo "watchdog.py"))) { throw "watchdog.py introuvable dans $Repo." }

$arguments = "watchdog.py --base `"$Repo`""
if (-not $NoRestart) { $arguments += " --restart" }

$action   = New-ScheduledTaskAction -Execute $python -Argument $arguments -WorkingDirectory $Repo
$trigger  = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
              -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
              -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Trading-AI : surveillance du scheduler, des stops broker et des erreurs (watchdog.py)." -Force | Out-Null

Write-Host "Tâche '$taskName' installée : toutes les $EveryMinutes min, relance automatique = $(-not $NoRestart)."
Write-Host "Vérifier les canaux d'alerte : $python watchdog.py --status"

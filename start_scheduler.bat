@echo off
chcp 65001 >nul
REM =====================================================
REM Script de Démarrage du Scheduler Trading AI
REM Démarre le cycle de trading autonome (8h30 - 18h00)
REM
REM GO-gate 6 (audit 2026-08-19 I2) : boucle de relance.
REM Toute exception non gérée tuait auparavant le scheduler
REM sans aucune relance (positions réelles sans surveillance).
REM Ce script relance schedule.py après 30s en cas de crash,
REM jusqu'à un Ctrl+C dans cette fenêtre. Un verrou d'instance
REM (scheduler.lock) empêche les doublons (GO-gate 6 / I1).
REM =====================================================

echo.
echo ======================================================
echo 📈 Trading AI - Live Scheduler
echo Démarrage du cycle de trading autonome (Nasdaq/WTI)
echo ======================================================
echo.

REM Vérifier si uv est installé
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ ERREUR: 'uv' n'est pas installé ou n'est pas dans le PATH.
    echo Veuillez l'installer depuis https://astral.sh/uv
    pause
    exit /b 1
)

echo.
echo 📅 Planification Active (Lundi - Vendredi) :
echo   - Fenêtre : 08:30 à 18:00 (Marché Ouvert)
echo   - Intervalle : Toutes les 30 minutes
echo   - Brief matinal : catch-up dès 01:00 si manqué
echo   - Mode : Trading 212 (DEMO/LIVE selon .env.t212)
echo.
echo 🛑 Pour arrêter le scheduler: Ctrl+C dans cette fenêtre
echo 🔁 Relance automatique après crash (30s de délai)
echo 📝 Logs : scheduler.log et trading.log
echo.

set RESTART_DELAY=30

:loop
echo [%date% %time%] Lancement du scheduler...
uv run schedule.py
if %ERRORLEVEL% equ 0 goto stopped
echo.
echo ======================================================
echo [%date% %time%] Scheduler CRASHÉ (code %ERRORLEVEL%).
echo Relance dans %RESTART_DELAY%s (Ctrl+C pour arrêter définitivement).
echo ======================================================
timeout /t %RESTART_DELAY% /nobreak >nul
goto loop

:stopped
echo.
echo ======================================================
echo [%date% %time%] Arrêt propre du scheduler (code 0) — superviseur terminé.
echo ======================================================
pause

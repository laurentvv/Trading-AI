@echo off
chcp 65001 >nul
REM =====================================================
REM Script de Démarrage du Scheduler Trading AI (Gemma 4)
REM Démarre le cycle de trading autonome (8h30 - 18h00)
REM =====================================================

echo.
echo ======================================================
echo 📈 Trading AI - Live Scheduler (Gemma 4)
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
echo   - Modèle : Gemma 4:e4b + AlphaEar News
echo   - Mode : Trading 212 (DEMO/REEL selon .env)
echo.
echo 🛑 Pour arrêter le scheduler: Ctrl+C dans cette fenêtre
echo 📝 Les logs sont disponibles dans: scheduler.log et trading.log
echo.

REM Démarrer le scheduler via uv
uv run schedule.py

REM Si on arrive ici, le scheduler s'est arrêté
echo.
echo ======================================================
echo Scheduler Trading AI arrêté
echo ======================================================

pause

@echo off
chcp 65001 >nul
REM =====================================================
REM Script de Demarrage du Scheduler Intelligent Trading AI
REM Demarre le systeme de supervision automatique
REM =====================================================

echo.
echo ======================================================
echo Trading AI - Scheduler Intelligent
echo Demarrage du systeme de supervision automatique
echo ======================================================
echo.



REM Activer l'environnement virtuel
echo Activation de l'environnement virtuel...
call .venv\Scripts\activate

REM Verifier que l'environnement est active
if defined VIRTUAL_ENV (
    echo ✅ Environnement virtuel active: %VIRTUAL_ENV%
) else (
    echo ❌ ERREUR: Impossible d'activer l'environnement virtuel
    echo Verifiez que le .venv existe dans %CD%
    pause
    exit /b 1
)

echo.
echo 📅 Demarrage du Scheduler Intelligent...
echo ⏰ Le systeme va automatiquement:
echo   - Executer l'analyse quotidienne a 18h00
echo   - Generer des rapports hebdomadaires le vendredi
echo   - Effectuer des evaluations de phase automatiques
echo   - Gerer les transitions entre phases selon le planning
echo.
echo 🛑 Pour arreter le scheduler: Ctrl+C
echo 📊 Pour voir le statut: python src/intelligent_scheduler.py --status
echo.

REM Afficher le repertoire courant
echo Repertoire courant: %CD%

REM Demarrer le scheduler intelligent
python src/intelligent_scheduler.py

REM Si on arrive ici, le scheduler s'est arrete
echo.
echo ======================================================
echo Scheduler Intelligent arrete
echo ======================================================

pause
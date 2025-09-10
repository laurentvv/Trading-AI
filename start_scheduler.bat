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

REM Vérifier que l'environnement est activé
if defined VIRTUAL_ENV (
    echo ✅ Environnement virtuel activé: %VIRTUAL_ENV%
) else (
    echo ❌ ERREUR: Impossible d'activer l'environnement virtuel
    echo Vérifiez que le .venv existe dans %CD%
    pause
    exit /b 1
)

echo.
echo 📅 Démarrage du Scheduler Intelligent...
echo ⏰ Le système va automatiquement:
echo   - Exécuter l'analyse quotidienne à 18h00
echo   - Générer des rapports hebdomadaires le vendredi
echo   - Effectuer des évaluations de phase automatiques
echo   - Gérer les transitions entre phases selon le planning
echo.
echo 🛑 Pour arrêter le scheduler: Ctrl+C
echo 📊 Pour voir le statut: python src/intelligent_scheduler.py --status
echo.

REM Démarrer le scheduler intelligent
python src/intelligent_scheduler.py

REM Si on arrive ici, le scheduler s'est arrêté
echo.
echo ======================================================
echo Scheduler Intelligent arrêté
echo ======================================================

pause
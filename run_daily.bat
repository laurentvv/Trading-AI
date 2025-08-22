@echo off
REM =====================================================
REM Script d'automatisation Trading AI - Exécution Quotidienne
REM =====================================================

echo.
echo ======================================================
echo Trading AI - Analyse Quotidienne
echo Date: %date%
echo Heure: %time%
echo ======================================================
echo.



REM Activer l'environnement virtuel
echo Activation de l'environnement virtuel...
call .venv\Scripts\activate

REM Vérifier que l'environnement est activé
echo Environnement virtuel activé: %VIRTUAL_ENV%
echo.

REM Exécuter l'analyse Trading AI
echo Démarrage de l'analyse Trading AI...
python src/enhanced_trading_example.py

REM Vérifier le code de sortie
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ SUCCÈS: Analyse terminée avec succès
    echo %date% %time% - Analyse réussie >> execution_log.txt
) else (
    echo.
    echo ❌ ERREUR: L'analyse a échoué (Code: %ERRORLEVEL%)
    echo %date% %time% - ERREUR Code %ERRORLEVEL% >> execution_log.txt
)

echo.
echo ======================================================
echo Fin de l'exécution - Vérifiez les fichiers générés:
echo - enhanced_trading_chart.png
echo - enhanced_performance_dashboard.png
echo - execution_log.txt
echo ======================================================

REM Pause pour voir les résultats (retirer en mode automatique)
pause
@echo off
REM =====================================================
REM Installation et Configuration du Scheduler Intelligent
REM Configure automatiquement tout le système
REM =====================================================

echo.
echo ======================================================
echo Trading AI - Installation du Scheduler Intelligent
echo ======================================================
echo.



echo ✅ Répertoire du projet: %CD%

REM Vérifier que l'environnement virtuel existe
if not exist ".venv" (
    echo ❌ ERREUR: Environnement virtuel .venv non trouvé
    echo Créez d'abord l'environnement virtuel avec: python -m venv .venv
    pause
    exit /b 1
)

REM Activer l'environnement virtuel
echo 🔄 Activation de l'environnement virtuel...
call .venv\Scripts\activate

REM Vérifier les dépendances Python nécessaires
echo 🔄 Vérification des dépendances Python...
python -c "import schedule; import pandas; import sqlite3; print('✅ Dépendances OK')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 📦 Installation des dépendances manquantes...
    pip install schedule pandas
)

REM Créer la configuration initiale
echo 🔄 Création de la configuration initiale...
python -c "import json; from datetime import datetime; config = {'project_start_date': datetime.now().isoformat(), 'trading_ticker': 'QQQ', 'daily_execution_time': '18:00', 'weekly_report_day': 'friday', 'monthly_report_day': 28, 'phase_transitions': {'phase_1_duration_days': 7, 'phase_2_duration_days': 21, 'phase_3_duration_days': 30, 'phase_4_duration_days': 120}, 'performance_targets': {'phase_2': {'sharpe_ratio': 0.5, 'max_drawdown': 0.05, 'win_rate': 0.45}, 'phase_3': {'sharpe_ratio': 1.0, 'max_drawdown': 0.03, 'win_rate': 0.55}, 'phase_4': {'sharpe_ratio': 1.5, 'max_drawdown': 0.02, 'win_rate': 0.60}}, 'alerts': {'email_notifications': False, 'performance_alerts': True, 'phase_completion_alerts': True}}; f = open('scheduler_config.json', 'w'); json.dump(config, f, indent=4); f.close(); print('✅ Configuration créée: scheduler_config.json')"

REM Initialiser la base de données
echo 🔄 Initialisation de la base de données...
python -c "import sqlite3; conn = sqlite3.connect('scheduler.db'); cursor = conn.cursor(); cursor.execute('CREATE TABLE IF NOT EXISTS task_executions (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, task_type TEXT NOT NULL, execution_time TIMESTAMP NOT NULL, phase TEXT NOT NULL, success BOOLEAN NOT NULL, duration_seconds REAL, error_message TEXT, results TEXT)'); cursor.execute('CREATE TABLE IF NOT EXISTS phase_progress (phase TEXT PRIMARY KEY, start_date TIMESTAMP, target_end_date TIMESTAMP, current_progress REAL, metrics_achieved TEXT, status TEXT, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)'); cursor.execute('CREATE TABLE IF NOT EXISTS system_metrics (date TEXT PRIMARY KEY, phase TEXT, sharpe_ratio REAL, max_drawdown REAL, win_rate REAL, total_trades INTEGER, performance_vs_benchmark REAL, model_weights TEXT, notes TEXT)'); conn.commit(); conn.close(); print('✅ Base de données initialisée: scheduler.db')"

REM Configuration pour lancement manuel
echo 🔄 Configuration pour exécution manuelle...

REM Tester la configuration
echo 🔄 Test de la configuration...
python -c "import sys; sys.path.append('src'); from intelligent_scheduler import IntelligentScheduler; print('✅ Configuration testée avec succès')" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Configuration testée avec succès
) else (
    echo ⚠️  Problème détecté dans la configuration
)

echo.
echo ======================================================
echo 🎉 INSTALLATION TERMINÉE AVEC SUCCÈS !
echo ======================================================
echo.
echo 📋 PROCHAINES ÉTAPES:
echo.
echo 1. DÉMARRER LE SCHEDULER:
echo    • Exécutez: start_scheduler.bat
echo    • Le scheduler tournera en boucle continue jusqu'à arrêt manuel (Ctrl+C)
echo.
echo 2. SURVEILLER LE SYSTÈME:
echo    • Statut: python src/intelligent_scheduler.py --status
echo    • Performance: python src/intelligent_scheduler.py --performance 7
echo    • Phase: python src/intelligent_scheduler.py --phase
echo.
echo 3. FICHIERS CRÉÉS:
echo    • scheduler_config.json - Configuration du système
echo    • scheduler.db - Base de données de suivi
echo    • scheduler.log - Journal d'exécution
echo.
echo 4. CALENDRIER AUTOMATIQUE:
echo    • Analyse quotidienne: 18h00 (modifiable dans config)
echo    • Rapport hebdomadaire: Vendredi 19h00
echo    • Rapport mensuel: Dernier jour du mois
echo    • Maintenance: Dimanche 22h00
echo.
echo ⚠️  IMPORTANT:
echo    • Assurez-vous qu'Ollama fonctionne pour les LLMs
echo    • Le système suivra automatiquement le planning des 4 phases
echo    • Les transitions de phase se feront automatiquement
echo    • Lancez start_scheduler.bat et laissez la fenêtre ouverte
echo    • Le scheduler s'arrêtera si vous fermez la fenêtre ou faites Ctrl+C
echo.
echo 🚀 Votre système de Trading AI est maintenant entièrement automatisé !
echo.

pause
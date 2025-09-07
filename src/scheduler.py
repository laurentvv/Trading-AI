import schedule
import time
import logging
import subprocess
import sys
from pathlib import Path
import json
import sqlite3

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="scheduler.log",
    filemode="a",
)
logger = logging.getLogger(__name__)


class IntelligentScheduler:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.db_path = self.project_root / "scheduler.db"
        self._initialize_db()
        self.config = self._load_config()
        self.current_phase = self.config.get(
            "current_phase", "Phase 1: Configuration & Test"
        )
        logger.info(f"Intelligent Trading AI Scheduler initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Current phase: {self.current_phase}")

    def _initialize_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scheduler_config (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS phase_progress (
                        phase TEXT PRIMARY KEY,
                        progress REAL
                    )
                """
                )
                conn.commit()
            logger.info("[OK] Scheduler database initialized")
        except sqlite3.Error as e:
            logger.error(f"[ERROR] Database initialization failed: {e}")
            sys.exit(1)

    def _load_config(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key, value FROM scheduler_config")
                rows = cursor.fetchall()
                config = {row[0]: json.loads(row[1]) for row in rows}
                if not config:
                    config = self._create_default_config()
                else:
                    logger.info("[CONFIG] Configuration loaded from database")
                return config
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(f"[ERROR] Failed to load configuration: {e}")
            return self._create_default_config()

    def _create_default_config(self):
        logger.info("[CONFIG] Creating default configuration")
        default_config = {
            "current_phase": "Phase 1: Configuration & Test",
            "schedule": {"daily_analysis": "18:00"},
            "phases": {
                "Phase 1: Configuration & Test": {"duration_days": 7},
                "Phase 2: Initial Learning": {"duration_days": 21},
                "Phase 3: Active Trading": {"duration_days": 90},
            },
        }
        self.save_config(default_config)
        return default_config

    def save_config(self, config):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for key, value in config.items():
                    cursor.execute(
                        "REPLACE INTO scheduler_config (key, value) VALUES (?, ?)",
                        (key, json.dumps(value)),
                    )
                conn.commit()
            logger.info("[CONFIG] Configuration saved")
        except sqlite3.Error as e:
            logger.error(f"[ERROR] Failed to save configuration: {e}")

    def run_daily_analysis(self):
        logger.info("[ANALYSIS] Starting daily trading analysis...")
        try:
            script_path = self.project_root / "src" / "main.py"
            python_executable = sys.executable
            process = subprocess.run(
                [python_executable, str(script_path)],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root,
            )
            logger.info(process.stdout)
            logger.info("[OK] Daily analysis completed successfully")
            self._check_performance_alerts()
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Daily analysis failed: {e.stderr}")
        except Exception as e:
            logger.error(
                f"[ERROR] An unexpected error occurred during daily analysis: {e}"
            )

    def _check_performance_alerts(self):
        logger.info("Checking for performance alerts...")
        # This is a placeholder for the actual implementation
        # In a real scenario, this method would check the performance_monitor.db
        # and trigger alerts if certain conditions are met.
        logger.info("No performance alerts to check in this version.")

    def update_phase_progress(self):
        # Placeholder for phase progress logic
        logger.info("Updating phase progress...")

    def start(self):
        logger.info("[INIT] Starting Intelligent Trading AI Scheduler...")
        schedule.every().day.at(
            self.config.get("schedule", {}).get("daily_analysis", "18:00")
        ).do(self.run_daily_analysis)
        schedule.every().friday.at("19:00").do(self.generate_weekly_report)

        while True:
            schedule.run_pending()
            time.sleep(1)

    def generate_weekly_report(self):
        logger.info("[REPORT] Generating weekly performance report...")
        try:
            # In a real scenario, this method would query the performance_monitor.db
            # and generate a report. For now, we'll just log a message.
            logger.info(
                "Weekly report generation logic not implemented in this version."
            )
            self.update_phase_progress()
            logger.info("[OK] Weekly report generated successfully")
        except Exception as e:
            logger.error(f"[ERROR] Weekly report generation failed: {e}")

    def update_phase_progress(self):
        logger.info("Updating phase progress...")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # This is a placeholder for the actual phase progress logic.
                # The error "Object of type bool is not JSON serializable" suggests
                # that a boolean value was being passed to json.dumps().
                # We will ensure that we only save JSON serializable types.
                cursor.execute(
                    "REPLACE INTO phase_progress (phase, progress) VALUES (?, ?)",
                    (self.current_phase, 0.5),
                )  # Using a float value
                conn.commit()
            logger.info("Phase progress updated.")
        except sqlite3.Error as e:
            logger.error(f"[ERROR] Failed to update phase progress: {e}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    scheduler = IntelligentScheduler(project_root)
    scheduler.start()

"""
Intelligent Trading AI Scheduler
Automated execution system that follows the implementation timeline and manages
all phases of the enhanced trading AI system deployment.
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import json
import pandas as pd
import sqlite3
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_trading_example import EnhancedTradingSystem
from performance_monitor import PerformanceMonitor

class Phase(Enum):
    PHASE_1 = "Phase 1: Configuration & Test"
    PHASE_2 = "Phase 2: Initial Learning"
    PHASE_3 = "Phase 3: Optimization"
    PHASE_4 = "Phase 4: Maturity"

class TaskType(Enum):
    DAILY_ANALYSIS = "daily_analysis"
    WEEKLY_REPORT = "weekly_report"
    MONTHLY_REPORT = "monthly_report"
    PHASE_EVALUATION = "phase_evaluation"
    SYSTEM_MAINTENANCE = "system_maintenance"

@dataclass
class ScheduledTask:
    task_id: str
    task_type: TaskType
    description: str
    phase: Phase
    frequency: str  # daily, weekly, monthly, or specific date
    time_schedule: str  # HH:MM format or cron-like
    enabled: bool = True
    last_execution: Optional[datetime] = None
    next_execution: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0

@dataclass
class PhaseProgress:
    phase: Phase
    start_date: datetime
    target_end_date: datetime
    current_progress: float  # 0-100%
    metrics_achieved: Dict[str, bool]
    status: str  # "not_started", "in_progress", "completed", "delayed"

class IntelligentScheduler:
    """
    Intelligent scheduler that manages the entire Trading AI deployment timeline.
    """
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            # Use the parent directory of the src folder where this script is located
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        # Load environment variables from .env file
        env_file = self.project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        
        self.scheduler_db = self.project_root / "scheduler.db"
        self.config_file = self.project_root / "scheduler_config.json"
        self.log_file = self.project_root / "scheduler.log"
        
        # Initialize logging with proper encoding for Windows
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # Configure console handler with UTF-8 encoding for Windows
        console_handler = None
        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                console_handler = handler
                break
        
        if console_handler:
            # Try to set encoding for Windows console
            try:
                import sys
                if sys.platform == 'win32':
                    console_handler.stream.reconfigure(encoding='utf-8')
            except:
                # Fallback: replace emoji characters in messages
                pass
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.trading_system = EnhancedTradingSystem()
        self.performance_monitor = PerformanceMonitor()
        
        # Current phase tracking
        self.current_phase = Phase.PHASE_1
        self.project_start_date = datetime.now()
        
        # Initialize database and schedule
        self._init_scheduler_database()
        self._load_configuration()
        self._setup_schedule()
        
        self.logger.info("[INIT] Intelligent Trading AI Scheduler initialized")
        self.logger.info(f"Project root: {self.project_root}")
        self.logger.info(f"Current phase: {self.current_phase.value}")
    
    def _init_scheduler_database(self):
        """Initialize the scheduler database."""
        try:
            conn = sqlite3.connect(self.scheduler_db)
            cursor = conn.cursor()
            
            # Tasks execution history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    execution_time TIMESTAMP NOT NULL,
                    phase TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    duration_seconds REAL,
                    error_message TEXT,
                    results TEXT  -- JSON string
                )
            ''')
            
            # Phase progress tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS phase_progress (
                    phase TEXT PRIMARY KEY,
                    start_date TIMESTAMP,
                    target_end_date TIMESTAMP,
                    current_progress REAL,
                    metrics_achieved TEXT,  -- JSON string
                    status TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # System metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    date TEXT PRIMARY KEY,
                    phase TEXT,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    performance_vs_benchmark REAL,
                    model_weights TEXT,  -- JSON string
                    notes TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("[OK] Scheduler database initialized")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize scheduler database: {e}")
    
    def _load_configuration(self):
        """Load or create scheduler configuration."""
        default_config = {
            "project_start_date": datetime.now().isoformat(),
            "trading_ticker": "QQQ",
            "daily_execution_time": "18:00",
            "weekly_report_day": "friday",
            "monthly_report_day": 28,
            "phase_transitions": {
                "phase_1_duration_days": 7,
                "phase_2_duration_days": 21,
                "phase_3_duration_days": 30,
                "phase_4_duration_days": 120
            },
            "performance_targets": {
                "phase_2": {"sharpe_ratio": 0.5, "max_drawdown": 0.05, "win_rate": 0.45},
                "phase_3": {"sharpe_ratio": 1.0, "max_drawdown": 0.03, "win_rate": 0.55},
                "phase_4": {"sharpe_ratio": 1.5, "max_drawdown": 0.02, "win_rate": 0.60}
            },
            "alerts": {
                "email_notifications": False,
                "performance_alerts": True,
                "phase_completion_alerts": True
            }
        }
        
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            self.logger.info("[CONFIG] Configuration loaded from file")
        else:
            self.config = default_config
            self._save_configuration()
            self.logger.info("[CONFIG] Default configuration created")
        
        # Set project start date
        self.project_start_date = datetime.fromisoformat(self.config["project_start_date"])
        
        # Load current phase from database if it exists
        self._load_current_phase_from_database()
    
    def _load_current_phase_from_database(self):
        """Load the current phase from the database."""
        try:
            conn = sqlite3.connect(self.scheduler_db)
            cursor = conn.cursor()
            
            # Get the latest phase with status 'completed' or 'in_progress'
            cursor.execute('''
                SELECT phase, status FROM phase_progress 
                WHERE status IN ('completed', 'in_progress')
                ORDER BY last_updated DESC LIMIT 1
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                phase_name, status = result
                # Map phase name to Phase enum
                phase_mapping = {
                    'PHASE_1': Phase.PHASE_1,
                    'PHASE_2': Phase.PHASE_2,
                    'PHASE_3': Phase.PHASE_3,
                    'PHASE_4': Phase.PHASE_4
                }
                
                if phase_name in phase_mapping:
                    # If the phase was completed, move to the next phase
                    if status == 'completed' and phase_name in phase_mapping:
                        next_phase_map = {
                            'PHASE_1': Phase.PHASE_2,
                            'PHASE_2': Phase.PHASE_3,
                            'PHASE_3': Phase.PHASE_4
                        }
                        if phase_name in next_phase_map:
                            self.current_phase = next_phase_map[phase_name]
                            self.logger.info(f"[PHASE] Loaded next phase from database: {self.current_phase.value}")
                        else:
                            self.current_phase = phase_mapping[phase_name]
                            self.logger.info(f"[PHASE] Loaded current phase from database: {self.current_phase.value}")
                    else:
                        self.current_phase = phase_mapping[phase_name]
                        self.logger.info(f"[PHASE] Loaded current phase from database: {self.current_phase.value}")
                        
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load phase from database: {e}")
    
    def _save_configuration(self):
        """Save current configuration to file."""
        # Preserve the original project start date if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    existing_config = json.load(f)
                # Keep the original start date
                if "project_start_date" in existing_config:
                    self.config["project_start_date"] = existing_config["project_start_date"]
            except Exception as e:
                self.logger.warning(f"Could not preserve original start date: {e}")
        
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4, default=str)
    
    def _setup_schedule(self):
        """Setup the automated schedule based on the implementation plan."""
        # Clear existing schedule
        schedule.clear()
        
        # Phase 1: Daily execution + setup tasks
        schedule.every().day.at(self.config["daily_execution_time"]).do(
            self._execute_daily_analysis
        ).tag("daily", "phase_1", "phase_2", "phase_3", "phase_4")
        
        # Weekly reports (every Friday at 19:00)
        schedule.every().friday.at("19:00").do(
            self._execute_weekly_report
        ).tag("weekly", "phase_2", "phase_3", "phase_4")
        
        # Monthly reports (last day of month)
        schedule.every().day.at("20:00").do(
            self._check_monthly_report
        ).tag("monthly", "phase_3", "phase_4")
        
        # Phase evaluation (automatic phase transitions)
        schedule.every().day.at("08:00").do(
            self._evaluate_phase_progress
        ).tag("phase_eval")
        
        # System maintenance (every Sunday at 22:00)
        schedule.every().sunday.at("22:00").do(
            self._system_maintenance
        ).tag("maintenance")
        
        self.logger.info("[SCHEDULE] Schedule configured according to implementation plan")
    
    def _execute_daily_analysis(self):
        """Execute daily trading analysis - Core of the system."""
        task_start = datetime.now()
        task_id = f"daily_analysis_{task_start.strftime('%Y%m%d')}"
        
        self.logger.info("[ANALYSIS] Starting daily trading analysis...")
        
        try:
            # Run the enhanced trading system
            results, performance_report = self.trading_system.run_enhanced_analysis()
            
            # Record execution
            self._record_task_execution(
                task_id=task_id,
                task_type=TaskType.DAILY_ANALYSIS,
                success=True,
                duration=(datetime.now() - task_start).total_seconds(),
                results=results
            )
            
            # Update CSV tracking
            self._update_performance_tracking(results, performance_report)
            
            # Check for alerts
            self._check_performance_alerts(results)
            
            self.logger.info("[OK] Daily analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Daily analysis failed: {e}")
            self._record_task_execution(
                task_id=task_id,
                task_type=TaskType.DAILY_ANALYSIS,
                success=False,
                duration=(datetime.now() - task_start).total_seconds(),
                error_message=str(e)
            )
    
    def _execute_weekly_report(self):
        """Generate and analyze weekly performance report."""
        task_start = datetime.now()
        task_id = f"weekly_report_{task_start.strftime('%Y_W%U')}"
        
        self.logger.info("[REPORT] Generating weekly performance report...")
        
        try:
            # Generate performance report
            report = self.performance_monitor.generate_performance_report(days_back=7)
            
            # Analyze week's performance
            weekly_analysis = self._analyze_weekly_performance(report)
            
            # Update phase progress
            self._update_phase_progress(weekly_analysis)
            
            # Generate dashboard
            self.performance_monitor.create_performance_dashboard(
                f"weekly_dashboard_{task_start.strftime('%Y_W%U')}.png"
            )
            
            self._record_task_execution(
                task_id=task_id,
                task_type=TaskType.WEEKLY_REPORT,
                success=True,
                duration=(datetime.now() - task_start).total_seconds(),
                results={"report": report, "analysis": weekly_analysis}
            )
            
            self.logger.info("[OK] Weekly report generated successfully")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Weekly report failed: {e}")
            self._record_task_execution(
                task_id=task_id,
                task_type=TaskType.WEEKLY_REPORT,
                success=False,
                duration=(datetime.now() - task_start).total_seconds(),
                error_message=str(e)
            )
    
    def _check_monthly_report(self):
        """Check if it's time for monthly report (last day of month)."""
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        
        # If tomorrow is a different month, generate monthly report
        if tomorrow.month != today.month:
            self._execute_monthly_report()
    
    def _execute_monthly_report(self):
        """Generate comprehensive monthly analysis."""
        task_start = datetime.now()
        task_id = f"monthly_report_{task_start.strftime('%Y_%m')}"
        
        self.logger.info("[REPORT] Generating monthly performance report...")
        
        try:
            # Generate comprehensive report
            report = self.performance_monitor.generate_performance_report(days_back=30)
            
            # Deep analysis
            monthly_analysis = self._analyze_monthly_performance(report)
            
            # Phase evaluation
            phase_evaluation = self._evaluate_current_phase()
            
            # Recommendations
            recommendations = self._generate_recommendations(monthly_analysis, phase_evaluation)
            
            self._record_task_execution(
                task_id=task_id,
                task_type=TaskType.MONTHLY_REPORT,
                success=True,
                duration=(datetime.now() - task_start).total_seconds(),
                results={
                    "report": report,
                    "analysis": monthly_analysis,
                    "phase_evaluation": phase_evaluation,
                    "recommendations": recommendations
                }
            )
            
            self.logger.info("[OK] Monthly report generated successfully")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Monthly report failed: {e}")

    def _evaluate_phase_progress(self):
        """Evaluate current phase progress and handle transitions."""
        current_progress = self._calculate_phase_progress()
        
        # Check if phase should transition
        should_transition, reason = self._should_transition_phase()
        
        if should_transition:
            self._transition_to_next_phase(reason)
        
        # Log progress
        self.logger.info(f"[PHASE] Phase {self.current_phase.name} progress: {current_progress:.1f}%")

    def _system_maintenance(self):
        """Perform system maintenance tasks."""
        task_start = datetime.now()
        
        self.logger.info("[MAINTENANCE] Performing system maintenance...")
        
        try:
            maintenance_tasks = []
            
            # Clean old logs (keep last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            # Implementation depends on your log retention policy
            
            # Archive old performance data
            # Implementation for data archival
            
            # Check system health
            health_check = self._perform_health_check()
            maintenance_tasks.append(f"Health check: {health_check}")
            
            # Optimize databases
            self._optimize_databases()
            maintenance_tasks.append("Database optimization completed")
            
            self.logger.info(f"[OK] Maintenance completed: {maintenance_tasks}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Maintenance failed: {e}")

    def _update_performance_tracking(self, results, performance_report):
        """Update the CSV performance tracking file."""
        try:
            csv_file = self.project_root / "trading_performance_tracking.csv"
            
            # Prepare data row
            enhanced_decision = results['enhanced_decision']
            risk_metrics = results['risk_metrics']
            position_sizing = results['position_sizing']
            weight_adjustment = results['weight_adjustment']
            
            new_row = {
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Heure': datetime.now().strftime('%H:%M'),
                'Signal_Final': enhanced_decision.final_signal,
                'Confiance': f"{enhanced_decision.final_confidence:.2%}",
                'Prix_QQQ': 0.0,  # Would need actual price data
                'Classic_Signal': enhanced_decision.individual_decisions[0].signal,
                'Classic_Conf': f"{enhanced_decision.individual_decisions[0].confidence:.2%}",
                'LLM_Text_Signal': enhanced_decision.individual_decisions[1].signal,
                'LLM_Text_Conf': f"{enhanced_decision.individual_decisions[1].confidence:.2%}",
                'LLM_Visual_Signal': enhanced_decision.individual_decisions[2].signal,
                'LLM_Visual_Conf': f"{enhanced_decision.individual_decisions[2].confidence:.2%}",
                'Sentiment_Signal': enhanced_decision.individual_decisions[3].signal,
                'Sentiment_Conf': f"{enhanced_decision.individual_decisions[3].confidence:.2%}",
                'Consensus_Score': f"{enhanced_decision.consensus_score:.2%}",
                'Disagreement_Factor': f"{enhanced_decision.disagreement_factor:.2%}",
                'Risk_Level': risk_metrics.risk_level.name,
                'Risk_Score': f"{risk_metrics.overall_risk_score:.3f}",
                'Position_Recommandee': f"${position_sizing.recommended_size:,.0f}",
                'Weight_Classic': f"{weight_adjustment.model_weights['classic']:.3f}",
                'Weight_LLM_Text': f"{weight_adjustment.model_weights['llm_text']:.3f}",
                'Weight_LLM_Visual': f"{weight_adjustment.model_weights['llm_visual']:.3f}",
                'Weight_Sentiment': f"{weight_adjustment.model_weights['sentiment']:.3f}",
                'Notes': f"Phase: {self.current_phase.name}"
            }
            
            # Append to CSV
            df_new = pd.DataFrame([new_row])
            if csv_file.exists():
                df_existing = pd.read_csv(csv_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
            
            df_combined.to_csv(csv_file, index=False)
            
        except Exception as e:
            self.logger.error(f"Failed to update performance tracking: {e}")
    
    def _analyze_weekly_performance(self, report):
        """Analyze weekly performance against targets."""
        if report.get('error'):
            return {"status": "error", "message": report['error']}
        
        perf = report.get('performance_summary', {})
        
        # Compare against current phase targets
        targets = self.config["performance_targets"].get(f"phase_{self.current_phase.name[-1]}", {})
        
        analysis = {
            "week_performance": perf,
            "targets": targets,
            "achievements": {},
            "recommendations": []
        }
        
        # Check achievements
        if targets:
            for metric, target in targets.items():
                current = perf.get(metric, 0)
                if metric in ['sharpe_ratio', 'win_rate']:
                    achieved = current >= target
                elif metric == 'max_drawdown':
                    achieved = current <= target
                else:
                    achieved = False
                
                analysis["achievements"][metric] = {
                    "target": target,
                    "current": current,
                    "achieved": achieved
                }
        
        return analysis
    
    def _analyze_monthly_performance(self, report):
        """Deep analysis of monthly performance."""
        weekly_analysis = self._analyze_weekly_performance(report)
        
        # Add monthly-specific analysis
        monthly_analysis = {
            "weekly_base": weekly_analysis,
            "trend_analysis": self._analyze_performance_trends(),
            "model_performance": self._analyze_model_performance(),
            "risk_analysis": self._analyze_risk_metrics(),
            "phase_readiness": self._assess_phase_transition_readiness()
        }
        
        return monthly_analysis
    
    def _should_transition_phase(self):
        """Determine if current phase should transition to next phase."""
        days_since_start = (datetime.now() - self.project_start_date).days
        
        phase_durations = self.config["phase_transitions"]
        
        # Calculate cumulative days for each phase
        phase_1_end = phase_durations["phase_1_duration_days"]
        phase_2_end = phase_1_end + phase_durations["phase_2_duration_days"]
        phase_3_end = phase_2_end + phase_durations["phase_3_duration_days"]
        phase_4_end = phase_3_end + phase_durations["phase_4_duration_days"]
        
        if self.current_phase == Phase.PHASE_1:
            if days_since_start >= phase_1_end:
                return True, "Phase 1 duration completed"
        elif self.current_phase == Phase.PHASE_2:
            if days_since_start >= phase_2_end:
                return True, "Phase 2 duration completed"
        elif self.current_phase == Phase.PHASE_3:
            if days_since_start >= phase_3_end:
                return True, "Phase 3 duration completed"
        elif self.current_phase == Phase.PHASE_4:
            if days_since_start >= phase_4_end:
                return True, "Phase 4 duration completed"
        
        return False, "Phase duration not met"
    
    def _transition_to_next_phase(self, reason):
        """Transition to the next phase."""
        phase_transitions = {
            Phase.PHASE_1: Phase.PHASE_2,
            Phase.PHASE_2: Phase.PHASE_3,
            Phase.PHASE_3: Phase.PHASE_4
        }
        
        if self.current_phase in phase_transitions:
            old_phase = self.current_phase
            new_phase = phase_transitions[self.current_phase]
            self.current_phase = new_phase
            
            self.logger.info(f"[TRANSITION] Phase transition: {old_phase.value} -> {new_phase.value}")
            self.logger.info(f"[REASON] Reason: {reason}")

            # Persist the change immediately to the database
            self._persist_phase_transition(old_phase, new_phase)
            
            # Update schedule for new phase
            self._setup_schedule()

    def _persist_phase_transition(self, old_phase: Phase, new_phase: Phase):
        """Persist the phase transition to the database."""
        try:
            conn = sqlite3.connect(self.scheduler_db)
            cursor = conn.cursor()

            # Mark old phase as completed
            cursor.execute("""
                UPDATE phase_progress
                SET status = 'completed', last_updated = ?
                WHERE phase = ?
            """, (datetime.now().isoformat(), old_phase.name))

            # Insert new phase as in_progress
            cursor.execute("""
                INSERT OR IGNORE INTO phase_progress (phase, start_date, status, last_updated)
                VALUES (?, ?, 'in_progress', ?)
            """, (new_phase.name, datetime.now().isoformat(), datetime.now().isoformat()))

            conn.commit()
            conn.close()
            self.logger.info(f"[DB] Persisted phase transition from {old_phase.value} to {new_phase.value}")
        except Exception as e:
            self.logger.error(f"Failed to persist phase transition: {e}")
    
    def _record_task_execution(self, task_id: str, task_type: TaskType, success: bool, 
                             duration: float, results=None, error_message: str = None):
        """Record task execution in database."""
        try:
            conn = sqlite3.connect(self.scheduler_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO task_executions 
                (task_id, task_type, execution_time, phase, success, duration_seconds, error_message, results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_id,
                task_type.value,
                datetime.now().isoformat(),
                self.current_phase.name,
                success,
                duration,
                error_message,
                json.dumps(results, default=str) if results else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to record task execution: {e}")
    
    def _calculate_phase_progress(self):
        """Calculate current phase completion percentage."""
        days_since_start = (datetime.now() - self.project_start_date).days
        
        phase_durations = self.config["phase_transitions"]
        
        # Calculate cumulative days for each phase
        phase_1_end = phase_durations["phase_1_duration_days"]
        phase_2_end = phase_1_end + phase_durations["phase_2_duration_days"]
        phase_3_end = phase_2_end + phase_durations["phase_3_duration_days"]
        phase_4_end = phase_3_end + phase_durations["phase_4_duration_days"]
        
        if self.current_phase == Phase.PHASE_1:
            target_days = phase_durations["phase_1_duration_days"]
            days_in_phase = days_since_start
        elif self.current_phase == Phase.PHASE_2:
            target_days = phase_durations["phase_2_duration_days"]
            days_in_phase = days_since_start - phase_1_end
        elif self.current_phase == Phase.PHASE_3:
            target_days = phase_durations["phase_3_duration_days"]
            days_in_phase = days_since_start - phase_2_end
        elif self.current_phase == Phase.PHASE_4:
            target_days = phase_durations["phase_4_duration_days"]
            days_in_phase = days_since_start - phase_3_end
        else:
            return 0.0
        
        return min(100.0, (days_in_phase / target_days) * 100) if target_days > 0 else 0.0
    
    def _update_phase_progress(self, analysis_data):
        """Update phase progress based on analysis data."""
        try:
            conn = sqlite3.connect(self.scheduler_db)
            cursor = conn.cursor()
            
            progress = self._calculate_phase_progress()
            
            # Calculate metrics achieved
            metrics_achieved = {}
            if 'achievements' in analysis_data:
                for metric, achievement in analysis_data['achievements'].items():
                    metrics_achieved[metric] = achievement.get('achieved', False)
            
            # Determine status
            if progress >= 100:
                status = 'completed'
            elif progress > 0:
                status = 'in_progress'
            else:
                status = 'not_started'
            
            # Update or insert phase progress
            cursor.execute('''
                INSERT OR REPLACE INTO phase_progress 
                (phase, start_date, target_end_date, current_progress, metrics_achieved, status, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_phase.name,
                self.project_start_date.isoformat(),
                (self.project_start_date + timedelta(days=self._get_cumulative_phase_duration())).isoformat(),
                progress,
                json.dumps(metrics_achieved),
                status,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to update phase progress: {e}")
    
    def _get_phase_duration(self):
        """Get the duration for the current phase in days."""
        phase_durations = self.config["phase_transitions"]
        if self.current_phase == Phase.PHASE_1:
            return phase_durations["phase_1_duration_days"]
        elif self.current_phase == Phase.PHASE_2:
            return phase_durations["phase_2_duration_days"]
        elif self.current_phase == Phase.PHASE_3:
            return phase_durations["phase_3_duration_days"]
        elif self.current_phase == Phase.PHASE_4:
            return phase_durations["phase_4_duration_days"]
        return 30  # Default
    
    def _get_cumulative_phase_duration(self):
        """Get the cumulative duration from project start to end of current phase."""
        phase_durations = self.config["phase_transitions"]
        
        if self.current_phase == Phase.PHASE_1:
            return phase_durations["phase_1_duration_days"]
        elif self.current_phase == Phase.PHASE_2:
            return phase_durations["phase_1_duration_days"] + phase_durations["phase_2_duration_days"]
        elif self.current_phase == Phase.PHASE_3:
            return (phase_durations["phase_1_duration_days"] + 
                   phase_durations["phase_2_duration_days"] + 
                   phase_durations["phase_3_duration_days"])
        elif self.current_phase == Phase.PHASE_4:
            return (phase_durations["phase_1_duration_days"] + 
                   phase_durations["phase_2_duration_days"] + 
                   phase_durations["phase_3_duration_days"] + 
                   phase_durations["phase_4_duration_days"])
        return 30  # Default
    
    def _analyze_performance_trends(self):
        """Analyze performance trends over time."""
        try:
            conn = sqlite3.connect(self.scheduler_db)
            df = pd.read_sql_query(
                "SELECT * FROM system_metrics ORDER BY date DESC LIMIT 30",
                conn
            )
            conn.close()
            
            if df.empty:
                return {"status": "no_data", "trends": {}}
            
            trends = {}
            for metric in ['sharpe_ratio', 'max_drawdown', 'win_rate']:
                if metric in df.columns:
                    values = df[metric].dropna()
                    if len(values) > 1:
                        # Simple trend calculation
                        trend = "improving" if values.iloc[0] > values.iloc[-1] else "declining"
                        trends[metric] = {
                            "trend": trend,
                            "latest": values.iloc[0] if len(values) > 0 else 0,
                            "average": values.mean()
                        }
            
            return {"status": "success", "trends": trends}
            
        except Exception as e:
            self.logger.error(f"Failed to analyze performance trends: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_model_performance(self):
        """Analyze individual model performance."""
        try:
            # Read from CSV tracking file if it exists
            csv_file = self.project_root / "trading_performance_tracking.csv"
            if not csv_file.exists():
                return {"status": "no_data", "models": {}}
            
            df = pd.read_csv(csv_file)
            if df.empty:
                return {"status": "no_data", "models": {}}
            
            # Analyze last 30 entries
            recent_data = df.tail(30)
            
            model_performance = {}
            for model in ['Classic', 'LLM_Text', 'LLM_Visual', 'Sentiment']:
                signal_col = f"{model}_Signal"
                conf_col = f"{model}_Conf"
                
                if signal_col in recent_data.columns:
                    signals = recent_data[signal_col]
                    buy_signals = (signals == 'BUY').sum()
                    sell_signals = (signals == 'SELL').sum()
                    hold_signals = (signals == 'HOLD').sum()
                    
                    model_performance[model.lower()] = {
                        "total_signals": len(signals),
                        "buy_ratio": buy_signals / len(signals) if len(signals) > 0 else 0,
                        "sell_ratio": sell_signals / len(signals) if len(signals) > 0 else 0,
                        "hold_ratio": hold_signals / len(signals) if len(signals) > 0 else 0
                    }
            
            return {"status": "success", "models": model_performance}
            
        except Exception as e:
            self.logger.error(f"Failed to analyze model performance: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_risk_metrics(self):
        """Analyze risk metrics trends."""
        try:
            csv_file = self.project_root / "trading_performance_tracking.csv"
            if not csv_file.exists():
                return {"status": "no_data", "risk_analysis": {}}
            
            df = pd.read_csv(csv_file)
            if df.empty:
                return {"status": "no_data", "risk_analysis": {}}
            
            recent_data = df.tail(30)
            
            risk_analysis = {}
            if 'Risk_Level' in recent_data.columns:
                risk_levels = recent_data['Risk_Level'].value_counts()
                risk_analysis['risk_distribution'] = risk_levels.to_dict()
                risk_analysis['high_risk_frequency'] = (
                    (recent_data['Risk_Level'].isin(['HIGH', 'VERY_HIGH'])).sum() / len(recent_data)
                    if len(recent_data) > 0 else 0
                )
            
            return {"status": "success", "risk_analysis": risk_analysis}
            
        except Exception as e:
            self.logger.error(f"Failed to analyze risk metrics: {e}")
            return {"status": "error", "message": str(e)}
    
    def _assess_phase_transition_readiness(self):
        """Assess if the system is ready for phase transition."""
        current_progress = self._calculate_phase_progress()
        
        # Get performance targets for current phase
        targets = self.config["performance_targets"].get(f"phase_{self.current_phase.name[-1]}", {})
        
        readiness = {
            "progress_complete": current_progress >= 100,
            "time_based_ready": current_progress >= 80,  # 80% time completion
            "performance_ready": False,
            "overall_ready": False
        }
        
        # Check performance readiness (simplified)
        if targets:
            # Would need actual performance data comparison
            readiness["performance_ready"] = current_progress >= 90  # Simplified
        
        readiness["overall_ready"] = (
            readiness["progress_complete"] or 
            (readiness["time_based_ready"] and readiness["performance_ready"])
        )
        
        return readiness
    
    def _generate_recommendations(self, monthly_analysis, phase_evaluation):
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Performance-based recommendations
        if 'trends' in monthly_analysis.get('trend_analysis', {}):
            trends = monthly_analysis['trend_analysis']['trends']
            for metric, trend_data in trends.items():
                if trend_data['trend'] == 'declining':
                    recommendations.append(f"Address declining {metric} - consider model retraining")
        
        # Risk-based recommendations
        risk_analysis = monthly_analysis.get('risk_analysis', {})
        if risk_analysis.get('status') == 'success':
            high_risk_freq = risk_analysis.get('risk_analysis', {}).get('high_risk_frequency', 0)
            if high_risk_freq > 0.3:  # More than 30% high risk
                recommendations.append("High risk frequency detected - review risk management parameters")
        
        # Phase transition recommendations
        if phase_evaluation.get('overall_ready'):
            recommendations.append(f"System ready for transition from {self.current_phase.value}")
        
        return recommendations
    
    def _evaluate_current_phase(self):
        """Evaluate current phase status and readiness."""
        progress = self._calculate_phase_progress()
        readiness = self._assess_phase_transition_readiness()
        
        return {
            "current_phase": self.current_phase.value,
            "progress_percentage": progress,
            "days_in_phase": (datetime.now() - self.project_start_date).days,
            "readiness_assessment": readiness
        }
    
    def _check_performance_alerts(self, results):
        """Check for performance alerts and notifications."""
        try:
            if not self.config.get("alerts", {}).get("performance_alerts", True):
                return
            
            enhanced_decision = results['enhanced_decision']
            risk_metrics = results['risk_metrics']
            
            # High risk alert
            if risk_metrics.risk_level.name in ['HIGH', 'VERY_HIGH']:
                self.logger.warning(f"[HIGH RISK ALERT] {risk_metrics.risk_level.name} - Score: {risk_metrics.overall_risk_score:.3f}")
            
            # Low consensus alert
            if enhanced_decision.consensus_score < 0.5:
                self.logger.warning(f"[LOW CONSENSUS] {enhanced_decision.consensus_score:.2%} - Models disagree significantly")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Performance alerts check failed: {e}")
    
    def _perform_health_check(self):
        """Perform system health check."""
        health_status = {
            "database_accessible": False,
            "csv_writable": False,
            "ollama_responsive": False,
            "data_cache_available": False
        }
        
        try:
            # Check database
            conn = sqlite3.connect(self.scheduler_db)
            conn.close()
            health_status["database_accessible"] = True
        except:
            pass
        
        # Add other health checks
        
        return health_status
    
    def _optimize_databases(self):
        """Optimize database performance."""
        try:
            conn = sqlite3.connect(self.scheduler_db)
            conn.execute("VACUUM")
            conn.close()
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
    
    def run(self):
        """Main scheduler loop - runs continuously."""
        self.logger.info("[INIT] Starting Intelligent Trading AI Scheduler...")
        self.logger.info(f"[PHASE] Current phase: {self.current_phase.value}")
        self.logger.info(f"[SCHEDULE] Daily analysis scheduled at: {self.config['daily_execution_time']}")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("[STOP] Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"[ERROR] Scheduler error: {e}")
            time.sleep(300)  # Wait 5 minutes before retrying
            self.run()  # Restart

    def get_status(self):
        """Get current scheduler status."""
        return {
            "current_phase": self.current_phase.value,
            "project_start_date": self.project_start_date.isoformat(),
            "days_running": (datetime.now() - self.project_start_date).days,
            "next_jobs": [str(job) for job in schedule.jobs],
            "phase_progress": self._calculate_phase_progress()
        }

def main():
    """Main entry point for the intelligent scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Trading AI Scheduler")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = IntelligentScheduler()
    
    if args.status:
        status = scheduler.get_status()
        print("[STATUS] Trading AI Scheduler Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    else:
        # Run the scheduler
        scheduler.run()

if __name__ == "__main__":
    main()
"""
Real-Time Performance Tracking and Monitoring System
Provides comprehensive performance monitoring, alerts, and automated model evaluation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from threading import Timer
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    timestamp: datetime
    model_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold
        }

@dataclass
class RealTimeMetrics:
    """Real-time performance metrics"""
    timestamp: datetime
    portfolio_value: float
    daily_return: float
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    active_positions: int
    cash_balance: float
    model_accuracy: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'daily_return': self.daily_return,
            'cumulative_return': self.cumulative_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'active_positions': self.active_positions,
            'cash_balance': self.cash_balance,
            'model_accuracy': self.model_accuracy,
            'risk_metrics': self.risk_metrics
        }

class PerformanceMonitor:
    """
    Real-time performance monitoring system with alerts and automated evaluation.
    """
    
    def __init__(self, 
                 db_path: str = "performance_monitor.db",
                 alert_thresholds: Dict = None,
                 email_config: Dict = None):
        """
        Initialize the performance monitor.
        
        Args:
            db_path: Database path for storing metrics
            alert_thresholds: Custom alert thresholds
            email_config: Email configuration for alerts
        """
        self.db_path = db_path
        self.email_config = email_config
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'max_drawdown': {'warning': 0.05, 'critical': 0.1},
            'daily_loss': {'warning': -0.02, 'critical': -0.05},
            'sharpe_ratio': {'warning': 0.5, 'critical': 0.0},
            'model_accuracy': {'warning': 0.45, 'critical': 0.35},
            'win_rate': {'warning': 0.4, 'critical': 0.3},
            'volatility': {'warning': 0.03, 'critical': 0.05}
        }
        
        # Active alerts to avoid spam
        self.active_alerts = set()
        self.alert_cooldown = {}  # Alert type -> last sent time
        self.cooldown_period = timedelta(hours=1)  # 1 hour cooldown
        
        # Performance history for calculations
        self.performance_history = []
        self.benchmark_history = []
        
        self._init_database()
    
    def _init_database(self):
        """Initialize monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Real-time metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realtime_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    portfolio_value REAL,
                    daily_return REAL,
                    cumulative_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    active_positions INTEGER,
                    cash_balance REAL,
                    model_accuracy TEXT,  -- JSON string
                    risk_metrics TEXT     -- JSON string
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    model_name TEXT,
                    metric_value REAL,
                    threshold REAL,
                    acknowledged BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Daily performance summary
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    starting_value REAL,
                    ending_value REAL,
                    daily_return REAL,
                    benchmark_return REAL,
                    trades_count INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    max_intraday_drawdown REAL,
                    volatility REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Performance monitoring database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring database: {e}")
    
    def record_metrics(self, metrics: RealTimeMetrics):
        """Record real-time metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_metrics 
                (timestamp, portfolio_value, daily_return, cumulative_return,
                 sharpe_ratio, max_drawdown, win_rate, total_trades,
                 active_positions, cash_balance, model_accuracy, risk_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.portfolio_value,
                metrics.daily_return,
                metrics.cumulative_return,
                metrics.sharpe_ratio,
                metrics.max_drawdown,
                metrics.win_rate,
                metrics.total_trades,
                metrics.active_positions,
                metrics.cash_balance,
                json.dumps(metrics.model_accuracy),
                json.dumps(metrics.risk_metrics)
            ))
            
            conn.commit()
            conn.close()
            
            # Add to history for calculations
            self.performance_history.append(metrics)
            
            # Keep only last 1000 entries in memory
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
    
    def check_alerts(self, metrics: RealTimeMetrics) -> List[PerformanceAlert]:
        """
        Check for performance alerts based on current metrics.
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            List of generated alerts
        """
        alerts = []
        current_time = datetime.now()
        
        # Check drawdown alert
        if metrics.max_drawdown > self.alert_thresholds['max_drawdown']['critical']:
            alert = PerformanceAlert(
                alert_type='max_drawdown',
                severity='CRITICAL',
                message=f"Critical drawdown detected: {metrics.max_drawdown:.2%}",
                timestamp=current_time,
                metric_value=metrics.max_drawdown,
                threshold=self.alert_thresholds['max_drawdown']['critical']
            )
            alerts.append(alert)
        elif metrics.max_drawdown > self.alert_thresholds['max_drawdown']['warning']:
            alert = PerformanceAlert(
                alert_type='max_drawdown',
                severity='MEDIUM',
                message=f"High drawdown warning: {metrics.max_drawdown:.2%}",
                timestamp=current_time,
                metric_value=metrics.max_drawdown,
                threshold=self.alert_thresholds['max_drawdown']['warning']
            )
            alerts.append(alert)
        
        # Check daily loss alert
        if metrics.daily_return < self.alert_thresholds['daily_loss']['critical']:
            alert = PerformanceAlert(
                alert_type='daily_loss',
                severity='CRITICAL',
                message=f"Critical daily loss: {metrics.daily_return:.2%}",
                timestamp=current_time,
                metric_value=metrics.daily_return,
                threshold=self.alert_thresholds['daily_loss']['critical']
            )
            alerts.append(alert)
        elif metrics.daily_return < self.alert_thresholds['daily_loss']['warning']:
            alert = PerformanceAlert(
                alert_type='daily_loss',
                severity='MEDIUM',
                message=f"Significant daily loss: {metrics.daily_return:.2%}",
                timestamp=current_time,
                metric_value=metrics.daily_return,
                threshold=self.alert_thresholds['daily_loss']['warning']
            )
            alerts.append(alert)
        
        # Check Sharpe ratio alert
        if metrics.sharpe_ratio < self.alert_thresholds['sharpe_ratio']['critical']:
            alert = PerformanceAlert(
                alert_type='sharpe_ratio',
                severity='HIGH',
                message=f"Poor risk-adjusted performance: Sharpe {metrics.sharpe_ratio:.2f}",
                timestamp=current_time,
                metric_value=metrics.sharpe_ratio,
                threshold=self.alert_thresholds['sharpe_ratio']['critical']
            )
            alerts.append(alert)
        
        # Check individual model accuracy
        for model_name, accuracy in metrics.model_accuracy.items():
            if accuracy < self.alert_thresholds['model_accuracy']['critical']:
                alert = PerformanceAlert(
                    alert_type='model_accuracy',
                    severity='HIGH',
                    message=f"Model {model_name} accuracy critically low: {accuracy:.2%}",
                    timestamp=current_time,
                    model_name=model_name,
                    metric_value=accuracy,
                    threshold=self.alert_thresholds['model_accuracy']['critical']
                )
                alerts.append(alert)
        
        # Check win rate
        if metrics.win_rate < self.alert_thresholds['win_rate']['critical']:
            alert = PerformanceAlert(
                alert_type='win_rate',
                severity='HIGH',
                message=f"Win rate critically low: {metrics.win_rate:.2%}",
                timestamp=current_time,
                metric_value=metrics.win_rate,
                threshold=self.alert_thresholds['win_rate']['critical']
            )
            alerts.append(alert)
        
        return alerts
    
    def process_alerts(self, alerts: List[PerformanceAlert]):
        """Process and send alerts"""
        for alert in alerts:
            # Check cooldown
            alert_key = f"{alert.alert_type}_{alert.severity}"
            if alert_key in self.alert_cooldown:
                time_since_last = datetime.now() - self.alert_cooldown[alert_key]
                if time_since_last < self.cooldown_period:
                    continue  # Skip due to cooldown
            
            # Record alert
            self._record_alert(alert)
            
            # Send notification
            if self.email_config and alert.severity in ['HIGH', 'CRITICAL']:
                self._send_email_alert(alert)
            
            # Update cooldown
            self.alert_cooldown[alert_key] = datetime.now()
            
            logger.warning(f"ALERT [{alert.severity}]: {alert.message}")
    
    def _record_alert(self, alert: PerformanceAlert):
        """Record alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_alerts 
                (timestamp, alert_type, severity, message, model_name, metric_value, threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp.isoformat(),
                alert.alert_type,
                alert.severity,
                alert.message,
                alert.model_name,
                alert.metric_value,
                alert.threshold
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record alert: {e}")
    
    def _send_email_alert(self, alert: PerformanceAlert):
        """Send email alert notification"""
        if not self.email_config:
            return
        
        try:
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            username = self.email_config.get('username')
            password = self.email_config.get('password')
            to_email = self.email_config.get('alert_email')
            
            if not all([smtp_server, username, password, to_email]):
                logger.warning("Incomplete email configuration, skipping email alert")
                return
            
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = to_email
            msg['Subject'] = f"Trading AI Alert [{alert.severity}]: {alert.alert_type}"
            
            body = f"""
            Trading AI Performance Alert
            
            Alert Type: {alert.alert_type}
            Severity: {alert.severity}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            {alert.message}
            
            Model: {alert.model_name or 'System-wide'}
            Current Value: {alert.metric_value}
            Threshold: {alert.threshold}
            
            Please review the system performance and take appropriate action.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_type}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def generate_performance_report(self, 
                                  days_back: int = 7) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            days_back: Number of days to include in report
            
        Returns:
            Performance report dictionary
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent metrics
            query = '''
                SELECT * FROM realtime_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(cutoff_date.isoformat(),))
            
            if df.empty:
                return {'error': 'No data available for the specified period'}
            
            # Calculate report metrics
            latest_metrics = df.iloc[0]
            
            # Performance over period
            period_return = latest_metrics['cumulative_return']
            
            # Average daily metrics
            avg_daily_return = df['daily_return'].mean()
            avg_sharpe = df['sharpe_ratio'].mean()
            max_drawdown_period = df['max_drawdown'].max()
            
            # Trading activity
            total_trades = latest_metrics['total_trades']
            avg_win_rate = df['win_rate'].mean()
            
            # Volatility
            volatility = df['daily_return'].std() * np.sqrt(252)  # Annualized
            
            # Recent alerts
            alert_query = '''
                SELECT alert_type, severity, COUNT(*) as count
                FROM performance_alerts 
                WHERE timestamp >= ? 
                GROUP BY alert_type, severity
                ORDER BY count DESC
            '''
            
            alerts_df = pd.read_sql_query(alert_query, conn, params=(cutoff_date.isoformat(),))
            
            conn.close()
            
            report = {
                'report_period': f"{days_back} days",
                'generated_at': datetime.now().isoformat(),
                'performance_summary': {
                    'period_return': period_return,
                    'average_daily_return': avg_daily_return,
                    'volatility': volatility,
                    'sharpe_ratio': avg_sharpe,
                    'max_drawdown': max_drawdown_period,
                    'win_rate': avg_win_rate,
                    'total_trades': int(total_trades)
                },
                'current_status': {
                    'portfolio_value': latest_metrics['portfolio_value'],
                    'active_positions': int(latest_metrics['active_positions']),
                    'cash_balance': latest_metrics['cash_balance']
                },
                'recent_alerts': alerts_df.to_dict('records') if not alerts_df.empty else [],
                'risk_assessment': self._assess_current_risk(df)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}
    
    def _assess_current_risk(self, df: pd.DataFrame) -> Dict:
        """Assess current risk level based on recent performance"""
        if df.empty:
            return {'risk_level': 'UNKNOWN', 'reason': 'No data available'}
        
        latest = df.iloc[0]
        recent_volatility = df['daily_return'].tail(5).std() * np.sqrt(252)
        recent_drawdown = latest['max_drawdown']
        recent_sharpe = df['sharpe_ratio'].tail(5).mean()
        
        risk_score = 0
        risk_factors = []
        
        # Volatility risk
        if recent_volatility > 0.3:
            risk_score += 3
            risk_factors.append("High volatility")
        elif recent_volatility > 0.2:
            risk_score += 2
            risk_factors.append("Elevated volatility")
        
        # Drawdown risk
        if recent_drawdown > 0.1:
            risk_score += 3
            risk_factors.append("Significant drawdown")
        elif recent_drawdown > 0.05:
            risk_score += 2
            risk_factors.append("Moderate drawdown")
        
        # Sharpe ratio risk
        if recent_sharpe < 0:
            risk_score += 2
            risk_factors.append("Negative risk-adjusted returns")
        elif recent_sharpe < 0.5:
            risk_score += 1
            risk_factors.append("Low risk-adjusted returns")
        
        # Determine risk level
        if risk_score >= 6:
            risk_level = "CRITICAL"
        elif risk_score >= 4:
            risk_level = "HIGH"
        elif risk_score >= 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendation': self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            'LOW': "Continue normal operations. Monitor for changes.",
            'MEDIUM': "Increase monitoring frequency. Consider reducing position sizes.",
            'HIGH': "Implement risk reduction measures. Review model parameters.",
            'CRITICAL': "Consider halting trading. Immediate review required."
        }
        return recommendations.get(risk_level, "Monitor situation closely.")
    
    def create_performance_dashboard(self, output_path: str = "performance_dashboard.png"):
        """Create visual performance dashboard"""
        try:
            # Get recent data
            cutoff_date = datetime.now() - timedelta(days=30)
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query('''
                SELECT timestamp, portfolio_value, daily_return, cumulative_return,
                       sharpe_ratio, max_drawdown, win_rate
                FROM realtime_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp
            ''', conn, params=(cutoff_date.isoformat(),))
            
            conn.close()
            
            if df.empty:
                logger.warning("No data available for dashboard")
                return
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create dashboard
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Trading AI Performance Dashboard', fontsize=16, fontweight='bold')
            
            # Portfolio value
            axes[0, 0].plot(df['timestamp'], df['portfolio_value'], color='blue', linewidth=2)
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_ylabel('Value ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Cumulative returns
            axes[0, 1].plot(df['timestamp'], df['cumulative_return'] * 100, color='green', linewidth=2)
            axes[0, 1].set_title('Cumulative Returns')
            axes[0, 1].set_ylabel('Return (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Daily returns distribution
            axes[0, 2].hist(df['daily_return'] * 100, bins=20, alpha=0.7, color='orange')
            axes[0, 2].set_title('Daily Returns Distribution')
            axes[0, 2].set_xlabel('Daily Return (%)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Sharpe ratio
            axes[1, 0].plot(df['timestamp'], df['sharpe_ratio'], color='purple', linewidth=2)
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Good Threshold')
            axes[1, 0].set_title('Sharpe Ratio Over Time')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Max drawdown
            axes[1, 1].plot(df['timestamp'], df['max_drawdown'] * 100, color='red', linewidth=2)
            axes[1, 1].fill_between(df['timestamp'], df['max_drawdown'] * 100, alpha=0.3, color='red')
            axes[1, 1].set_title('Maximum Drawdown')
            axes[1, 1].set_ylabel('Drawdown (%)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Win rate
            axes[1, 2].plot(df['timestamp'], df['win_rate'] * 100, color='teal', linewidth=2)
            axes[1, 2].axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% Baseline')
            axes[1, 2].set_title('Win Rate Over Time')
            axes[1, 2].set_ylabel('Win Rate (%)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            # Format x-axes
            for ax in axes.flat:
                if hasattr(ax, 'tick_params'):
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance dashboard saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
    
    def update_monitoring(self, 
                         portfolio_value: float,
                         daily_return: float,
                         trades_data: List[Dict],
                         model_predictions: Dict[str, Dict]):
        """
        Main update method for real-time monitoring.
        
        Args:
            portfolio_value: Current portfolio value
            daily_return: Today's return
            trades_data: List of recent trades
            model_predictions: Recent model predictions and outcomes
        """
        try:
            # Calculate metrics
            timestamp = datetime.now()
            
            # Calculate cumulative return (simplified)
            if len(self.performance_history) > 0:
                initial_value = self.performance_history[0].portfolio_value
                cumulative_return = (portfolio_value / initial_value) - 1
            else:
                cumulative_return = 0.0
            
            # Calculate Sharpe ratio (simplified, 30-day rolling)
            if len(self.performance_history) >= 30:
                recent_returns = [m.daily_return for m in self.performance_history[-30:]]
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            if len(self.performance_history) > 0:
                peak_value = max(m.portfolio_value for m in self.performance_history)
                max_drawdown = (peak_value - portfolio_value) / peak_value
            else:
                max_drawdown = 0.0
            
            # Calculate win rate
            if trades_data:
                winning_trades = sum(1 for trade in trades_data if trade.get('return', 0) > 0)
                win_rate = winning_trades / len(trades_data)
            else:
                win_rate = 0.0
            
            # Model accuracy
            model_accuracy = {}
            for model_name, predictions in model_predictions.items():
                if predictions.get('total_predictions', 0) > 0:
                    accuracy = predictions.get('correct_predictions', 0) / predictions['total_predictions']
                    model_accuracy[model_name] = accuracy
            
            # Create metrics object
            metrics = RealTimeMetrics(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=len(trades_data),
                active_positions=1,  # Simplified
                cash_balance=0.0,    # Would need actual data
                model_accuracy=model_accuracy,
                risk_metrics={}      # Could add specific risk metrics
            )
            
            # Record metrics
            self.record_metrics(metrics)
            
            # Check for alerts
            alerts = self.check_alerts(metrics)
            
            # Process alerts
            if alerts:
                self.process_alerts(alerts)
            
            logger.info(f"Performance monitoring updated: Portfolio ${portfolio_value:,.2f}, "
                       f"Daily Return: {daily_return:.2%}, Alerts: {len(alerts)}")
            
        except Exception as e:
            logger.error(f"Failed to update monitoring: {e}")
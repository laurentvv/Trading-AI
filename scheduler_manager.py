"""
Trading AI Scheduler Manager
Command-line interface for managing the intelligent scheduler.
"""

import json
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import sys

class SchedulerManager:
    """Manager for monitoring and configuring the Trading AI scheduler."""
    
    def __init__(self, project_root="c:/test/Trading-AI"):
        self.project_root = Path(project_root)
        self.scheduler_db = self.project_root / "scheduler.db"
        self.config_file = self.project_root / "scheduler_config.json"
    
    def show_status(self):
        """Display current scheduler status."""
        print("📊 TRADING AI SCHEDULER STATUS")
        print("=" * 50)
        
        # Check if configuration exists
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            start_date = datetime.fromisoformat(config["project_start_date"])
            days_running = (datetime.now() - start_date).days
            
            print(f"🚀 Project Start Date: {start_date.strftime('%Y-%m-%d')}")
            print(f"📅 Days Running: {days_running}")
            print(f"⏰ Daily Analysis Time: {config['daily_execution_time']}")
            print(f"📊 Trading Ticker: {config['trading_ticker']}")
        else:
            print("❌ Configuration file not found")
            return
        
        # Check database statistics
        if self.scheduler_db.exists():
            self._show_execution_stats()
            self._show_recent_activity()
        else:
            print("❌ Scheduler database not found")
    
    def _show_execution_stats(self):
        """Show execution statistics from database."""
        try:
            conn = sqlite3.connect(self.scheduler_db)
            
            # Total executions
            cursor = conn.execute("SELECT COUNT(*) FROM task_executions")
            total_executions = cursor.fetchone()[0]
            
            # Success rate
            cursor = conn.execute("SELECT COUNT(*) FROM task_executions WHERE success = 1")
            successful_executions = cursor.fetchone()[0]
            
            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
            
            # Recent executions (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM task_executions WHERE execution_time >= ?",
                (week_ago,)
            )
            recent_executions = cursor.fetchone()[0]
            
            print(f"\n📈 EXECUTION STATISTICS")
            print(f"   Total Executions: {total_executions}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Last 7 Days: {recent_executions} executions")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error reading execution stats: {e}")
    
    def _show_recent_activity(self):
        """Show recent activity."""
        try:
            conn = sqlite3.connect(self.scheduler_db)
            
            cursor = conn.execute("""
                SELECT task_type, execution_time, success, duration_seconds
                FROM task_executions 
                ORDER BY execution_time DESC 
                LIMIT 10
            """)
            
            recent_tasks = cursor.fetchall()
            
            if recent_tasks:
                print(f"\n⏰ RECENT ACTIVITY (Last 10)")
                print("-" * 50)
                for task_type, exec_time, success, duration in recent_tasks:
                    status = "✅" if success else "❌"
                    time_str = datetime.fromisoformat(exec_time).strftime('%m-%d %H:%M')
                    duration_str = f"{duration:.1f}s" if duration else "N/A"
                    print(f"   {status} {task_type} | {time_str} | {duration_str}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error reading recent activity: {e}")
    
    def show_performance_summary(self, days=7):
        """Show performance summary from CSV tracking."""
        csv_file = self.project_root / "trading_performance_tracking.csv"
        
        if not csv_file.exists():
            print("❌ Performance tracking file not found")
            return
        
        try:
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                print("📊 No performance data available yet")
                return
            
            # Filter recent data
            df['Date'] = pd.to_datetime(df['Date'])
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_df = df[df['Date'] >= cutoff_date]
            
            print(f"\n📊 PERFORMANCE SUMMARY (Last {days} days)")
            print("=" * 50)
            print(f"Total Trading Days: {len(recent_df)}")
            
            if len(recent_df) > 0:
                # Signal distribution
                signals = recent_df['Signal_Final'].value_counts()
                print(f"\nSignal Distribution:")
                for signal, count in signals.items():
                    percentage = count / len(recent_df) * 100
                    print(f"   {signal}: {count} ({percentage:.1f}%)")
                
                # Risk levels
                if 'Risk_Level' in recent_df.columns:
                    risk_levels = recent_df['Risk_Level'].value_counts()
                    print(f"\nRisk Level Distribution:")
                    for level, count in risk_levels.items():
                        percentage = count / len(recent_df) * 100
                        print(f"   {level}: {count} ({percentage:.1f}%)")
                
                # Average confidence
                if 'Confiance' in recent_df.columns:
                    try:
                        # Remove % sign and convert to float
                        confidences = recent_df['Confiance'].str.rstrip('%').astype(float)
                        avg_confidence = confidences.mean()
                        print(f"\nAverage Confidence: {avg_confidence:.1f}%")
                    except:
                        pass
        
        except Exception as e:
            print(f"❌ Error reading performance data: {e}")
    
    def show_phase_progress(self):
        """Show current phase progress."""
        if not self.scheduler_db.exists():
            print("❌ Scheduler database not found")
            return
        
        try:
            conn = sqlite3.connect(self.scheduler_db)
            cursor = conn.execute("SELECT * FROM phase_progress ORDER BY last_updated DESC LIMIT 1")
            phase_data = cursor.fetchone()
            
            if phase_data:
                phase, start_date, target_end, progress, metrics_json, status, last_updated = phase_data
                
                print(f"\n📈 PHASE PROGRESS")
                print("=" * 30)
                print(f"Current Phase: {phase}")
                print(f"Progress: {progress:.1f}%")
                print(f"Status: {status}")
                print(f"Start Date: {start_date}")
                if target_end:
                    print(f"Target End: {target_end}")
                
                if metrics_json:
                    try:
                        metrics = json.loads(metrics_json)
                        print(f"Metrics Achieved:")
                        for metric, achieved in metrics.items():
                            status_icon = "✅" if achieved else "⏳"
                            print(f"   {status_icon} {metric}")
                    except:
                        pass
            else:
                print("📊 No phase progress data available")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error reading phase progress: {e}")
    
    def configure(self):
        """Interactive configuration setup."""
        print("⚙️ SCHEDULER CONFIGURATION")
        print("=" * 30)
        
        # Load existing config or defaults
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            print("Current configuration loaded.")
        else:
            config = {
                "daily_execution_time": "18:00",
                "trading_ticker": "QQQ",
                "project_start_date": datetime.now().isoformat()
            }
        
        # Interactive configuration
        print(f"\n1. Daily Execution Time (current: {config['daily_execution_time']})")
        new_time = input("Enter new time (HH:MM format) or press Enter to keep current: ").strip()
        if new_time:
            config["daily_execution_time"] = new_time
        
        print(f"\n2. Trading Ticker (current: {config['trading_ticker']})")
        new_ticker = input("Enter new ticker or press Enter to keep current: ").strip()
        if new_ticker:
            config["trading_ticker"] = new_ticker
        
        # Save configuration
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4, default=str)
        
        print("✅ Configuration saved!")
    
    def export_data(self, output_file=None):
        """Export all scheduler data to CSV."""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            output_file = f"scheduler_export_{timestamp}.csv"
        
        try:
            conn = sqlite3.connect(self.scheduler_db)
            
            # Export task executions
            df = pd.read_sql_query("SELECT * FROM task_executions", conn)
            df.to_csv(output_file, index=False)
            
            conn.close()
            
            print(f"✅ Data exported to: {output_file}")
            print(f"   Records exported: {len(df)}")
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Trading AI Scheduler Manager")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--performance", type=int, default=7, metavar="DAYS", 
                       help="Show performance summary for last N days")
    parser.add_argument("--phase", action="store_true", help="Show phase progress")
    parser.add_argument("--configure", action="store_true", help="Configure scheduler")
    parser.add_argument("--export", type=str, metavar="FILE", help="Export data to CSV")
    
    args = parser.parse_args()
    
    manager = SchedulerManager()
    
    if args.status:
        manager.show_status()
    elif args.performance:
        manager.show_performance_summary(args.performance)
    elif args.phase:
        manager.show_phase_progress()
    elif args.configure:
        manager.configure()
    elif args.export:
        manager.export_data(args.export)
    else:
        # Show all information by default
        manager.show_status()
        manager.show_performance_summary()
        manager.show_phase_progress()

if __name__ == "__main__":
    main()
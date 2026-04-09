import time
import os
import sqlite3
from datetime import datetime, timedelta
from src.performance_monitor import PerformanceMonitor, PerformanceAlert

def test_alert_performance():
    # Setup
    db_path = "test_perf_monitor.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    monitor = PerformanceMonitor(db_path=db_path)

    # Create test alerts
    alerts = []
    for i in range(100):
        alerts.append(
            PerformanceAlert(
                alert_type="TEST",
                severity="LOW",
                message=f"Test alert {i}",
                timestamp=datetime.now(),
                metric_value=i,
                threshold=0
            )
        )

    # Baseline measurement
    start_time = time.time()
    monitor.process_alerts(alerts)
    end_time = time.time()

    print(f"Time taken for 100 alerts: {end_time - start_time:.4f} seconds")

    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    test_alert_performance()

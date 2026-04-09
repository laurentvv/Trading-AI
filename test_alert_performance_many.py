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
    # Make them 1000 alerts to see the difference clearly
    for i in range(1000):
        # We need to use unique alerts so they don't get skipped by cooldown
        alerts.append(
            PerformanceAlert(
                alert_type=f"TEST_{i}",
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

    print(f"Time taken for 1000 alerts: {end_time - start_time:.4f} seconds")

    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    test_alert_performance()

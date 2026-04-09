import sqlite3
import time
import os
from datetime import datetime
from src.performance_monitor import PerformanceMonitor, PerformanceAlert

def test_optimization():
    db_path = 'test_perf_monitor.db'
    if os.path.exists(db_path): os.remove(db_path)
    monitor = PerformanceMonitor(db_path=db_path)

    alerts = []
    for i in range(1000):
        alerts.append(
            PerformanceAlert(
                alert_type=f'TEST_{i}',
                severity='LOW',
                message=f'Test alert {i}',
                timestamp=datetime.now(),
                metric_value=i,
                threshold=0
            )
        )

    # Simulate optimized behavior
    start = time.time()

    try:
        conn = sqlite3.connect(monitor.db_path)
        cursor = conn.cursor()

        records = []
        for alert in alerts:
            alert_key = f"{alert.alert_type}_{alert.severity}"
            if alert_key in monitor.alert_cooldown:
                time_since_last = datetime.now() - monitor.alert_cooldown[alert_key]
                if time_since_last < monitor.cooldown_period:
                    continue

            # Add to batch
            records.append((
                alert.timestamp.isoformat(),
                alert.alert_type,
                alert.severity,
                alert.message,
                alert.model_name,
                alert.metric_value,
                alert.threshold
            ))

            if monitor.email_config and alert.severity in ['HIGH', 'CRITICAL']:
                monitor._send_email_alert(alert)

            monitor.alert_cooldown[alert_key] = datetime.now()

        # Bulk insert
        if records:
            cursor.executemany('''
                INSERT INTO performance_alerts
                (timestamp, alert_type, severity, message, model_name, metric_value, threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()

        conn.close()

    except Exception as e:
        print(f"Error: {e}")

    end = time.time()
    print(f'Optimized time: {end - start:.4f}s')

    if os.path.exists(db_path): os.remove(db_path)

test_optimization()

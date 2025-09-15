import sqlite3
import pandas as pd

def analyze_databases():
    """
    Analyzes the scheduler and model performance databases to extract
    current phase and performance metrics.
    """
    # Analyze scheduler.db
    try:
        conn_scheduler = sqlite3.connect("scheduler.db")
        print("--- Scheduler Status ---")
        
        # Get current phase
        df_phase = pd.read_sql_query("SELECT * FROM phase_progress ORDER BY last_updated DESC LIMIT 1", conn_scheduler)
        if not df_phase.empty:
            print("Current Phase:")
            print(df_phase.to_string(index=False))
        else:
            print("No phase progress information found.")
            
        # Get last task executions
        print("\n--- Last 5 Task Executions ---")
        df_tasks = pd.read_sql_query("SELECT task_id, task_type, execution_time, success, duration_seconds, error_message FROM task_executions ORDER BY execution_time DESC LIMIT 5", conn_scheduler)
        if not df_tasks.empty:
            print(df_tasks.to_string(index=False))
        else:
            print("No task execution history found.")

        conn_scheduler.close()

    except Exception as e:
        print(f"Error reading scheduler.db: {e}")

    # Analyze performance_monitor.db
    try:
        conn_perf = sqlite3.connect("performance_monitor.db")
        print("\n--- Model Performance ---")

        # Get latest realtime metrics
        print("\n--- Last 5 Real-Time Metrics ---")
        df_realtime = pd.read_sql_query("SELECT * FROM realtime_metrics ORDER BY timestamp DESC LIMIT 5", conn_perf)
        if not df_realtime.empty:
            print(df_realtime.to_string(index=False))
        else:
            print("No real-time metrics found.")

        conn_perf.close()

    except Exception as e:
        print(f"Error reading performance_monitor.db: {e}")

if __name__ == "__main__":
    analyze_databases()
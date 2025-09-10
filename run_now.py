
import sys
from pathlib import Path
# Add src directory to path to allow imports from the src folder
sys.path.append(str(Path(__file__).parent / 'src'))

from intelligent_scheduler import IntelligentScheduler

def run_analysis_now():
    """
    Initializes the IntelligentScheduler and runs the daily analysis task immediately.
    """
    print("Initializing the scheduler to run the analysis immediately...")
    scheduler = IntelligentScheduler()
    print("Scheduler initialized. Forcing daily analysis task...")
    scheduler._execute_daily_analysis()
    print("Daily analysis task has been executed.")

if __name__ == "__main__":
    run_analysis_now()

"""Run all Task 3 scripts in sequence."""
from __future__ import annotations
import subprocess
import sys


def run(cmd):
    """Run a command as a subprocess, printing it first."""
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    run([sys.executable, "scripts/task3_generate_forecast.py"])
    run([sys.executable, "scripts/task3_plot_forecast.py"])
    run([sys.executable, "scripts/task3_analyze_trends.py"])
    print("Task 3 complete.")

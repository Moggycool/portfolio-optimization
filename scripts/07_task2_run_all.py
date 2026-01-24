""" A script to run all Task 2 scripts in sequence. """
# scripts/07_task2_run_all.py
from __future__ import annotations

import subprocess
import sys

SCRIPTS = [
    "scripts/03_task2_make_splits_and_features.py",
    "scripts/04_task2_train_arima.py",
    "scripts/05_task2_train_lstm.py",
    "scripts/06_task2_compare_models.py",
]


def main():
    """ Main function to run all Task 2 scripts in order. """
    for s in SCRIPTS:
        print(f"\n=== Running {s} ===")
        r = subprocess.run([sys.executable, s], check=False)
        if r.returncode != 0:
            raise SystemExit(f"Script failed: {s}")

    print("\nAll Task 2 scripts completed successfully.")


if __name__ == "__main__":
    main()

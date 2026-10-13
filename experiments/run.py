"""
Entry point for running experiments.

Usage:
    python experiments/run.py

Configure via .env (copy .env.example and fill in values).
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.config import load_config
from experiments.pipeline import run_pipeline

if __name__ == "__main__":
    config = load_config()
    run_pipeline(config)

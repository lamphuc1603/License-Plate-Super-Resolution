"""
ICIP-XLPSR — inference entry point.

Reads sequences from the input directory and writes results to the output
directory (both configured in config.yaml). Run with:

    python src/inference.py
"""

import sys
import os

# Add src/ to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import InferencePipeline


def main():
    pipeline = InferencePipeline()
    pipeline.run()


if __name__ == "__main__":
    main()

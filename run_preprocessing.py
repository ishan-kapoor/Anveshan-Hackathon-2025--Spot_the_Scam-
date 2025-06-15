# File: run_preprocessing.py

"""
Script to preprocess raw job data and save cleaned train/test splits.
Run this BEFORE training or predictions.
"""

from src.preprocessing import save_processed_data

if __name__ == "__main__":
    try:
        save_processed_data()
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")

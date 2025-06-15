# File: main.py

"""
Entrypoint script for batch prediction on the processed test set.
Loads model and vectorizer, runs prediction, and saves results to outputs/predictions.csv.
"""

import os
import pandas as pd
from src.predict import predict_on_csv

TEST_PATH = "data/processed/processed_test.csv"
OUTPUT_PATH = "outputs/predictions.csv"


def main():
    print("🔄 Loading test data...")

    if not os.path.exists(TEST_PATH):
        print(f"🚫 Test file not found at {TEST_PATH}")
        return

    test_df = pd.read_csv(TEST_PATH)

    print("🤖 Running fraud prediction...")
    predictions = predict_on_csv(test_df)

    print(f"💾 Saving predictions to {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    predictions.to_csv(OUTPUT_PATH, index=False)
    print("✅ Done! Predictions saved.")


if __name__ == "__main__":
    main()

"""
SageMaker Processing Job Script — Data Preprocessing
=====================================================
Runs inside a SageMaker Processing Job container.

Input:  /opt/ml/processing/input/  (train.csv, test.csv)
Output: /opt/ml/processing/output/ (train_processed.csv,
                                    test_processed.csv,
                                    label_encoder.pkl)
"""

import os
import re
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder


INPUT_DIR  = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, remove punctuation, add length features, drop nulls."""
    df = df.copy()
    df["input_text"] = df["input_text"].str.lower()
    df["input_text"] = df["input_text"].str.replace(
        r"[^\w\s]", "", regex=True
    )
    df["text_length"] = df["input_text"].str.len()
    df["word_count"]  = df["input_text"].str.split().str.len()
    df = df.dropna(subset=["input_text", "output_text"])
    return df


def main():
    # ── Load ──────────────────────────────────────────────────────────────
    df_train = pd.read_csv(f"{INPUT_DIR}/train.csv")
    df_test  = pd.read_csv(f"{INPUT_DIR}/test.csv")
    print(f"Loaded — train: {len(df_train)}, test: {len(df_test)}")

    # ── Clean ─────────────────────────────────────────────────────────────
    df_train = clean_text(df_train)
    df_test  = clean_text(df_test)

    # ── Encode labels ─────────────────────────────────────────────────────
    le = LabelEncoder()
    df_train["label"] = le.fit_transform(df_train["output_text"])
    df_test["label"]  = le.transform(df_test["output_text"])
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_train.to_csv(f"{OUTPUT_DIR}/train_processed.csv", index=False)
    df_test.to_csv(f"{OUTPUT_DIR}/test_processed.csv",  index=False)

    with open(f"{OUTPUT_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"Saved to {OUTPUT_DIR}")
    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")


if __name__ == "__main__":
    main()

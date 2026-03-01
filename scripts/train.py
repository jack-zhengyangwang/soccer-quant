"""
Train logistic regression models for Premier League match prediction.

Two models:
1. Match outcome (multinomial) — Home Win / Draw / Away Win
2. Over/Under 2.5 goals (binary)

Uses temporal train/test split (NOT random) to prevent data leakage.
Train: seasons 2021-2024, Test: season 2025 (i.e., 2024-25).
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

FEATURE_COLS = [
    "home_rolling_xg", "away_rolling_xg",
    "home_rolling_xga", "away_rolling_xga",
    "home_rolling_points", "away_rolling_points",
    "home_rolling_poss", "away_rolling_poss",
    "home_rolling_sot_ratio", "away_rolling_sot_ratio",
    "xg_diff", "xga_diff",
]


def load_data():
    """Load feature-engineered dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from data/features.csv")
    return df


def temporal_split(df, test_season=2025):
    """
    Split data temporally by season.

    Train: all seasons before test_season.
    Test: test_season only.
    Prevents future data from leaking into training.
    """
    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]
    print(f"Train: {len(train)} rows (seasons {sorted(train['season'].unique())})")
    print(f"Test:  {len(test)} rows (season {test_season})")
    return train, test


def train_model(X_train, y_train, X_test, y_test, model_name, labels=None):
    """Train a logistic regression model and evaluate."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 50}")
    print(f"{model_name}")
    print(f"{'=' * 50}")
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    return model, scaler, y_prob


def main():
    df = load_data()

    # Temporal split
    train, test = temporal_split(df, test_season=2025)

    X_train = train[FEATURE_COLS]
    X_test = test[FEATURE_COLS]

    # Model 1: Match Outcome (multinomial)
    print("\n--- Training Match Outcome Model ---")
    outcome_model, outcome_scaler, outcome_probs = train_model(
        X_train, train["outcome"],
        X_test, test["outcome"],
        model_name="MATCH OUTCOME (Multinomial Logistic Regression)",
        labels=["Home Win", "Draw", "Away Win"],
    )

    # Naive baseline: always predict most common class
    baseline_class = train["outcome"].mode()[0]
    baseline_acc = (test["outcome"] == baseline_class).mean()
    print(f"Naive baseline (always predict class {baseline_class}): {baseline_acc:.4f}")

    # Model 2: Over/Under 2.5 Goals (binary)
    print("\n--- Training Over/Under 2.5 Goals Model ---")
    ou_model, ou_scaler, ou_probs = train_model(
        X_train, train["over_2_5"],
        X_test, test["over_2_5"],
        model_name="OVER/UNDER 2.5 GOALS (Binary Logistic Regression)",
        labels=["Under 2.5", "Over 2.5"],
    )

    baseline_class_ou = train["over_2_5"].mode()[0]
    baseline_acc_ou = (test["over_2_5"] == baseline_class_ou).mean()
    print(f"Naive baseline (always predict class {baseline_class_ou}): {baseline_acc_ou:.4f}")

    # Sample predictions
    print(f"\n{'=' * 50}")
    print("SAMPLE PREDICTIONS (first 10 test matches)")
    print(f"{'=' * 50}")
    sample = test.head(10)[["date", "HomeTeam", "AwayTeam"]].copy()
    sample["P(H)"] = outcome_probs[:10, 0].round(3)
    sample["P(D)"] = outcome_probs[:10, 1].round(3)
    sample["P(A)"] = outcome_probs[:10, 2].round(3)
    sample["P(Over)"] = ou_probs[:10, 1].round(3)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()

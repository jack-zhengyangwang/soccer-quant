"""
Feature engineering pipeline for Premier League match prediction.

Data flow:
    FBref xG data (team-perspective, 3,800 rows)
    → Normalize team names
    → Compute rolling features per team (last 5 matches)
    → Pivot to match-level (~1,900 matches)
    → Add target variables
    → Output: data/features.csv (~1,860 rows, 12 features + 2 targets)
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from import_data import download_match_xg, load_match_xg

# Comprehensive team name mapping: FBref variants → match-results convention.
# Covers inconsistencies between FBref's `team` column (full names)
# and `opponent` column (abbreviated names).
TEAM_NAME_MAP = {
    # FBref team column (full names)
    "Brighton And Hove Albion": "Brighton",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Norwich City": "Norwich",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Luton Town": "Luton",
    # FBref opponent column (abbreviated variants)
    "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle",
    "Nott'ham Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
}


def normalize_team_names(df):
    """Map FBref team and opponent names to a consistent convention."""
    df = df.copy()
    df["team"] = df["team"].replace(TEAM_NAME_MAP)
    df["opponent"] = df["opponent"].replace(TEAM_NAME_MAP)
    return df


def compute_rolling_features(df):
    """
    Compute rolling 5-match averages per team.

    Uses shift(1) to prevent data leakage (excludes current match).
    Uses min_periods=3 to require at least 3 past matches.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    # Derive per-match stats
    df["points"] = df["result"].map({"W": 3, "D": 1, "L": 0})
    df["sot_ratio"] = df["sot"] / df["sh"].replace(0, np.nan)

    rolling_cols = {
        "xg": "rolling_xg",
        "xga": "rolling_xga",
        "points": "rolling_points",
        "poss": "rolling_poss",
        "sot_ratio": "rolling_sot_ratio",
    }

    for src_col, dst_col in rolling_cols.items():
        df[dst_col] = df.groupby("team")[src_col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=3).mean()
        )

    return df


def pivot_to_match_level(df):
    """
    Convert team-perspective rows to match-level rows.

    Home rows (venue=='Home') provide home_* features.
    Away rows (venue=='Away') provide away_* features.
    Merge on (date, HomeTeam, AwayTeam).
    """
    rolling_cols = [
        "rolling_xg",
        "rolling_xga",
        "rolling_points",
        "rolling_poss",
        "rolling_sot_ratio",
    ]

    home = df[df["venue"] == "Home"].copy()
    away = df[df["venue"] == "Away"].copy()

    # Rename rolling columns with home_/away_ prefix
    home = home.rename(columns={col: f"home_{col}" for col in rolling_cols})
    home = home.rename(columns={"team": "HomeTeam", "opponent": "AwayTeam"})

    away = away.rename(columns={col: f"away_{col}" for col in rolling_cols})
    away = away.rename(columns={"team": "AwayTeam", "opponent": "HomeTeam"})

    # Select columns for merge
    home_feature_cols = [f"home_{col}" for col in rolling_cols]
    away_feature_cols = [f"away_{col}" for col in rolling_cols]

    home_keep = ["date", "HomeTeam", "AwayTeam", "gf", "ga", "result", "season"] + home_feature_cols
    away_keep = ["date", "HomeTeam", "AwayTeam"] + away_feature_cols

    matches = home[home_keep].merge(
        away[away_keep],
        on=["date", "HomeTeam", "AwayTeam"],
        how="inner",
    )

    print(f"Match-level rows: {len(matches)}")
    return matches


def add_targets(df):
    """Add prediction target columns."""
    df = df.copy()

    # Match outcome from home perspective: W=Home Win(0), D=Draw(1), L=Away Win(2)
    df["outcome"] = df["result"].map({"W": 0, "D": 1, "L": 2})

    # Over/Under 2.5 goals
    df["total_goals"] = df["gf"] + df["ga"]
    df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)

    return df


def add_diff_features(df):
    """Add derived difference features."""
    df = df.copy()
    df["xg_diff"] = df["home_rolling_xg"] - df["away_rolling_xg"]
    df["xga_diff"] = df["home_rolling_xga"] - df["away_rolling_xga"]
    return df


def main():
    # Load FBref data
    print("Downloading FBref xG data...")
    xg_path = download_match_xg()
    xg = load_match_xg(xg_path)
    print(f"\nRaw FBref data: {len(xg)} rows")

    # Normalize team names
    xg = normalize_team_names(xg)

    # Verify normalization: team and opponent should use the same name set
    team_names = set(xg["team"].unique())
    opp_names = set(xg["opponent"].unique())
    mismatches = (team_names - opp_names) | (opp_names - team_names)
    if mismatches:
        print(f"WARNING: Name mismatches after normalization: {mismatches}")
    else:
        print("Team name normalization OK — team and opponent names match")

    # Compute rolling features
    print("Computing rolling features...")
    xg = compute_rolling_features(xg)

    # Pivot to match level
    print("Pivoting to match level...")
    matches = pivot_to_match_level(xg)

    # Add targets and difference features
    matches = add_targets(matches)
    matches = add_diff_features(matches)

    # Select final columns
    feature_cols = [
        "home_rolling_xg", "away_rolling_xg",
        "home_rolling_xga", "away_rolling_xga",
        "home_rolling_points", "away_rolling_points",
        "home_rolling_poss", "away_rolling_poss",
        "home_rolling_sot_ratio", "away_rolling_sot_ratio",
        "xg_diff", "xga_diff",
    ]
    meta_cols = ["date", "season", "HomeTeam", "AwayTeam"]
    target_cols = ["outcome", "over_2_5"]

    output = matches[meta_cols + feature_cols + target_cols].copy()

    # Drop rows with missing rolling features
    before = len(output)
    output = output.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"Dropped {before - len(output)} rows with insufficient rolling history")
    print(f"Final dataset: {len(output)} rows, {len(feature_cols)} features, {len(target_cols)} targets")

    # Verify no NaN in features or targets
    nan_count = output[feature_cols + target_cols].isnull().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaN values in features/targets!"

    # Save
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
    output.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Summary stats
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Rows: {len(output)}")
    print(f"Date range: {output['date'].min()} to {output['date'].max()}")
    print(f"Seasons: {sorted(output['season'].unique())}")
    print(f"\nOutcome distribution:")
    print(output["outcome"].value_counts().sort_index().rename({0: "Home Win", 1: "Draw", 2: "Away Win"}))
    print(f"\nOver 2.5 goals distribution:")
    print(output["over_2_5"].value_counts().sort_index().rename({0: "Under", 1: "Over"}))


if __name__ == "__main__":
    main()

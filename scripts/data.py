"""Data loading, model training, and statistical query helpers."""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")

FEATURE_COLS = [
    "home_rolling_xg", "away_rolling_xg",
    "home_rolling_xga", "away_rolling_xga",
    "home_rolling_points", "away_rolling_points",
    "home_rolling_poss", "away_rolling_poss",
    "home_rolling_sot_ratio", "away_rolling_sot_ratio",
    "xg_diff", "xga_diff",
]

RADAR_AXES = ["xG", "xGA (inv)", "Points", "Poss%", "SoT Ratio"]
HOME_STAT_COLS = [
    "home_rolling_xg", "home_rolling_xga", "home_rolling_points",
    "home_rolling_poss", "home_rolling_sot_ratio",
]
AWAY_STAT_COLS = [
    "away_rolling_xg", "away_rolling_xga", "away_rolling_points",
    "away_rolling_poss", "away_rolling_sot_ratio",
]

# ---------------------------------------------------------------------------
# Module-level state (loaded once at import time)
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

train = df[df["season"] < 2025].copy()
X_train = train[FEATURE_COLS]
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

outcome_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
outcome_model.fit(X_train_sc, train["outcome"])

ou_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
ou_model.fit(X_train_sc, train["over_2_5"])

all_teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))

# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def get_latest_stats(team):
    """Return the 5 rolling stats from the team's most recent match, or None."""
    home_mask = df["HomeTeam"] == team
    away_mask = df["AwayTeam"] == team
    home_rows = df.loc[home_mask].sort_values("date")
    away_rows = df.loc[away_mask].sort_values("date")

    latest_home = home_rows.iloc[-1] if len(home_rows) > 0 else None
    latest_away = away_rows.iloc[-1] if len(away_rows) > 0 else None

    if latest_home is None and latest_away is None:
        return None

    if latest_home is not None and latest_away is not None:
        if latest_home["date"] >= latest_away["date"]:
            row, cols = latest_home, HOME_STAT_COLS
        else:
            row, cols = latest_away, AWAY_STAT_COLS
    elif latest_home is not None:
        row, cols = latest_home, HOME_STAT_COLS
    else:
        row, cols = latest_away, AWAY_STAT_COLS

    return {
        "xg": row[cols[0]],
        "xga": row[cols[1]],
        "points": row[cols[2]],
        "poss": row[cols[3]],
        "sot_ratio": row[cols[4]],
    }


def build_feature_vector(home_stats, away_stats):
    """Assemble the 12-feature vector and scale it."""
    vec = pd.DataFrame([[
        home_stats["xg"], away_stats["xg"],
        home_stats["xga"], away_stats["xga"],
        home_stats["points"], away_stats["points"],
        home_stats["poss"], away_stats["poss"],
        home_stats["sot_ratio"], away_stats["sot_ratio"],
        home_stats["xg"] - away_stats["xg"],
        home_stats["xga"] - away_stats["xga"],
    ]], columns=FEATURE_COLS)
    return scaler.transform(vec)


def get_quick_prediction(home, away):
    """Return (label, probability) for quick fixture-row tip, or None."""
    home_stats = get_latest_stats(home)
    away_stats = get_latest_stats(away)
    if home_stats is None or away_stats is None:
        return None
    try:
        X_vec = build_feature_vector(home_stats, away_stats)
        probs = outcome_model.predict_proba(X_vec)[0]
        p_home, p_draw, p_away = probs[0], probs[1], probs[2]
        if p_home >= p_draw and p_home >= p_away:
            return ("Home", p_home)
        elif p_away >= p_draw:
            return ("Away", p_away)
        else:
            return ("Draw", p_draw)
    except Exception:
        return None


def generate_insights(home_team, away_team):
    """Auto-generate insight strings from data."""
    insights = []

    h2h_mask = (
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team))
        | ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    )
    h2h = df.loc[h2h_mask].sort_values("date", ascending=False).head(5)
    if not h2h.empty:
        wins = sum(
            1 for _, r in h2h.iterrows()
            if (r["outcome"] == 0 and r["HomeTeam"] == home_team)
            or (r["outcome"] == 2 and r["AwayTeam"] == home_team)
        )
        insights.append(f"{home_team} won {wins}/{len(h2h)} recent H2H meetings")

    home_stats = get_latest_stats(home_team)
    away_stats = get_latest_stats(away_team)

    if home_stats:
        pts = home_stats["points"]
        insights.append(f"{home_team} avg points/match: {pts:.1f}")
        insights.append(f"{home_team} rolling xG: {home_stats['xg']:.2f}")
    if away_stats:
        insights.append(f"{away_team} rolling xGA: {away_stats['xga']:.2f}")
        insights.append(f"{away_team} avg points/match: {away_stats['points']:.1f}")

    home_form = get_form_string(home_team)
    if home_form:
        insights.append(f"{home_team} form: {home_form}")
    away_form = get_form_string(away_team)
    if away_form:
        insights.append(f"{away_team} form: {away_form}")

    return insights


def get_form_string(team, n=5):
    """Return last n results as a string like 'WWDLW'."""
    mask = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    matches = df.loc[mask].sort_values("date", ascending=False).head(n)
    if matches.empty:
        return ""
    results = []
    for _, r in matches.iterrows():
        if r["outcome"] == 1:
            results.append("D")
        elif (r["outcome"] == 0 and r["HomeTeam"] == team) or \
             (r["outcome"] == 2 and r["AwayTeam"] == team):
            results.append("W")
        else:
            results.append("L")
    return "".join(results)

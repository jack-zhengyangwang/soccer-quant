"""
Import Premier League data from Kaggle using kagglehub.

Two datasets:
1. Match results (2000-2025) — individual match scores, shots, cards, etc.
2. Team stats (2016-2025) — season-level xG, passes, crosses, etc.
"""

import kagglehub
import pandas as pd
import os
import glob


def download_match_results():
    """
    Download EPL match-level results (2000-2025).
    Source: football-data.co.uk via Kaggle.
    Contains: HomeTeam, AwayTeam, goals, shots, corners, cards per match.
    """
    print("Downloading EPL match results (2000-2025)...")
    path = kagglehub.dataset_download("marcohuiii/english-premier-league-epl-match-data-2000-2025")
    print(f"Downloaded to: {path}")
    return path


def download_match_xg():
    """
    Download Premier League match-level xG data (2021-2025).
    Source: FBref via Kaggle.
    Contains: xG, xGA, possession, shots, formation per match.
    """
    print("Downloading match-level xG data (2021-2025)...")
    path = kagglehub.dataset_download("armin2080/premier-league-matches-dataset-2021-to-2025")
    print(f"Downloaded to: {path}")
    return path


def download_team_stats():
    """
    Download Premier League team stats (2016-2025).
    Source: premierleague.com via Kaggle.
    Contains: xG, shots, passes, season club stats, gameweek tables.
    """
    print("Downloading Premier League team stats (2016-2025)...")
    path = kagglehub.dataset_download("danielijezie/premier-league-data-from-2016-to-2024")
    print(f"Downloaded to: {path}")
    return path


def load_match_results(path):
    """Load and combine all match result CSVs into a single DataFrame."""
    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)

    if not csv_files:
        print("No CSV files found!")
        return None

    # Try loading — could be one combined file or multiple season files
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} matches from {len(dfs)} file(s)")
    return combined


def load_match_xg(path):
    """Load match-level xG data from FBref dataset."""
    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)

    if not csv_files:
        print("No CSV files found!")
        return None

    df = pd.read_csv(csv_files[0])
    print(f"Loaded {len(df)} match records with xG data")
    return df


def load_season_stats(path):
    """Load season-level club stats (contains xG and advanced metrics)."""
    stat_files = glob.glob(os.path.join(path, "**/*season_club_stats*.csv"), recursive=True)

    if not stat_files:
        print("No season stats files found!")
        return None

    dfs = []
    for f in sorted(stat_files):
        try:
            df = pd.read_csv(f)
            # Extract season year from filename (e.g., 2024_season_club_stats.csv)
            basename = os.path.basename(f)
            year = basename.split("_")[0]
            df["season_start_year"] = int(year)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} team-season records from {len(dfs)} season files")
    return combined


def inspect_dataframe(df, name="DataFrame"):
    """Print summary info about a DataFrame."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])


if __name__ == "__main__":
    # Download all datasets
    match_path = download_match_results()
    xg_path = download_match_xg()
    stats_path = download_team_stats()

    # Load match results
    matches = load_match_results(match_path)
    if matches is not None:
        inspect_dataframe(matches, "MATCH RESULTS (2000-2025)")

    # Load match-level xG data
    match_xg = load_match_xg(xg_path)
    if match_xg is not None:
        inspect_dataframe(match_xg, "MATCH-LEVEL xG (2021-2025, FBref)")

    # Load season stats (xG, shots, etc.)
    season_stats = load_season_stats(stats_path)
    if season_stats is not None:
        inspect_dataframe(season_stats, "SEASON CLUB STATS (2016-2025)")

    # Quick summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    if matches is not None:
        print(f"  Match results: {matches.shape[0]} matches, {matches.shape[1]} columns")
        if "Season" in matches.columns:
            print(f"  Seasons: {matches['Season'].nunique()} ({matches['Season'].min()} to {matches['Season'].max()})")
        if "HomeTeam" in matches.columns:
            print(f"  Teams: {matches['HomeTeam'].nunique()}")
    if match_xg is not None:
        print(f"  Match xG: {match_xg.shape[0]} records, {match_xg.shape[1]} columns")
        if "season" in match_xg.columns:
            print(f"  Seasons: {match_xg['season'].nunique()} ({match_xg['season'].min()} to {match_xg['season'].max()})")
    if season_stats is not None:
        print(f"  Season stats: {season_stats.shape[0]} team-seasons, {season_stats.shape[1]} columns")

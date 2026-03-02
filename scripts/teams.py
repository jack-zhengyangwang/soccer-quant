"""Team colors, FPL API integration, and badge helpers."""

from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Team colors
# ---------------------------------------------------------------------------

TEAM_COLORS = {
    "Arsenal": "#EF0107",
    "Aston Villa": "#670E36",
    "Bournemouth": "#DA291C",
    "Brentford": "#E30613",
    "Brighton": "#0057B8",
    "Burnley": "#6C1D45",
    "Chelsea": "#034694",
    "Crystal Palace": "#1B458F",
    "Everton": "#003399",
    "Fulham": "#000000",
    "Ipswich": "#0044AA",
    "Leeds": "#FFCD00",
    "Leicester": "#003090",
    "Liverpool": "#C8102E",
    "Luton": "#F78F1E",
    "Man City": "#6CABDD",
    "Man United": "#DA291C",
    "Newcastle": "#241F20",
    "Norwich": "#FFF200",
    "Nott'm Forest": "#DD0000",
    "Sheffield United": "#EE2737",
    "Southampton": "#D71920",
    "Sunderland": "#EB172B",
    "Tottenham": "#132257",
    "Watford": "#FBEE23",
    "West Brom": "#122F67",
    "West Ham": "#7A263A",
    "Wolves": "#FDB913",
}

TEAM_COLORS_AWAY = {
    "Arsenal": "#E8D59E",
    "Aston Villa": "#95BFE5",
    "Bournemouth": "#000000",
    "Brentford": "#FBB800",
    "Brighton": "#FFCD00",
    "Burnley": "#99D6EA",
    "Chelsea": "#F5A623",
    "Crystal Palace": "#C4122E",
    "Everton": "#F5A623",
    "Fulham": "#CC0000",
    "Ipswich": "#FF6B00",
    "Leeds": "#1D428A",
    "Leicester": "#FDBE11",
    "Liverpool": "#00B2A9",
    "Luton": "#002D62",
    "Man City": "#1C2C5B",
    "Man United": "#FBE122",
    "Newcastle": "#41B6E6",
    "Norwich": "#00A650",
    "Nott'm Forest": "#FFFFFF",
    "Sheffield United": "#1A1A1A",
    "Southampton": "#FFC20E",
    "Sunderland": "#211E1F",
    "Tottenham": "#BFBFBF",
    "Watford": "#000000",
    "West Brom": "#FFFFFF",
    "West Ham": "#1BB1E7",
    "Wolves": "#231F20",
}

_COLOR_DIST_THRESHOLD = 120

# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------


def hex_to_rgb(h):
    """Convert '#RRGGBB' to (r, g, b)."""
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def hex_to_rgba(hex_color, alpha):
    """Convert hex color to rgba string."""
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def color_distance(c1, c2):
    """Simple Euclidean RGB distance."""
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def get_matchup_colors(home_team, away_team):
    """Return (home_color, away_color) guaranteed to be visually distinct."""
    home_c = TEAM_COLORS.get(home_team, "#2ecc71")
    away_c = TEAM_COLORS_AWAY.get(away_team, "#e74c3c")

    if color_distance(home_c, away_c) < _COLOR_DIST_THRESHOLD:
        away_c = TEAM_COLORS.get(away_team, "#e74c3c")

    if color_distance(home_c, away_c) < _COLOR_DIST_THRESHOLD:
        away_c = "#e74c3c" if home_c != "#e74c3c" else "#3498db"

    return home_c, away_c


# ---------------------------------------------------------------------------
# FPL API integration
# ---------------------------------------------------------------------------

FPL_NAME_MAP = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}

_fpl_fixtures_cache = None
_fpl_teams_cache = None
_fpl_badges_cache = None


def fetch_fpl_teams():
    """Fetch team id->name mapping and badge URLs from FPL bootstrap-static."""
    global _fpl_teams_cache, _fpl_badges_cache
    if _fpl_teams_cache is not None:
        return _fpl_teams_cache
    try:
        resp = requests.get(
            "https://fantasy.premierleague.com/api/bootstrap-static/",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        mapping = {}
        badges = {}
        for t in data["teams"]:
            name = t["name"]
            mapped = FPL_NAME_MAP.get(name, name)
            mapping[t["id"]] = mapped
            code = t["code"]
            badges[mapped] = (
                f"https://resources.premierleague.com/premierleague/badges/50/t{code}.png"
            )
        _fpl_teams_cache = mapping
        _fpl_badges_cache = badges
        return mapping
    except Exception:
        return {}


def get_badge_url(team):
    """Return the badge image URL for a team, or None."""
    if _fpl_badges_cache is None:
        fetch_fpl_teams()
    if _fpl_badges_cache:
        return _fpl_badges_cache.get(team)
    return None


def fetch_fpl_fixtures():
    """Fetch all fixtures for the current season from FPL API."""
    global _fpl_fixtures_cache
    if _fpl_fixtures_cache is not None:
        return _fpl_fixtures_cache
    try:
        teams = fetch_fpl_teams()
        if not teams:
            return []
        resp = requests.get(
            "https://fantasy.premierleague.com/api/fixtures/",
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        fixtures = []
        for f in raw:
            home_id = f.get("team_h")
            away_id = f.get("team_a")
            home_name = teams.get(home_id, f"Team {home_id}")
            away_name = teams.get(away_id, f"Team {away_id}")
            kickoff = f.get("kickoff_time")
            dt = None
            if kickoff:
                try:
                    dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
            fixtures.append({
                "home": home_name,
                "away": away_name,
                "kickoff": dt,
                "gameweek": f.get("event"),
                "finished": f.get("finished", False),
                "home_score": f.get("team_h_score"),
                "away_score": f.get("team_a_score"),
            })
        _fpl_fixtures_cache = fixtures
        return fixtures
    except Exception:
        return []


def get_gameweeks(fixtures):
    """Return sorted list of unique gameweek numbers."""
    return sorted({f["gameweek"] for f in fixtures if f["gameweek"] is not None})


def detect_current_gameweek(fixtures):
    """Detect the current/next gameweek based on dates."""
    now = datetime.now(timezone.utc)
    for f in sorted(fixtures, key=lambda x: (x["kickoff"] or datetime.min.replace(tzinfo=timezone.utc))):
        if not f["finished"] and f["gameweek"] is not None:
            return f["gameweek"]
    gws = get_gameweeks(fixtures)
    return gws[-1] if gws else 1

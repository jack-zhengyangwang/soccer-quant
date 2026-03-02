"""Soccer Quant — Fixture Calendar + Match Detail (two-page dashboard).

Routes:
    /                              → Landing page (fixture calendar)
    /match?home=Arsenal&away=Chelsea → Match detail page
"""

import os
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from urllib.parse import parse_qs, quote, urlencode

import numpy as np
import pandas as pd
import requests
from dash import Dash, Input, Output, callback, dcc, html
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Data & model setup
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

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Temporal split
train = df[df["season"] < 2025].copy()

X_train = train[FEATURE_COLS]
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

# Outcome model (multinomial: 0=H, 1=D, 2=A)
outcome_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
outcome_model.fit(X_train_sc, train["outcome"])

# Over/Under model (binary: 0=Under, 1=Over)
ou_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
ou_model.fit(X_train_sc, train["over_2_5"])

all_teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))

# Primary team colors (used when team is home, or the default)
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

# Secondary / away colors (kit alternates — guaranteed distinct from primary)
TEAM_COLORS_AWAY = {
    "Arsenal": "#E8D59E",       # gold away
    "Aston Villa": "#95BFE5",   # sky blue
    "Bournemouth": "#000000",   # black
    "Brentford": "#FBB800",     # yellow
    "Brighton": "#FFCD00",      # yellow
    "Burnley": "#99D6EA",       # light blue
    "Chelsea": "#F5A623",       # amber
    "Crystal Palace": "#C4122E", # red
    "Everton": "#F5A623",       # amber
    "Fulham": "#CC0000",        # red
    "Ipswich": "#FF6B00",       # orange
    "Leeds": "#1D428A",         # blue
    "Leicester": "#FDBE11",     # gold
    "Liverpool": "#00B2A9",     # teal
    "Luton": "#002D62",         # navy
    "Man City": "#1C2C5B",      # navy
    "Man United": "#FBE122",    # yellow
    "Newcastle": "#41B6E6",     # sky blue
    "Norwich": "#00A650",       # green
    "Nott'm Forest": "#FFFFFF", # white
    "Sheffield United": "#1A1A1A", # black
    "Southampton": "#FFC20E",   # yellow
    "Sunderland": "#211E1F",    # black
    "Tottenham": "#BFBFBF",     # silver/white
    "Watford": "#000000",       # black
    "West Brom": "#FFFFFF",     # white
    "West Ham": "#1BB1E7",      # sky blue
    "Wolves": "#231F20",        # black
}

# Minimum perceptual distance (simple RGB) to consider two colors "different"
_COLOR_DIST_THRESHOLD = 120


def _hex_to_rgb(h):
    """Convert '#RRGGBB' to (r, g, b)."""
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _color_distance(c1, c2):
    """Simple Euclidean RGB distance."""
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def get_matchup_colors(home_team, away_team):
    """Return (home_color, away_color) guaranteed to be visually distinct."""
    home_c = TEAM_COLORS.get(home_team, "#2ecc71")
    away_c = TEAM_COLORS_AWAY.get(away_team, "#e74c3c")

    # If the away secondary is still too close to home primary, swap to primary
    if _color_distance(home_c, away_c) < _COLOR_DIST_THRESHOLD:
        away_c = TEAM_COLORS.get(away_team, "#e74c3c")

    # Last resort: if primary vs primary is also too close, force a fallback
    if _color_distance(home_c, away_c) < _COLOR_DIST_THRESHOLD:
        away_c = "#e74c3c" if home_c != "#e74c3c" else "#3498db"

    return home_c, away_c


def _hex_to_rgba(hex_color, alpha):
    """Convert hex color to rgba string."""
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"

# ---------------------------------------------------------------------------
# FPL API integration
# ---------------------------------------------------------------------------

# Map FPL short names → names used in our dataset
FPL_NAME_MAP = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nott'm Forest",
    "Man City": "Man City",
    "Newcastle": "Newcastle",
    "Wolves": "Wolves",
    "Brighton": "Brighton",
    "West Ham": "West Ham",
}

_fpl_fixtures_cache = None
_fpl_teams_cache = None
_fpl_badges_cache = None  # team name → badge URL


def _fetch_fpl_teams():
    """Fetch team id→name mapping and badge URLs from FPL bootstrap-static."""
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
        _fetch_fpl_teams()
    if _fpl_badges_cache:
        return _fpl_badges_cache.get(team)
    return None


def fetch_fpl_fixtures():
    """Fetch all fixtures for the current season from FPL API."""
    global _fpl_fixtures_cache
    if _fpl_fixtures_cache is not None:
        return _fpl_fixtures_cache
    try:
        teams = _fetch_fpl_teams()
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


def _get_gameweeks(fixtures):
    """Return sorted list of unique gameweek numbers."""
    gws = sorted({f["gameweek"] for f in fixtures if f["gameweek"] is not None})
    return gws


def _detect_current_gameweek(fixtures):
    """Detect the current/next gameweek based on dates."""
    now = datetime.now(timezone.utc)
    # Find the first gameweek with an unfinished fixture
    for f in sorted(fixtures, key=lambda x: (x["kickoff"] or datetime.min.replace(tzinfo=timezone.utc))):
        if not f["finished"] and f["gameweek"] is not None:
            return f["gameweek"]
    # Fallback: last gameweek
    gws = _get_gameweeks(fixtures)
    return gws[-1] if gws else 1


# ---------------------------------------------------------------------------
# Shared styles & layout helpers
# ---------------------------------------------------------------------------

DARK_CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0e0"),
)

CHART_CARD_STYLE = {
    "background": "#1e1e2f",
    "borderRadius": "12px",
    "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
    "margin": "8px",
    "padding": "16px",
}

_SECTION_PAD = "0 40px"


def _empty_fig(message):
    """Blank figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="#999"),
    )
    fig.update_layout(
        height=380,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        **DARK_CHART_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Step 1: Reusable data/chart functions
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


def make_header(subtitle="Fixture Calendar"):
    """Shared page header."""
    return html.Div([
        html.H1(
            ["Soccer Quant", html.Span(
                f"  |  {subtitle}",
                style={"fontWeight": "300", "fontSize": "0.65em"},
            )],
            style={"margin": "0", "padding": "24px 40px", "color": "#fff"},
        ),
        html.A(
            "GitHub",
            href="https://github.com/jack-zhengyangwang/soccer-quant",
            target="_blank",
            className="github-link",
            style={
                "position": "absolute", "right": "40px", "top": "50%",
                "transform": "translateY(-50%)",
                "color": "#ccc", "textDecoration": "none",
                "fontSize": "0.9em", "fontWeight": "500",
                "border": "1px solid #555", "borderRadius": "6px",
                "padding": "6px 14px", "transition": "border-color 0.2s",
            },
        ),
    ], style={
        "background": "linear-gradient(135deg, #1a1a2e, #16213e, #0f3460)",
        "marginBottom": "32px", "boxShadow": "0 2px 12px rgba(0,0,0,0.5)",
        "position": "relative",
    })


def make_prediction_cards(home_team, away_team, home_stats, away_stats):
    """Build the outcome probability bar + Over 2.5 pill."""
    X_vec = build_feature_vector(home_stats, away_stats)
    outcome_probs = outcome_model.predict_proba(X_vec)[0]
    ou_probs = ou_model.predict_proba(X_vec)[0]

    p_home, p_draw, p_away = outcome_probs[0], outcome_probs[1], outcome_probs[2]
    p_over = ou_probs[1]

    home_color, away_color = get_matchup_colors(home_team, away_team)
    draw_color = "#555"

    # Build the 3-segment probability bar
    def _segment(label, prob, bg, pos):
        """One segment of the bar. pos = 'left' | 'center' | 'right'."""
        radius = {"left": "10px 0 0 10px", "right": "0 10px 10px 0",
                   "center": "0"}
        # Show label only if segment is wide enough
        show_label = prob >= 0.12
        return html.Div(
            [
                html.Div(f"{prob:.0%}", style={
                    "fontSize": "1.3em", "fontWeight": "bold", "color": "#fff",
                }),
                html.Div(label, style={
                    "fontSize": "0.75em", "color": "rgba(255,255,255,0.8)",
                    "marginTop": "2px",
                }) if show_label else None,
            ],
            className="prob-segment",
            style={
                "flex": f"{prob}",
                "background": bg,
                "borderRadius": radius[pos],
                "display": "flex", "flexDirection": "column",
                "alignItems": "center", "justifyContent": "center",
                "padding": "14px 4px",
                "minWidth": "36px",
                "transition": "flex 0.3s ease",
            },
        )

    # Badge row above the bar
    home_badge = get_badge_url(home_team)
    away_badge = get_badge_url(away_team)
    badge_size = "40px"

    badge_row = html.Div([
        html.Div([
            html.Img(src=home_badge, style={
                "height": badge_size, "width": badge_size,
                "objectFit": "contain",
            }) if home_badge else None,
            html.Span(home_team, style={
                "color": "#ccc", "fontSize": "0.85em", "marginLeft": "8px",
                "fontWeight": "500",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Span("vs", style={
            "color": "#555", "fontSize": "0.9em", "fontWeight": "bold",
        }),
        html.Div([
            html.Span(away_team, style={
                "color": "#ccc", "fontSize": "0.85em", "marginRight": "8px",
                "fontWeight": "500",
            }),
            html.Img(src=away_badge, style={
                "height": badge_size, "width": badge_size,
                "objectFit": "contain",
            }) if away_badge else None,
        ], style={"display": "flex", "alignItems": "center"}),
    ], className="badge-row", style={
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "center", "padding": "0 8px",
        "marginBottom": "10px",
    })

    prob_bar = html.Div([
        _segment(home_team, p_home, home_color, "left"),
        _segment("Draw", p_draw, draw_color, "center"),
        _segment(away_team, p_away, away_color, "right"),
    ], className="prob-bar", style={
        "display": "flex", "width": "100%", "borderRadius": "10px",
        "overflow": "hidden", "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
        "marginBottom": "12px",
    })

    # Over 2.5 pill underneath
    over_pill = html.Div([
        html.Span("Over 2.5 Goals", style={
            "color": "#aaa", "fontSize": "0.8em", "marginRight": "8px",
        }),
        html.Span(f"{p_over:.0%}", style={
            "color": "#fff", "fontWeight": "bold", "fontSize": "0.95em",
        }),
    ], className="over-pill", style={
        "display": "inline-flex", "alignItems": "center",
        "background": f"linear-gradient(90deg, #1e1e2f {100*(1-p_over):.0f}%, #3498db 100%)",
        "borderRadius": "20px", "padding": "8px 20px",
        "border": "1px solid #3a3a5c",
    })

    return html.Div([
        badge_row, prob_bar,
        html.Div(over_pill, style={"textAlign": "center"}),
    ]), (p_home, p_draw, p_away, p_over)


def make_radar_chart(home_team, away_team, home_stats, away_stats):
    """Build radar comparison chart."""
    raw_home = [home_stats["xg"], home_stats["xga"], home_stats["points"],
                home_stats["poss"], home_stats["sot_ratio"]]
    raw_away = [away_stats["xg"], away_stats["xga"], away_stats["points"],
                away_stats["poss"], away_stats["sot_ratio"]]

    all_vals = np.array([raw_home, raw_away])
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    norm_home = ((np.array(raw_home) - mins) / ranges).tolist()
    norm_away = ((np.array(raw_away) - mins) / ranges).tolist()
    norm_home[1] = 1 - norm_home[1]
    norm_away[1] = 1 - norm_away[1]

    home_color, away_color = get_matchup_colors(home_team, away_team)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_home + [norm_home[0]], theta=RADAR_AXES + [RADAR_AXES[0]],
        fill="toself", name=home_team,
        fillcolor=_hex_to_rgba(home_color, 0.15),
        line=dict(color=home_color),
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_away + [norm_away[0]], theta=RADAR_AXES + [RADAR_AXES[0]],
        fill="toself", name=away_team,
        fillcolor=_hex_to_rgba(away_color, 0.15),
        line=dict(color=away_color),
    ))
    fig.update_layout(
        title="Form Comparison (5-match rolling)",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333"),
            angularaxis=dict(gridcolor="#333"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=400, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08,
                    xanchor="center", x=0.5),
        **DARK_CHART_LAYOUT,
    )
    return fig


def make_stat_chart(home_team, away_team, home_stats, away_stats):
    """Build horizontal grouped bar stat comparison."""
    stat_labels = ["xG", "xGA", "Points", "Poss%", "SoT Ratio"]
    raw_home = [home_stats["xg"], home_stats["xga"], home_stats["points"],
                home_stats["poss"], home_stats["sot_ratio"]]
    raw_away = [away_stats["xg"], away_stats["xga"], away_stats["points"],
                away_stats["poss"], away_stats["sot_ratio"]]

    home_color, away_color = get_matchup_colors(home_team, away_team)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=stat_labels, x=raw_home, orientation="h", name=home_team,
        marker_color=home_color,
    ))
    fig.add_trace(go.Bar(
        y=stat_labels, x=raw_away, orientation="h", name=away_team,
        marker_color=away_color,
    ))
    fig.update_layout(
        title="Stat Comparison (latest rolling values)",
        barmode="group", height=400, xaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.08,
                    xanchor="center", x=0.5),
        margin=dict(l=100),
        **DARK_CHART_LAYOUT,
    )
    return fig


def make_h2h_chart(home_team, away_team):
    """Build head-to-head history chart."""
    h2h_mask = (
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team))
        | ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    )
    h2h = df.loc[h2h_mask].sort_values("date", ascending=False).head(10)

    if h2h.empty:
        return _empty_fig("No head-to-head history found"), (0, 0, 0)

    home_color, away_color = get_matchup_colors(home_team, away_team)

    labels, vals, colors, texts = [], [], [], []
    wins, draws, losses = 0, 0, 0
    for _, r in h2h.iterrows():
        date_str = r["date"].strftime("%Y-%m-%d")
        label = f"{r['HomeTeam']} vs {r['AwayTeam']} ({date_str})"
        labels.append(label)

        if r["outcome"] == 1:
            color, val, text = "#95a5a6", 0.5, "D"
            draws += 1
        elif (r["outcome"] == 0 and r["HomeTeam"] == home_team) or \
             (r["outcome"] == 2 and r["AwayTeam"] == home_team):
            color, val, text = home_color, 1.0, "W"
            wins += 1
        else:
            color, val, text = away_color, 1.0, "L"
            losses += 1

        vals.append(val)
        colors.append(color)
        texts.append(text)

    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h",
        marker_color=colors, text=texts,
        textposition="inside", textfont=dict(color="#fff", size=14),
    ))
    fig.update_layout(
        title=f"Head-to-Head (last {len(h2h)}) — "
              f"{home_team}: {wins}W {draws}D {losses}L",
        height=400,
        xaxis=dict(visible=False),
        margin=dict(l=260),
        showlegend=False,
        **DARK_CHART_LAYOUT,
    )
    return fig, (wins, draws, losses)


# ---------------------------------------------------------------------------
# Step 3: New helpers
# ---------------------------------------------------------------------------

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

    # H2H record
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

    # Home form (points)
    home_stats = get_latest_stats(home_team)
    away_stats = get_latest_stats(away_team)

    if home_stats:
        pts = home_stats["points"]
        insights.append(f"{home_team} avg points/match: {pts:.1f}")
        insights.append(f"{home_team} rolling xG: {home_stats['xg']:.2f}")
    if away_stats:
        insights.append(f"{away_team} rolling xGA: {away_stats['xga']:.2f}")
        insights.append(f"{away_team} avg points/match: {away_stats['points']:.1f}")

    # Home recent form string
    home_form = _get_form_string(home_team)
    if home_form:
        insights.append(f"{home_team} form: {home_form}")
    away_form = _get_form_string(away_team)
    if away_form:
        insights.append(f"{away_team} form: {away_form}")

    return insights


def _get_form_string(team, n=5):
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


def make_form_badges(team, n=5):
    """Return colored W/D/L badge elements for last n results."""
    form = _get_form_string(team, n)
    if not form:
        return html.Span("N/A", style={"color": "#666"})
    color_map = {"W": "#2ecc71", "D": "#f1c40f", "L": "#e74c3c"}
    badges = []
    for ch in form:
        badges.append(html.Span(ch, className="form-badge", style={
            "display": "inline-block",
            "width": "28px", "height": "28px", "lineHeight": "28px",
            "textAlign": "center", "borderRadius": "4px",
            "background": color_map.get(ch, "#555"),
            "color": "#fff", "fontWeight": "bold", "fontSize": "0.8em",
            "marginRight": "4px",
        }))
    return html.Div(badges, style={"display": "inline-flex", "gap": "2px"})


def make_fixture_row(fixture, prediction):
    """Build one clickable fixture row for the landing page."""
    home = fixture["home"]
    away = fixture["away"]
    finished = fixture["finished"]

    # Kickoff time
    ko = fixture["kickoff"]
    time_str = ko.strftime("%H:%M") if ko else "TBC"

    # Score or prediction tip
    if finished and fixture["home_score"] is not None:
        result_text = f"{fixture['home_score']} - {fixture['away_score']}"
        result_style = {"color": "#ccc", "fontWeight": "bold"}
    elif prediction:
        label, prob = prediction
        result_text = f"{label} {prob:.0%}"
        result_style = {"color": "#6c63ff", "fontSize": "0.85em"}
    else:
        result_text = "N/A"
        result_style = {"color": "#666", "fontSize": "0.85em"}

    # Link to match detail
    params = urlencode({"home": home, "away": away})
    href = f"/match?{params}"

    return dcc.Link(
        html.Div([
            html.Span(time_str, className="fixture-time", style={
                "flex": "0 0 60px", "color": "#888", "fontSize": "0.9em",
            }),
            html.Span(home, className="fixture-home", style={
                "flex": "1", "textAlign": "right", "fontWeight": "500",
                "color": "#e0e0e0",
            }),
            html.Span("vs", style={
                "flex": "0 0 40px", "textAlign": "center",
                "color": "#555", "fontSize": "0.85em",
            }),
            html.Span(away, className="fixture-away", style={
                "flex": "1", "textAlign": "left", "fontWeight": "500",
                "color": "#e0e0e0",
            }),
            html.Span(result_text, className="fixture-tip", style={
                **result_style, "flex": "0 0 100px", "textAlign": "right",
            }),
            html.Span("\u203a", style={
                "flex": "0 0 24px", "textAlign": "center",
                "color": "#555", "fontSize": "1.4em",
            }),
        ], className="fixture-row"),
        href=href,
        style={"textDecoration": "none"},
    )


def make_stat_glossary():
    """Collapsible stat glossary."""
    return html.Div([
        html.Details([
            html.Summary("What do these stats mean?"),
            html.Div([
                html.Div([
                    html.Span("xG", style={"fontWeight": "bold", "color": "#ddd"}),
                    " (Expected Goals) \u2014 Average goal-scoring quality of shots taken. "
                    "Higher means more dangerous chances created.",
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Span("xGA", style={"fontWeight": "bold", "color": "#ddd"}),
                    " (Expected Goals Against) \u2014 Average quality of shots conceded. "
                    "Lower means a stronger defence.",
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Span("Points", style={"fontWeight": "bold", "color": "#ddd"}),
                    " \u2014 Average league points per match (3 for a win, 1 for a draw). "
                    "Captures recent form regardless of underlying metrics.",
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Span("Poss%", style={"fontWeight": "bold", "color": "#ddd"}),
                    " (Possession) \u2014 Share of total match possession. "
                    "Indicates control of the game tempo.",
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Span("SoT Ratio", style={"fontWeight": "bold", "color": "#ddd"}),
                    " (Shots on Target Ratio) \u2014 Proportion of shots that hit the target. "
                    "Reflects shooting accuracy and finishing quality.",
                ]),
            ], style={"padding": "10px 0 4px", "color": "#999",
                       "fontSize": "0.88em", "lineHeight": "1.6"}),
        ]),
    ], style={"padding": "0 40px 24px"})


# ---------------------------------------------------------------------------
# Step 4: Dash app with dcc.Location routing
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Soccer Quant"

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
], style={
    "fontFamily": "'Inter', 'Segoe UI', Arial, sans-serif",
    "maxWidth": "1400px", "margin": "0 auto",
    "background": "#0f1117", "minHeight": "100vh",
    "color": "#e0e0e0", "paddingBottom": "48px",
})


# ---------------------------------------------------------------------------
# Step 5: Landing page (fixture calendar)
# ---------------------------------------------------------------------------

def landing_layout():
    """Build the fixture calendar landing page."""
    fixtures = fetch_fpl_fixtures()
    gameweeks = _get_gameweeks(fixtures)
    current_gw = _detect_current_gameweek(fixtures) if fixtures else 1

    gw_options = [{"label": f"Gameweek {gw}", "value": gw} for gw in gameweeks]

    return html.Div([
        make_header("Fixture Calendar"),

        # Controls row
        html.Div([
            html.Div([
                html.Label("Gameweek", style={
                    "fontWeight": "600", "marginBottom": "8px", "color": "#aaa",
                    "fontSize": "0.8em", "textTransform": "uppercase",
                    "letterSpacing": "0.08em", "display": "block",
                }),
                dcc.Dropdown(
                    id="gw-dropdown",
                    options=gw_options,
                    value=current_gw,
                    clearable=False,
                    searchable=False,
                    placeholder="Select gameweek...",
                ),
            ], style={"flex": "1", "minWidth": "0", "maxWidth": "280px"}),

            html.Div([
                html.Label("Filter by date", style={
                    "fontWeight": "600", "marginBottom": "8px", "color": "#aaa",
                    "fontSize": "0.8em", "textTransform": "uppercase",
                    "letterSpacing": "0.08em", "display": "block",
                }),
                dcc.DatePickerSingle(
                    id="date-picker",
                    placeholder="All dates",
                    clearable=True,
                    style={"width": "100%"},
                ),
            ], style={"flex": "0 0 200px"}),
        ], style={
            "padding": "0 40px 24px", "display": "flex",
            "alignItems": "flex-end", "gap": "24px",
        }),

        # Fixture table
        html.Div(id="fixture-table", style={"padding": "0 40px"}),

        # Fallback message if API is down
        html.Div(
            id="fixture-fallback",
            style={"padding": "0 40px", "display": "none"},
        ),
    ])


# ---------------------------------------------------------------------------
# Step 6: Match detail page
# ---------------------------------------------------------------------------

def match_detail_layout(home_team, away_team):
    """Build the match detail page for a given matchup."""
    # Back link
    back = dcc.Link(
        "\u2190 Back to Fixtures",
        href="/",
        className="back-link",
        style={
            "color": "#6c63ff", "textDecoration": "none",
            "fontSize": "0.95em", "fontWeight": "500",
            "display": "inline-block", "padding": "0 40px 16px",
        },
    )

    header = make_header("Match Predictor")

    # Check for data availability
    home_stats = get_latest_stats(home_team)
    away_stats = get_latest_stats(away_team)

    if home_stats is None or away_stats is None:
        missing = []
        if home_stats is None:
            missing.append(home_team)
        if away_stats is None:
            missing.append(away_team)
        no_data_msg = html.Div([
            header, back,
            html.Div([
                html.H2(f"{home_team} vs {away_team}", style={
                    "color": "#e0e0e0", "textAlign": "center",
                    "marginBottom": "16px",
                }),
                html.Div([
                    html.P(
                        f"No historical data available for: {', '.join(missing)}",
                        style={"color": "#f1c40f", "fontSize": "1.1em",
                               "textAlign": "center"},
                    ),
                    html.P(
                        "This team may be newly promoted or not in our dataset.",
                        style={"color": "#888", "textAlign": "center"},
                    ),
                ], style={
                    "background": "#1e1e2f", "borderRadius": "12px",
                    "padding": "40px", "margin": "20px 40px",
                    "textAlign": "center",
                }),
            ]),
        ])
        return no_data_msg

    # Build components
    pred_cards, (p_home, p_draw, p_away, p_over) = make_prediction_cards(
        home_team, away_team, home_stats, away_stats,
    )
    insights = generate_insights(home_team, away_team)
    radar_fig = make_radar_chart(home_team, away_team, home_stats, away_stats)
    stat_fig = make_stat_chart(home_team, away_team, home_stats, away_stats)
    h2h_fig, _ = make_h2h_chart(home_team, away_team)

    # Find fixture info from FPL
    fixture_info = _find_fixture_info(home_team, away_team)
    kickoff_str = ""
    gw_str = ""
    if fixture_info:
        if fixture_info["kickoff"]:
            kickoff_str = fixture_info["kickoff"].strftime("%a %d %b, %H:%M")
        if fixture_info["gameweek"]:
            gw_str = f"GW {fixture_info['gameweek']}"

    return html.Div([
        header,
        back,

        # Two-column top section
        html.Div([
            # LEFT column: matchup + predictions + form
            html.Div([
                # Matchup header
                html.Div([
                    html.H2(f"{home_team} vs {away_team}", style={
                        "margin": "0 0 8px", "color": "#e0e0e0",
                        "fontSize": "1.5em",
                    }),
                    html.Div([
                        html.Span(kickoff_str, style={
                            "color": "#888", "fontSize": "0.9em",
                        }) if kickoff_str else None,
                        html.Span(f"  \u2022  {gw_str}", style={
                            "color": "#888", "fontSize": "0.9em",
                        }) if gw_str else None,
                    ], style={"marginBottom": "20px"}),
                ], style={"padding": "0 0 8px"}),

                # Prediction cards
                html.Div(pred_cards, style={
                    "textAlign": "center", "marginBottom": "24px",
                }),

                # Form badges
                html.Div([
                    html.Div([
                        html.Span(f"{home_team}:", style={
                            "color": "#aaa", "fontSize": "0.85em",
                            "display": "inline-block", "width": "140px",
                            "textAlign": "right", "marginRight": "12px",
                        }),
                        make_form_badges(home_team),
                    ], style={
                        "display": "flex", "alignItems": "center",
                        "marginBottom": "8px",
                    }),
                    html.Div([
                        html.Span(f"{away_team}:", style={
                            "color": "#aaa", "fontSize": "0.85em",
                            "display": "inline-block", "width": "140px",
                            "textAlign": "right", "marginRight": "12px",
                        }),
                        make_form_badges(away_team),
                    ], style={
                        "display": "flex", "alignItems": "center",
                    }),
                ], style={
                    "background": "#1e1e2f", "borderRadius": "12px",
                    "padding": "16px 20px", "marginBottom": "16px",
                }),
            ], className="detail-left", style={
                "flex": "1", "minWidth": "0", "padding": "0 8px",
            }),

            # RIGHT column: insights + news placeholder
            html.Div([
                # Key Insights panel
                html.Div([
                    html.H3("Key Insights", style={
                        "color": "#e0e0e0", "marginTop": "0",
                        "marginBottom": "12px", "fontSize": "1.1em",
                    }),
                    html.Ul([
                        html.Li(insight, style={
                            "color": "#ccc", "marginBottom": "8px",
                            "fontSize": "0.9em", "lineHeight": "1.5",
                        })
                        for insight in insights
                    ], style={"paddingLeft": "18px", "margin": "0"}),
                ], className="insights-panel", style={
                    "background": "#1e1e2f", "borderRadius": "12px",
                    "padding": "20px 24px", "marginBottom": "16px",
                    "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
                }),

                # News placeholder
                html.Div([
                    html.H3("News & Updates", style={
                        "color": "#bbb", "marginTop": "0",
                    }),
                    html.P("Coming soon...", style={
                        "color": "#666", "fontStyle": "italic",
                        "textAlign": "center", "padding": "20px 0",
                    }),
                ], style={
                    "border": "2px dashed #444", "borderRadius": "12px",
                    "padding": "20px 24px",
                }),
            ], className="detail-right", style={
                "flex": "1", "minWidth": "0", "padding": "0 8px",
            }),
        ], className="detail-top-row", style={
            "display": "flex", "gap": "16px", "padding": "0 32px 24px",
        }),

        # Charts row: Radar + Stat comparison
        html.Div([
            html.Div(
                html.Div(dcc.Graph(figure=radar_fig), style=CHART_CARD_STYLE),
                style={"flex": "1", "minWidth": "0"},
            ),
            html.Div(
                html.Div(dcc.Graph(figure=stat_fig), style=CHART_CARD_STYLE),
                style={"flex": "1", "minWidth": "0"},
            ),
        ], style={"display": "flex", "gap": "16px", "padding": "0 32px 24px"}),

        # H2H full width
        html.Div(
            html.Div(dcc.Graph(figure=h2h_fig), style=CHART_CARD_STYLE),
            style={"padding": "0 32px 24px"},
        ),

        # Stat glossary
        make_stat_glossary(),
    ])


def _find_fixture_info(home_team, away_team):
    """Find the FPL fixture entry for this matchup."""
    fixtures = fetch_fpl_fixtures()
    for f in fixtures:
        if f["home"] == home_team and f["away"] == away_team:
            return f
    return None


# ---------------------------------------------------------------------------
# Routing callback
# ---------------------------------------------------------------------------

@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("url", "search"),
)
def display_page(pathname, search):
    """Route to the correct page based on URL."""
    if pathname == "/match":
        params = parse_qs(search.lstrip("?")) if search else {}
        home = params.get("home", [None])[0]
        away = params.get("away", [None])[0]
        if home and away:
            return match_detail_layout(home, away)
        # Missing params — redirect to landing
        return landing_layout()
    # Default: landing page
    return landing_layout()


# ---------------------------------------------------------------------------
# Landing page callback: update fixture table
# ---------------------------------------------------------------------------

@callback(
    Output("fixture-table", "children"),
    Input("gw-dropdown", "value"),
    Input("date-picker", "date"),
)
def update_fixture_table(selected_gw, selected_date):
    """Filter and render fixture rows for the selected gameweek/date."""
    fixtures = fetch_fpl_fixtures()

    if not fixtures:
        return html.Div([
            html.P("Fixtures unavailable — could not reach the FPL API.",
                   style={"color": "#f1c40f", "fontSize": "1.1em",
                           "textAlign": "center", "padding": "40px"}),
            html.P("Try refreshing the page or check your internet connection.",
                   style={"color": "#888", "textAlign": "center"}),
        ])

    # Filter by gameweek
    filtered = [f for f in fixtures if f["gameweek"] == selected_gw]

    # Filter by date if selected
    if selected_date:
        try:
            sel = datetime.fromisoformat(selected_date).date()
            filtered = [
                f for f in filtered
                if f["kickoff"] and f["kickoff"].date() == sel
            ]
        except (ValueError, TypeError):
            pass

    if not filtered:
        return html.P(
            "No fixtures found for this selection.",
            style={"color": "#888", "textAlign": "center", "padding": "40px"},
        )

    # Sort by kickoff time
    filtered.sort(key=lambda f: f["kickoff"] or datetime.min.replace(tzinfo=timezone.utc))

    # Build rows with quick predictions
    rows = []
    for fix in filtered:
        pred = get_quick_prediction(fix["home"], fix["away"])
        rows.append(make_fixture_row(fix, pred))

    return html.Div(rows, className="fixture-list")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8050))
    URL = f"http://127.0.0.1:{PORT}"
    print(f"Starting Soccer Quant at {URL}")

    if "--no-open" not in sys.argv and not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.2, webbrowser.open, args=[URL]).start()

    app.run(host="0.0.0.0", port=PORT, debug=True)

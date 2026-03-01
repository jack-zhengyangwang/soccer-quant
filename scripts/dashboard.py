"""Match Predictor dashboard — pick Home & Away teams, see predictions."""

import os

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Data & model setup (unchanged pattern: load CSV, temporal split, train)
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

# Radar axes: the 5 rolling stats we compare teams on
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

# ---------------------------------------------------------------------------
# Team list & helpers
# ---------------------------------------------------------------------------

all_teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))


def get_latest_stats(team):
    """Return the 5 rolling stats from the team's most recent match."""
    home_mask = df["HomeTeam"] == team
    away_mask = df["AwayTeam"] == team
    home_rows = df.loc[home_mask].sort_values("date")
    away_rows = df.loc[away_mask].sort_values("date")

    latest_home = home_rows.iloc[-1] if len(home_rows) > 0 else None
    latest_away = away_rows.iloc[-1] if len(away_rows) > 0 else None

    # Pick whichever appearance is more recent
    if latest_home is not None and latest_away is not None:
        if latest_home["date"] >= latest_away["date"]:
            row = latest_home
            cols = HOME_STAT_COLS
        else:
            row = latest_away
            cols = AWAY_STAT_COLS
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
        home_stats["xg"] - away_stats["xg"],      # xg_diff
        home_stats["xga"] - away_stats["xga"],     # xga_diff
    ]], columns=FEATURE_COLS)
    return scaler.transform(vec)


def _empty_fig(message):
    """Blank figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="#999"),
    )
    fig.update_layout(
        template="plotly_white", height=380,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "Soccer Quant | Match Predictor"

# Shared styles
CARD_BASE = {
    "display": "inline-block", "width": "23%", "textAlign": "center",
    "borderRadius": "10px", "padding": "18px 8px", "margin": "0 0.5%",
    "color": "#fff", "fontFamily": "Arial, sans-serif",
}

app.layout = html.Div([
    # Header
    html.Div([
        html.H1(
            ["Soccer Quant", html.Span("  |  Match Predictor",
             style={"fontWeight": "300", "fontSize": "0.65em"})],
            style={"margin": "0", "padding": "18px 24px", "color": "#fff"},
        ),
    ], style={"background": "linear-gradient(135deg, #1a1a2e, #16213e)",
              "marginBottom": "20px"}),

    # Team selectors
    html.Div([
        html.Div([
            html.Label("Home Team", style={"fontWeight": "bold", "marginBottom": "4px"}),
            dcc.Dropdown(
                id="home-team",
                options=[{"label": t, "value": t} for t in all_teams],
                value="Arsenal",
                clearable=False,
            ),
        ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top"}),

        html.Div("vs.", style={
            "width": "20%", "display": "inline-block", "textAlign": "center",
            "fontSize": "1.6em", "fontWeight": "bold", "color": "#888",
            "verticalAlign": "middle", "paddingTop": "24px",
        }),

        html.Div([
            html.Label("Away Team", style={"fontWeight": "bold", "marginBottom": "4px"}),
            dcc.Dropdown(
                id="away-team",
                options=[{"label": t, "value": t} for t in all_teams],
                value="Chelsea",
                clearable=False,
            ),
        ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"padding": "0 24px 16px", "textAlign": "center"}),

    # Prediction cards
    html.Div(id="prediction-cards", style={"padding": "0 24px 16px", "textAlign": "center"}),

    # Stat glossary
    html.Div([
        html.Details([
            html.Summary("What do these stats mean?", style={
                "cursor": "pointer", "fontWeight": "bold", "color": "#555",
                "fontSize": "0.95em",
            }),
            html.Div([
                html.Div([
                    html.Span("xG", style={"fontWeight": "bold"}),
                    " (Expected Goals) — Average goal-scoring quality of shots taken. "
                    "Higher means more dangerous chances created.",
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Span("xGA", style={"fontWeight": "bold"}),
                    " (Expected Goals Against) — Average quality of shots conceded. "
                    "Lower means a stronger defence.",
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Span("Points", style={"fontWeight": "bold"}),
                    " — Average league points per match (3 for a win, 1 for a draw). "
                    "Captures recent form regardless of underlying metrics.",
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Span("Poss%", style={"fontWeight": "bold"}),
                    " (Possession) — Share of total match possession. "
                    "Indicates control of the game tempo.",
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Span("SoT Ratio", style={"fontWeight": "bold"}),
                    " (Shots on Target Ratio) — Proportion of shots that hit the target. "
                    "Reflects shooting accuracy and finishing quality.",
                ]),
            ], style={"padding": "10px 0 4px", "color": "#666",
                       "fontSize": "0.88em", "lineHeight": "1.6"}),
        ]),
    ], style={"padding": "0 28px 12px"}),

    # Charts row: Radar + Stat comparison
    html.Div([
        html.Div(dcc.Graph(id="radar-chart"),
                 style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div(dcc.Graph(id="stat-comparison"),
                 style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
    ]),

    # Bottom row: H2H + Qualitative placeholder
    html.Div([
        html.Div(dcc.Graph(id="h2h-chart"),
                 style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            html.Div([
                html.H3("Qualitative Intelligence", style={"color": "#555", "marginTop": "0"}),
                html.Ul([
                    html.Li("Injury / suspension tracker"),
                    html.Li("Manager tactical notes"),
                    html.Li("Transfer window impact"),
                    html.Li("Fixture congestion analysis"),
                ], style={"color": "#999", "lineHeight": "2"}),
                html.P("Coming Soon", style={
                    "textAlign": "center", "color": "#bbb",
                    "fontSize": "1.1em", "fontStyle": "italic",
                }),
            ], style={
                "border": "2px dashed #ddd", "borderRadius": "12px",
                "padding": "24px", "margin": "20px",
                "minHeight": "300px",
            }),
        ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
    ]),

], style={"fontFamily": "Arial, sans-serif", "maxWidth": "1400px", "margin": "0 auto",
          "background": "#fafafa", "minHeight": "100vh"})


# ---------------------------------------------------------------------------
# Single callback — both dropdowns → all outputs
# ---------------------------------------------------------------------------

@app.callback(
    Output("prediction-cards", "children"),
    Output("radar-chart", "figure"),
    Output("stat-comparison", "figure"),
    Output("h2h-chart", "figure"),
    Input("home-team", "value"),
    Input("away-team", "value"),
)
def update_all(home_team, away_team):
    # Guard: same team
    if home_team == away_team:
        msg = "Please select two different teams."
        empty_cards = html.P(msg, style={"color": "#e74c3c", "fontSize": "1.2em",
                                         "padding": "20px"})
        return empty_cards, _empty_fig(msg), _empty_fig(msg), _empty_fig(msg)

    # --- Fetch latest stats ---
    home_stats = get_latest_stats(home_team)
    away_stats = get_latest_stats(away_team)

    # --- Predict ---
    X_vec = build_feature_vector(home_stats, away_stats)
    outcome_probs = outcome_model.predict_proba(X_vec)[0]  # [P(H), P(D), P(A)]
    ou_probs = ou_model.predict_proba(X_vec)[0]             # [P(Under), P(Over)]

    p_home = outcome_probs[0]
    p_draw = outcome_probs[1]
    p_away = outcome_probs[2]
    p_over = ou_probs[1]

    # --- Prediction cards ---
    def _card(label, prob, base_color):
        alpha = 0.4 + 0.6 * prob  # intensity proportional to confidence
        r, g, b = base_color
        bg = f"rgba({r},{g},{b},{alpha:.2f})"
        return html.Div([
            html.Div(f"{prob:.0%}", style={"fontSize": "2em", "fontWeight": "bold"}),
            html.Div(label, style={"fontSize": "0.9em", "marginTop": "4px"}),
        ], style={**CARD_BASE, "background": bg})

    cards = html.Div([
        _card(f"{home_team} Win", p_home, (46, 204, 113)),    # green
        _card("Draw", p_draw, (241, 196, 15)),                 # yellow
        _card(f"{away_team} Win", p_away, (231, 76, 60)),      # red
        _card("Over 2.5", p_over, (52, 152, 219)),             # blue
    ])

    # --- Radar chart ---
    raw_home = [home_stats["xg"], home_stats["xga"], home_stats["points"],
                home_stats["poss"], home_stats["sot_ratio"]]
    raw_away = [away_stats["xg"], away_stats["xga"], away_stats["points"],
                away_stats["poss"], away_stats["sot_ratio"]]

    # Min-max normalize across both teams for the radar
    all_vals = np.array([raw_home, raw_away])
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid division by zero

    norm_home = ((np.array(raw_home) - mins) / ranges).tolist()
    norm_away = ((np.array(raw_away) - mins) / ranges).tolist()

    # Invert xGA axis (lower is better)
    norm_home[1] = 1 - norm_home[1]
    norm_away[1] = 1 - norm_away[1]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=norm_home + [norm_home[0]], theta=RADAR_AXES + [RADAR_AXES[0]],
        fill="toself", name=home_team, fillcolor="rgba(46,204,113,0.15)",
        line=dict(color="#2ecc71"),
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=norm_away + [norm_away[0]], theta=RADAR_AXES + [RADAR_AXES[0]],
        fill="toself", name=away_team, fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="#e74c3c"),
    ))
    radar_fig.update_layout(
        title="Form Comparison (5-match rolling)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white", height=400, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
    )

    # --- Stat comparison (horizontal grouped bar) ---
    stat_labels = ["xG", "xGA", "Points", "Poss%", "SoT Ratio"]
    stat_fig = go.Figure()
    stat_fig.add_trace(go.Bar(
        y=stat_labels, x=raw_home, orientation="h", name=home_team,
        marker_color="#2ecc71",
    ))
    stat_fig.add_trace(go.Bar(
        y=stat_labels, x=raw_away, orientation="h", name=away_team,
        marker_color="#e74c3c",
    ))
    stat_fig.update_layout(
        title="Stat Comparison (latest rolling values)",
        barmode="group", template="plotly_white", height=400,
        xaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        margin=dict(l=100),
    )

    # --- Head-to-head history ---
    h2h_mask = (
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team))
        | ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    )
    h2h = df.loc[h2h_mask].sort_values("date", ascending=False).head(10)

    if h2h.empty:
        h2h_fig = _empty_fig("No head-to-head history found")
    else:
        h2h_labels, h2h_vals, h2h_colors, h2h_texts = [], [], [], []
        wins, draws, losses = 0, 0, 0
        for _, r in h2h.iterrows():
            date_str = r["date"].strftime("%Y-%m-%d")
            label = f"{r['HomeTeam']} vs {r['AwayTeam']} ({date_str})"
            h2h_labels.append(label)

            # Determine result from home_team's perspective
            if r["outcome"] == 1:  # draw
                color, val, text = "#95a5a6", 0.5, "D"
                draws += 1
            elif (r["outcome"] == 0 and r["HomeTeam"] == home_team) or \
                 (r["outcome"] == 2 and r["AwayTeam"] == home_team):
                color, val, text = "#2ecc71", 1.0, "W"
                wins += 1
            else:
                color, val, text = "#e74c3c", 1.0, "L"
                losses += 1

            h2h_vals.append(val)
            h2h_colors.append(color)
            h2h_texts.append(text)

        h2h_fig = go.Figure(go.Bar(
            y=h2h_labels, x=h2h_vals, orientation="h",
            marker_color=h2h_colors, text=h2h_texts,
            textposition="inside", textfont=dict(color="#fff", size=14),
        ))
        h2h_fig.update_layout(
            title=f"Head-to-Head (last {len(h2h)}) — "
                  f"{home_team}: {wins}W {draws}D {losses}L",
            template="plotly_white", height=400,
            xaxis=dict(visible=False),
            margin=dict(l=260),
            showlegend=False,
        )

    return cards, radar_fig, stat_fig, h2h_fig


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Soccer Quant — Match Predictor at http://127.0.0.1:8050")
    app.run(debug=True)

"""Interactive Plotly Dash dashboard for Soccer Quant predictions."""

import os

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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

FEATURE_LABELS = [
    "Home xG", "Away xG",
    "Home xGA", "Away xGA",
    "Home Points", "Away Points",
    "Home Poss%", "Away Poss%",
    "Home SoT Ratio", "Away SoT Ratio",
    "xG Diff", "xGA Diff",
]

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Temporal split
train = df[df["season"] < 2025].copy()
test = df[df["season"] == 2025].copy()

X_train = train[FEATURE_COLS]
X_test = test[FEATURE_COLS]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Outcome model (multinomial: 0=H, 1=D, 2=A)
outcome_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
outcome_model.fit(X_train_sc, train["outcome"])

# Over/Under model (binary: 0=Under, 1=Over)
ou_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
ou_model.fit(X_train_sc, train["over_2_5"])

# Pre-compute test-set predictions for confusion matrix
y_pred = outcome_model.predict(X_test_sc)
cm = confusion_matrix(test["outcome"], y_pred, labels=[0, 1, 2])

# Pre-compute prediction probabilities for every row (used in callbacks)
X_all_sc = scaler.transform(df[FEATURE_COLS])
df["p_home"] = outcome_model.predict_proba(X_all_sc)[:, 0]
df["p_draw"] = outcome_model.predict_proba(X_all_sc)[:, 1]
df["p_away"] = outcome_model.predict_proba(X_all_sc)[:, 2]

# ---------------------------------------------------------------------------
# Static charts
# ---------------------------------------------------------------------------


def make_feature_importance_fig():
    """Horizontal grouped bar: |coef| averaged across outcome classes + O/U."""
    outcome_importance = np.abs(outcome_model.coef_).mean(axis=0)
    ou_importance = np.abs(ou_model.coef_).squeeze()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=FEATURE_LABELS, x=outcome_importance,
        orientation="h", name="Match Outcome",
        marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        y=FEATURE_LABELS, x=ou_importance,
        orientation="h", name="Over/Under 2.5",
        marker_color="#EF553B",
    ))
    fig.update_layout(
        title="Feature Importance (|coefficient| magnitude)",
        xaxis_title="Mean |Coefficient|",
        barmode="group",
        template="plotly_white",
        height=480,
        margin=dict(l=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_confusion_matrix_fig():
    """Heatmap of outcome predictions vs actuals on 2024-25 test set."""
    labels = ["Home Win", "Draw", "Away Win"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=18),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Confusion Matrix — 2024-25 Test Set",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        height=480,
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Dropdown / checklist options
# ---------------------------------------------------------------------------

all_teams = sorted(df["HomeTeam"].unique())
all_seasons = sorted(df["season"].unique())

# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "Soccer Quant Dashboard"

app.layout = html.Div([
    html.H1("Soccer Quant Dashboard", style={"textAlign": "center", "padding": "16px 0 0"}),
    html.P(
        "Interactive exploration of Premier League prediction models",
        style={"textAlign": "center", "color": "#666", "marginBottom": "20px"},
    ),

    # Controls row
    html.Div([
        html.Div([
            html.Label("Team", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="team-dropdown",
                options=[{"label": t, "value": t} for t in all_teams],
                value="Arsenal",
                clearable=False,
            ),
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "20px"}),

        html.Div([
            html.Label("Seasons", style={"fontWeight": "bold"}),
            dcc.Checklist(
                id="season-checklist",
                options=[{"label": str(s), "value": s} for s in all_seasons],
                value=all_seasons,
                inline=True,
                labelStyle={"marginRight": "12px"},
            ),
        ], style={"width": "65%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"padding": "0 24px 12px"}),

    # Charts row 1
    html.Div([
        html.Div(dcc.Graph(id="feature-importance", figure=make_feature_importance_fig()),
                 style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div(dcc.Graph(id="rolling-form"),
                 style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
    ]),

    # Charts row 2
    html.Div([
        html.Div(dcc.Graph(id="match-predictions"),
                 style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div(dcc.Graph(id="confusion-matrix", figure=make_confusion_matrix_fig()),
                 style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
    ]),
], style={"fontFamily": "Arial, sans-serif", "maxWidth": "1400px", "margin": "0 auto"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _team_matches(team, seasons):
    """Return rows where *team* played (home or away), filtered by seasons."""
    mask = (
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))
        & df["season"].isin(seasons)
    )
    return df.loc[mask].sort_values("date").copy()


def _extract_team_stats(matches, team):
    """Extract rolling stats from the team's perspective regardless of venue."""
    rows = []
    for _, r in matches.iterrows():
        if r["HomeTeam"] == team:
            rows.append({
                "date": r["date"],
                "xg": r["home_rolling_xg"],
                "xga": r["home_rolling_xga"],
                "points": r["home_rolling_points"],
                "opponent": r["AwayTeam"],
                "venue": "Home",
            })
        else:
            rows.append({
                "date": r["date"],
                "xg": r["away_rolling_xg"],
                "xga": r["away_rolling_xga"],
                "points": r["away_rolling_points"],
                "opponent": r["HomeTeam"],
                "venue": "Away",
            })
    return pd.DataFrame(rows)


@app.callback(
    Output("rolling-form", "figure"),
    Input("team-dropdown", "value"),
    Input("season-checklist", "value"),
)
def update_rolling_form(team, seasons):
    matches = _team_matches(team, seasons)
    if matches.empty:
        return _empty_fig(f"No matches found for {team}")

    stats = _extract_team_stats(matches, team)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=stats["date"], y=stats["xg"], name="Rolling xG",
        mode="lines", line=dict(color="#636EFA"),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=stats["date"], y=stats["xga"], name="Rolling xGA",
        mode="lines", line=dict(color="#EF553B"),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=stats["date"], y=stats["points"], name="Rolling Points",
        mode="lines", line=dict(color="#00CC96", dash="dot"),
    ), secondary_y=True)

    fig.update_layout(
        title=f"{team} — Rolling Form (5-match avg)",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="xG / xGA", secondary_y=False)
    fig.update_yaxes(title_text="Points", secondary_y=True)
    return fig


@app.callback(
    Output("match-predictions", "figure"),
    Input("team-dropdown", "value"),
    Input("season-checklist", "value"),
)
def update_match_predictions(team, seasons):
    matches = _team_matches(team, seasons)
    if matches.empty:
        return _empty_fig(f"No matches found for {team}")

    labels, p_win, p_draw, p_loss = [], [], [], []
    for _, r in matches.iterrows():
        if r["HomeTeam"] == team:
            labels.append(f"vs {r['AwayTeam']} (H)")
            p_win.append(r["p_home"])
            p_draw.append(r["p_draw"])
            p_loss.append(r["p_away"])
        else:
            labels.append(f"vs {r['HomeTeam']} (A)")
            p_win.append(r["p_away"])
            p_draw.append(r["p_draw"])
            p_loss.append(r["p_home"])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=p_win, name="Win", marker_color="#00CC96"))
    fig.add_trace(go.Bar(x=labels, y=p_draw, name="Draw", marker_color="#FFA15A"))
    fig.add_trace(go.Bar(x=labels, y=p_loss, name="Loss", marker_color="#EF553B"))
    fig.update_layout(
        barmode="stack",
        title=f"{team} — Match Prediction Probabilities",
        yaxis_title="Probability",
        template="plotly_white",
        height=480,
        xaxis=dict(tickangle=-45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _empty_fig(message):
    """Return a blank figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16, color="#999"))
    fig.update_layout(template="plotly_white", height=480,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Soccer Quant Dashboard at http://127.0.0.1:8050")
    app.run(debug=True)

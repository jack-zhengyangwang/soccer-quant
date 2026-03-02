"""Soccer Quant — Fixture Calendar + Match Detail + User Profile (multi-page dashboard).

Routes:
    /                              → Landing page (fixture calendar)
    /match?home=Arsenal&away=Chelsea → Match detail page
    /profile                       → User profile (login / register / settings)
"""

import os
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from urllib.parse import parse_qs

from dash import Dash, Input, Output, State, callback, ctx, dcc, html, ALL

from data import (
    generate_insights,
    get_latest_stats,
    get_quick_prediction,
)
from teams import fetch_fpl_fixtures, get_gameweeks, detect_current_gameweek
from components import (
    CHART_CARD_STYLE,
    make_fixture_row,
    make_form_badges,
    make_h2h_panel,
    make_header,
    make_news_panel,
    make_prediction_cards,
    make_stat_chart,
    make_stat_glossary,
)
from qualitative import get_match_news, get_news_highlights
from auth import register_user, login_user, get_user, update_user
from user_profile import login_layout, profile_layout

# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Soccer Quant"
server = app.server  # WSGI entry point for gunicorn

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="user-session", storage_type="session"),
    html.Div(id="page-content"),
], style={
    "fontFamily": "'Inter', 'Segoe UI', Arial, sans-serif",
    "maxWidth": "1400px", "margin": "0 auto",
    "background": "#0f1117", "minHeight": "100vh",
    "color": "#e0e0e0", "paddingBottom": "48px",
})


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------


def landing_layout():
    """Build the fixture calendar landing page."""
    fixtures = fetch_fpl_fixtures()
    gameweeks = get_gameweeks(fixtures)
    current_gw = detect_current_gameweek(fixtures) if fixtures else 1

    gw_options = [{"label": f"Gameweek {gw}", "value": gw} for gw in gameweeks]

    return html.Div([
        make_header("Fixture Calendar"),

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

        html.Div(id="fixture-table", style={"padding": "0 40px"}),
    ])


# ---------------------------------------------------------------------------
# Match detail page
# ---------------------------------------------------------------------------


def _find_fixture_info(home_team, away_team):
    """Find the FPL fixture entry for this matchup."""
    fixtures = fetch_fpl_fixtures()
    for f in fixtures:
        if f["home"] == home_team and f["away"] == away_team:
            return f
    return None


def match_detail_layout(home_team, away_team):
    """Build the match detail page for a given matchup."""
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

    home_stats = get_latest_stats(home_team)
    away_stats = get_latest_stats(away_team)

    if home_stats is None or away_stats is None:
        missing = []
        if home_stats is None:
            missing.append(home_team)
        if away_stats is None:
            missing.append(away_team)
        return html.Div([
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

    pred_cards, (p_home, p_draw, p_away, p_over) = make_prediction_cards(
        home_team, away_team, home_stats, away_stats,
    )
    insights = generate_insights(home_team, away_team)
    news_data = get_match_news(home_team, away_team)
    news_highlights = get_news_highlights(news_data)
    if news_highlights:
        insights = insights + news_highlights
    stat_fig = make_stat_chart(home_team, away_team, home_stats, away_stats)
    h2h_panel = make_h2h_panel(home_team, away_team)

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

        # Matchup header + prediction + form
        html.Div([
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

            html.Div(pred_cards, style={
                "textAlign": "center", "marginBottom": "24px",
            }),

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
        ], style={"padding": "0 32px 24px"}),

        # Key Highlights + News & Updates side by side
        html.Div([
            # Key Highlights — 3 sub-boxes
            html.Div([
                html.H3("Key Highlights", style={
                    "color": "#e0e0e0", "marginTop": "0",
                    "marginBottom": "16px", "fontSize": "1.1em",
                }),
                # Top row: Key Stats (left) + Key News (right)
                html.Div([
                    html.Div([
                        html.H4("Key Stats", style={
                            "color": "#6c63ff", "margin": "0 0 10px",
                            "fontSize": "0.9em", "fontWeight": "600",
                        }),
                        html.Ul([
                            html.Li(ins, style={
                                "color": "#ccc", "marginBottom": "6px",
                                "fontSize": "0.85em", "lineHeight": "1.5",
                            })
                            for ins in insights
                        ], style={"paddingLeft": "16px", "margin": "0"}),
                    ], className="insights-panel", style={
                        "background": "#151524", "borderRadius": "10px",
                        "padding": "14px 18px", "flex": "1", "minWidth": "0",
                    }),
                    html.Div([
                        html.H4("Key News", style={
                            "color": "#2ecc71", "margin": "0 0 10px",
                            "fontSize": "0.9em", "fontWeight": "600",
                        }),
                        html.Ul([
                            html.Li(h, style={
                                "color": "#ccc", "marginBottom": "6px",
                                "fontSize": "0.85em", "lineHeight": "1.5",
                            })
                            for h in news_highlights
                        ], style={"paddingLeft": "16px", "margin": "0"})
                        if news_highlights else
                        html.P("No news highlights yet.", style={
                            "color": "#555", "fontStyle": "italic",
                            "fontSize": "0.85em", "margin": "0",
                        }),
                    ], style={
                        "background": "#151524", "borderRadius": "10px",
                        "padding": "14px 18px", "flex": "1", "minWidth": "0",
                    }),
                ], style={
                    "display": "flex", "gap": "12px", "marginBottom": "12px",
                }),
                # Bottom box: Betting Suggestion
                html.Div([
                    html.H4("Betting Suggestion", style={
                        "color": "#f39c12", "margin": "0 0 10px",
                        "fontSize": "0.9em", "fontWeight": "600",
                    }),
                    html.P("Coming soon...", style={
                        "color": "#555", "fontStyle": "italic",
                        "fontSize": "0.85em", "margin": "0",
                    }),
                ], style={
                    "background": "#151524", "borderRadius": "10px",
                    "padding": "14px 18px",
                }),
            ], style={
                "background": "#1e1e2f", "borderRadius": "12px",
                "padding": "20px 24px",
                "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
                "flex": "1", "minWidth": "0",
            }),
            # News & Updates panel
            html.Div(
                make_news_panel(news_data),
                style={"flex": "1", "minWidth": "0"},
            ),
        ], className="detail-top-row", style={
            "display": "flex", "gap": "16px", "padding": "0 32px 24px",
            "alignItems": "flex-start",
        }),

        # H2H + Stat chart side by side
        html.Div([
            html.Div(
                html.Div(h2h_panel, style=CHART_CARD_STYLE),
                style={"flex": "1", "minWidth": "0"},
            ),
            html.Div(
                html.Div(dcc.Graph(figure=stat_fig), style=CHART_CARD_STYLE),
                style={"flex": "1", "minWidth": "0"},
            ),
        ], style={"display": "flex", "gap": "16px", "padding": "0 32px 24px"}),

        make_stat_glossary(),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("url", "search"),
    State("user-session", "data"),
)
def display_page(pathname, search, session):
    """Route to the correct page based on URL."""
    if pathname == "/match":
        params = parse_qs(search.lstrip("?")) if search else {}
        home = params.get("home", [None])[0]
        away = params.get("away", [None])[0]
        if home and away:
            return match_detail_layout(home, away)
        return landing_layout()
    if pathname == "/profile":
        if session and session.get("email"):
            user = get_user(session["email"])
            if user:
                return html.Div([make_header("Profile"), profile_layout(user)])
        return html.Div([make_header("Profile"), login_layout()])
    return landing_layout()


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

    filtered = [f for f in fixtures if f["gameweek"] == selected_gw]

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

    filtered.sort(key=lambda f: f["kickoff"] or datetime.min.replace(tzinfo=timezone.utc))

    rows = []
    for fix in filtered:
        pred = get_quick_prediction(fix["home"], fix["away"])
        rows.append(make_fixture_row(fix, pred))

    return html.Div(rows, className="fixture-list")


# ---------------------------------------------------------------------------
# Profile callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("user-session", "data", allow_duplicate=True),
    Output("auth-error", "children"),
    Output("url", "pathname", allow_duplicate=True),
    Input("btn-login", "n_clicks"),
    Input("btn-register", "n_clicks"),
    State("auth-email", "value"),
    State("auth-password", "value"),
    State("user-session", "data"),
    prevent_initial_call=True,
)
def handle_auth(login_clicks, register_clicks, email, password, session):
    """Handle Sign In and Register button clicks."""
    trigger = ctx.triggered_id
    if trigger not in ("btn-login", "btn-register"):
        return session, "", "/profile"
    if not email or not password:
        return session, "Please enter both email and password.", "/profile"

    if trigger == "btn-register":
        result = register_user(email, password)
        if isinstance(result, str):
            return session, result, "/profile"
        return {"email": result["email"]}, "", "/profile"

    # Login
    user = login_user(email, password)
    if user is None:
        return session, "Invalid email or password.", "/profile"
    return {"email": user["email"]}, "", "/profile"


@callback(
    Output("save-teams-msg", "children"),
    Output("fav-team-links", "children"),
    Input("btn-save-teams", "n_clicks"),
    State("fav-teams-dropdown", "value"),
    State("user-session", "data"),
    prevent_initial_call=True,
)
def save_favorite_teams(n_clicks, teams, session):
    """Persist selected favorite teams."""
    if not session or not session.get("email"):
        return "", []
    teams = teams or []
    update_user(session["email"], {"favorite_teams": teams})
    from user_profile import _fav_links
    return "Saved!", _fav_links(teams)


@callback(
    Output("tier-msg", "children"),
    Output("url", "pathname", allow_duplicate=True),
    Input({"type": "btn-tier", "tier": ALL}, "n_clicks"),
    State("user-session", "data"),
    prevent_initial_call=True,
)
def change_tier(n_clicks_list, session):
    """Update the user's subscription tier."""
    if not session or not session.get("email"):
        return "", "/profile"
    triggered = ctx.triggered_id
    if triggered is None:
        return "", "/profile"
    tier = triggered["tier"]
    update_user(session["email"], {"subscription": tier})
    return f"Switched to {tier.title()}!", "/profile"


@callback(
    Output("user-session", "data", allow_duplicate=True),
    Output("url", "pathname", allow_duplicate=True),
    Input("btn-signout", "n_clicks"),
    prevent_initial_call=True,
)
def sign_out(n_clicks):
    """Clear the session and redirect to login form."""
    return None, "/profile"


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

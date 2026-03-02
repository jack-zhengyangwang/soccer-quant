"""Reusable Dash layout builders and chart functions."""

from urllib.parse import urlencode

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from data import (
    build_feature_vector,
    df,
    get_form_string,
    get_latest_stats,
    outcome_model,
    ou_model,
)
from teams import get_badge_url, get_matchup_colors

# ---------------------------------------------------------------------------
# Shared styles
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

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def empty_fig(message):
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
        html.Div([
            dcc.Link(
                html.Span("\u2699", style={"fontSize": "1.2em"}),
                href="/profile",
                className="profile-link",
                style={
                    "color": "#ccc", "textDecoration": "none",
                    "fontSize": "0.9em", "fontWeight": "500",
                    "border": "1px solid #555", "borderRadius": "6px",
                    "padding": "6px 12px", "transition": "border-color 0.2s",
                    "display": "inline-flex", "alignItems": "center",
                    "gap": "6px",
                },
            ),
            html.A(
                "GitHub",
                href="https://github.com/jack-zhengyangwang/soccer-quant",
                target="_blank",
                className="github-link",
                style={
                    "color": "#ccc", "textDecoration": "none",
                    "fontSize": "0.9em", "fontWeight": "500",
                    "border": "1px solid #555", "borderRadius": "6px",
                    "padding": "6px 14px", "transition": "border-color 0.2s",
                },
            ),
        ], style={
            "position": "absolute", "right": "40px", "top": "50%",
            "transform": "translateY(-50%)",
            "display": "flex", "gap": "10px", "alignItems": "center",
        }),
    ], style={
        "background": "linear-gradient(135deg, #1a1a2e, #16213e, #0f3460)",
        "marginBottom": "32px", "boxShadow": "0 2px 12px rgba(0,0,0,0.5)",
        "position": "relative",
    })


# ---------------------------------------------------------------------------
# Prediction bar
# ---------------------------------------------------------------------------


def make_prediction_cards(home_team, away_team, home_stats, away_stats):
    """Build the outcome probability bar + Over 2.5 pill."""
    X_vec = build_feature_vector(home_stats, away_stats)
    outcome_probs = outcome_model.predict_proba(X_vec)[0]
    ou_probs = ou_model.predict_proba(X_vec)[0]

    p_home, p_draw, p_away = outcome_probs[0], outcome_probs[1], outcome_probs[2]
    p_over = ou_probs[1]

    home_color, away_color = get_matchup_colors(home_team, away_team)
    draw_color = "#555"

    def _segment(label, prob, bg, pos):
        radius = {"left": "10px 0 0 10px", "right": "0 10px 10px 0",
                   "center": "0"}
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


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def make_stat_chart(home_team, away_team, home_stats, away_stats):
    """Build per-stat stacked bar comparison with team names and values."""
    stat_labels = ["xG", "xGA", "Points", "Poss%", "SoT Ratio"]
    raw_home = [home_stats["xg"], home_stats["xga"], home_stats["points"],
                home_stats["poss"], home_stats["sot_ratio"]]
    raw_away = [away_stats["xg"], away_stats["xga"], away_stats["points"],
                away_stats["poss"], away_stats["sot_ratio"]]

    home_color, away_color = get_matchup_colors(home_team, away_team)

    fig = make_subplots(
        rows=len(stat_labels), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    for i, (label, hv, av) in enumerate(zip(stat_labels, raw_home, raw_away)):
        row = i + 1
        total = hv + av if (hv + av) > 0 else 1
        h_pct = hv / total * 100
        a_pct = av / total * 100

        fig.add_trace(go.Bar(
            x=[h_pct], y=[label], orientation="h", name=home_team,
            marker_color=home_color, showlegend=(i == 0),
            text=f"  {home_team}  {hv:.2f}",
            textposition="inside", insidetextanchor="start",
            textfont=dict(color="#fff", size=13, family="Inter, sans-serif"),
            legendgroup="home",
        ), row=row, col=1)

        fig.add_trace(go.Bar(
            x=[a_pct], y=[label], orientation="h", name=away_team,
            marker_color=away_color, showlegend=(i == 0),
            text=f"  {away_team}  {av:.2f}",
            textposition="inside", insidetextanchor="start",
            textfont=dict(color="#fff", size=13, family="Inter, sans-serif"),
            legendgroup="away",
        ), row=row, col=1)

        fig.update_yaxes(
            showticklabels=False, row=row, col=1,
            gridcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(
            visible=False, range=[0, 100], row=row, col=1,
        )

    # Left-aligned stat labels as annotations
    n = len(stat_labels)
    for i, label in enumerate(stat_labels):
        # Each subplot occupies an equal vertical band; compute its midpoint
        row_h = 1.0 / n
        y_mid = 1.0 - (i + 0.5) * row_h
        fig.add_annotation(
            text=f"<b>{label}</b>",
            xref="paper", yref="paper",
            x=-0.18, y=y_mid,
            xanchor="left", yanchor="middle",
            showarrow=False,
            font=dict(color="#ccc", size=14, family="Inter, sans-serif"),
        )

    fig.update_layout(
        title="Stat Comparison (5-match rolling)",
        barmode="stack",
        height=50 + 80 * n,
        showlegend=False,
        margin=dict(l=100, r=20, t=50, b=20),
        **DARK_CHART_LAYOUT,
    )
    return fig


def make_h2h_panel(home_team, away_team):
    """Build an EPL-style H2H panel: summary bar + list of past meetings."""
    h2h_mask = (
        ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team))
        | ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    )
    h2h = df.loc[h2h_mask].sort_values("date", ascending=False).head(10)

    if h2h.empty:
        return html.Div(
            html.P("No head-to-head history found",
                   style={"color": "#666", "textAlign": "center", "padding": "60px 0"}),
        )

    home_color, away_color = get_matchup_colors(home_team, away_team)
    home_badge = get_badge_url(home_team)
    away_badge = get_badge_url(away_team)

    wins, draws, losses = 0, 0, 0
    for _, r in h2h.iterrows():
        if r["outcome"] == 1:
            draws += 1
        elif (r["outcome"] == 0 and r["HomeTeam"] == home_team) or \
             (r["outcome"] == 2 and r["AwayTeam"] == home_team):
            wins += 1
        else:
            losses += 1

    total = len(h2h)
    w_pct = wins / total * 100 if total else 0
    d_pct = draws / total * 100 if total else 0
    l_pct = losses / total * 100 if total else 0

    badge_style = {"height": "24px", "width": "24px", "objectFit": "contain"}

    # Summary bar
    summary = html.Div([
        html.Div([
            html.Img(src=home_badge, style=badge_style) if home_badge else None,
            html.Span(f"  {wins}", style={
                "color": "#fff", "fontWeight": "bold", "fontSize": "1.1em",
                "marginLeft": "8px",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Div(style={
                "flex": f"{w_pct}", "background": home_color,
                "borderRadius": "4px 0 0 4px", "height": "8px",
                "minWidth": "2px" if wins else "0",
            }),
            html.Div(style={
                "flex": f"{d_pct}", "background": "#555",
                "height": "8px",
                "minWidth": "2px" if draws else "0",
            }),
            html.Div(style={
                "flex": f"{l_pct}", "background": away_color,
                "borderRadius": "0 4px 4px 0", "height": "8px",
                "minWidth": "2px" if losses else "0",
            }),
        ], style={
            "display": "flex", "flex": "1", "margin": "0 12px",
            "borderRadius": "4px", "overflow": "hidden",
        }),
        html.Div([
            html.Span(f"{losses}  ", style={
                "color": "#fff", "fontWeight": "bold", "fontSize": "1.1em",
                "marginRight": "8px",
            }),
            html.Img(src=away_badge, style=badge_style) if away_badge else None,
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "alignItems": "center",
        "padding": "12px 0", "marginBottom": "4px",
    })

    draws_label = html.Div(
        f"{draws} Draw{'s' if draws != 1 else ''}",
        style={
            "textAlign": "center", "color": "#888", "fontSize": "0.8em",
            "marginBottom": "12px",
        },
    )

    # Match rows
    result_colors = {"W": "#2ecc71", "D": "#888", "L": "#e74c3c"}
    rows = []
    for _, r in h2h.iterrows():
        date_str = r["date"].strftime("%d %b %Y")
        ht, at = r["HomeTeam"], r["AwayTeam"]

        # Score: gf/ga are from the home team's perspective
        home_goals = int(r["gf"]) if pd.notna(r.get("gf")) else "?"
        away_goals = int(r["ga"]) if pd.notna(r.get("ga")) else "?"
        score_str = f"{home_goals} - {away_goals}"

        if r["outcome"] == 1:
            res = "D"
        elif (r["outcome"] == 0 and r["HomeTeam"] == home_team) or \
             (r["outcome"] == 2 and r["AwayTeam"] == home_team):
            res = "W"
        else:
            res = "L"

        ht_badge = get_badge_url(ht)
        at_badge = get_badge_url(at)
        small_badge = {"height": "18px", "width": "18px", "objectFit": "contain"}

        row = html.Div([
            html.Span(date_str, style={
                "flex": "0 0 90px", "color": "#888", "fontSize": "0.8em",
            }),
            html.Div([
                html.Img(src=ht_badge, style=small_badge) if ht_badge else None,
                html.Span(ht, style={
                    "color": "#ccc", "fontSize": "0.85em", "marginLeft": "6px",
                }),
            ], style={
                "flex": "1", "display": "flex", "alignItems": "center",
                "justifyContent": "flex-end",
            }),
            html.Span(score_str, style={
                "flex": "0 0 56px", "textAlign": "center",
                "color": "#fff", "fontWeight": "bold", "fontSize": "0.85em",
            }),
            html.Div([
                html.Img(src=at_badge, style=small_badge) if at_badge else None,
                html.Span(at, style={
                    "color": "#ccc", "fontSize": "0.85em", "marginLeft": "6px",
                }),
            ], style={"flex": "1", "display": "flex", "alignItems": "center"}),
            html.Span(res, style={
                "flex": "0 0 28px", "textAlign": "center",
                "fontWeight": "bold", "fontSize": "0.85em",
                "color": result_colors.get(res, "#888"),
                "background": "rgba(255,255,255,0.05)",
                "borderRadius": "4px", "padding": "2px 0",
            }),
        ], style={
            "display": "flex", "alignItems": "center", "gap": "4px",
            "padding": "8px 0",
            "borderBottom": "1px solid rgba(255,255,255,0.05)",
        })
        rows.append(row)

    return html.Div([
        html.H4("Head-to-Head", style={
            "color": "#e0e0e0", "margin": "0 0 8px", "fontSize": "1em",
        }),
        summary,
        draws_label,
        html.Div(rows),
    ])


# ---------------------------------------------------------------------------
# Form badges
# ---------------------------------------------------------------------------


def make_form_badges(team, n=5):
    """Return colored W/D/L badge elements for last n results."""
    form = get_form_string(team, n)
    if not form:
        return html.Span("N/A", style={"color": "#666"})
    color_map = {"W": "#2ecc71", "D": "#888", "L": "#e74c3c"}
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


# ---------------------------------------------------------------------------
# Fixture row (landing page)
# ---------------------------------------------------------------------------


def make_fixture_row(fixture, prediction):
    """Build one clickable fixture row for the landing page."""
    home = fixture["home"]
    away = fixture["away"]
    finished = fixture["finished"]

    ko = fixture["kickoff"]
    time_str = ko.strftime("%H:%M") if ko else "TBC"

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


# ---------------------------------------------------------------------------
# News panel
# ---------------------------------------------------------------------------

_SECTION_META = {
    "injury": {"label": "Injury Updates", "icon": "\U0001f3e5", "color": "#e74c3c"},
    "team_report": {"label": "Team Report", "icon": "\U0001f4cb", "color": "#3498db"},
    "sentiment": {"label": "Sentiment", "icon": "\U0001f4ca", "color": "#2ecc71"},
    "tactical": {"label": "Tactical Preview", "icon": "\u2694\ufe0f", "color": "#f39c12"},
}

_SECTION_ORDER = ["injury", "team_report", "sentiment", "tactical"]


def _make_source_links(sources):
    """Render a list of clickable source links."""
    if not sources:
        return None
    links = []
    for s in sources[:3]:
        links.append(html.A(
            s.get("title", "Source")[:50],
            href=s.get("url", "#"),
            target="_blank",
            rel="noopener noreferrer",
            style={
                "color": "#6c63ff", "fontSize": "0.8em",
                "textDecoration": "none", "marginRight": "12px",
            },
        ))
    return html.Div(links, style={"marginTop": "6px"})


def _make_section(key, bullets):
    """Render one subsection (e.g. Injury Updates) with its bullets."""
    meta = _SECTION_META.get(key, {"label": key, "icon": "", "color": "#888"})

    if not bullets:
        return html.Div([
            html.Div([
                html.Span(f"{meta['icon']} ", style={"marginRight": "4px"}),
                html.Span(meta["label"], style={
                    "fontWeight": "600", "color": meta["color"],
                }),
            ], style={"marginBottom": "6px"}),
            html.P("No updates available.", style={
                "color": "#555", "fontStyle": "italic",
                "fontSize": "0.85em", "margin": "0 0 0 4px",
            }),
        ], className="news-section", style={"marginBottom": "16px"})

    items = []
    for b in bullets:
        detail_children = [b.get("detail", "")]
        source_links = _make_source_links(b.get("sources", []))
        if source_links:
            detail_children.append(source_links)

        items.append(
            html.Details([
                html.Summary(b.get("headline", "")),
                html.Div(
                    detail_children,
                    style={
                        "padding": "6px 0 8px 20px", "color": "#999",
                        "fontSize": "0.88em", "lineHeight": "1.6",
                    },
                ),
            ], className="news-item")
        )

    return html.Div([
        html.Div([
            html.Span(f"{meta['icon']} ", style={"marginRight": "4px"}),
            html.Span(meta["label"], style={
                "fontWeight": "600", "color": meta["color"],
            }),
        ], style={"marginBottom": "6px"}),
        html.Div(items),
    ], className="news-section", style={"marginBottom": "16px"})


def make_news_panel(news_data):
    """Render the News & Updates panel with 4 subsections and source links."""
    source = news_data.get("source", "unavailable")
    sections = news_data.get("sections", {})
    fetched_at = news_data.get("fetched_at", "")

    if source == "unavailable" or not sections:
        return html.Div([
            html.H3("News & Updates", style={
                "color": "#e0e0e0", "marginTop": "0",
                "marginBottom": "12px", "fontSize": "1.1em",
            }),
            html.P(
                "News unavailable \u2014 set TAVILY_API_KEY and ANTHROPIC_API_KEY "
                "to enable live updates.",
                style={
                    "color": "#666", "fontStyle": "italic",
                    "textAlign": "center", "padding": "20px 0",
                },
            ),
        ], className="news-panel", style={
            "background": "#1e1e2f", "borderRadius": "12px",
            "padding": "20px 24px", "marginBottom": "16px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
        })

    section_divs = [
        _make_section(key, sections.get(key, []))
        for key in _SECTION_ORDER
    ]

    # Timestamp footer
    footer_parts = []
    if fetched_at:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(fetched_at)
            footer_parts.append(ts.strftime("%d %b %Y, %H:%M UTC"))
        except (ValueError, TypeError):
            pass
    if source == "cache":
        footer_parts.append("cached")

    footer_text = " \u2022 ".join(footer_parts) if footer_parts else ""

    return html.Div([
        html.H3("News & Updates", style={
            "color": "#e0e0e0", "marginTop": "0",
            "marginBottom": "16px", "fontSize": "1.1em",
        }),
        html.Div(section_divs),
        html.Div(footer_text, style={
            "color": "#555", "fontSize": "0.75em",
            "textAlign": "right", "marginTop": "8px",
        }) if footer_text else None,
    ], className="news-panel", style={
        "background": "#1e1e2f", "borderRadius": "12px",
        "padding": "20px 24px", "marginBottom": "16px",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
    })


# ---------------------------------------------------------------------------
# Stat glossary
# ---------------------------------------------------------------------------


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

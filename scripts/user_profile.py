"""Profile page layout builders — login form and signed-in profile view."""

from dash import dcc, html

from teams import TEAM_COLORS

# All teams available for the favourite-teams picker
ALL_TEAMS = sorted(TEAM_COLORS.keys())

TIER_INFO = {
    "free": {
        "name": "Free",
        "features": ["Match stats", "H2H history"],
        "color": "#888",
    },
    "pro": {
        "name": "Pro",
        "features": ["Match stats", "H2H history", "Live news", "Match alerts"],
        "color": "#6c63ff",
    },
    "premium": {
        "name": "Premium",
        "features": [
            "Match stats", "H2H history", "Live news",
            "Match alerts", "Betting suggestions", "Priority support",
        ],
        "color": "#f39c12",
    },
}


# ------------------------------------------------------------------
# Login / Register form (signed-out state)
# ------------------------------------------------------------------

def login_layout():
    return html.Div([
        html.Div([
            html.H2("Sign In or Register", style={
                "color": "#e0e0e0", "marginBottom": "24px", "textAlign": "center",
            }),

            html.Label("Email", style={
                "color": "#aaa", "fontSize": "0.85em", "display": "block",
                "marginBottom": "4px",
            }),
            dcc.Input(
                id="auth-email", type="email", placeholder="you@example.com",
                className="auth-input",
                style={
                    "width": "100%", "padding": "10px 14px",
                    "background": "#151524", "border": "1px solid #3a3a5c",
                    "borderRadius": "8px", "color": "#e0e0e0",
                    "marginBottom": "16px", "boxSizing": "border-box",
                },
            ),

            html.Label("Password", style={
                "color": "#aaa", "fontSize": "0.85em", "display": "block",
                "marginBottom": "4px",
            }),
            dcc.Input(
                id="auth-password", type="password", placeholder="Password",
                className="auth-input",
                style={
                    "width": "100%", "padding": "10px 14px",
                    "background": "#151524", "border": "1px solid #3a3a5c",
                    "borderRadius": "8px", "color": "#e0e0e0",
                    "marginBottom": "24px", "boxSizing": "border-box",
                },
            ),

            html.Div([
                html.Button("Sign In", id="btn-login", n_clicks=0,
                            className="auth-btn auth-btn-primary"),
                html.Button("Register", id="btn-register", n_clicks=0,
                            className="auth-btn auth-btn-secondary"),
            ], style={
                "display": "flex", "gap": "12px", "justifyContent": "center",
            }),

            html.Div(id="auth-error", style={
                "color": "#e74c3c", "textAlign": "center",
                "marginTop": "16px", "fontSize": "0.9em",
            }),
        ], className="auth-card", style={
            "background": "#1e1e2f", "borderRadius": "14px",
            "padding": "40px 36px", "maxWidth": "400px", "margin": "0 auto",
            "boxShadow": "0 8px 32px rgba(0,0,0,0.5)",
        }),
    ], style={"padding": "60px 40px"})


# ------------------------------------------------------------------
# Signed-in profile view
# ------------------------------------------------------------------

def _tier_card(tier_key, current_tier):
    info = TIER_INFO[tier_key]
    is_current = tier_key == current_tier
    border_color = info["color"] if is_current else "#3a3a5c"
    badge = html.Span("Current Plan", style={
        "fontSize": "0.75em", "background": info["color"],
        "color": "#fff", "borderRadius": "12px",
        "padding": "2px 10px", "marginLeft": "8px",
    }) if is_current else None

    features_list = html.Ul([
        html.Li(f, style={
            "color": "#bbb", "fontSize": "0.85em", "marginBottom": "4px",
        }) for f in info["features"]
    ], style={"paddingLeft": "18px", "margin": "12px 0"})

    button = html.Div("Current", style={
        "color": "#888", "textAlign": "center", "fontSize": "0.85em",
    }) if is_current else html.Button(
        "Select", id={"type": "btn-tier", "tier": tier_key},
        n_clicks=0, className="auth-btn auth-btn-secondary",
        style={"width": "100%", "marginTop": "4px"},
    )

    return html.Div([
        html.H4([info["name"], badge], style={
            "color": info["color"], "margin": "0 0 4px",
        }),
        features_list,
        button,
    ], className="tier-card", style={
        "flex": "1", "minWidth": "0",
        "background": "#151524", "borderRadius": "12px",
        "padding": "20px", "border": f"2px solid {border_color}",
        "transition": "border-color 0.2s",
    })


def profile_layout(user):
    email = user["email"]
    favorite_teams = user.get("favorite_teams", [])
    subscription = user.get("subscription", "free")

    team_options = [{"label": t, "value": t} for t in ALL_TEAMS]

    return html.Div([
        # --- Favorite Teams ---
        html.Div([
            html.H3("Favorite Teams", style={
                "color": "#e0e0e0", "marginTop": "0", "marginBottom": "12px",
            }),
            html.P("Select your teams for quick fixture access.", style={
                "color": "#888", "fontSize": "0.85em", "marginBottom": "12px",
            }),
            dcc.Dropdown(
                id="fav-teams-dropdown",
                options=team_options,
                value=favorite_teams,
                multi=True,
                placeholder="Choose teams...",
            ),
            html.Div([
                html.Button("Save Teams", id="btn-save-teams", n_clicks=0,
                            className="auth-btn auth-btn-primary",
                            style={"marginTop": "12px"}),
                html.Span(id="save-teams-msg", style={
                    "color": "#2ecc71", "marginLeft": "12px",
                    "fontSize": "0.85em", "alignSelf": "center",
                }),
            ], style={"display": "flex", "alignItems": "center"}),

            # Quick links to favorite team fixtures
            html.Div(id="fav-team-links", children=_fav_links(favorite_teams), style={
                "marginTop": "16px", "display": "flex", "flexWrap": "wrap", "gap": "8px",
            }),
        ], className="profile-section", style={
            "background": "#1e1e2f", "borderRadius": "14px",
            "padding": "24px 28px", "marginBottom": "20px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
        }),

        # --- Subscription Tiers ---
        html.Div([
            html.H3("Subscription", style={
                "color": "#e0e0e0", "marginTop": "0", "marginBottom": "16px",
            }),
            html.Div([
                _tier_card("free", subscription),
                _tier_card("pro", subscription),
                _tier_card("premium", subscription),
            ], className="tier-row", style={
                "display": "flex", "gap": "16px",
            }),
            html.Div(id="tier-msg", style={
                "color": "#2ecc71", "fontSize": "0.85em",
                "textAlign": "center", "marginTop": "12px",
            }),
        ], className="profile-section", style={
            "background": "#1e1e2f", "borderRadius": "14px",
            "padding": "24px 28px", "marginBottom": "20px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.4)",
        }),

        # --- Account info + Sign Out ---
        html.Div([
            html.Span(f"Signed in as {email}", style={
                "color": "#888", "fontSize": "0.85em",
            }),
            html.Button("Sign Out", id="btn-signout", n_clicks=0,
                        className="auth-btn auth-btn-danger",
                        style={"marginLeft": "auto"}),
        ], style={
            "display": "flex", "alignItems": "center",
            "padding": "16px 28px",
        }),
    ], style={"padding": "32px 40px", "maxWidth": "820px", "margin": "0 auto"})


def _fav_links(teams):
    """Build quick-link pills for each favorite team's fixtures."""
    if not teams:
        return []
    return [
        dcc.Link(
            t, href=f"/?team={t}",
            style={
                "background": "#2a2a4a", "color": "#ccc",
                "borderRadius": "16px", "padding": "4px 14px",
                "fontSize": "0.82em", "textDecoration": "none",
                "transition": "background 0.15s",
            },
        )
        for t in teams
    ]

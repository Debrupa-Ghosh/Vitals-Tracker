"""
Agent 5: Frontend / UI Agent
Serves dynamic UI configuration — theme, form schema, dashboard layout.
React frontend fetches these configs and renders dynamically.
"""

import json
from db import get_connection


def _get_config(key: str) -> dict:
    """Fetch a UI config from the database by key."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT config_value FROM ui_config WHERE config_key = %s", (key,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row:
        val = row["config_value"]
        if isinstance(val, str):
            return json.loads(val)
        return val
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_theme() -> dict:
    """Return the current theme configuration."""
    theme = _get_config("theme")
    if not theme:
        # Fallback defaults
        theme = {
            "mode": "dark",
            "colors": {
                "background": "#0f1117",
                "backgroundAlt": "#161822",
                "surface": "rgba(255,255,255,0.05)",
                "surfaceHover": "rgba(255,255,255,0.08)",
                "primary": "#4f8cff",
                "healthy": "#00d4aa",
                "healthyGlow": "rgba(0,212,170,0.15)",
                "warning": "#ffb347",
                "warningGlow": "rgba(255,179,71,0.15)",
                "danger": "#ff6b6b",
                "dangerGlow": "rgba(255,107,107,0.15)",
                "text": "#e8e6df",
                "textMuted": "#8a8880",
                "textDim": "#5a5850",
                "border": "rgba(255,255,255,0.08)",
                "borderHover": "rgba(255,255,255,0.15)",
            },
            "fonts": {"primary": "Inter", "mono": "JetBrains Mono"},
            "borderRadius": "12px",
            "glassmorphism": {
                "blur": "20px",
                "border": "rgba(255,255,255,0.08)",
            },
        }
    return theme


def get_form_schema() -> dict:
    """Return the dynamic form field definitions."""
    schema = _get_config("form_schema")
    if not schema:
        schema = {"steps": []}
    return schema


def get_dashboard_layout() -> dict:
    """Return the dashboard widget layout configuration."""
    layout = _get_config("dashboard_layout")
    if not layout:
        layout = {"widgets": [], "severityColors": {}}
    return layout


def get_full_config() -> dict:
    """Return all UI configuration combined."""
    return {
        "theme": get_theme(),
        "formSchema": get_form_schema(),
        "dashboardLayout": get_dashboard_layout(),
    }

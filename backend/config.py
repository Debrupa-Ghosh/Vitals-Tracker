"""
Application configuration for Vitals Tracker backend.
"""

import os


class Config:
    """Central configuration class."""

    # MySQL Database
    DB_HOST = os.environ.get("VT_DB_HOST", "localhost")
    DB_PORT = int(os.environ.get("VT_DB_PORT", 3306))
    DB_USER = os.environ.get("VT_DB_USER")
    DB_PASSWORD = os.environ.get("VT_DB_PASSWORD")
    DB_NAME = os.environ.get("VT_DB_NAME", "vitals_tracker")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    # Flask
    DEBUG = os.environ.get("VT_DEBUG", "true").lower() == "true"
    HOST = os.environ.get("VT_HOST", "0.0.0.0")
    PORT = int(os.environ.get("VT_PORT", 5001))

    # Frontend
    FRONTEND_PORT = int(os.environ.get("VT_FRONTEND_PORT", 5173))
    FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

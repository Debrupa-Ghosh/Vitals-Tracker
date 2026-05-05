"""
Vitals Tracker — Flask Backend Application
Main entry point with all API routes.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.config import Config
from backend.db import init_db
from backend.services import integration

from backend.services import input as input_svc
from backend.services import analysis
from backend.services import reporting
from backend.services import assistant
from backend.services import frontend

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# In-memory cache for the latest analysis (per user session)
# In production, you'd use Redis or session-based storage
_analysis_cache = {}


# ---------------------------------------------------------------------------
# Pipeline Route — Runs Agents 1 → 2 → 3 → 4 in sequence
# ---------------------------------------------------------------------------

@app.route("/api/vitals", methods=["POST"])
def submit_vitals():
    """
    Submit health vitals. Runs the full agent pipeline:
    Agent 1 (validate + process) → Agent 2 (analyze) → Agent 4 (recommendations + doctors)
    """
    raw_data = request.get_json()
    print(f"DEBUG: Received raw_data: {raw_data}")
    if not raw_data:
        return jsonify({"error": "No data provided"}), 400

    # Agent 1: Input & Processing
    result = input_svc.process(raw_data)
    if not result["valid"]:
        return jsonify({"valid": False, "errors": result["errors"]}), 422

    processed = result["data"]

    # Agent 2: Health Analysis & Prediction
    analysis_res = analysis.analyze(processed)

    # Agent 4: Recommendations & Doctor suggestions
    recommendations = assistant.get_recommendations(analysis_res)
    doctors = assistant.suggest_doctors(analysis_res)

    # Cache analysis for chat context
    _analysis_cache[analysis_res["user_id"]] = analysis_res

    # Agent 3: Get trends
    trends = reporting.get_trends(analysis_res["user_id"])

    return jsonify({
        "valid": True,
        "analysis": analysis_res,
        "recommendations": recommendations,
        "doctors": doctors,
        "trends": trends,
    })


# ---------------------------------------------------------------------------
# Agent 3: Reporting Routes
# ---------------------------------------------------------------------------

@app.route("/api/history/<int:user_id>", methods=["GET"])
def get_history(user_id):
    """Get 7-day trend data for a user."""
    trends = reporting.get_trends(user_id)
    return jsonify(trends)


@app.route("/api/report/<int:record_id>", methods=["GET"])
def get_report(record_id):
    """Get structured report data for a health record."""
    report = reporting.get_report(record_id)
    if "error" in report:
        return jsonify(report), 404
    return jsonify(report)


# ---------------------------------------------------------------------------
# Agent 4: Assistant Routes
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    """Send a message to the chat assistant."""
    data = request.get_json()
    message = data.get("message", "")
    user_id = data.get("user_id")

    # Get cached analysis context
    context = _analysis_cache.get(user_id)

    response = assistant.chat(message, context)
    return jsonify(response)


@app.route("/api/doctors/<int:record_id>", methods=["GET"])
def get_doctors(record_id):
    """Get doctor suggestions for a specific record."""
    # Fetch analysis from cache or reconstruct from DB
    report = reporting.get_report(record_id)
    if "error" in report:
        return jsonify({"error": "Record not found"}), 404

    # Build a minimal analysis dict for doctor matching
    analysis = {
        "bmi": {"category": report["vitals"]["bmi"]["category"]},
        "bloodPressure": {"severity": "critical" if report["vitals"]["bloodPressure"]["category"].startswith("High") else "healthy",
                          "category": report["vitals"]["bloodPressure"]["category"]},
        "cholesterol": {"severity": "critical" if report["vitals"]["cholesterol"]["category"] == "High" else "moderate" if report["vitals"]["cholesterol"]["category"] == "Borderline High" else "healthy",
                        "category": report["vitals"]["cholesterol"]["category"]},
        "glucose": {"category": report["vitals"]["glucose"]["category"]},
        "heartRisk": {"level": report["risk"]["heartRiskLevel"]},
        "lifestyle": report["lifestyle"],
    }
    doctors = assistant.suggest_doctors(analysis)
    return jsonify(doctors)


@app.route("/api/recommendations/<int:record_id>", methods=["GET"])
def get_recommendations(record_id):
    """Get health recommendations for a specific record."""
    report = reporting.get_report(record_id)
    if "error" in report:
        return jsonify({"error": "Record not found"}), 404

    analysis = {
        "bmi": {"category": report["vitals"]["bmi"]["category"]},
        "bloodPressure": {"category": report["vitals"]["bloodPressure"]["category"],
                          "severity": "critical" if report["vitals"]["bloodPressure"]["category"].startswith("High") else "healthy"},
        "cholesterol": {"category": report["vitals"]["cholesterol"]["category"],
                        "severity": "critical" if report["vitals"]["cholesterol"]["category"] == "High" else "moderate" if report["vitals"]["cholesterol"]["category"] == "Borderline High" else "healthy"},
        "glucose": {"category": report["vitals"]["glucose"]["category"]},
        "heartRisk": {"level": report["risk"]["heartRiskLevel"]},
        "lifestyle": report["lifestyle"],
    }
    recs = assistant.get_recommendations(analysis)
    return jsonify(recs)


# ---------------------------------------------------------------------------
# Agent 5: Frontend/UI Routes
# ---------------------------------------------------------------------------

@app.route("/api/ui/config", methods=["GET"])
def get_ui_config():
    """Get full UI configuration (theme + form schema + dashboard layout)."""
    return jsonify(frontend.get_full_config())


@app.route("/api/ui/theme", methods=["GET"])
def get_theme():
    """Get current theme tokens."""
    return jsonify(frontend.get_theme())


@app.route("/api/ui/form-schema", methods=["GET"])
def get_form_schema():
    """Get dynamic form field definitions."""
    return jsonify(frontend.get_form_schema())


@app.route("/api/ui/dashboard-layout", methods=["GET"])
def get_dashboard_layout():
    """Get dashboard widget layout configuration."""
    return jsonify(frontend.get_dashboard_layout())


# ---------------------------------------------------------------------------
# Agent 6: System Status Route
# ---------------------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def get_status():
    """Get system health status."""
    return jsonify(integration.get_status())


# ---------------------------------------------------------------------------
# App Startup
# ---------------------------------------------------------------------------

import os

if __name__ == "__main__":
    print("\n🏥 Starting Vitals Tracker Backend...")
    init_db()
    integration.mark_started()

    port = int(os.environ.get("PORT", Config.PORT))
    print(f"🚀 Flask server running on port {port}\n")

    app.run(host="0.0.0.0", port=port)

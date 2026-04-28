"""
Database seeding script.
Populates doctors table and UI config with initial data.
"""

from db import init_db, get_connection
from data.doctors_seed import DOCTORS
import json


def seed_doctors():
    """Insert sample doctors if the table is empty."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM doctors")
    count = cursor.fetchone()[0]

    if count == 0:
        print("  Seeding doctors table...")
        for doc in DOCTORS:
            cursor.execute(
                "INSERT INTO doctors (name, specialty, hospital, phone, rating, distance_km, available) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (doc["name"], doc["specialty"], doc["hospital"], doc["phone"],
                 doc["rating"], doc["distance_km"], doc["available"])
            )
        conn.commit()
        print(f"  ✓ Inserted {len(DOCTORS)} doctors")
    else:
        print(f"  ✓ Doctors table already has {count} records")

    cursor.close()
    conn.close()


def seed_ui_config():
    """Insert default UI configuration if not present."""
    conn = get_connection()
    cursor = conn.cursor()

    configs = {
        "theme": json.dumps({
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
                "borderHover": "rgba(255,255,255,0.15)"
            },
            "fonts": {"primary": "Inter", "mono": "JetBrains Mono"},
            "borderRadius": "12px",
            "glassmorphism": {"blur": "20px", "border": "rgba(255,255,255,0.08)"}
        }),
        "form_schema": json.dumps({
            "steps": [
                {
                    "id": "personal",
                    "title": "Personal Info",
                    "icon": "user",
                    "fields": [
                        {"name": "name", "type": "text", "label": "Full Name", "placeholder": "Enter your full name", "required": True},
                        {"name": "age", "type": "number", "label": "Age", "placeholder": "e.g. 35", "min": 1, "max": 120, "required": True, "unit": "years"},
                        {"name": "gender", "type": "select", "label": "Gender", "options": [{"value": "male", "label": "Male"}, {"value": "female", "label": "Female"}, {"value": "other", "label": "Other"}], "required": True}
                    ]
                },
                {
                    "id": "vitals",
                    "title": "Vitals",
                    "icon": "heart",
                    "fields": [
                        {"name": "height", "type": "number", "label": "Height", "placeholder": "e.g. 175", "min": 50, "max": 300, "required": True, "unit": "cm", "step": 0.1},
                        {"name": "weight", "type": "number", "label": "Weight", "placeholder": "e.g. 72", "min": 10, "max": 500, "required": True, "unit": "kg", "step": 0.1},
                        {"name": "systolic", "type": "number", "label": "Systolic BP", "placeholder": "e.g. 120", "min": 60, "max": 250, "required": True, "unit": "mmHg"},
                        {"name": "diastolic", "type": "number", "label": "Diastolic BP", "placeholder": "e.g. 80", "min": 40, "max": 150, "required": True, "unit": "mmHg"},
                        {"name": "cholesterol", "type": "number", "label": "Total Cholesterol", "placeholder": "e.g. 200", "min": 50, "max": 500, "required": True, "unit": "mg/dL"},
                        {"name": "glucose", "type": "number", "label": "Fasting Glucose", "placeholder": "e.g. 95", "min": 30, "max": 600, "required": True, "unit": "mg/dL"}
                    ]
                },
                {
                    "id": "lifestyle",
                    "title": "Lifestyle",
                    "icon": "activity",
                    "fields": [
                        {"name": "smoking", "type": "toggle", "label": "Do you smoke?", "required": False},
                        {"name": "alcohol", "type": "select", "label": "Alcohol Consumption", "options": [{"value": "none", "label": "None"}, {"value": "light", "label": "Light (1-2/week)"}, {"value": "moderate", "label": "Moderate (3-5/week)"}, {"value": "heavy", "label": "Heavy (daily)"}], "required": True},
                        {"name": "exercise", "type": "select", "label": "Exercise Level", "options": [{"value": "sedentary", "label": "Sedentary"}, {"value": "light", "label": "Light (1-2x/week)"}, {"value": "moderate", "label": "Moderate (3-4x/week)"}, {"value": "active", "label": "Active (5+/week)"}], "required": True},
                        {"name": "diet", "type": "select", "label": "Diet Quality", "options": [{"value": "poor", "label": "Poor"}, {"value": "average", "label": "Average"}, {"value": "balanced", "label": "Balanced"}, {"value": "excellent", "label": "Excellent"}], "required": True}
                    ]
                }
            ]
        }),
        "dashboard_layout": json.dumps({
            "widgets": [
                {"id": "health-score", "type": "gauge", "title": "Overall Health", "position": {"row": 1, "col": 1, "span": 2}},
                {"id": "bmi", "type": "metric-card", "title": "BMI", "position": {"row": 1, "col": 3}},
                {"id": "blood-pressure", "type": "metric-card", "title": "Blood Pressure", "position": {"row": 1, "col": 4}},
                {"id": "cholesterol", "type": "metric-card", "title": "Cholesterol", "position": {"row": 2, "col": 1}},
                {"id": "glucose", "type": "metric-card", "title": "Glucose", "position": {"row": 2, "col": 2}},
                {"id": "heart-risk", "type": "risk-banner", "title": "Heart Disease Risk", "position": {"row": 2, "col": 3, "span": 2}},
                {"id": "trends", "type": "chart-panel", "title": "7-Day Trends", "position": {"row": 3, "col": 1, "span": 4}}
            ],
            "severityColors": {
                "healthy": "#00d4aa",
                "moderate": "#ffb347",
                "critical": "#ff6b6b"
            }
        })
    }

    for key, value in configs.items():
        cursor.execute(
            "INSERT INTO ui_config (config_key, config_value) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE config_value = VALUES(config_value)",
            (key, value)
        )

    conn.commit()
    print(f"  ✓ UI configs seeded ({len(configs)} entries)")
    cursor.close()
    conn.close()


def run_seed():
    """Run all seed operations."""
    print("\n🌱 Seeding database...")
    init_db()
    seed_doctors()
    seed_ui_config()
    print("✅ Seeding complete!\n")


if __name__ == "__main__":
    run_seed()

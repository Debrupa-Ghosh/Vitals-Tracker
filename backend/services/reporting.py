"""
Agent 3: Insights & Reporting Agent
Generates 7-day trend data and structured report payloads.
"""

import random
from datetime import datetime, timedelta
from db import get_connection


def _fetch_history(user_id: int, limit: int = 7) -> list:
    """Fetch the last N health records for a user from MySQL."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM health_records WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
        (user_id, limit)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    # Reverse so oldest is first (chronological order)
    return list(reversed(rows))


def _simulate_history(current_record: dict, count: int = 7) -> list:
    """
    Generate simulated historical data points based on the current record
    with realistic daily variance for trend visualization.
    """
    simulated = []
    now = datetime.now()

    for i in range(count - 1, -1, -1):
        day = now - timedelta(days=i)
        # Add realistic variance
        variance = {
            "bmi": random.uniform(-0.5, 0.5),
            "systolic": random.randint(-8, 8),
            "diastolic": random.randint(-5, 5),
            "cholesterol": random.randint(-10, 10),
            "glucose": random.randint(-8, 8),
            "overall_health_score": random.randint(-5, 5),
        }

        entry = {
            "date": day.strftime("%Y-%m-%d"),
            "label": day.strftime("%a"),  # Mon, Tue, etc.
            "bmi": round(current_record.get("bmi", 24.0) + variance["bmi"], 1),
            "systolic": max(70, current_record.get("systolic", 120) + variance["systolic"]),
            "diastolic": max(45, current_record.get("diastolic", 80) + variance["diastolic"]),
            "cholesterol": max(80, current_record.get("cholesterol", 200) + variance["cholesterol"]),
            "glucose": max(50, current_record.get("glucose", 95) + variance["glucose"]),
            "health_score": max(0, min(100,
                current_record.get("overall_health_score", 75) + variance["overall_health_score"]
            )),
        }
        simulated.append(entry)

    return simulated


def get_trends(user_id: int) -> dict:
    """
    Agent 3: Get 7-day trend data for a user.
    Uses real DB records if available (≥7), otherwise simulates.
    """
    records = _fetch_history(user_id, limit=7)

    if len(records) >= 7:
        # Use real data
        trends = []
        for r in records:
            created = r["created_at"]
            if isinstance(created, str):
                created = datetime.fromisoformat(created)
            trends.append({
                "date": created.strftime("%Y-%m-%d"),
                "label": created.strftime("%a"),
                "bmi": float(r["bmi"]),
                "systolic": int(r["systolic"]),
                "diastolic": int(r["diastolic"]),
                "cholesterol": int(r["cholesterol"]),
                "glucose": int(r["glucose"]),
                "health_score": int(r.get("overall_health_score") or 75),
            })
    else:
        # Simulate based on latest record
        latest = records[-1] if records else {
            "bmi": 24.0, "systolic": 120, "diastolic": 80,
            "cholesterol": 200, "glucose": 95, "overall_health_score": 75
        }
        trends = _simulate_history(latest, count=7)

    return {
        "user_id": user_id,
        "period": "7 days",
        "data_source": "database" if len(records) >= 7 else "simulated",
        "trends": trends,
    }


def get_report(record_id: int) -> dict:
    """
    Agent 3: Get a structured report for a specific health record.
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Fetch record with user info
    cursor.execute("""
        SELECT hr.*, u.name, u.age, u.gender
        FROM health_records hr
        JOIN users u ON hr.user_id = u.id
        WHERE hr.id = %s
    """, (record_id,))
    record = cursor.fetchone()
    cursor.close()
    conn.close()

    if not record:
        return {"error": "Record not found"}

    created = record["created_at"]
    if isinstance(created, datetime):
        created = created.isoformat()

    return {
        "record_id": record_id,
        "generated_at": datetime.now().isoformat(),
        "title": "Vitals Tracker Health Report",
        "patient": {
            "name": record["name"],
            "age": record["age"],
            "gender": record["gender"],
        },
        "vitals": {
            "height": float(record["height"]),
            "weight": float(record["weight"]),
            "bmi": {"value": float(record["bmi"]), "category": record["bmi_category"]},
            "bloodPressure": {
                "systolic": int(record["systolic"]),
                "diastolic": int(record["diastolic"]),
                "category": record["bp_category"],
            },
            "cholesterol": {"value": int(record["cholesterol"]), "category": record["cholesterol_category"]},
            "glucose": {"value": int(record["glucose"]), "category": record["glucose_category"]},
        },
        "risk": {
            "heartRiskLevel": record["heart_risk_level"],
            "heartRiskScore": float(record["heart_risk_score"]) if record["heart_risk_score"] else 0,
        },
        "overall": {
            "score": int(record["overall_health_score"]) if record["overall_health_score"] else 0,
            "grade": record["overall_health_grade"] or "N/A",
        },
        "lifestyle": {
            "smoking": bool(record["smoking"]),
            "alcohol": record["alcohol"],
            "exercise": record["exercise"],
            "diet": record["diet"],
        },
        "recorded_at": created,
    }

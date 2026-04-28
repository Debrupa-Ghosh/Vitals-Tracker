"""
Agent 2: Health Analysis & Prediction Agent
Classifies vitals, predicts heart disease risk, and computes an overall health score.
"""

from db import get_connection
import os
import joblib
import numpy as np

# Load the ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "heart_risk_model.joblib")
if os.path.exists(MODEL_PATH):
    try:
        heart_risk_model = joblib.load(MODEL_PATH)
        print("✅ ML Model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading ML model: {e}")
        heart_risk_model = None
else:
    heart_risk_model = None



# ---------------------------------------------------------------------------
# Classification Functions
# ---------------------------------------------------------------------------

def _classify_bmi(bmi: float) -> dict:
    if bmi < 18.5:
        return {"value": bmi, "category": "Underweight", "severity": "moderate"}
    elif bmi < 25:
        return {"value": bmi, "category": "Normal", "severity": "healthy"}
    elif bmi < 30:
        return {"value": bmi, "category": "Overweight", "severity": "moderate"}
    else:
        return {"value": bmi, "category": "Obese", "severity": "critical"}


def _classify_bp(systolic: int, diastolic: int) -> dict:
    if systolic < 90 or diastolic < 60:
        cat, sev = "Low", "moderate"
    elif systolic < 120 and diastolic < 80:
        cat, sev = "Normal", "healthy"
    elif systolic < 130 and diastolic < 80:
        cat, sev = "Elevated", "moderate"
    elif systolic < 140 or diastolic < 90:
        cat, sev = "High Stage 1", "critical"
    else:
        cat, sev = "High Stage 2", "critical"
    return {"systolic": systolic, "diastolic": diastolic, "category": cat, "severity": sev}


def _classify_cholesterol(value: int) -> dict:
    if value < 200:
        return {"value": value, "category": "Desirable", "severity": "healthy"}
    elif value < 240:
        return {"value": value, "category": "Borderline High", "severity": "moderate"}
    else:
        return {"value": value, "category": "High", "severity": "critical"}


def _classify_glucose(value: int) -> dict:
    if value < 100:
        return {"value": value, "category": "Normal", "severity": "healthy"}
    elif value < 126:
        return {"value": value, "category": "Prediabetic", "severity": "moderate"}
    else:
        return {"value": value, "category": "Diabetic", "severity": "critical"}


# ---------------------------------------------------------------------------
# Heart Disease Risk Prediction
# ---------------------------------------------------------------------------

def _predict_heart_risk(data: dict, bmi_cls: dict, bp_cls: dict,
                        chol_cls: dict, glu_cls: dict) -> dict:
    """
    Predicts heart disease risk using the trained ML model if available.
    Falls back to a clinical heuristic if the model is missing.
    """
    demographics = data["demographics"]
    lifestyle = data["lifestyle"]
    vitals = data["vitals"]

    # 1. Prepare features for ML Model
    age = demographics["age"]
    gender = 1 if demographics["gender"] == "male" else 0
    bmi = vitals["bmi"]
    systolic = vitals["systolic"]
    diastolic = vitals["diastolic"]
    cholesterol = vitals["cholesterol"]
    glucose = vitals["glucose"]
    
    smoking = 1 if lifestyle["smoking"] else 0
    alcohol_map = {"none": 0, "light": 1, "moderate": 2, "heavy": 3}
    alcohol = alcohol_map.get(lifestyle["alcohol"], 0)
    exercise_map = {"sedentary": 0, "light": 1, "moderate": 2, "active": 3}
    exercise = exercise_map.get(lifestyle["exercise"], 0)
    diet_map = {"poor": 0, "average": 1, "balanced": 2, "excellent": 3}
    diet = diet_map.get(lifestyle["diet"], 1)

    factors = []
    # Identify key risk factors for UI display
    if age > 55: factors.append("age")
    if bmi >= 30: factors.append("obesity")
    if systolic >= 130 or diastolic >= 80: factors.append("blood pressure")
    if cholesterol >= 240: factors.append("high cholesterol")
    if glucose >= 126: factors.append("high glucose")
    if smoking == 1: factors.append("smoking")
    if not factors: factors = ["no significant risk factors"]

    if heart_risk_model is not None:
        try:
            # Create feature array matching training data
            X = np.array([[age, gender, bmi, systolic, diastolic, cholesterol, glucose,
                           smoking, alcohol, exercise, diet]])
            
            # Predict probabilities
            probs = heart_risk_model.predict_proba(X)[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            
            # Map predicted class
            classes = ["Low", "Medium", "High"]
            level = classes[pred_class]
            
            # Interpolate a 0-1 score based on class and confidence
            if pred_class == 0: # Low
                score = 0.25 * (1.0 - confidence)
            elif pred_class == 1: # Medium
                score = 0.25 + 0.30 * confidence
            else: # High
                score = 0.55 + 0.45 * confidence
                
            return {
                "level": level,
                "score": round(score, 3),
                "confidence": round(confidence, 2),
                "factors": factors
            }
        except Exception as e:
            print(f"⚠️ ML Prediction failed: {e}. Falling back to heuristic.")
            
    # --- Fallback Heuristic ---
    score = 0.0
    if age > 65: score += 0.15
    elif age > 55: score += 0.10
    elif age > 45: score += 0.06

    if gender == 1 and age > 45: score += 0.05
    if bmi >= 30: score += 0.15
    elif bmi >= 25: score += 0.08
    if systolic >= 140 or diastolic >= 90: score += 0.20
    elif systolic >= 130 or diastolic >= 80: score += 0.14
    if cholesterol >= 240: score += 0.15
    elif cholesterol >= 200: score += 0.08
    if glucose >= 126: score += 0.10
    elif glucose >= 100: score += 0.05
    if smoking == 1: score += 0.10
    if alcohol == 3: score += 0.05
    if exercise == 0: score += 0.05
    elif exercise == 3: score -= 0.05
    if diet == 0: score += 0.03
    elif diet >= 2: score -= 0.03

    score = max(0.0, min(1.0, score))
    if score < 0.25: level = "Low"
    elif score < 0.55: level = "Medium"
    else: level = "High"
    
    confidence = min(0.95, 0.60 + abs(score - 0.40) * 1.2)

    return {
        "level": level,
        "score": round(score, 3),
        "confidence": round(confidence, 2),
        "factors": factors
    }


# ---------------------------------------------------------------------------
# Overall Health Score
# ---------------------------------------------------------------------------

def _compute_overall_score(bmi_cls, bp_cls, chol_cls, glu_cls, heart_risk, lifestyle):
    """Compute a 0-100 overall health score and letter grade."""
    score = 100

    # Deductions for each category
    severity_penalty = {"healthy": 0, "moderate": 10, "critical": 20}
    score -= severity_penalty.get(bmi_cls["severity"], 0)
    score -= severity_penalty.get(bp_cls["severity"], 0)
    score -= severity_penalty.get(chol_cls["severity"], 0)
    score -= severity_penalty.get(glu_cls["severity"], 0)

    # Heart risk deduction
    risk_penalty = {"Low": 0, "Medium": 10, "High": 25}
    score -= risk_penalty.get(heart_risk["level"], 0)

    # Lifestyle bonuses / penalties
    if lifestyle["smoking"]:
        score -= 8
    if lifestyle["exercise"] == "active":
        score += 5
    elif lifestyle["exercise"] == "sedentary":
        score -= 5
    if lifestyle["diet"] in ("balanced", "excellent"):
        score += 3
    elif lifestyle["diet"] == "poor":
        score -= 3

    score = max(0, min(100, score))

    # Letter grade
    if score >= 93:
        grade = "A+"
    elif score >= 85:
        grade = "A"
    elif score >= 78:
        grade = "B+"
    elif score >= 70:
        grade = "B"
    elif score >= 63:
        grade = "C+"
    elif score >= 55:
        grade = "C"
    elif score >= 45:
        grade = "D"
    else:
        grade = "F"

    # Summary sentence
    if score >= 85:
        summary = "Excellent health profile. Keep up your healthy lifestyle!"
    elif score >= 70:
        summary = "Good overall health with some areas for improvement."
    elif score >= 55:
        summary = "Fair health status. Several risk factors need attention."
    else:
        summary = "Health needs significant attention. Please consult a healthcare professional."

    return {"score": score, "grade": grade, "summary": summary}


# ---------------------------------------------------------------------------
# Database Update
# ---------------------------------------------------------------------------

def _update_record(record_id, analysis):
    """Persist analysis results back into the health_records row."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE health_records SET "
        "bmi_category=%s, bp_category=%s, cholesterol_category=%s, glucose_category=%s, "
        "heart_risk_level=%s, heart_risk_score=%s, overall_health_score=%s, overall_health_grade=%s "
        "WHERE id=%s",
        (
            analysis["bmi"]["category"],
            analysis["bloodPressure"]["category"],
            analysis["cholesterol"]["category"],
            analysis["glucose"]["category"],
            analysis["heartRisk"]["level"],
            analysis["heartRisk"]["score"],
            analysis["overallHealth"]["score"],
            analysis["overallHealth"]["grade"],
            record_id,
        )
    )
    conn.commit()
    cursor.close()
    conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(processed_data: dict) -> dict:
    """
    Agent 2 entry point.
    Takes ProcessedData from Agent 1, performs full health analysis.

    Returns: AnalysisResult dict
    """
    vitals = processed_data["vitals"]
    lifestyle = processed_data["lifestyle"]

    bmi_cls = _classify_bmi(vitals["bmi"])
    bp_cls = _classify_bp(vitals["systolic"], vitals["diastolic"])
    chol_cls = _classify_cholesterol(vitals["cholesterol"])
    glu_cls = _classify_glucose(vitals["glucose"])
    heart_risk = _predict_heart_risk(processed_data, bmi_cls, bp_cls, chol_cls, glu_cls)
    overall = _compute_overall_score(bmi_cls, bp_cls, chol_cls, glu_cls, heart_risk, lifestyle)

    analysis = {
        "record_id": processed_data["record_id"],
        "user_id": processed_data["user_id"],
        "demographics": processed_data["demographics"],
        "vitals": vitals,
        "lifestyle": lifestyle,
        "bmi": bmi_cls,
        "bloodPressure": bp_cls,
        "cholesterol": chol_cls,
        "glucose": glu_cls,
        "heartRisk": heart_risk,
        "overallHealth": overall,
    }

    # Persist analysis results
    _update_record(processed_data["record_id"], analysis)

    return analysis

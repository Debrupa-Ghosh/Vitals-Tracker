"""
Agent 4: Assistant & Recommendation Agent
Provides personalized health recommendations, chatbot responses, and doctor suggestions.
"""

import re
from db import get_connection
from data.health_tips import RECOMMENDATIONS, HEALTHY_TIPS, CHAT_FAQ
import google.generativeai as genai
from config import Config

try:
    genai.configure(api_key=Config.GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    print("Warning: Gemini not configured properly.", e)
    gemini_model = None


# ---------------------------------------------------------------------------
# Recommendations Engine
# ---------------------------------------------------------------------------

def get_recommendations(analysis: dict) -> list:
    """
    Generate prioritized health recommendations based on analysis results.
    Returns a list of recommendation dicts sorted by priority.
    """
    recs = []

    # BMI-based recommendations
    bmi_cat = analysis["bmi"]["category"]
    if bmi_cat in RECOMMENDATIONS["bmi"]:
        recs.extend(RECOMMENDATIONS["bmi"][bmi_cat])

    # Blood Pressure recommendations
    bp_cat = analysis["bloodPressure"]["category"]
    if bp_cat in RECOMMENDATIONS["blood_pressure"]:
        recs.extend(RECOMMENDATIONS["blood_pressure"][bp_cat])

    # Cholesterol recommendations
    chol_cat = analysis["cholesterol"]["category"]
    if chol_cat in RECOMMENDATIONS["cholesterol"]:
        recs.extend(RECOMMENDATIONS["cholesterol"][chol_cat])

    # Glucose recommendations
    glu_cat = analysis["glucose"]["category"]
    if glu_cat in RECOMMENDATIONS["glucose"]:
        recs.extend(RECOMMENDATIONS["glucose"][glu_cat])

    # Heart risk recommendations
    risk_level = analysis["heartRisk"]["level"]
    if risk_level in RECOMMENDATIONS["heart_risk"]:
        recs.extend(RECOMMENDATIONS["heart_risk"][risk_level])

    # Lifestyle-specific recommendations
    lifestyle = analysis["lifestyle"]
    if lifestyle["smoking"]:
        recs.extend(RECOMMENDATIONS["lifestyle"]["smoking"])
    if lifestyle["alcohol"] == "heavy":
        recs.extend(RECOMMENDATIONS["lifestyle"]["alcohol_heavy"])
    if lifestyle["exercise"] == "sedentary":
        recs.extend(RECOMMENDATIONS["lifestyle"]["sedentary"])
    if lifestyle["diet"] == "poor":
        recs.extend(RECOMMENDATIONS["lifestyle"]["poor_diet"])

    # If everything is healthy, add maintenance tips
    if not recs:
        recs.extend(HEALTHY_TIPS)

    # Sort by priority: high → medium → low
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: priority_order.get(r["priority"], 1))

    # Remove duplicates by text
    seen = set()
    unique_recs = []
    for r in recs:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique_recs.append(r)

    return unique_recs


# ---------------------------------------------------------------------------
# Chat Assistant
# ---------------------------------------------------------------------------

def chat(message: str, analysis_context: dict = None) -> dict:
    """
    Agent 4 chatbot powered by Gemini API.
    Responds to user messages with context-aware answers based on their health profile.
    """
    if not gemini_model:
        return {
            "response": "I'm sorry, my AI model is currently offline. Please check the API configuration.",
            "type": "error"
        }

    # Build context string from analysis
    context_str = "No specific health data provided."
    if analysis_context:
        vitals = []
        if "bmi" in analysis_context: vitals.append(f"BMI: {analysis_context['bmi']['value']} ({analysis_context['bmi']['category']})")
        if "bloodPressure" in analysis_context: vitals.append(f"Blood Pressure: {analysis_context['bloodPressure']['systolic']}/{analysis_context['bloodPressure']['diastolic']} ({analysis_context['bloodPressure']['category']})")
        if "cholesterol" in analysis_context: vitals.append(f"Cholesterol: {analysis_context['cholesterol']['value']} ({analysis_context['cholesterol']['category']})")
        if "glucose" in analysis_context: vitals.append(f"Glucose: {analysis_context['glucose']['value']} ({analysis_context['glucose']['category']})")
        
        risk = analysis_context.get("heartRisk", {}).get("level", "Unknown")
        
        context_str = f"User Vitals:\n" + "\n".join(vitals) + f"\nHeart Disease Risk: {risk}"

    # System prompt + user message
    prompt = f"""You are 'HealthBot', a friendly and professional AI medical assistant for the 'Vitals Tracker' app. 
Your goal is to answer the user's questions clearly, concisely, and empathetically.

Here is the user's current health context:
{context_str}

Important Guidelines:
1. Do not give direct medical diagnoses or prescribe medications. Always advise consulting a doctor for serious issues.
2. If the user asks about their own health, refer to the provided context.
3. Keep the response concise, usually under 3-4 paragraphs. Use markdown formatting (like bolding and bullet points) for readability.
4. If the user asks about recommendations or doctors, inform them that the 'Doctor Suggestions' are located on the Dashboard, and general 'Recommendations' are on the Assistant tab next to the chat.

User Message: {message}
"""

    try:
        response = gemini_model.generate_content(prompt)
        ai_text = response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        ai_text = f"Sorry, I encountered an error while trying to process your request."

    return {
        "response": ai_text,
        "type": "ai_response"
    }


# ---------------------------------------------------------------------------
# Doctor Suggestions
# ---------------------------------------------------------------------------

def _map_risk_to_specialties(analysis: dict) -> list:
    """Map health risk factors to relevant medical specialties."""
    specialties = set()

    # Always suggest general medicine
    specialties.add("General Medicine")

    # Heart risk
    if analysis["heartRisk"]["level"] in ("Medium", "High"):
        specialties.add("Cardiology")

    # Blood pressure
    if analysis["bloodPressure"]["severity"] == "critical":
        specialties.add("Cardiology")
        specialties.add("Nephrology")

    # Glucose
    if analysis["glucose"]["category"] in ("Prediabetic", "Diabetic"):
        specialties.add("Endocrinology")

    # Cholesterol
    if analysis["cholesterol"]["severity"] in ("moderate", "critical"):
        specialties.add("Cardiology")

    # BMI
    if analysis["bmi"]["category"] in ("Overweight", "Obese", "Underweight"):
        specialties.add("Nutrition & Dietetics")

    # Lifestyle
    if analysis["lifestyle"]["smoking"]:
        specialties.add("Pulmonology")

    if analysis["lifestyle"]["exercise"] in ("moderate", "active"):
        specialties.add("Sports Medicine")

    return list(specialties)


def suggest_doctors(analysis: dict) -> list:
    """
    Query the doctors table filtered by relevant specialties.
    Returns list of doctor dicts sorted by rating.
    """
    specialties = _map_risk_to_specialties(analysis)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    placeholders = ", ".join(["%s"] * len(specialties))
    cursor.execute(
        f"SELECT * FROM doctors WHERE specialty IN ({placeholders}) AND available = TRUE "
        "ORDER BY rating DESC, distance_km ASC",
        specialties
    )
    doctors = cursor.fetchall()
    cursor.close()
    conn.close()

    # Format for frontend
    return [
        {
            "id": d["id"],
            "name": d["name"],
            "specialty": d["specialty"],
            "hospital": d["hospital"],
            "phone": d["phone"],
            "rating": float(d["rating"]),
            "distance": f"{d['distance_km']} km",
            "available": bool(d["available"]),
        }
        for d in doctors
    ]

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# PDF generation imports
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# App config
# -------------------------------
st.set_page_config(page_title="Vitals Tracker & Risk", page_icon="🏥", layout="wide")

# -------------------------------
# PDF helpers
# -------------------------------
def build_report_pdf(payload: dict) -> bytes:
    """
    Build a PDF report from the given payload and return bytes.
    payload keys: name, age, height_cm, weight_kg, bmi, bp_systolic, bp_diastolic,
                  pred_tier, proba (list or None), generated_at (str),
                  doctor_name, doctor_location, doctor_phone, recs (list)
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("Vitals Tracker — Risk Report", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))

    meta_tbl = Table(
        [
            ["Name", payload["name"], "Generated at", payload["generated_at"]],
            ["Age", str(payload["age"]), "BMI", f"{payload['bmi']:.2f}"],
            ["Height (cm)", str(payload["height_cm"]), "Weight (kg)", f"{payload['weight_kg']:.1f}"],
            ["Systolic (mmHg)", str(payload["bp_systolic"]), "Diastolic (mmHg)", str(payload["bp_diastolic"])],
        ],
        colWidths=[110, 150, 110, 150],
    )
    meta_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ]
        )
    )
    story.append(meta_tbl)
    story.append(Spacer(1, 16))

    pred_para = Paragraph(f"Predicted Tier: <b>{payload['pred_tier']}</b>", styles["Heading2"])
    story.append(pred_para)
    if payload.get("proba"):
        p = payload["proba"]
        proba_text = f"Probabilities — Low: {p[0]:.2f}, Medium: {p[1]:.2f}, High: {p[2]:.2f}"
        story.append(Paragraph(proba_text, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Recommendations", styles["Heading2"]))
    for rec in payload.get("recs", []):
        story.append(Paragraph(f"• {rec}", styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Doctor Contact", styles["Heading2"]))
    doc_tbl = Table(
        [
            ["Doctor", payload["doctor_name"]],
            ["Location", payload["doctor_location"]],
            ["Phone", payload["doctor_phone"]],
        ],
        colWidths=[110, 300],
    )
    doc_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ]
        )
    )
    story.append(doc_tbl)

    disclaimer = Paragraph(
        "Note: This report is informational and not a medical diagnosis.", styles["Italic"]
    )
    story.append(Spacer(1, 12))
    story.append(disclaimer)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def make_doctor_card(city_hint: str = "Your City"):
    # Placeholder for hackathon demo; replace with real directory/lookup if needed.
    return {
        "doctor_name": "Dr. Ananya Kulkarni, MD (Internal Medicine)",
        "doctor_location": f"{city_hint} — Shivajinagar Clinic",
        "doctor_phone": "+91-98765-43210",
    }

# -------------------------------
# Session state init
# -------------------------------
if "hist_df" not in st.session_state:
    # Example starter history for Trends (date, systolic, diastolic, weight, height)
    today = pd.to_datetime(datetime.now().date())
    dates = pd.date_range(end=today, periods=14, freq="D")
    st.session_state.hist_df = pd.DataFrame(
        {
            "date": dates,
            "bp_systolic": np.random.randint(110, 140, len(dates)),
            "bp_diastolic": np.random.randint(70, 90, len(dates)),
            "weight_kg": np.random.uniform(60, 80, len(dates)).round(1),
            "height_cm": np.full(len(dates), 170),
        }
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {"role": "user"/"assistant", "content": str}

if "model" not in st.session_state:
    # Train a tiny demo model on synthetic data
    rng = np.random.default_rng(42)
    X_demo = pd.DataFrame(
        {
            "age": rng.integers(20, 80, 400),
            "bmi": rng.uniform(18, 38, 400),
            "bp_systolic": rng.integers(90, 180, 400),
            "bp_diastolic": rng.integers(55, 120, 400),
        }
    )
    y_demo = (
        ((X_demo["bmi"] >= 30) | (X_demo["bp_systolic"] >= 140) | (X_demo["bp_diastolic"] >= 90)).astype(int)
        + (X_demo["age"] > 55).astype(int)
    ).clip(0, 2)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),  # sparse-friendly; harmless here
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    pipeline.fit(X_demo, y_demo)
    st.session_state.model = pipeline

# Utilities
def calc_bmi(weight_kg: float, height_cm: float) -> float:
    h_m = max(height_cm, 1) / 100.0
    return float(weight_kg) / (h_m ** 2)

def risk_label(pred: int) -> str:
    return {0: "Low Risk ✅", 1: "Medium Risk ⚠", 2: "High Risk 🔴"}.get(int(pred), "Unknown")

def validate_ranges(age, height_cm, weight_kg, sys, dia) -> list:
    errs = []
    if not (1 <= age <= 120):
        errs.append("Age must be between 1 and 120.")
    if not (50 <= height_cm <= 250):
        errs.append("Height must be between 50 and 250 cm.")
    if not (20 <= weight_kg <= 300):
        errs.append("Weight must be between 20 and 300 kg.")
    if not (70 <= sys <= 250):
        errs.append("Systolic BP must be between 70 and 250 mmHg.")
    if not (40 <= dia <= 150):
        errs.append("Diastolic BP must be between 40 and 150 mmHg.")
    if sys < dia:
        errs.append("Systolic BP must be greater than Diastolic BP.")
    return errs

# -------------------------------
# Tabs
# -------------------------------
tab_input, tab_trends, tab_risk, tab_chat = st.tabs(["Input", "Trends", "Risk", "Chat"])

# -------------------------------
# Input tab (st.form with validation)
# -------------------------------
with tab_input:
    st.header("Enter today's vitals")

    with st.form("vitals_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Full Name", value=st.session_state.get("name", ""))
            age = st.number_input(
                "Age (years)", min_value=1, max_value=120, value=int(st.session_state.get("age", 30))
            )
        with col2:
            height_cm = st.number_input(
                "Height (cm)", min_value=50, max_value=250, value=int(st.session_state.get("height_cm", 170))
            )
            weight_kg = st.number_input(
                "Weight (kg)", min_value=20.0, max_value=300.0, value=float(st.session_state.get("weight_kg", 70.0))
            )
        with col3:
            bp_systolic = st.number_input(
                "Systolic BP (mmHg)", min_value=70, max_value=250, value=int(st.session_state.get("bp_systolic", 120))
            )
            bp_diastolic = st.number_input(
                "Diastolic BP (mmHg)", min_value=40, max_value=150, value=int(st.session_state.get("bp_diastolic", 80))
            )

        submitted = st.form_submit_button("Save vitals")

    if submitted:
        errors = validate_ranges(age, height_cm, weight_kg, bp_systolic, bp_diastolic)
        if not name.strip():
            errors.append("Name is required.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            # Persist to session and append to history for Trends
            st.session_state.name = name
            st.session_state.age = int(age)
            st.session_state.height_cm = int(height_cm)
            st.session_state.weight_kg = float(weight_kg)
            st.session_state.bp_systolic = int(bp_systolic)
            st.session_state.bp_diastolic = int(bp_diastolic)

            new_row = {
                "date": pd.to_datetime(datetime.now().date()),
                "bp_systolic": int(bp_systolic),
                "bp_diastolic": int(bp_diastolic),
                "weight_kg": float(weight_kg),
                "height_cm": int(height_cm),
            }
            df = st.session_state.hist_df
            # Avoid duplicate date rows for demo; update or append
            if (df["date"] == new_row["date"]).any():
                idx = df.index[df["date"] == new_row["date"]][0]
                for k in ["bp_systolic", "bp_diastolic", "weight_kg", "height_cm"]:
                    df.loc[idx, k] = new_row[k]
            else:
                st.session_state.hist_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            st.success("Vitals saved. Go to Risk tab to compute prediction, or view Trends.")

    # Inline BMI metric
    curr_weight = st.session_state.get("weight_kg", 70.0)
    curr_height = st.session_state.get("height_cm", 170)
    bmi_now = calc_bmi(curr_weight, curr_height)
    st.metric("Current BMI", f"{bmi_now:.2f}")

# -------------------------------
# Trends tab (time series + rolling)
# -------------------------------
with tab_trends:
    st.header("Trends and rolling averages")

    df = st.session_state.hist_df.copy().sort_values("date")
    df["bmi"] = df.apply(lambda r: calc_bmi(r["weight_kg"], r["height_cm"]), axis=1)
    df["bp_sys_roll7"] = df["bp_systolic"].rolling(7, min_periods=1).mean()
    df["bp_dia_roll7"] = df["bp_diastolic"].rolling(7, min_periods=1).mean()
    df["bmi_roll7"] = df["bmi"].rolling(7, min_periods=1).mean()

    st.write("Tip: Hover charts for values; use the Input tab daily to build history.")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Systolic BP (7d rolling)")
        st.line_chart(df.set_index("date")[["bp_systolic", "bp_sys_roll7"]])
    with colB:
        st.subheader("Diastolic BP (7d rolling)")
        st.line_chart(df.set_index("date")[["bp_diastolic", "bp_dia_roll7"]])

    st.subheader("BMI (7d rolling)")
    st.line_chart(df.set_index("date")[["bmi", "bmi_roll7"]])

# -------------------------------
# Risk tab (predict + feature importance + PDF download)
# -------------------------------
with tab_risk:
    st.header("Risk assessment")

    name = st.session_state.get("name", "User")
    age = int(st.session_state.get("age", 30))
    height_cm = int(st.session_state.get("height_cm", 170))
    weight_kg = float(st.session_state.get("weight_kg", 70.0))
    bp_systolic = int(st.session_state.get("bp_systolic", 120))
    bp_diastolic = int(st.session_state.get("bp_diastolic", 80))

    bmi = calc_bmi(weight_kg, height_cm)

    # Show inputs summary
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Age (y)", age)
    i2.metric("BMI", f"{bmi:.2f}")
    i3.metric("Systolic", f"{bp_systolic} mmHg")
    i4.metric("Diastolic", f"{bp_diastolic} mmHg")

    model = st.session_state.model
    X_user = pd.DataFrame(
        [{"age": age, "bmi": bmi, "bp_systolic": bp_systolic, "bp_diastolic": bp_diastolic}]
    )

    pred = int(model.predict(X_user)[0])
    proba = model.predict_proba(X_user)[0] if hasattr(model, "predict_proba") else None

    st.subheader(f"Predicted tier: {risk_label(pred)}")
    if proba is not None and len(proba) == 3:
        st.write(f"Probabilities → Low: {proba[0]:.2f} • Medium: {proba[1]:.2f} • High: {proba[2]:.2f}")

    # Feature importance from RF (best-effort; pipeline step name 'rf')
    try:
        rf = model.named_steps["rf"]
        importances = rf.feature_importances_
        feat_names = ["age", "bmi", "bp_systolic", "bp_diastolic"]
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        st.subheader("Top drivers (global model)")
        st.bar_chart(imp_df.set_index("feature"))
    except Exception:
        st.info("Feature importances unavailable for this model configuration.")

    # Recommendations (simple rules)
    recs = []
    if bmi >= 25:
        recs += [
            "Aim for 150–300 minutes of moderate activity per week.",
            "Reduce calorie-dense, low-nutrient foods; increase fiber and protein.",
        ]
    if bp_systolic >= 130 or bp_diastolic >= 85:
        recs += [
            "Limit sodium to ~2,300 mg/day; prefer fresh foods over processed.",
            "Practice stress management (breathing, brief walks, sleep hygiene).",
        ]
    if age >= 40:
        recs += ["Schedule periodic health checkups and monitor BP regularly."]
    if not recs:
        recs = [
            "Maintain a balanced diet and regular physical activity.",
            "Sleep 7–9 hours and stay hydrated.",
        ]
    st.subheader("Personalized suggestions")
    for r in recs:
        st.write("• " + r)

    # ------- PDF Download -------
    city_hint = "Your City"
    doctor_info = make_doctor_card(city_hint)
    payload = {
        "name": name,
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "pred_tier": risk_label(pred),
        "proba": proba.tolist() if proba is not None else None,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "recs": recs,
        **doctor_info,
    }
    pdf_bytes = build_report_pdf(payload)
    st.download_button(
        label="📄 Download PDF report",
        data=pdf_bytes,
        file_name=f"vitals_risk_report_{name.replace(' ', '_')}.pdf",
        mime="application/pdf",
    )

# -------------------------------
# Chat tab (Streamlit chat elements, grounded on session data)
# -------------------------------
with tab_chat:
    st.header("Assistant")
    st.write(
        "Ask about your inputs, risk result, or how to improve. This demo uses rules and app context (no external API)."
    )

    def answer_with_context(message: str) -> str:
        """Very simple rule-based responder grounded in current session values."""
        name = st.session_state.get("name", "User")
        age = int(st.session_state.get("age", 30))
        height_cm = int(st.session_state.get("height_cm", 170))
        weight_kg = float(st.session_state.get("weight_kg", 70.0))
        bp_systolic = int(st.session_state.get("bp_systolic", 120))
        bp_diastolic = int(st.session_state.get("bp_diastolic", 80))
        bmi = calc_bmi(weight_kg, height_cm)

        model = st.session_state.model
        X_user = pd.DataFrame(
            [{"age": age, "bmi": bmi, "bp_systolic": bp_systolic, "bp_diastolic": bp_diastolic}]
        )
        pred = int(model.predict(X_user)[0])

        msg = message.lower().strip()

        if "bmi" in msg:
            return f"{name}, your current BMI is {bmi:.2f}. A BMI of 18.5–24.9 is generally considered in the healthy range."
        if "blood" in msg or "bp" in msg or "pressure" in msg:
            return f"Your latest BP is {bp_systolic}/{bp_diastolic} mmHg. Values ≥130/85 often warrant lifestyle attention; consult a clinician for personalized guidance."
        if "risk" in msg or "result" in msg or "prediction" in msg:
            return f"Your model tier is {risk_label(pred)}. Key drivers typically include BMI and blood pressure. Improving activity, diet quality, sleep, and stress can shift risk down."
        if "improve" in msg or "recommend" in msg or "diet" in msg or "exercise" in msg:
            tips = []
            if bmi >= 25:
                tips.append("Target a modest weekly deficit and prioritize protein and fiber.")
            if bp_systolic >= 130 or bp_diastolic >= 85:
                tips.append("Reduce sodium and manage stress; add daily walks.")
            if not tips:
                tips.append("Maintain consistent routines: 7–9h sleep, regular meals, and activity.")
            return "Suggestions: " + " ".join(tips)
        if "hello" in msg or "hi" in msg:
            return f"Hi {name}! You can ask about BMI, BP, your risk result, or how to improve."
        return "You can ask about your BMI, BP, risk result, or how to improve. This assistant is informational only."

    # render history
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    prompt = st.chat_input("Type your question")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        reply = answer_with_context(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# Footer note
st.caption("This app is a demo and does not provide medical diagnosis.")

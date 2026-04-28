"""
Health tips and recommendation knowledge base.
Used by Agent 4 (Assistant & Recommendation) to generate personalized advice.
"""

RECOMMENDATIONS = {
    "bmi": {
        "Underweight": [
            {"category": "Diet", "priority": "high", "icon": "🥗", "text": "Increase caloric intake with nutrient-dense foods like nuts, avocados, whole grains, and lean proteins."},
            {"category": "Exercise", "priority": "medium", "icon": "🏋️", "text": "Focus on strength training exercises to build muscle mass. Avoid excessive cardio."},
            {"category": "Medical", "priority": "high", "icon": "🏥", "text": "Consult a nutritionist to rule out underlying causes like thyroid disorders or malabsorption."},
        ],
        "Overweight": [
            {"category": "Diet", "priority": "high", "icon": "🥗", "text": "Reduce processed food intake. Focus on vegetables, lean protein, and whole grains. Aim for a 500-calorie daily deficit."},
            {"category": "Exercise", "priority": "high", "icon": "🏃", "text": "Aim for 150+ minutes of moderate aerobic activity per week. Walking, swimming, or cycling are excellent starting points."},
            {"category": "Lifestyle", "priority": "medium", "icon": "😴", "text": "Ensure 7-8 hours of quality sleep. Poor sleep disrupts hunger hormones and promotes weight gain."},
        ],
        "Obese": [
            {"category": "Medical", "priority": "high", "icon": "🏥", "text": "Schedule an appointment with your physician for a comprehensive metabolic assessment."},
            {"category": "Diet", "priority": "high", "icon": "🥗", "text": "Work with a registered dietitian to create a structured meal plan. Consider Mediterranean or DASH diet."},
            {"category": "Exercise", "priority": "high", "icon": "🏃", "text": "Start with low-impact activities like swimming or walking. Gradually increase duration and intensity."},
            {"category": "Stress", "priority": "medium", "icon": "🧘", "text": "Practice stress management techniques. Emotional eating often contributes to weight challenges."},
        ],
    },
    "blood_pressure": {
        "Low": [
            {"category": "Diet", "priority": "medium", "icon": "🧂", "text": "Slightly increase salt intake. Stay well-hydrated with water and electrolyte drinks."},
            {"category": "Lifestyle", "priority": "medium", "icon": "💧", "text": "Stand up slowly from sitting or lying positions. Wear compression stockings if dizzy."},
        ],
        "Elevated": [
            {"category": "Diet", "priority": "high", "icon": "🥗", "text": "Reduce sodium intake to less than 2,300 mg/day. Increase potassium-rich foods like bananas and spinach."},
            {"category": "Exercise", "priority": "high", "icon": "🏃", "text": "Regular aerobic exercise (30 min/day, 5 days/week) can lower systolic BP by 5-8 mmHg."},
        ],
        "High Stage 1": [
            {"category": "Medical", "priority": "high", "icon": "🏥", "text": "Monitor BP daily and consult your doctor. Lifestyle changes are critical at this stage."},
            {"category": "Diet", "priority": "high", "icon": "🥗", "text": "Adopt the DASH diet. Limit alcohol and caffeine. Reduce sodium to under 1,500 mg/day."},
            {"category": "Stress", "priority": "high", "icon": "🧘", "text": "Practice daily relaxation: deep breathing, meditation, or yoga. Chronic stress elevates BP."},
        ],
        "High Stage 2": [
            {"category": "Medical", "priority": "high", "icon": "🚨", "text": "See your doctor immediately. You may need medication in addition to lifestyle changes."},
            {"category": "Diet", "priority": "high", "icon": "🥗", "text": "Strict DASH diet adherence. Eliminate processed foods, reduce sodium below 1,500 mg/day."},
            {"category": "Lifestyle", "priority": "high", "icon": "🚭", "text": "If you smoke, quitting is the single most impactful change you can make for your heart."},
        ],
    },
    "cholesterol": {
        "Borderline High": [
            {"category": "Diet", "priority": "high", "icon": "🐟", "text": "Increase omega-3 fatty acids (fish, flaxseed). Reduce saturated fats and trans fats."},
            {"category": "Exercise", "priority": "medium", "icon": "🏃", "text": "Regular exercise raises HDL (good cholesterol). Aim for 30 minutes of brisk walking daily."},
        ],
        "High": [
            {"category": "Medical", "priority": "high", "icon": "🏥", "text": "Consult your doctor about cholesterol management. Medication may be necessary."},
            {"category": "Diet", "priority": "high", "icon": "🥑", "text": "Adopt a heart-healthy diet: oats, nuts, olive oil, fatty fish. Avoid fried foods and red meat."},
            {"category": "Lifestyle", "priority": "high", "icon": "🚭", "text": "If you smoke, quitting can improve HDL cholesterol by up to 10% within a year."},
        ],
    },
    "glucose": {
        "Prediabetic": [
            {"category": "Diet", "priority": "high", "icon": "🍎", "text": "Reduce refined carbohydrates and sugary beverages. Choose complex carbs with low glycemic index."},
            {"category": "Exercise", "priority": "high", "icon": "🏃", "text": "150 minutes of moderate exercise weekly can reduce diabetes risk by 58%."},
            {"category": "Medical", "priority": "medium", "icon": "🔬", "text": "Get an HbA1c test every 6 months. Early intervention prevents progression to diabetes."},
        ],
        "Diabetic": [
            {"category": "Medical", "priority": "high", "icon": "🚨", "text": "Consult an endocrinologist immediately. Regular glucose monitoring is essential."},
            {"category": "Diet", "priority": "high", "icon": "🥗", "text": "Follow a diabetes-friendly meal plan. Monitor carbohydrate intake at every meal."},
            {"category": "Lifestyle", "priority": "high", "icon": "📊", "text": "Monitor blood glucose regularly. Keep a log to identify patterns and triggers."},
        ],
    },
    "heart_risk": {
        "Medium": [
            {"category": "Medical", "priority": "high", "icon": "❤️", "text": "Schedule a cardiac screening including ECG and lipid panel within the next month."},
            {"category": "Lifestyle", "priority": "high", "icon": "🧘", "text": "Prioritize stress reduction. Chronic stress is a major contributor to cardiovascular disease."},
        ],
        "High": [
            {"category": "Medical", "priority": "high", "icon": "🚨", "text": "Urgent: Schedule a comprehensive cardiac evaluation. Share this report with your cardiologist."},
            {"category": "Lifestyle", "priority": "high", "icon": "🚭", "text": "Eliminate all tobacco and limit alcohol. These are the most impactful changes you can make."},
            {"category": "Exercise", "priority": "medium", "icon": "🚶", "text": "Begin a doctor-supervised exercise program. Start gently and increase gradually."},
        ],
    },
    "lifestyle": {
        "smoking": [
            {"category": "Lifestyle", "priority": "high", "icon": "🚭", "text": "Quitting smoking reduces heart disease risk by 50% within 1 year. Consider nicotine replacement therapy."},
        ],
        "alcohol_heavy": [
            {"category": "Lifestyle", "priority": "high", "icon": "🍷", "text": "Reduce alcohol to ≤1 drink/day for women, ≤2 for men. Heavy drinking raises blood pressure and heart risk."},
        ],
        "sedentary": [
            {"category": "Exercise", "priority": "high", "icon": "🏃", "text": "Break up long sitting periods. Even 5-minute walks every hour significantly reduce health risks."},
        ],
        "poor_diet": [
            {"category": "Diet", "priority": "high", "icon": "🥗", "text": "Transition to whole foods. Start by adding one serving of vegetables to each meal."},
        ],
    },
}

# General healthy tips for people with good results
HEALTHY_TIPS = [
    {"category": "Maintenance", "priority": "low", "icon": "✅", "text": "Great job! Maintain your current healthy lifestyle. Continue regular exercise and balanced nutrition."},
    {"category": "Prevention", "priority": "low", "icon": "🩺", "text": "Schedule annual health checkups even when feeling healthy. Prevention is better than cure."},
    {"category": "Hydration", "priority": "low", "icon": "💧", "text": "Stay hydrated with 2-3 liters of water daily. Proper hydration supports cardiovascular health."},
    {"category": "Sleep", "priority": "low", "icon": "😴", "text": "Prioritize 7-8 hours of quality sleep. Good sleep is foundational to long-term health."},
    {"category": "Mental Health", "priority": "low", "icon": "🧠", "text": "Don't neglect mental health. Practice mindfulness, maintain social connections, and seek help if needed."},
]


# Chat FAQ knowledge base
CHAT_FAQ = {
    "bmi": "BMI (Body Mass Index) is calculated as weight (kg) divided by height (m) squared. It's a screening tool for weight categories: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (≥30).",
    "blood_pressure": "Blood pressure is measured as systolic/diastolic (mmHg). Normal is below 120/80. Elevated is 120-129/<80. High Stage 1 is 130-139/80-89. High Stage 2 is ≥140/≥90.",
    "cholesterol": "Total cholesterol below 200 mg/dL is desirable. 200-239 is borderline high. 240+ is high. High cholesterol increases risk of heart disease and stroke.",
    "glucose": "Fasting blood glucose below 100 mg/dL is normal. 100-125 indicates prediabetes. 126+ may indicate diabetes. Regular monitoring is important for early detection.",
    "heart_disease": "Heart disease risk depends on multiple factors: age, gender, blood pressure, cholesterol, glucose, BMI, smoking status, and exercise habits. Our algorithm weighs these factors to estimate your risk level.",
    "dash_diet": "The DASH (Dietary Approaches to Stop Hypertension) diet emphasizes fruits, vegetables, whole grains, lean protein, and low-fat dairy while limiting sodium, saturated fat, and sweets.",
    "exercise": "The American Heart Association recommends at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week, plus muscle-strengthening activities 2+ days/week.",
    "sleep": "Adults need 7-9 hours of sleep per night. Poor sleep increases risk of obesity, diabetes, cardiovascular disease, and mental health disorders.",
    "stress": "Chronic stress contributes to high blood pressure, heart disease, obesity, and diabetes. Effective stress management includes exercise, meditation, adequate sleep, and social support.",
}

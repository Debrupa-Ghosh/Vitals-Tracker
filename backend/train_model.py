"""
train_model.py
Generates a synthetic dataset based on clinical rules, trains a RandomForestClassifier,
and saves the model as a .joblib file for the Health Analysis Agent.
"""

import os
import random
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def generate_synthetic_data(num_samples=10000):
    print(f"Generating {num_samples} synthetic patient records...")
    X = []
    y = []

    for _ in range(num_samples):
        # 1. Generate realistic basic demographics
        age = random.randint(18, 90)
        gender = random.choice([0, 1])  # 0: Female/Other, 1: Male

        # 2. Generate vitals with some correlation to age/lifestyle
        bmi = round(random.uniform(16.0, 45.0), 1)
        systolic = random.randint(90, 180)
        diastolic = random.randint(60, 110)
        cholesterol = random.randint(120, 300)
        glucose = random.randint(70, 200)

        # 3. Generate lifestyle factors
        smoking = random.choice([0, 1])  # 0: No, 1: Yes
        # Alcohol: 0: None, 1: Light, 2: Moderate, 3: Heavy
        alcohol = random.choice([0, 1, 2, 3])
        # Exercise: 0: Sedentary, 1: Light, 2: Moderate, 3: Active
        exercise = random.choice([0, 1, 2, 3])
        # Diet: 0: Poor, 1: Average, 2: Balanced, 3: Excellent
        diet = random.choice([0, 1, 2, 3])

        # 4. Calculate a synthetic "true" risk score (0-1) using clinical heuristic logic
        # This acts as our ground truth for training
        score = 0.0

        if age > 65: score += 0.15
        elif age > 55: score += 0.10
        elif age > 45: score += 0.06

        if gender == 1 and age > 45: score += 0.05

        if bmi >= 30: score += 0.15
        elif bmi >= 25: score += 0.08

        if systolic >= 140 or diastolic >= 90: score += 0.20
        elif systolic >= 130 or diastolic >= 80: score += 0.14
        elif systolic >= 120 and diastolic < 80: score += 0.07

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

        # Add some random noise to make the dataset realistic and the model robust
        noise = random.uniform(-0.05, 0.05)
        score = max(0.0, min(1.0, score + noise))

        # 5. Classify the risk level (Target variable)
        # 0: Low, 1: Medium, 2: High
        if score < 0.25:
            target = 0
        elif score < 0.55:
            target = 1
        else:
            target = 2

        # Append feature array and target
        X.append([
            age, gender, bmi, systolic, diastolic, cholesterol, glucose,
            smoking, alcohol, exercise, diet
        ])
        y.append(target)

    return np.array(X), np.array(y)


def main():
    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Get Data
    X, y = generate_synthetic_data(10000)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Initialize Model
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # 4. Train
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk (0)", "Medium Risk (1)", "High Risk (2)"]))

    # 6. Save Model
    model_path = os.path.join(models_dir, "heart_risk_model.joblib")
    joblib.dump(clf, model_path)
    print(f"\n✅ Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()

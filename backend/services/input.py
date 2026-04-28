"""
Agent 1: Input & Processing Agent
Validates user input, calculates BMI, and saves structured data to MySQL.
"""

from db import get_connection


# ---------------------------------------------------------------------------
# Validation Rules
# ---------------------------------------------------------------------------

VALIDATION_RULES = {
    "name":        {"type": str,   "required": True,  "min_len": 1, "max_len": 100},
    "age":         {"type": int,   "required": True,  "min": 1,     "max": 120},
    "gender":      {"type": str,   "required": True,  "allowed": ["male", "female", "other"]},
    "height":      {"type": float, "required": True,  "min": 50,    "max": 300},
    "weight":      {"type": float, "required": True,  "min": 10,    "max": 500},
    "systolic":    {"type": int,   "required": True,  "min": 60,    "max": 250},
    "diastolic":   {"type": int,   "required": True,  "min": 40,    "max": 150},
    "cholesterol": {"type": int,   "required": True,  "min": 50,    "max": 500},
    "glucose":     {"type": int,   "required": True,  "min": 30,    "max": 600},
    "smoking":     {"type": bool,  "required": False},
    "alcohol":     {"type": str,   "required": True,  "allowed": ["none", "light", "moderate", "heavy"]},
    "exercise":    {"type": str,   "required": True,  "allowed": ["sedentary", "light", "moderate", "active"]},
    "diet":        {"type": str,   "required": True,  "allowed": ["poor", "average", "balanced", "excellent"]},
}


def _validate(raw: dict) -> list:
    """Validate raw input data. Returns a list of error strings (empty = valid)."""
    errors = []
    for field, rules in VALIDATION_RULES.items():
        value = raw.get(field)

        # Required check
        if rules.get("required") and (value is None or value == ""):
            errors.append(f"{field} is required.")
            continue

        if value is None or value == "":
            continue

        # Type coercion & check
        try:
            if rules["type"] == int:
                value = int(value)
            elif rules["type"] == float:
                value = float(value)
            elif rules["type"] == bool:
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = bool(value)
            else:
                value = str(value).strip()
            raw[field] = value
        except (ValueError, TypeError):
            errors.append(f"{field} must be a valid {rules['type'].__name__}.")
            continue

        # Range check
        if "min" in rules and value < rules["min"]:
            errors.append(f"{field} must be at least {rules['min']}.")
        if "max" in rules and value > rules["max"]:
            errors.append(f"{field} must be at most {rules['max']}.")
        if "min_len" in rules and len(str(value)) < rules["min_len"]:
            errors.append(f"{field} is too short.")
        if "max_len" in rules and len(str(value)) > rules["max_len"]:
            errors.append(f"{field} is too long (max {rules['max_len']} chars).")

        # Allowed values
        if "allowed" in rules and value not in rules["allowed"]:
            errors.append(f"{field} must be one of: {', '.join(rules['allowed'])}.")

    return errors


def _calculate_bmi(weight: float, height: float) -> float:
    """Calculate BMI from weight (kg) and height (cm)."""
    height_m = height / 100.0
    return round(weight / (height_m ** 2), 1)


def _save_to_db(data: dict) -> dict:
    """Save user and health record to MySQL. Returns dict with user_id and record_id."""
    conn = get_connection()
    cursor = conn.cursor()

    # Upsert user (find by name + age + gender, or create)
    cursor.execute(
        "SELECT id FROM users WHERE name = %s AND age = %s AND gender = %s",
        (data["name"], data["age"], data["gender"])
    )
    row = cursor.fetchone()
    if row:
        user_id = row[0]
    else:
        cursor.execute(
            "INSERT INTO users (name, age, gender) VALUES (%s, %s, %s)",
            (data["name"], data["age"], data["gender"])
        )
        user_id = cursor.lastrowid

    # Insert health record
    cursor.execute(
        "INSERT INTO health_records "
        "(user_id, height, weight, bmi, systolic, diastolic, cholesterol, glucose, "
        " smoking, alcohol, exercise, diet) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (user_id, data["height"], data["weight"], data["bmi"],
         data["systolic"], data["diastolic"], data["cholesterol"], data["glucose"],
         data.get("smoking", False), data.get("alcohol", "none"),
         data.get("exercise", "sedentary"), data.get("diet", "average"))
    )
    record_id = cursor.lastrowid

    conn.commit()
    cursor.close()
    conn.close()

    return {"user_id": user_id, "record_id": record_id}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process(raw_data: dict) -> dict:
    """
    Agent 1 entry point.
    Validates input, calculates BMI, saves to DB, returns processed data.

    Returns:
        On success: {"valid": True, "data": ProcessedData}
        On failure: {"valid": False, "errors": [str]}
    """
    errors = _validate(raw_data)
    if errors:
        return {"valid": False, "errors": errors}

    bmi = _calculate_bmi(raw_data["weight"], raw_data["height"])
    raw_data["bmi"] = bmi

    ids = _save_to_db(raw_data)

    processed = {
        "valid": True,
        "data": {
            "record_id": ids["record_id"],
            "user_id": ids["user_id"],
            "demographics": {
                "name": raw_data["name"],
                "age": int(raw_data["age"]),
                "gender": raw_data["gender"],
            },
            "vitals": {
                "height": float(raw_data["height"]),
                "weight": float(raw_data["weight"]),
                "bmi": bmi,
                "systolic": int(raw_data["systolic"]),
                "diastolic": int(raw_data["diastolic"]),
                "cholesterol": int(raw_data["cholesterol"]),
                "glucose": int(raw_data["glucose"]),
            },
            "lifestyle": {
                "smoking": bool(raw_data.get("smoking", False)),
                "alcohol": raw_data.get("alcohol", "none"),
                "exercise": raw_data.get("exercise", "sedentary"),
                "diet": raw_data.get("diet", "average"),
            },
        }
    }
    return processed

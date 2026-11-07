"""Find lab values and vital signs in a report and flag abnormal ones.

Reference ranges are typical adult values used for screening only; real labs
print their own ranges, and sex or age can change them.
"""

import re

# name, aliases, low, high, unit, plain-language name
LAB_TESTS = [
    ("Hemoglobin", ["hemoglobin", "haemoglobin", "hb", "hgb"], 12.0, 17.0, "g/dL", "red blood cell protein"),
    ("WBC count", ["total leukocyte count", "total leucocyte count", "white blood cell count", "wbc count",
                   "wbc", "tlc", "total wbc count"], 4000, 11000, "/µL", "white blood cells (infection fighters)"),
    ("Platelet count", ["platelet count", "platelets", "plt"], 150000, 450000, "/µL", "platelets (clotting cells)"),
    ("ESR", ["esr", "erythrocyte sedimentation rate"], 0, 20, "mm/hr", "inflammation marker (ESR)"),
    ("CRP", ["c-reactive protein", "crp"], 0, 10, "mg/L", "inflammation marker (CRP)"),
    ("Fasting blood sugar", ["fasting blood sugar", "fasting blood glucose", "fasting plasma glucose",
                             "fasting glucose"], 70, 100, "mg/dL", "fasting sugar"),
    ("Post-prandial blood sugar", ["post-prandial blood sugar", "postprandial blood sugar",
                                   "post prandial blood sugar"], 70, 140, "mg/dL", "sugar after meals"),
    ("Random blood sugar", ["random blood sugar", "random blood glucose", "random glucose", "blood sugar",
                            "blood glucose"], 70, 140, "mg/dL", "blood sugar"),
    ("HbA1c", ["hba1c", "glycated hemoglobin", "glycosylated hemoglobin"], 4.0, 5.7, "%",
     "3-month average sugar (HbA1c)"),
    ("Serum creatinine", ["serum creatinine", "creatinine", "s. creatinine"], 0.6, 1.3, "mg/dL",
     "kidney function (creatinine)"),
    ("Blood urea", ["blood urea nitrogen", "blood urea", "urea", "bun"], 7, 40, "mg/dL", "kidney waste (urea)"),
    ("Sodium", ["serum sodium", "sodium", "na+"], 135, 145, "mmol/L", "sodium (salt)"),
    ("Potassium", ["serum potassium", "potassium", "k+"], 3.5, 5.1, "mmol/L", "potassium"),
    ("Total cholesterol", ["total cholesterol", "serum cholesterol", "cholesterol"], 0, 200, "mg/dL",
     "total cholesterol"),
    ("LDL cholesterol", ["ldl cholesterol", "ldl-c", "ldl"], 0, 100, "mg/dL", "bad cholesterol (LDL)"),
    ("HDL cholesterol", ["hdl cholesterol", "hdl-c", "hdl"], 40, 100, "mg/dL", "good cholesterol (HDL)"),
    ("Triglycerides", ["triglycerides", "tg"], 0, 150, "mg/dL", "blood fats (triglycerides)"),
    ("TSH", ["tsh", "thyroid stimulating hormone"], 0.4, 4.0, "mIU/L", "thyroid hormone signal (TSH)"),
    ("ALT (SGPT)", ["sgpt", "alt", "alanine aminotransferase"], 7, 56, "U/L", "liver enzyme (ALT)"),
    ("AST (SGOT)", ["sgot", "ast", "aspartate aminotransferase"], 10, 40, "U/L", "liver enzyme (AST)"),
    ("Total bilirubin", ["total bilirubin", "serum bilirubin", "bilirubin"], 0.2, 1.2, "mg/dL",
     "bilirubin (jaundice marker)"),
    ("Albumin", ["serum albumin", "albumin"], 3.5, 5.0, "g/dL", "blood protein (albumin)"),
    ("Uric acid", ["serum uric acid", "uric acid"], 3.4, 7.0, "mg/dL", "uric acid"),
    ("Troponin I", ["troponin i", "troponin-i", "trop i", "troponin"], 0, 0.04, "ng/mL",
     "heart damage marker (troponin)"),
    ("Vitamin D", ["vitamin d3", "vitamin d", "25-oh vitamin d"], 30, 100, "ng/mL", "vitamin D"),
    ("Ejection fraction", ["left ventricular ejection fraction", "ejection fraction"], 55, 70, "%",
     "heart pumping strength"),
    ("Heart rate", ["heart rate", "pulse rate", "pulse"], 60, 100, "bpm", "heart rate"),
    ("Respiratory rate", ["respiratory rate"], 12, 20, "/min", "breathing rate"),
    ("Oxygen saturation", ["oxygen saturation", "spo2", "sao2"], 95, 100, "%", "oxygen level"),
    ("Temperature", ["temperature", "temp"], 97.0, 99.5, "°F", "body temperature"),
]

_NUMBER = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_GAP = r"[^\d\n.]{0,25}?"  # words like " level was ", " (fasting): " between name and value
_TIME_AFTER = re.compile(r"\s*(hours?|hrs?|days?|weeks?|months?|years?|times?|cigarettes)\b", re.IGNORECASE)
_BP_RE = re.compile(r"\bblood pressure\b[^\d\n]{0,20}?(\d{2,3})\s*/\s*(\d{2,3})", re.IGNORECASE)


def _normalise(name, value, raw_after):
    """Convert values written in other common units to the reference unit."""
    after = raw_after.lower()
    if name == "WBC count" and value < 200:  # written as 11.2 x10^3/µL
        value *= 1000
    elif name == "Platelet count":
        if "lakh" in after or value < 10:  # 1.5 lakh
            value *= 100000
        elif value < 1000:  # 150 x10^3/µL
            value *= 1000
    elif name == "Temperature" and value < 45:  # Celsius
        value = value * 9 / 5 + 32
    value = round(value, 2)
    return int(value) if value.is_integer() else value


def find_lab_values(text):
    """Return a list of lab/vital readings with a NORMAL, LOW or HIGH flag."""
    candidates = []
    for name, aliases, low, high, unit, plain in LAB_TESTS:
        for alias in aliases:
            pattern = r"(?<![\w-])%s(?![\w])%s(?<![\w.])%s" % (re.escape(alias), _GAP, _NUMBER)
            for m in re.finditer(pattern, text, re.IGNORECASE):
                candidates.append((m.start(), m.end(), len(alias), name, low, high, unit, plain, m))

    # Longest alias wins when two matches overlap ("LDL cholesterol" over "cholesterol").
    candidates.sort(key=lambda c: (-c[2], c[0]))
    taken, results, seen = [], [], set()
    for start, end, _, name, low, high, unit, plain, m in candidates:
        if any(start < t_end and end > t_start for t_start, t_end in taken):
            continue
        if _TIME_AFTER.match(text, end):  # "HbA1c after 3 months" is not a value
            continue
        taken.append((start, end))
        value = _normalise(name, float(m.group(1).replace(",", "")), text[m.end():m.end() + 12])
        if (name, value) in seen:
            continue
        seen.add((name, value))
        status = "LOW" if value < low else "HIGH" if value > high else "NORMAL"
        results.append({
            "test": name, "value": value, "unit": unit, "low": low, "high": high,
            "status": status, "plain": plain, "source": m.group(0).strip(), "pos": start,
        })

    for m in _BP_RE.finditer(text):
        sys_bp, dia_bp = int(m.group(1)), int(m.group(2))
        status = "HIGH" if sys_bp >= 140 or dia_bp >= 90 else "LOW" if sys_bp < 90 or dia_bp < 60 else "NORMAL"
        results.append({
            "test": "Blood pressure", "value": "%d/%d" % (sys_bp, dia_bp), "unit": "mmHg", "low": "90/60",
            "high": "139/89", "status": status, "plain": "blood pressure", "source": m.group(0).strip(),
            "pos": m.start(),
        })

    results.sort(key=lambda r: r["pos"])
    return results


def abnormal(results):
    return [r for r in results if r["status"] != "NORMAL"]

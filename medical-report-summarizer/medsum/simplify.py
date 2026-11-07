"""Step 5: turn the clinical summary into a plain-language patient summary.

Uses glossary-based lexical simplification (jargon -> everyday words), rewrites
third-person phrases into second person, and adds short explanations for
abnormal test results. Readability is measured with the Flesch Reading Ease
score (higher = easier; 60+ is plain English).
"""

import re

from .lexicon import GLOSSARY

_PERSON_RULES = [
    (r"\bthe patient was\b", "you were"),
    (r"\bpatient was\b", "you were"),
    (r"\bthe patient is\b", "you are"),
    (r"\bpatient is\b", "you are"),
    (r"\bthe patient has\b", "you have"),
    (r"\bpatient has\b", "you have"),
    (r"\bthe patient's\b", "your"),
    (r"\bpatient's\b", "your"),
    (r"\bthe patient\b", "you"),
    (r"\bpatient\b", "you"),
]

_VERB_FIX = {"has": "have", "is": "are", "was": "were", "lives": "live", "presents": "present", "denies": "deny",
             "complains": "complain", "reports": "report", "needs": "need", "takes": "take", "feels": "feel"}

_GLOSSARY_RE = re.compile(
    r"(?<![\w-])(%s)(?![\w-])" % "|".join(re.escape(k) for k in sorted(GLOSSARY, key=len, reverse=True)),
    re.IGNORECASE,
)


def _capitalise_sentences(text):
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def simplify_text(text):
    """Replace medical jargon with plain words and speak to the patient directly."""
    text = _GLOSSARY_RE.sub(lambda m: GLOSSARY[m.group(0).lower()], text)
    for pattern, repl in _PERSON_RULES:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\byou were (\d+[- ]year[- ]old)", r"you are a \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(you) (has|is|was|lives|presents|denies|complains|reports|needs|takes|feels)\b",
                  lambda m: m.group(1) + " " + _VERB_FIX[m.group(2).lower()], text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return _capitalise_sentences(text)


def explain_labs(abnormal_results):
    """One short, plain sentence per abnormal lab value."""
    lines = []
    for r in abnormal_results:
        word = "higher" if r["status"] == "HIGH" else "lower"
        if r["test"] == "Blood pressure":
            normal = "below 140/90"
        else:
            normal = "%s-%s" % (_fmt(r["low"]), _fmt(r["high"]))
        lines.append("%s: %s than normal, %s %s (normal %s %s)." % (
            r["plain"][0].upper() + r["plain"][1:], word, _fmt(r["value"]), r["unit"], normal, r["unit"]))
    return lines


def _fmt(value):
    if isinstance(value, float):
        return ("%d" % value) if value.is_integer() and value >= 100 else ("%g" % value)
    return str(value)


def patient_summary(clinical_summary, groups, abnormal_results):
    """Build the patient-facing summary: simplified text + key points."""
    text = simplify_text(clinical_summary)
    conditions = [GLOSSARY.get(d["label"].lower(), d["label"]) for d in groups.get("Disease", []) if not d["negated"]]
    medicines = [m["label"] for m in groups.get("Medication", []) if not m["negated"]]
    return {
        "text": text,
        "conditions": _unique(conditions),
        "lab_notes": explain_labs(abnormal_results),
        "medicines": medicines,
    }


def _unique(items):
    """Drop repeats and names already contained in a longer one ("diabetes" vs "diabetes (high blood sugar)")."""
    lowered = [item.lower() for item in items]
    seen, out = set(), []
    for item, low in zip(items, lowered):
        if low in seen or any(low != other and low in other for other in lowered):
            continue
        seen.add(low)
        out.append(item)
    return out


def _syllables(word):
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1 and not word.endswith("le"):
        count -= 1
    return max(1, count)


def flesch_reading_ease(text):
    """Flesch Reading Ease = 206.835 - 1.015 (words/sentences) - 84.6 (syllables/words)."""
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
    return round(score, 1)

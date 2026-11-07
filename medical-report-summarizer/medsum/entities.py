"""Step 3: medical Named Entity Recognition (NER).

Two methods are combined:
  1. A dictionary (gazetteer) matcher over curated lists of diseases, symptoms,
     drugs and procedures, plus regular expressions for drug dosages.
  2. An optional transformer model (BERT fine-tuned on biomedical NER,
     default "d4data/biomedical-ner-all") that finds terms not in the lists.
"""

import re

from .lexicon import DISEASES, DRUGS, PROCEDURES, SYMPTOMS

CATEGORIES = {
    "Disease": DISEASES,
    "Symptom": SYMPTOMS,
    "Medication": DRUGS,
    "Procedure": PROCEDURES,
}

# Map labels of the transformer model to our categories.
MODEL_LABELS = {
    "Disease_disorder": "Disease",
    "Sign_symptom": "Symptom",
    "Medication": "Medication",
    "Therapeutic_procedure": "Procedure",
}

_DOSE_RE = re.compile(
    r"(?P<dose>\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?|iu))"
    r"(?:\s+(?P<freq>once daily|twice daily|three times daily|four times daily|at bedtime|"
    r"as needed|if needed|weekly|daily|stat))?",
    re.IGNORECASE,
)

_NEGATION_RE = re.compile(r"\b(no|not|denies|denied|without|negative for|ruled out|rule out|absence of)\b[^.;]{0,40}$",
                          re.IGNORECASE)

_ner_pipeline = None
_ner_error = None


def _patterns():
    compiled = []
    for category, terms in CATEGORIES.items():
        for term in terms:
            compiled.append((category, term, re.compile(r"(?<![\w-])%s(?![\w])" % re.escape(term), re.IGNORECASE)))
    compiled.sort(key=lambda item: -len(item[1]))  # prefer longest match
    return compiled


_PATTERNS = _patterns()


def dictionary_entities(text):
    """Find lexicon terms in text. Returns dicts with text, category, position and negation flag."""
    found, taken = [], []
    for category, term, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            entity = {
                "text": m.group(0),
                "norm": term.lower(),
                "category": category,
                "start": m.start(),
                "negated": bool(_NEGATION_RE.search(text[max(0, m.start() - 60):m.start()])),
                "source": "dictionary",
            }
            if category == "Medication":
                dose = _DOSE_RE.match(text[m.end():m.end() + 40].lstrip())
                if dose:
                    entity["dose"] = dose.group("dose")
                    entity["frequency"] = dose.group("freq") or ""
            found.append(entity)
    return found


def _load_model(model_name):
    global _ner_pipeline, _ner_error
    if _ner_pipeline is None and _ner_error is None:
        try:
            from transformers import pipeline

            _ner_pipeline = pipeline("token-classification", model=model_name, aggregation_strategy="simple")
        except Exception as exc:  # model not downloaded, no internet, etc.
            _ner_error = str(exc)
    return _ner_pipeline


def model_entities(text, model_name):
    """Run the transformer NER model; returns [] if it is not available."""
    ner = _load_model(model_name)
    if ner is None:
        return []
    found = []
    # The model accepts ~512 tokens, so run it paragraph by paragraph.
    offset = 0
    for chunk in _chunks(text, 1200):
        for item in ner(chunk):
            category = MODEL_LABELS.get(item["entity_group"])
            word = item["word"].strip()
            if not category or len(word) < 4 or item["score"] < 0.75 or word.startswith("#"):
                continue
            start, end = offset + item["start"], offset + item["end"]
            # Skip word pieces such as "dia" from "diaphoretic".
            if (start > 0 and text[start - 1].isalpha()) or (end < len(text) and text[end].isalpha()):
                continue
            found.append({
                "text": text[start:end],
                "norm": word.lower(),
                "category": category,
                "start": start,
                "negated": bool(_NEGATION_RE.search(text[max(0, start - 60):start])),
                "source": "model",
            })
        offset += len(chunk)
    return found


def _chunks(text, size):
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(". ", start, end)
            end = cut + 2 if cut > start else end
        yield text[start:end]
        start = end


def extract_entities(text, use_model=False, model_name="d4data/biomedical-ner-all"):
    """Combine dictionary and (optional) model entities, removing duplicates."""
    entities = dictionary_entities(text)
    if use_model:
        known = {(e["norm"], e["category"]) for e in entities}
        spans = [(e["start"], e["start"] + len(e["text"])) for e in entities]
        for e in model_entities(text, model_name):
            end = e["start"] + len(e["text"])
            if (e["norm"], e["category"]) in known or any(e["start"] < s_end and end > s for s, s_end in spans):
                continue
            known.add((e["norm"], e["category"]))
            entities.append(e)
    entities.sort(key=lambda e: e["start"])
    return entities


def group_entities(entities):
    """Unique entity names per category, keeping first-seen order and negation."""
    groups = {name: [] for name in CATEGORIES}
    seen = set()
    for e in entities:
        key = (e["category"], e["norm"])
        if key in seen:
            continue
        seen.add(key)
        label = e["text"]
        if e.get("dose"):
            label += " " + e["dose"] + (" " + e["frequency"] if e.get("frequency") else "")
        groups[e["category"]].append({"label": label, "negated": e["negated"], "source": e["source"]})
    return groups


def model_status():
    return _ner_error

"""Runs the full summarization pipeline on one report.

    raw text -> pre-processing -> NER + lab flags -> summarization
             -> fact check -> doctor summary + patient summary
"""

import re
import time

from . import abstractive, extractive, factcheck, simplify
from .entities import dictionary_entities, extract_entities, group_entities
from .labs import abnormal, find_lab_values
from .preprocess import preprocess

METHODS = ("extractive", "abstractive", "hybrid")

# The BART model was trained on news, where the first sentences matter most, so
# the most important sections are given to it first.
SECTION_ORDER = ["Diagnosis", "Impression", "Chief Complaint", "History", "Findings", "Hospital Course",
                 "Procedures", "Treatment", "Investigations", "Examination", "Medications", "Advice",
                 "Follow-up", "Past History", "General"]


_HEADER_RE = re.compile(r"(hospital|laborator|diagnostics|imaging|centre|center|report|summary|synthetic|"
                        r"patient\s*(name)?\s*:|age\s*/?\s*sex|referred by|date of)", re.IGNORECASE)


_DOSE_LINE_RE = re.compile(r"\d+(\.\d+)?\s?(mg|mcg|g|ml|units?|iu)\b", re.IGNORECASE)


def _is_list_line(text):
    return len(text.split()) <= 8 and bool(_DOSE_LINE_RE.search(text) or find_lab_values(text))


def _lower_first(text):
    return text if text[1:2].isupper() else text[:1].lower() + text[1:]


def _diagnosis_sentence(all_sentences):
    """Join short diagnosis (or impression) fragments into one full sentence."""
    for heading, template in (("Diagnosis", "The patient was diagnosed with %s."), ("Impression", "The report shows %s.")):
        parts = [s["text"].rstrip(". ") for s in all_sentences if s["section"] == heading]
        if parts:
            parts = [_lower_first(p) for p in parts]
            joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
            return template % joined
    return ""


def _ordered_for_model(items, all_sentences):
    """Model input: diagnosis first (always included), then other sections by importance, no header lines."""
    items = [s for s in items if not (s["section"] == "General" and _HEADER_RE.search(s["text"]))]
    # Bare lab and medicine lines ("LDL cholesterol 168 mg/dL.", "Aspirin 75 mg once daily.") are
    # shown in their own tables; the model tends to garble units, so it only gets narrative sentences.
    items = [s for s in items if not _is_list_line(s["text"])] or items
    rank = {name: i for i, name in enumerate(SECTION_ORDER)}
    items = sorted(items, key=lambda s: rank.get(s["section"], len(rank)))
    diagnosis = _diagnosis_sentence(all_sentences)
    rest = [s["text"] for s in items if s["section"] not in ("Diagnosis", "Impression")]
    return ([diagnosis] if diagnosis else []) + rest, diagnosis


def _guard_diagnosis(summary, diagnosis, entities):
    """If the model left out most diagnosed conditions, put the diagnosis sentence first."""
    if not diagnosis or diagnosis in summary:
        return summary
    diseases = {e["norm"] for e in dictionary_entities(diagnosis) if e["category"] == "Disease"}
    covered = sum(d in summary.lower() for d in diseases)
    if diseases and covered < len(diseases) / 2:
        return diagnosis + " " + summary
    return summary


def summarize_report(raw_text, method="extractive", ratio=0.3, use_ner_model=False):
    """Summarize a report. method is 'extractive', 'abstractive' or 'hybrid'.

    hybrid = TextRank first selects the key sentences, then the transformer
    rewrites them into a fluent summary (shorter input, fewer off-topic details).
    """
    if method not in METHODS:
        raise ValueError("method must be one of %s" % (METHODS,))
    timings, warnings = {}, []

    t = time.perf_counter()
    prep = preprocess(raw_text)
    timings["preprocess"] = time.perf_counter() - t

    t = time.perf_counter()
    entities = extract_entities(prep["expanded"], use_model=use_ner_model)
    labs = find_lab_values(prep["expanded"])
    flagged = abnormal(labs)
    groups = group_entities(entities)
    timings["ner"] = time.perf_counter() - t

    t = time.perf_counter()
    ext = extractive.summarize(prep["sentences"], entities, labs, ratio=ratio)
    timings["extractive"] = time.perf_counter() - t

    clinical, used = ext["summary"], "extractive"
    if method in ("abstractive", "hybrid"):
        t = time.perf_counter()
        items = (prep["sentences"] if method == "abstractive"
                 else extractive.summarize(prep["sentences"], entities, labs, ratio=0.5, max_sentences=16)["sentences"])
        source, diagnosis = _ordered_for_model(items, prep["sentences"])
        try:
            clinical, used = _guard_diagnosis(abstractive.summarize(source), diagnosis, entities), method
        except abstractive.ModelUnavailable as exc:
            warnings.append("Transformer model unavailable, showing the extractive summary instead. (%s)" % exc)
        timings["abstractive"] = time.perf_counter() - t

    t = time.perf_counter()
    model_input = " ".join(source) if used != "extractive" else ""
    facts = factcheck.check(prep["expanded"] + " " + model_input, clinical, entities, flagged)
    patient = simplify.patient_summary(clinical, groups, flagged)
    timings["postprocess"] = time.perf_counter() - t

    source_words = len(prep["clean"].split())
    summary_words = len(clinical.split())
    return {
        "method": used,
        "clinical_summary": clinical,
        "extractive_sentences": ext["sentences"],
        "patient": patient,
        "entities": groups,
        "labs": labs,
        "abnormal_labs": flagged,
        "factcheck": facts,
        "sections": [name for name, _ in prep["sections"]],
        "stats": {
            "source_words": source_words,
            "summary_words": summary_words,
            "compression": round(100.0 * (1 - summary_words / source_words), 1) if source_words else 0.0,
            "readability_source": simplify.flesch_reading_ease(prep["clean"]),
            "readability_clinical": simplify.flesch_reading_ease(clinical),
            "readability_patient": simplify.flesch_reading_ease(patient["text"]),
            "seconds": round(sum(timings.values()), 2),
        },
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "warnings": warnings,
        "model": abstractive.model_name() if used != "extractive" else "TextRank",
    }

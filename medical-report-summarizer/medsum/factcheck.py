"""Step 6: check that the summary is faithful to the source report.

Two checks:
  * Hallucination check: every number and medical entity in the summary must
    also appear in the source report. Anything else is flagged as unsupported.
  * Coverage check: what share of the important source entities (diagnoses,
    medicines, abnormal labs) the summary mentions.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .entities import dictionary_entities

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text):
    return set(_NUM_RE.findall(text.replace(",", "")))


def check(source_text, summary_text, source_entities, abnormal_results):
    source_lower = source_text.lower()
    summary_lower = summary_text.lower()

    unsupported_numbers = sorted(_numbers(summary_text) - _numbers(source_text), key=float)
    summary_entities = dictionary_entities(summary_text)
    unsupported_entities = sorted({e["norm"] for e in summary_entities if e["norm"] not in source_lower})

    key_entities = {e["norm"] for e in source_entities
                    if not e["negated"] and e["category"] in ("Disease", "Medication", "Procedure")}
    covered = {term for term in key_entities if term in summary_lower}
    missing = sorted(key_entities - covered)

    lab_total = len(abnormal_results)
    lab_covered = sum(1 for r in abnormal_results if _lab_mentioned(r, summary_lower))

    total = len(key_entities) + lab_total
    coverage = round(100.0 * (len(covered) + lab_covered) / total, 1) if total else 100.0
    # Content words the model introduced that are not in the report (may be a typo or a made-up detail).
    source_words = set(re.findall(r"[a-z]+", source_lower))
    new_words = sorted({w for w in re.findall(r"[a-z]+", summary_lower)
                        if len(w) > 2 and w not in source_words and w not in ENGLISH_STOP_WORDS})

    return {
        "faithful": not unsupported_numbers and not unsupported_entities,
        "new_words": new_words,
        "unsupported_numbers": unsupported_numbers,
        "unsupported_entities": unsupported_entities,
        "coverage": coverage,
        "covered": sorted(covered),
        "missing": missing,
        "labs_covered": lab_covered,
        "labs_total": lab_total,
    }


def _lab_mentioned(result, summary_lower):
    value = str(result["value"])
    if value.endswith(".0"):
        value = value[:-2]
    return result["test"].split(" ")[0].lower() in summary_lower or value in summary_lower

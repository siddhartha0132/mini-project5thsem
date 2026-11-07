"""Step 2: clean the text, expand abbreviations, split sections and sentences."""

import re

from .lexicon import ABBREVIATIONS, UPPERCASE_ONLY

# Section headings commonly found in discharge summaries and reports.
SECTION_NAMES = {
    "chief complaint": "Chief Complaint",
    "presenting complaint": "Chief Complaint",
    "complaints": "Chief Complaint",
    "history of present illness": "History",
    "history": "History",
    "past medical history": "Past History",
    "past history": "Past History",
    "clinical history": "History",
    "examination": "Examination",
    "on examination": "Examination",
    "physical examination": "Examination",
    "vitals": "Examination",
    "investigations": "Investigations",
    "lab results": "Investigations",
    "laboratory results": "Investigations",
    "results": "Investigations",
    "findings": "Findings",
    "observations": "Findings",
    "impression": "Impression",
    "conclusion": "Impression",
    "diagnosis": "Diagnosis",
    "final diagnosis": "Diagnosis",
    "diagnoses": "Diagnosis",
    "hospital course": "Hospital Course",
    "course in hospital": "Hospital Course",
    "treatment given": "Treatment",
    "treatment": "Treatment",
    "procedure": "Procedures",
    "procedures": "Procedures",
    "medications": "Medications",
    "discharge medications": "Medications",
    "medication on discharge": "Medications",
    "advice": "Advice",
    "advice on discharge": "Advice",
    "follow up": "Follow-up",
    "follow-up": "Follow-up",
    "recommendation": "Advice",
    "recommendations": "Advice",
}

_HEADING_RE = re.compile(
    r"^\s*(%s)\s*[:\-]\s*" % "|".join(sorted((re.escape(k) for k in SECTION_NAMES), key=len, reverse=True)),
    re.IGNORECASE,
)

# Lines that are boilerplate in printed reports.
_NOISE_RE = re.compile(
    r"^\s*(page \d+( of \d+)?|printed on.*|this is a computer generated report.*|"
    r"signature.*|dr\.?\s.*mbbs.*|-{3,}|={3,}|\*{3,})\s*$",
    re.IGNORECASE,
)

# Words that end with a period but do not end a sentence.
_NO_SPLIT = ("dr", "mr", "mrs", "ms", "vs", "no", "approx", "e.g", "i.e", "etc", "st", "fig")


def clean_text(text):
    """Remove noise lines, odd characters and repeated whitespace."""
    text = text.replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[•●▪‣]", "-", text)  # bullet symbols
    text = re.sub(r"[^\x09-\x7e\n°µ±≤≥]", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line and not _NOISE_RE.match(line)]
    text = "\n".join(lines)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def expand_abbreviations(text):
    """Replace clinical abbreviations with their full forms (e.g. HTN -> hypertension)."""

    def replace(match):
        word = match.group(0)
        key = word.lower()
        if key in UPPERCASE_ONLY and not word.isupper():
            return word
        return ABBREVIATIONS[key]

    keys = sorted(ABBREVIATIONS, key=len, reverse=True)
    pattern = r"(?<![\w/])(%s)(?![\w/])" % "|".join(re.escape(k) for k in keys)
    return re.sub(pattern, replace, text, flags=re.IGNORECASE)


def _join_lines(lines):
    """Join lines into a paragraph, ending list-style lines with a full stop.

    A line without closing punctuation that is followed by a line starting with
    a capital letter, digit or dash is treated as a separate item (e.g. one lab
    result per line); otherwise it is a wrapped line and is joined with a space.
    """
    parts = []
    for i, line in enumerate(lines):
        line = line.lstrip("- ").strip()
        if not line:
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if line[-1] not in ".!?;:," and (not nxt or re.match(r"[A-Z0-9\-(]", nxt)):
            line += "."
        parts.append(line)
    return " ".join(parts)


def split_sections(text):
    """Split a report into an ordered list of (section name, text) pairs."""
    sections = []
    current, buffer = "General", []
    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            if buffer:
                sections.append((current, _join_lines(buffer)))
            current = SECTION_NAMES[match.group(1).lower()]
            rest = line[match.end():].strip()
            buffer = [rest] if rest else []
        else:
            buffer.append(line)
    if buffer:
        sections.append((current, _join_lines(buffer)))
    return [(name, body) for name, body in sections if body]


def split_sentences(text):
    """Rule-based sentence splitter that keeps numbers and titles intact."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9(])", text)
    sentences = []
    for piece in pieces:
        if sentences and sentences[-1].lower().rstrip(".").split(" ")[-1] in _NO_SPLIT:
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    # Very long lines without punctuation (lists) are split on " - ".
    result = []
    for sentence in sentences:
        if len(sentence) > 300 and " - " in sentence:
            result.extend(part.strip() for part in sentence.split(" - ") if part.strip())
        else:
            result.append(sentence)
    return [s for s in result if len(s.split()) >= 2]


def preprocess(raw_text):
    """Run the full pre-processing step and return a dict used by later stages."""
    cleaned = clean_text(raw_text)
    expanded = expand_abbreviations(cleaned)
    sections = split_sections(expanded)
    sentences = []
    for name, body in sections:
        for sentence in split_sentences(body):
            sentences.append({"text": sentence, "section": name})
    return {"clean": cleaned, "expanded": expanded, "sections": sections, "sentences": sentences}

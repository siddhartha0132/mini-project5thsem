# AI-Based Medical Report Summarization Using NLP

A web application that reads a medical report (discharge summary, lab report or scan report) and produces:

- a **clinical summary** for doctors, with abnormal values highlighted
- a **plain-language summary** for patients, with each abnormal result explained
- the **medical entities** in the report (diseases, symptoms, medicines with doses, procedures)
- a **fact check** that confirms every number and medical term in the summary exists in the original report

Input can be typed text, a PDF, or a scanned image (OCR).

## Pipeline

```
Upload ─► Text extraction ─► Pre-processing ─► Medical NER + lab flags ─► Summarization ─► Fact check ─► Doctor summary
          (PyMuPDF / OCR)    (clean, expand      (dictionary + BERT,       (TextRank / BART /               └► Patient summary
                              abbreviations,      negation detection,       hybrid)                            (simplification)
                              sections)           reference ranges)
```

| Step | File | Technique |
| --- | --- | --- |
| 1. Text extraction | `medsum/extract.py` | PyMuPDF for PDFs; Tesseract OCR for scans and image-only PDFs, after grayscale, upscaling, contrast and thresholding |
| 2. Pre-processing | `medsum/preprocess.py` | Noise removal, 80+ clinical abbreviations expanded (HTN → hypertension), section detection, rule-based sentence splitting |
| 3. Named Entity Recognition | `medsum/entities.py` | Medical dictionary matcher with longest-match and NegEx-style negation ("no history of asthma"), dose and frequency extraction, plus an optional BERT model (`d4data/biomedical-ner-all`) |
| 3b. Lab analysis | `medsum/labs.py` | Regex extraction of 30 lab tests and vitals, unit normalisation (lakh, ×10³, °C → °F), comparison with reference ranges |
| 4a. Extractive summary | `medsum/extractive.py` | TextRank: TF-IDF vectors → cosine-similarity graph → PageRank, boosted by section weight, entity density and abnormal labs; redundancy filter |
| 4b. Abstractive summary | `medsum/abstractive.py` | BART encoder-decoder (`sshleifer/distilbart-cnn-12-6`), long reports chunked, beam search with a repetition penalty |
| 4c. Hybrid | `medsum/pipeline.py` | TextRank selects the key sentences, then BART rewrites them |
| 5. Patient summary | `medsum/simplify.py` | Glossary-based simplification (100+ terms), third-person to second-person rewriting, Flesch Reading Ease score |
| 6. Fact check | `medsum/factcheck.py` | Unsupported numbers and entities (hallucination check) and key-fact coverage |
| Storage | `medsum/db.py` | SQLite history of reports and summaries |
| Web app | `app.py`, `templates/`, `static/` | Flask: upload, results, history, download, JSON API |

### TextRank formula

```
WS(Vi) = (1 − d) + d · Σ_{Vj ∈ In(Vi)} [ w_ji / Σ_{Vk ∈ Out(Vj)} w_jk ] · WS(Vj)        d = 0.85
```

## Setup

Requires Python 3.9 or newer.

```bash
cd medical-report-summarizer
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000. On macOS or Linux, `./run.sh` does all of this in one step.

- The first **Hybrid** or **Abstractive** run downloads the BART model (about 1.2 GB) and is slow; later runs are fast.
- For scanned images, install the Tesseract program: `brew install tesseract` on macOS, `sudo apt install tesseract-ocr` on Ubuntu, or the Windows installer from the Tesseract GitHub page.
- Extractive mode works without PyTorch or the model download.

## Using the app

1. Upload a report, paste its text, or click one of the five sample reports.
2. Choose a method: **Extractive**, **Hybrid** (recommended) or **Abstractive**, and a summary length.
3. The results page shows both summaries, the abnormal values, entities, all lab values, the fact check and the TextRank score of each sentence.
4. Download the summary as `.txt` or print it. Past reports are listed under **History**.

JSON API:

```bash
curl -X POST http://127.0.0.1:5000/api/summarize \
     -H "Content-Type: application/json" \
     -d '{"text": "Diagnosis: Pneumonia. Hemoglobin 9.2 g/dL ...", "method": "extractive"}'
```

## Evaluation

```bash
python scripts/evaluate.py                 # lead-5 baseline vs TextRank
python scripts/evaluate.py --abstractive   # adds hybrid and abstractive (BART)
python -m unittest discover tests          # 20 unit tests
```

The script compares each method's summary with the hand-written reference summaries in `data/references.json`
using ROUGE-1, ROUGE-2 and ROUGE-L (optionally BERTScore with `--bertscore`). It also reports fact coverage,
faithfulness, length, readability and time. Results are written to `results/evaluation.md`.

### Results (5 sample reports, Apple M5, distilbart-cnn-12-6 without fine-tuning)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Faithful | Avg words | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Lead baseline (opening sentences) | 0.320 | 0.157 | 0.219 | – | 77 | 0.00 |
| **Extractive (TextRank)** | **0.525** | **0.302** | **0.376** | 5/5 | 76 | 0.02 |
| Hybrid (TextRank + BART) | 0.429 | 0.192 | 0.273 | 5/5 | 44 | 3.0 |
| Abstractive (BART) | 0.392 | 0.155 | 0.266 | 5/5 | 51 | 2.0 |

- All methods beat the baseline. TextRank with medical boosting scores highest, partly because the reference
  summaries reuse much of the reports' wording.
- The hybrid and abstractive summaries are shorter (44 to 51 words) and more fluent. The BART model was
  trained on news articles, not medical text, so it sometimes changes words (for example "ureter" to "ureters",
  or "hydronephrosis" to "hydronphrosis"). The fact checker flags these as new words, which shows why the check
  is needed. Fine-tuning on clinical data (`scripts/finetune.py`) is the expected fix.
- Lab extraction found 58 lab values and vitals across the samples, and flagged 41 as abnormal.
- The patient summaries are easier to read: the average Flesch Reading Ease score rose from 40.2 (clinical
  summary, "difficult") to 59.0 (patient summary, "fairly easy" / plain English).

## Fine-tuning on medical data

The default BART model was trained on news articles. To adapt it to clinical text, prepare a CSV with `report`
and `summary` columns (for example from MIMIC-III discharge summaries, which require PhysioNet credentialing, or
from MTSamples) and run:

```bash
python scripts/finetune.py --data data/train.csv --epochs 3
```

The best checkpoint is saved to `models/medsum-bart`, and the app uses it automatically. A GPU or Google Colab is
recommended.

## Project structure

```
medical-report-summarizer/
├── app.py                  Flask web application
├── medsum/                 NLP pipeline package
│   ├── extract.py          PDF / image / text extraction (OCR)
│   ├── preprocess.py       cleaning, abbreviations, sections, sentences
│   ├── lexicon.py          medical vocabularies and plain-language glossary
│   ├── entities.py         medical NER (dictionary + BERT)
│   ├── labs.py             lab values and reference ranges
│   ├── extractive.py       TextRank summarizer
│   ├── abstractive.py      BART summarizer
│   ├── simplify.py         patient-friendly summary and readability
│   ├── factcheck.py        hallucination and coverage check
│   ├── pipeline.py         runs all steps
│   └── db.py               SQLite storage
├── templates/, static/     web pages and styles
├── data/samples/           5 synthetic sample reports
├── data/references.json    reference summaries for evaluation
├── scripts/evaluate.py     ROUGE evaluation
├── scripts/finetune.py     fine-tuning script
├── tests/                  unit tests
└── requirements.txt
```

## Limitations

- The sample reports are synthetic. No real patient data is included.
- The reference ranges are typical adult values. Real ranges vary by laboratory, age and sex.
- The pre-trained BART model was not trained on medical text, so fine-tuning would improve the abstractive summaries.
- This is an educational project, not a medical device. Summaries must be checked by a qualified clinician.

## References

1. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
2. Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension," ACL 2020.
3. Mihalcea and Tarau, "TextRank: Bringing Order into Texts," EMNLP 2004.
4. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," ACL Workshop 2004.
5. Lee et al., "BioBERT: a pre-trained biomedical language representation model for biomedical text mining," Bioinformatics 2020.
6. Chapman et al., "A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries" (NegEx), Journal of Biomedical Informatics 2001.
7. Johnson et al., "MIMIC-III, a freely accessible critical care database," Scientific Data 2016.
8. Hugging Face Transformers: https://huggingface.co/docs/transformers

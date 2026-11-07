"""Evaluate the summarization methods with ROUGE (and BERTScore if installed).

Usage:
    python scripts/evaluate.py                    # extractive + baselines
    python scripts/evaluate.py --abstractive      # also hybrid and abstractive (needs the BART model)

Writes a results table to results/evaluation.md and results/evaluation.json.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rouge_score import rouge_scorer  # noqa: E402

from medsum.pipeline import summarize_report  # noqa: E402
from medsum.preprocess import preprocess  # noqa: E402
from medsum.simplify import flesch_reading_ease  # noqa: E402


def lead_baseline(text, words=75):
    """Baseline: the opening sentences of the report, up to about the same length as the TextRank summary."""
    picked, count = [], 0
    for sentence in preprocess(text)["sentences"]:
        if count >= words:
            break
        picked.append(sentence["text"])
        count += len(sentence["text"].split())
    return " ".join(picked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abstractive", action="store_true", help="also evaluate hybrid and abstractive methods")
    parser.add_argument("--bertscore", action="store_true", help="also compute BERTScore (pip install bert-score)")
    args = parser.parse_args()

    with open(os.path.join(ROOT, "data", "references.json"), encoding="utf-8") as f:
        references = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    methods = ["lead baseline", "extractive"] + (["hybrid", "abstractive"] if args.abstractive else [])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = {m: [] for m in methods}
    outputs = {}

    for name, reference in references.items():
        with open(os.path.join(ROOT, "data", "samples", name), encoding="utf-8") as f:
            text = f.read()
        for method in methods:
            start = time.perf_counter()
            if method == "lead baseline":
                summary, facts = lead_baseline(text), None
            else:
                result = summarize_report(text, method=method)
                summary, facts = result["clinical_summary"], result["factcheck"]
            seconds = time.perf_counter() - start
            scores = scorer.score(reference, summary)
            rows[method].append({
                "report": name,
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
                "coverage": facts["coverage"] if facts else None,
                "faithful": facts["faithful"] if facts else None,
                "words": len(summary.split()),
                "readability": flesch_reading_ease(summary),
                "seconds": seconds,
            })
            outputs.setdefault(name, {})[method] = summary
            print("%-32s %-16s R1=%.3f R2=%.3f RL=%.3f" % (name, method, scores["rouge1"].fmeasure,
                                                          scores["rouge2"].fmeasure, scores["rougeL"].fmeasure))

    if args.bertscore:
        from bert_score import score as bert_score

        for method in methods:
            cands = [outputs[r["report"]][method] for r in rows[method]]
            refs = [references[r["report"]] for r in rows[method]]
            _, _, f1 = bert_score(cands, refs, lang="en", verbose=False)
            for r, value in zip(rows[method], f1.tolist()):
                r["bertscore"] = value

    def avg(method, key):
        values = [r[key] for r in rows[method] if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    lines = ["# Evaluation results", "",
             "Average over %d reports. ROUGE values are F1 scores (0 to 1, higher is better)." % len(references), "",
             "| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |%s Fact coverage | Faithful | Avg words | Readability | Time (s) |"
             % (" BERTScore |" if args.bertscore else ""),
             "| --- | --- | --- | --- |%s --- | --- | --- | --- | --- |" % (" --- |" if args.bertscore else "")]
    for m in methods:
        faithful = [r["faithful"] for r in rows[m] if r["faithful"] is not None]
        lines.append("| %s | %.3f | %.3f | %.3f |%s %s | %s | %.0f | %.1f | %.2f |" % (
            m, avg(m, "rouge1"), avg(m, "rouge2"), avg(m, "rougeL"),
            (" %.3f |" % avg(m, "bertscore")) if args.bertscore else "",
            ("%.1f%%" % avg(m, "coverage")) if avg(m, "coverage") is not None else "-",
            ("%d/%d" % (sum(faithful), len(faithful))) if faithful else "-",
            avg(m, "words"), avg(m, "readability"), avg(m, "seconds")))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "evaluation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(ROOT, "results", "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump({"scores": rows, "summaries": outputs}, f, indent=2)
    print("\n" + "\n".join(lines))
    print("\nSaved results/evaluation.md and results/evaluation.json")


if __name__ == "__main__":
    main()

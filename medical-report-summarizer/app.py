"""Flask web application for AI-based medical report summarization.

Run:  python app.py   then open http://127.0.0.1:5000
"""

import os

from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, url_for

from medsum import abstractive
from medsum.extract import ALLOWED_EXT, ExtractionError, extract_text
from medsum.db import ReportStore
from medsum.pipeline import METHODS, summarize_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("MEDSUM_SECRET", "medsum-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads
store = ReportStore(os.path.join(BASE_DIR, "medsum.db"))


def sample_names():
    return sorted(f for f in os.listdir(SAMPLES_DIR) if f.endswith(".txt"))


def sample_title(name):
    return name[3:-4].replace("_", " ").capitalize() if name[:2].isdigit() else name[:-4]


@app.route("/")
def index():
    samples = [(name, sample_title(name)) for name in sample_names()]
    return render_template("index.html", samples=samples, methods=METHODS,
                           allowed=", ".join(sorted(ALLOWED_EXT)))


@app.route("/sample/<name>")
def sample(name):
    if name not in sample_names():
        abort(404)
    with open(os.path.join(SAMPLES_DIR, name), encoding="utf-8") as f:
        return jsonify({"name": name, "title": sample_title(name), "text": f.read()})


@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form.get("text", "").strip()
    title = request.form.get("title", "").strip()
    upload = request.files.get("file")
    try:
        if upload and upload.filename:
            text = extract_text(upload.filename, upload.read())
            title = title or upload.filename
    except ExtractionError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))

    if len(text.split()) < 15:
        flash("Please upload a report or paste at least a few sentences of report text.", "error")
        return redirect(url_for("index"))

    method = request.form.get("method", "extractive")
    ratio = float(request.form.get("ratio", 0.3))
    use_ner_model = request.form.get("ner_model") == "on"
    result = summarize_report(text, method=method, ratio=ratio, use_ner_model=use_ner_model)
    report_id = store.save(title or "Pasted report", result["method"], text, result)
    return redirect(url_for("report", report_id=report_id))


@app.route("/report/<int:report_id>")
def report(report_id):
    item = store.get(report_id)
    if item is None:
        abort(404)
    return render_template("result.html", report=item, r=item["result"])


@app.route("/report/<int:report_id>/download")
def download(report_id):
    item = store.get(report_id)
    if item is None:
        abort(404)
    r = item["result"]
    lines = ["MEDICAL REPORT SUMMARY", "Report: %s" % item["title"], "Method: %s" % r["method"], "",
             "CLINICAL SUMMARY (for doctors)", r["clinical_summary"], ""]
    if r["abnormal_labs"]:
        lines.append("ABNORMAL VALUES")
        lines += ["- %s: %s %s (%s)" % (l["test"], l["value"], l["unit"], l["status"]) for l in r["abnormal_labs"]]
        lines.append("")
    lines += ["PATIENT SUMMARY (plain language)", r["patient"]["text"], ""]
    lines += ["- " + note for note in r["patient"]["lab_notes"]]
    lines += ["", "Generated automatically by MedSum. Always confirm with your doctor."]
    filename = "summary_%d.txt" % report_id
    return Response("\n".join(lines), mimetype="text/plain",
                    headers={"Content-Disposition": "attachment; filename=%s" % filename})


@app.route("/report/<int:report_id>/delete", methods=["POST"])
def delete(report_id):
    store.delete(report_id)
    flash("Report deleted.", "info")
    return redirect(url_for("history"))


@app.route("/history")
def history():
    return render_template("history.html", items=store.recent())


@app.route("/about")
def about():
    return render_template("about.html", model=abstractive.model_name())


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """JSON API: {"text": "...", "method": "extractive|abstractive|hybrid", "ratio": 0.3}"""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    method = data.get("method", "extractive")
    if method not in METHODS:
        return jsonify({"error": "method must be one of %s" % ", ".join(METHODS)}), 400
    return jsonify(summarize_report(text, method=method, ratio=float(data.get("ratio", 0.3))))


@app.errorhandler(413)
def too_large(_):
    flash("File is too large (limit 10 MB).", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))

"""SQLite storage for uploaded reports and their summaries."""

import json
import sqlite3
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    title       TEXT NOT NULL,
    method      TEXT NOT NULL,
    source_text TEXT NOT NULL,
    result_json TEXT NOT NULL
);
"""


class ReportStore:
    def __init__(self, path):
        self.path = path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, title, method, source_text, result):
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO reports (created_at, title, method, source_text, result_json) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), title, method, source_text, json.dumps(result)),
            )
            return cur.lastrowid

    def get(self, report_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            return None
        report = dict(row)
        report["result"] = json.loads(report.pop("result_json"))
        return report

    def recent(self, limit=50):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, title, method, result_json FROM reports ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        items = []
        for row in rows:
            result = json.loads(row["result_json"])
            items.append({
                "id": row["id"], "created_at": row["created_at"], "title": row["title"], "method": row["method"],
                "abnormal": len(result.get("abnormal_labs", [])),
                "compression": result.get("stats", {}).get("compression"),
            })
        return items

    def delete(self, report_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))

"""Flask routes for the Manufacturing Process Recommender."""
import csv
import os
import threading
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd
from flask import Flask, render_template, request

from config import *
from rules import validate

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-change-me")

_lock = threading.Lock()
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

SESSION_HEADER = (["session_id", "timestamp_utc"] + FEATURE_COLUMNS
                  + [f"top{i}{s}" for i in (1, 2, 3) for s in ("", "_score")])
FEEDBACK_HEADER = ["session_id", "timestamp_utc", "useful"]


def append_row(path, header, row):
    with _lock:
        path.parent.mkdir(exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow(row)


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_form(form):
    """Return (values, raw, error). Numeric fields become int."""
    values, raw = {}, {}
    for name in FEATURE_COLUMNS:
        raw[name] = form.get(name, "").strip()
        v = raw[name]
        if name != "material_type":
            try:
                v = int(v)
            except ValueError:
                return values, raw, f"Please choose a valid answer for: {FIELDS[name][0]}."
        values[name] = v
    return values, raw, validate(values)


def render_form(raw=None, error=None):
    status = 400 if error else 200
    return render_template("index.html", fields=FIELDS, steps=STEPS, answers=raw or {},
                           error=error), status


@app.route("/")
def index():
    if MODEL is None:
        return render_template("missing_model.html"), 503
    return render_form()


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return render_template("missing_model.html"), 503
    values, raw, error = parse_form(request.form)
    if error:
        return render_form(raw, error)

    df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    scores = MODEL.predict_proba(df)[0]
    top = sorted(zip(MODEL.classes_, scores), key=lambda t: t[1], reverse=True)[:3]
    results = [{"name": CLASSES[c][0], "description": CLASSES[c][1],
                "score": round(float(s) * 100, 1)} for c, s in top]

    session_id = uuid.uuid4().hex
    log = [session_id, now()] + [values[c] for c in FEATURE_COLUMNS]
    for c, s in top:
        log += [c, round(float(s) * 100, 1)]
    append_row(SESSIONS_LOG, SESSION_HEADER, log)

    readable = [(FIELDS[c][0], FIELDS[c][2][values[c]]) for c in FEATURE_COLUMNS]
    return render_template("result.html", top=results[0], others=results[1:],
                           answers=readable, session_id=session_id)


@app.route("/feedback", methods=["POST"])
def feedback():
    useful = request.form.get("useful", "")
    session_id = request.form.get("session_id", "")
    if useful not in ("yes", "no") or not session_id.isalnum():
        return "Bad request", 400
    append_row(FEEDBACK_LOG, FEEDBACK_HEADER, [session_id, now(), useful])
    return render_template("feedback_thanks.html")


if __name__ == "__main__":
    app.run()

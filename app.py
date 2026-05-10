import os
import uuid
import json
import sqlite3
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file
from transformers import pipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = os.path.join("static", "plots")
DB_PATH = "analysis.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# -----------------------------
# DATABASE SETUP
# -----------------------------


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            filename TEXT,
            db REAL,
            severity TEXT,
            source TEXT,
            features TEXT,
            stats TEXT,
            report TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# -----------------------------
# MODEL
# -----------------------------
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
sound_classifier = pipeline("audio-classification", model=MODEL_ID, top_k=10)

CONF_THRESHOLD = 0.40




# -----------------------------
# HELPERS
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"wav", "mp3"}


def preprocess_audio(y):
    return librosa.util.normalize(y)



# -----------------------------
# FEATURES
# -----------------------------
def extract_features(y, sr):
    rms = np.sqrt(np.mean(y ** 2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return {
        "rms": float(rms),
        "zcr": float(zcr),
        "spectral_centroid": float(centroid),
        "mfcc_mean": np.mean(mfcc, axis=1).tolist()
    }


# -----------------------------
# NOISE LEVEL
# -----------------------------
def compute_noise_level(y):
    rms = np.sqrt(np.mean(y ** 2))
    dbfs = 20 * np.log10(rms + 1e-10)
    return float(np.clip(80 + (dbfs * 1.8), 0, 120))


def classify_noise(db):
    if db < 50:
        return "Safe"
    elif db <= 70:
        return "Moderate"
    return "Harmful"


# -----------------------------
# TIME SERIES (NEW SPEC FEATURE)
# -----------------------------
def compute_db_timeseries(y):
    rms = librosa.feature.rms(y=y)[0]
    db_series = librosa.amplitude_to_db(rms, ref=np.max)
    return db_series.tolist()


def plot_timeseries(db_series, uid):
    path = os.path.join(PLOT_FOLDER, f"{uid}_timeseries.png")

    plt.figure(figsize=(10, 3))
    plt.plot(db_series)
    plt.title("Noise Level Over Time (dB)")
    plt.xlabel("Time Frame")
    plt.ylabel("dB")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


# -----------------------------
# SOURCE
# -----------------------------
def get_source(preds):
    return preds[0]["label"] if preds else "unknown"


def filter_predictions(preds):
    return [
        {"label": p["label"], "score": round(p["score"] * 100, 2)}
        for p in preds if p["score"] >= CONF_THRESHOLD
    ]


# -----------------------------
# VISUALS
# -----------------------------
def save_plots(y, sr, uid):
    wav = os.path.join(PLOT_FOLDER, f"{uid}_waveform.png")
    spec = os.path.join(PLOT_FOLDER, f"{uid}_spectrogram.png")

    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.tight_layout()
    plt.savefig(wav)
    plt.close()

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(spec)
    plt.close()

    return wav, spec


# -----------------------------
# SAVE TO DB
# -----------------------------
def save_to_db(uid, filename, db, severity, source, features, stats, report):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO analyses VALUES (?,?,?,?,?,?,?, ?,?)
    """, (
        uid, filename, db, severity, source,
        json.dumps(features),
        json.dumps(stats),
        json.dumps(report),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files.get("audio_file")
        if not file or file.filename == "":
            return redirect(request.url)

        if not allowed_file(file.filename):
            return redirect(request.url)

        uid = uuid.uuid4().hex
        filename = file.filename
        path = os.path.join(UPLOAD_FOLDER, f"{uid}.wav")
        file.save(path)

        y, sr = librosa.load(path, sr=44100, mono=True)
        y = preprocess_audio(y)

        features = extract_features(y, sr)
        db = compute_noise_level(y)
        severity = classify_noise(db)
        stats = {
            "avg_db": float(np.mean(compute_db_timeseries(y))),
            "max_db": float(np.max(compute_db_timeseries(y))),
            "min_db": float(np.min(compute_db_timeseries(y)))
        }

        preds = sound_classifier(path)
        source = get_source(preds)
        predictions = filter_predictions(preds)

        wav, spec = save_plots(y, sr, uid)
        ts_path = plot_timeseries(compute_db_timeseries(y), uid)

        report = {
            "db": db,
            "severity": severity,
            "source": source,
            "stats": stats,
            "predictions": predictions
        }

        save_to_db(uid, filename, db, severity, source, features, stats, report)

        return render_template(
            "result.html",
            id=uid,
            db_value=round(db, 2),
            severity=severity,
            source=source,
            features=features,
            stats=stats,
            predictions=predictions,
            report=json.dumps(report, indent=2),
            waveform_img=url_for("static", filename=f"plots/{uid}_waveform.png"),
            spectrogram_img=url_for("static", filename=f"plots/{uid}_spectrogram.png"),
            timeseries_img=url_for("static", filename=f"plots/{uid}_timeseries.png")
        )

    return render_template("index.html")


# -----------------------------
# HISTORY
# -----------------------------
@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM analyses ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()

    return render_template("history.html", rows=rows)


# -----------------------------
# DOWNLOAD REPORT
# -----------------------------
@app.route("/download/<uid>")
def download(uid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT report FROM analyses WHERE id=?", (uid,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "Not found"

    path = f"{uid}_report.json"
    with open(path, "w") as f:
        f.write(row[0])

    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
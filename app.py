"""Flask web app: upload an image or video, get back annotated detections."""
import os
import secrets
import uuid
from pathlib import Path

import cv2
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from detector import WeaponDetector, annotate, scan_video

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(16)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB upload limit

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_detector = None
_detector_error = None


def get_detector():
    """Load the model once, on first use, and remember a missing-weights error for the UI."""
    global _detector, _detector_error
    if _detector is None and _detector_error is None:
        try:
            _detector = WeaponDetector()
        except FileNotFoundError as exc:
            _detector_error = str(exc)
    return _detector


@app.route("/", methods=["GET"])
def index():
    get_detector()
    return render_template("index.html", model_error=_detector_error)


@app.route("/detect", methods=["POST"])
def detect():
    detector = get_detector()
    if detector is None:
        flash(_detector_error)
        return redirect(url_for("index"))

    file = request.files.get("file")
    if not file or not file.filename:
        flash("Please choose an image or video to upload.")
        return redirect(url_for("index"))

    ext = Path(secure_filename(file.filename)).suffix.lower()
    if ext not in IMAGE_EXT | VIDEO_EXT:
        flash(f"Unsupported file type '{ext or '?'}'. Use an image ({', '.join(sorted(IMAGE_EXT))}) "
              f"or a video ({', '.join(sorted(VIDEO_EXT))}).")
        return redirect(url_for("index"))

    # Never trust the uploaded name for the path on disk
    job = uuid.uuid4().hex[:12]
    upload_path = UPLOAD_DIR / f"{job}{ext}"
    file.save(upload_path)

    try:
        if ext in IMAGE_EXT:
            frame = cv2.imread(str(upload_path))
            if frame is None:
                raise ValueError("That image could not be read.")
            detections = detector.detect(frame)
            out_name = f"{job}.jpg"
            cv2.imwrite(str(RESULT_DIR / out_name), annotate(frame, detections))
            results = [{"time": None, "image": out_name, "detections": detections}]
            frames_checked = 1
        else:
            hits, frames_checked, _fps = scan_video(detector, str(upload_path))
            results = []
            for n, (t, img, dets) in enumerate(hits):
                out_name = f"{job}_{n}.jpg"
                cv2.imwrite(str(RESULT_DIR / out_name), img)
                results.append({"time": t, "image": out_name, "detections": dets})
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("index"))
    finally:
        upload_path.unlink(missing_ok=True)  # don't keep user uploads around

    weapon_found = any(r["detections"] for r in results)
    return render_template("result.html", results=results, weapon_found=weapon_found,
                           frames_checked=frames_checked, is_video=ext in VIDEO_EXT,
                           filename=secure_filename(file.filename))


@app.errorhandler(413)
def too_large(_err):
    flash("That file is larger than the 100 MB limit.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")

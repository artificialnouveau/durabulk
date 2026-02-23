import os
import uuid
import shutil
import threading
import tempfile
import zipfile
import io
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
import instaloader
from ultralytics import YOLO
import easyocr

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
DURA_DIR = DOWNLOADS_DIR / "dura_bulk"
NON_DURA_DIR = DOWNLOADS_DIR / "non_dura_bulk"

# Ensure output dirs exist
DURA_DIR.mkdir(parents=True, exist_ok=True)
NON_DURA_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store
jobs = {}

# Lazy-loaded models
_yolo_model = None
_ocr_reader = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def fuzzy_match_dura_bulk(text):
    """Check if text contains something close to 'dura bulk'."""
    text = text.lower().strip()
    # Direct substring check
    if "dura" in text and "bulk" in text:
        return True
    # Check with spaces/punctuation removed
    cleaned = re.sub(r"[^a-z0-9]", "", text)
    if "durabulk" in cleaned:
        return True
    return False


def run_pipeline(job_id, name, start_date, end_date, max_posts=100, is_hashtag=False, ig_username="", ig_password=""):
    """Background pipeline: scrape profile/hashtag → detect boats → OCR → sort."""
    job = jobs[job_id]

    try:
        # --- Step 1: Scrape ---
        job["step"] = "scraping"
        label = f"#{name}" if is_hashtag else f"@{name}"
        job["detail"] = f"Fetching posts from {label}..."

        tmp_dir = tempfile.mkdtemp(prefix="dura_bulk_")
        L = instaloader.Instaloader(
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            post_metadata_txt_pattern="",
        )

        # Login if credentials provided (required for hashtag scraping)
        if ig_username and ig_password:
            try:
                job["detail"] = f"Logging in as @{ig_username}..."
                L.login(ig_username, ig_password)
                job["detail"] = f"Logged in. Fetching posts from {label}..."
            except Exception as e:
                job["step"] = "error"
                job["detail"] = f"Instagram login failed: {e}"
                return

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        image_paths = []
        job["total"] = max_posts
        job["current"] = 0
        try:
            if is_hashtag:
                posts = instaloader.Hashtag.from_name(L.context, name).get_posts()
            else:
                profile = instaloader.Profile.from_username(L.context, name)
                posts = profile.get_posts()

            count = 0
            skipped = 0
            for post in posts:
                if count >= max_posts:
                    break
                post_date = post.date_utc
                if post_date.date() > end_dt.date():
                    skipped += 1
                    job["detail"] = f"Scanning posts from {label}... ({skipped} skipped, {count} downloaded)"
                    continue
                if post_date.date() < start_dt.date():
                    break

                if post.is_video:
                    continue

                filename = f"{post.date_utc.strftime('%Y%m%d_%H%M%S')}_{post.shortcode}.jpg"
                filepath = os.path.join(tmp_dir, filename)

                try:
                    L.download_pic(filepath, post.url, post.date_utc)
                    # download_pic may append extension
                    if os.path.exists(filepath):
                        image_paths.append(Path(filepath))
                    elif os.path.exists(filepath + ".jpg"):
                        shutil.move(filepath + ".jpg", filepath)
                        image_paths.append(Path(filepath))
                    count += 1
                    job["current"] = count
                    job["detail"] = f"Downloading from {label}: {count} of ~{max_posts} images..."
                except Exception:
                    continue

        except Exception as e:
            job["step"] = "error"
            job["detail"] = f"Scrape error: {e}"
            return

        job["total"] = len(image_paths)
        job["current"] = 0
        job["detail"] = f"Downloaded {len(image_paths)} images. Starting analysis..."

        if not image_paths:
            job["step"] = "done"
            job["detail"] = "No images found for this profile/date range."
            job["results"] = {"dura_bulk": [], "non_dura_bulk": []}
            return

        # --- Step 2 & 3: Detect boats + OCR ---
        job["step"] = "detecting"
        model = get_yolo()
        reader = get_ocr()

        dura_files = []
        non_dura_files = []
        job["partial_results"] = {"dura_bulk": [], "non_dura_bulk": []}

        for i, img_path in enumerate(image_paths):
            job["current"] = i + 1
            job["detail"] = f"Processing image {i + 1}/{len(image_paths)}"

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            # YOLO detection
            results = model(img, verbose=False)
            is_dura = False

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != 8:  # 8 = boat in COCO
                        continue

                    # Crop boat region
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    crop = img.crop((x1, y1, x2, y2))

                    # OCR on crop
                    crop_path = str(img_path) + "_crop.jpg"
                    crop.save(crop_path)
                    try:
                        ocr_results = reader.readtext(crop_path)
                        all_text = " ".join([r[1] for r in ocr_results])
                        if fuzzy_match_dura_bulk(all_text):
                            is_dura = True
                            break
                    except Exception:
                        pass
                    finally:
                        if os.path.exists(crop_path):
                            os.remove(crop_path)

                if is_dura:
                    break

            # Sort image
            dest_dir = DURA_DIR if is_dura else NON_DURA_DIR
            dest_name = img_path.name
            shutil.copy2(img_path, dest_dir / dest_name)

            if is_dura:
                dura_files.append(dest_name)
                job["partial_results"]["dura_bulk"].append(dest_name)
            else:
                non_dura_files.append(dest_name)
                job["partial_results"]["non_dura_bulk"].append(dest_name)

        # Cleanup temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # --- Step 4: Done ---
        job["step"] = "done"
        job["detail"] = (
            f"Done! {len(dura_files)} Dura Bulk, {len(non_dura_files)} other."
        )
        job["results"] = {
            "dura_bulk": dura_files,
            "non_dura_bulk": non_dura_files,
        }

    except Exception as e:
        job["step"] = "error"
        job["detail"] = str(e)


def run_upload_pipeline(job_id, tmp_dir):
    """Pipeline for uploaded images: detect boats → OCR → sort."""
    job = jobs[job_id]

    try:
        image_paths = [
            f for f in Path(tmp_dir).iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ]

        job["total"] = len(image_paths)
        job["step"] = "detecting"
        job["detail"] = f"Processing {len(image_paths)} images..."

        if not image_paths:
            job["step"] = "done"
            job["detail"] = "No valid images found."
            job["results"] = {"dura_bulk": [], "non_dura_bulk": []}
            return

        model = get_yolo()
        reader = get_ocr()

        dura_files = []
        non_dura_files = []
        job["partial_results"] = {"dura_bulk": [], "non_dura_bulk": []}

        for i, img_path in enumerate(image_paths):
            job["current"] = i + 1
            job["detail"] = f"Processing image {i + 1}/{len(image_paths)}"

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            results = model(img, verbose=False)
            is_dura = False

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != 8:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    crop = img.crop((x1, y1, x2, y2))

                    crop_path = str(img_path) + "_crop.jpg"
                    crop.save(crop_path)
                    try:
                        ocr_results = reader.readtext(crop_path)
                        all_text = " ".join([r[1] for r in ocr_results])
                        if fuzzy_match_dura_bulk(all_text):
                            is_dura = True
                            break
                    except Exception:
                        pass
                    finally:
                        if os.path.exists(crop_path):
                            os.remove(crop_path)

                if is_dura:
                    break

            dest_dir = DURA_DIR if is_dura else NON_DURA_DIR
            dest_name = f"upload_{i:04d}{img_path.suffix}"
            shutil.copy2(img_path, dest_dir / dest_name)

            if is_dura:
                dura_files.append(dest_name)
                job["partial_results"]["dura_bulk"].append(dest_name)
            else:
                non_dura_files.append(dest_name)
                job["partial_results"]["non_dura_bulk"].append(dest_name)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        job["step"] = "done"
        job["detail"] = (
            f"Done! {len(dura_files)} Dura Bulk, {len(non_dura_files)} other."
        )
        job["results"] = {
            "dura_bulk": dura_files,
            "non_dura_bulk": non_dura_files,
        }

    except Exception as e:
        job["step"] = "error"
        job["detail"] = str(e)


# --- Routes ---


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/scrape", methods=["POST"])
def start_scrape():
    data = request.json
    raw_input = data.get("profile", "").strip()
    start_date = data.get("start_date", "")
    end_date = data.get("end_date", "")
    max_posts = int(data.get("max_posts", 100))
    ig_username = data.get("ig_username", "").strip()
    ig_password = data.get("ig_password", "")

    if not raw_input or not start_date or not end_date:
        return jsonify({"error": "Missing required fields"}), 400

    # Detect hashtag vs profile
    is_hashtag = raw_input.startswith("#")
    name = raw_input.lstrip("@#")

    if not name:
        return jsonify({"error": "Missing required fields"}), 400

    if is_hashtag and (not ig_username or not ig_password):
        return jsonify({"error": "Instagram login required for hashtag scraping"}), 400

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "step": "queued",
        "detail": "Starting...",
        "current": 0,
        "total": 0,
        "results": None,
    }

    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, name, start_date, end_date, max_posts, is_hashtag, ig_username, ig_password),
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/upload", methods=["POST"])
def upload_images():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    tmp_dir = tempfile.mkdtemp(prefix="dura_bulk_upload_")
    for f in files:
        if f.filename:
            safe_name = os.path.basename(f.filename)
            f.save(os.path.join(tmp_dir, safe_name))

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "step": "queued",
        "detail": "Starting...",
        "current": 0,
        "total": 0,
        "results": None,
    }

    thread = threading.Thread(
        target=run_upload_pipeline, args=(job_id, tmp_dir)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/images/<category>/<filename>")
def serve_image(category, filename):
    if category == "dura_bulk":
        folder = DURA_DIR
    elif category == "non_dura_bulk":
        folder = NON_DURA_DIR
    else:
        return "Not found", 404
    return send_from_directory(folder, filename)


@app.route("/api/download/<category>")
def download_zip(category):
    if category == "dura_bulk":
        folder = DURA_DIR
    elif category == "non_dura_bulk":
        folder = NON_DURA_DIR
    else:
        return "Not found", 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in folder.iterdir():
            if f.is_file():
                zf.write(f, f.name)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{category}.zip",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)

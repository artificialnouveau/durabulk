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

import requests as http_requests
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
from apify_client import ApifyClient
from ultralytics import YOLO
import easyocr

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
DURA_DIR = DOWNLOADS_DIR / "dura_bulk"
NON_DURA_DIR = DOWNLOADS_DIR / "non_dura_bulk"

# Apify API token (set via Render environment variables)
APIFY_TOKEN = os.environ.get("APIFY_TOKEN", "")

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


def run_pipeline(job_id, name, start_date, end_date, max_posts=100, is_hashtag=False):
    """Background pipeline: scrape via Apify → detect boats → OCR → sort."""
    job = jobs[job_id]

    try:
        # --- Step 1: Scrape via Apify ---
        job["step"] = "scraping"
        label = f"#{name}" if is_hashtag else f"@{name}"
        job["detail"] = f"Fetching posts from {label} via Apify..."

        if not APIFY_TOKEN:
            job["step"] = "error"
            job["detail"] = "Apify API token not configured. Set APIFY_TOKEN env var."
            return

        tmp_dir = tempfile.mkdtemp(prefix="dura_bulk_")
        client = ApifyClient(APIFY_TOKEN)

        # Use the Instagram Scraper actor
        run_input = {
            "resultsLimit": max_posts,
        }

        if is_hashtag:
            run_input["directUrls"] = [f"https://www.instagram.com/explore/tags/{name}/"]
            run_input["resultsType"] = "posts"
        else:
            run_input["directUrls"] = [f"https://www.instagram.com/{name}/"]
            run_input["resultsType"] = "posts"

        job["detail"] = f"Running Apify scraper for {label}... (this may take a minute)"
        job["total"] = max_posts
        job["current"] = 0

        try:
            run = client.actor("apify/instagram-scraper").call(run_input=run_input)
        except Exception as e:
            job["step"] = "error"
            job["detail"] = f"Apify scraper failed: {e}"
            return

        # Download images from results
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        image_paths = []
        count = 0

        job["detail"] = f"Downloading images from {label}..."

        # Collect all items first to inspect structure
        all_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        job["detail"] = f"Got {len(all_items)} items from Apify. Processing..."

        if all_items:
            first = all_items[0]
            job["debug_keys"] = list(first.keys())
            print(f"[DEBUG] {len(all_items)} items. First item keys: {list(first.keys())}", flush=True)
            # Log a sample of values for key fields
            for key in ["type", "displayUrl", "imageUrl", "url", "display_url", "timestamp", "shortCode"]:
                val = first.get(key)
                if val:
                    print(f"[DEBUG] first['{key}'] = {str(val)[:200]}", flush=True)
        else:
            print(f"[DEBUG] Dataset returned 0 items!", flush=True)
            job["step"] = "done"
            job["detail"] = "Apify returned 0 items from dataset."
            job["results"] = {"dura_bulk": [], "non_dura_bulk": []}
            return

        items_seen = 0
        skipped_date = 0
        skipped_video = 0
        skipped_no_url = 0

        for item in all_items:
            items_seen += 1
            if count >= max_posts:
                break

            # Filter by date if timestamp available
            timestamp = item.get("timestamp")
            if timestamp:
                try:
                    post_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date()
                    if post_date > end_dt.date() or post_date < start_dt.date():
                        skipped_date += 1
                        continue
                except Exception:
                    pass

            # Skip videos
            if item.get("type") == "Video":
                skipped_video += 1
                continue

            # Get image URL — try multiple field names used by different Apify actors
            image_url = (
                item.get("displayUrl")
                or item.get("imageUrl")
                or item.get("url")
                or item.get("display_url")
                or ""
            )
            if not image_url:
                # Try first image in carousel
                images = item.get("images") or item.get("childPosts") or []
                if images:
                    first_img = images[0]
                    if isinstance(first_img, str):
                        image_url = first_img
                    elif isinstance(first_img, dict):
                        image_url = first_img.get("url", "") or first_img.get("displayUrl", "") or first_img.get("imageUrl", "")

            if not image_url:
                skipped_no_url += 1
                continue

            # Build filename: YYYYMMDD_profilename_NNN.jpg
            date_prefix = ""
            if timestamp:
                try:
                    post_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    date_prefix = post_dt.strftime("%Y%m%d")
                except Exception:
                    pass
            if not date_prefix:
                date_prefix = "nodate"
            filename = f"{date_prefix}_{name}_{count:03d}.jpg"
            filepath = os.path.join(tmp_dir, filename)

            try:
                resp = http_requests.get(image_url, timeout=30)
                if resp.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    image_paths.append(Path(filepath))
                    count += 1
                    job["current"] = count
                    job["detail"] = f"Downloading from {label}: {count} images..."
            except Exception:
                continue

        print(f"[DEBUG] Items seen: {items_seen}, downloaded: {count}, skipped_date: {skipped_date}, skipped_video: {skipped_video}, skipped_no_url: {skipped_no_url}", flush=True)
        job["debug_stats"] = f"seen:{items_seen} dl:{count} skip_date:{skipped_date} skip_vid:{skipped_video} skip_nourl:{skipped_no_url}"

        job["total"] = len(image_paths)
        job["current"] = 0
        job["detail"] = f"Downloaded {len(image_paths)} images. Starting analysis..."

        if not image_paths:
            job["step"] = "done"
            job["detail"] = f"No images downloaded. {items_seen} items from Apify: {skipped_date} filtered by date, {skipped_video} videos, {skipped_no_url} had no image URL."
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

    if not raw_input or not start_date or not end_date:
        return jsonify({"error": "Missing required fields"}), 400

    # Detect hashtag vs profile
    is_hashtag = raw_input.startswith("#")
    name = raw_input.lstrip("@#")

    if not name:
        return jsonify({"error": "Missing required fields"}), 400

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
        args=(job_id, name, start_date, end_date, max_posts, is_hashtag),
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

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
from apify_client import ApifyClient

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Apify API token (set via Render environment variables)
APIFY_TOKEN = os.environ.get("APIFY_TOKEN", "")

# In-memory job store
jobs = {}


def run_pipeline(job_id, name, start_date, end_date, max_posts=100, is_hashtag=False):
    """Background pipeline: scrape via Apify → download images."""
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

        # Create a job-specific download folder
        job_dir = DOWNLOADS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        client = ApifyClient(APIFY_TOKEN)

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

        all_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        job["detail"] = f"Got {len(all_items)} items from Apify. Downloading..."

        if not all_items:
            job["step"] = "done"
            job["detail"] = "Apify returned 0 items from dataset."
            job["results"] = {"images": []}
            return

        count = 0
        downloaded = []
        skipped_date = 0
        skipped_video = 0
        skipped_no_url = 0

        for item in all_items:
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

            # Get image URL
            image_url = (
                item.get("displayUrl")
                or item.get("imageUrl")
                or item.get("url")
                or item.get("display_url")
                or ""
            )
            if not image_url:
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

            owner = item.get("ownerUsername") or name
            filename = f"{date_prefix}_{owner}_{count:03d}.jpg"
            filepath = job_dir / filename

            try:
                resp = http_requests.get(image_url, timeout=30)
                if resp.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    downloaded.append(filename)
                    count += 1
                    job["current"] = count
                    job["detail"] = f"Downloading from {label}: {count} images..."
            except Exception:
                continue

        # --- Done ---
        job["step"] = "done"
        if not downloaded:
            job["detail"] = f"No images downloaded. {len(all_items)} items: {skipped_date} filtered by date, {skipped_video} videos, {skipped_no_url} had no image URL."
        else:
            job["detail"] = f"Done! Downloaded {len(downloaded)} images."
        job["total"] = len(downloaded)
        job["current"] = len(downloaded)
        job["results"] = {"images": downloaded}

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


@app.route("/api/status/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/images/<job_id>/<filename>")
def serve_image(job_id, filename):
    folder = DOWNLOADS_DIR / job_id
    if not folder.exists():
        return "Not found", 404
    return send_from_directory(folder, filename)


@app.route("/api/download/<job_id>")
def download_zip(job_id):
    folder = DOWNLOADS_DIR / job_id
    if not folder.exists():
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
        download_name=f"images_{job_id}.zip",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)

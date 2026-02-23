#!/usr/bin/env python3
"""
Download images from @durabulk on Instagram.
Run this once locally before deploying the static site.

Usage:
    pip install instaloader
    python download_images.py
"""

import instaloader
import json
import os
import shutil
from datetime import datetime

PROFILE = "durabulk"
OUTPUT_DIR = "images"
MAX_POSTS = 100
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    L = instaloader.Instaloader(
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
    )

    # Optional: login for private profiles or higher rate limits
    # L.login("your_username", "your_password")

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")

    print(f"Fetching posts from @{PROFILE}...")
    profile = instaloader.Profile.from_username(L.context, PROFILE)

    image_files = []
    count = 0

    for post in profile.get_posts():
        if count >= MAX_POSTS:
            break
        post_date = post.date_utc
        if post_date.date() > end_dt.date():
            continue
        if post_date.date() < start_dt.date():
            break
        if post.is_video:
            continue

        filename = f"{post.date_utc.strftime('%Y%m%d_%H%M%S')}_{post.shortcode}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(filepath):
            try:
                stem = os.path.splitext(filepath)[0]
                L.download_pic(stem, post.url, post.date_utc)
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    candidate = stem + ext
                    if os.path.exists(candidate) and candidate != filepath:
                        shutil.move(candidate, filepath)
                        break
            except Exception as e:
                print(f"  Skipped: {e}")
                continue

        if os.path.exists(filepath):
            image_files.append(filename)
            count += 1
            print(f"  [{count}/{MAX_POSTS}] {filename}")

    print(f"\nDownloaded {len(image_files)} images.")

    # Generate image list JSON for the static site
    all_images = sorted(
        f for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )

    with open("image-list.json", "w") as f:
        json.dump(all_images, f, indent=2)

    print(f"Wrote image-list.json with {len(all_images)} entries.")


if __name__ == "__main__":
    main()

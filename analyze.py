#!/usr/bin/env python3
"""
Offline analysis of downloaded @durabulk images using YOLOv8 + EasyOCR.
Reads image-list.json, processes each image, writes results.json.

Usage:
    pip install ultralytics easyocr pillow
    python analyze.py
"""

import json
import os
import re
import sys

from PIL import Image
from ultralytics import YOLO
import easyocr

IMAGES_DIR = "images"
IMAGE_LIST = "image-list.json"
RESULTS_FILE = "results.json"


def fuzzy_match_dura_bulk(text):
    """Check if text contains something close to 'dura bulk'."""
    lower = text.lower().strip()
    if "dura" in lower and "bulk" in lower:
        return True
    cleaned = re.sub(r"[^a-z0-9]", "", lower)
    return "durabulk" in cleaned


def analyze_image(model, reader, img_path):
    """Run YOLOv8 boat detection + EasyOCR on a single image.
    Returns (is_dura_bulk, details_string).
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return False, f"Could not open image: {e}"

    results = model(img, verbose=False)
    boats_found = 0
    all_ocr_text = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 8:  # 8 = boat in COCO
                continue

            boats_found += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = img.crop((x1, y1, x2, y2))

            # Save temp crop for EasyOCR
            crop_path = img_path + "_crop_tmp.jpg"
            try:
                crop.save(crop_path)
                ocr_results = reader.readtext(crop_path)
                text = " ".join([r[1] for r in ocr_results])
                if text.strip():
                    all_ocr_text.append(text.strip())
            except Exception:
                pass
            finally:
                if os.path.exists(crop_path):
                    os.remove(crop_path)

    combined_text = " | ".join(all_ocr_text)
    is_dura = fuzzy_match_dura_bulk(combined_text) if combined_text else False

    details = f"boats={boats_found}"
    if combined_text:
        details += f", ocr_text=\"{combined_text}\""

    return is_dura, details


def main():
    if not os.path.exists(IMAGE_LIST):
        print(f"Error: {IMAGE_LIST} not found. Run download_images.py first.")
        sys.exit(1)

    with open(IMAGE_LIST) as f:
        image_names = json.load(f)

    if not image_names:
        print("No images listed in image-list.json.")
        sys.exit(0)

    print(f"Loading models...")
    model = YOLO("yolov8n.pt")
    reader = easyocr.Reader(["en"], gpu=False)

    print(f"Analyzing {len(image_names)} images...\n")
    results = {}

    for i, name in enumerate(image_names):
        img_path = os.path.join(IMAGES_DIR, name)
        if not os.path.exists(img_path):
            print(f"  [{i+1}/{len(image_names)}] SKIP {name} (file not found)")
            results[name] = {"dura_bulk": False, "details": "file not found"}
            continue

        is_dura, details = analyze_image(model, reader, img_path)
        label = "DURA BULK" if is_dura else "other"
        print(f"  [{i+1}/{len(image_names)}] {label:>10}  {name}  ({details})")
        results[name] = {"dura_bulk": is_dura, "details": details}

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    dura_count = sum(1 for r in results.values() if r["dura_bulk"])
    print(f"\nDone. {dura_count} Dura Bulk, {len(results) - dura_count} other.")
    print(f"Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()

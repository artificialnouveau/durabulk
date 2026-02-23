"""
Analyze downloaded images locally for "Dura Bulk" text using OCR.

Usage:
    python local_analyze.py
    python local_analyze.py --input my_images
    python local_analyze.py --input images --output results

Images containing "Dura Bulk" (or partial matches) are copied to
<output>/dura_bulk/ and the rest to <output>/non_dura_bulk/.
"""

import argparse
import re
from pathlib import Path

import easyocr
from PIL import Image


def fuzzy_match_dura_bulk(text):
    """Check if text contains 'dura bulk' or partial matches."""
    text = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9]", "", text)

    if "dura" in text and "bulk" in text:
        return True
    if "durabulk" in cleaned:
        return True

    # Partial matches (4+ char substrings of "durabulk")
    target = "durabulk"
    for length in range(4, len(target) + 1):
        for start in range(len(target) - length + 1):
            if target[start:start + length] in cleaned:
                return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Analyze images locally for Dura Bulk text")
    parser.add_argument("--input", default="images", help="Input directory with images (default: images)")
    parser.add_argument("--output", default="results", help="Output directory (default: results)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    dura_dir = output_dir / "dura_bulk"
    non_dura_dir = output_dir / "non_dura_bulk"

    if not input_dir.exists():
        print(f"Error: input directory '{input_dir}' not found.")
        print("Run local_download_images.py first to download images.")
        return

    image_paths = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ])

    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return

    dura_dir.mkdir(parents=True, exist_ok=True)
    non_dura_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OCR engine...")
    reader = easyocr.Reader(["en"], gpu=False)

    dura_count = 0
    non_dura_count = 0

    print(f"Analyzing {len(image_paths)} images...\n")

    for i, img_path in enumerate(image_paths):
        try:
            ocr_results = reader.readtext(str(img_path))
            all_text = " ".join([r[1] for r in ocr_results])
            is_dura = fuzzy_match_dura_bulk(all_text)
        except Exception as e:
            print(f"  [{i+1}/{len(image_paths)}] Error on {img_path.name}: {e}")
            is_dura = False

        if is_dura:
            dest = dura_dir / img_path.name
            dura_count += 1
            label = "DURA BULK"
        else:
            dest = non_dura_dir / img_path.name
            non_dura_count += 1
            label = "other"

        # Copy file to result folder
        import shutil
        shutil.copy2(img_path, dest)

        ocr_preview = all_text[:60].replace("\n", " ") if all_text.strip() else "(no text)"
        print(f"  [{i+1}/{len(image_paths)}] {img_path.name} → {label}  |  OCR: {ocr_preview}")

    print(f"\nDone! {dura_count} Dura Bulk, {non_dura_count} other.")
    print(f"  Dura Bulk images: {dura_dir}/")
    print(f"  Other images:     {non_dura_dir}/")


if __name__ == "__main__":
    main()

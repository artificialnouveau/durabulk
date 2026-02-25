"""
Download Instagram images locally using instaloader.

Usage:
    # Set your username so you don't have to pass --login every time:
    export INSTA_USERNAME=your_username  # add to ~/.zshrc to persist

    python local_download_images.py durabulk
    python local_download_images.py durabulk --start 2025-01-01 --end 2025-12-31
    python local_download_images.py "#durabulk" --max 50

    # Or pass --login explicitly to override:
    python local_download_images.py "#durabulk" --login OTHER_USERNAME --max 50

Images are saved to ./images/ as YYYYMMDD_profilename_NNN.jpg
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import instaloader


def main():
    parser = argparse.ArgumentParser(description="Download Instagram images locally")
    parser.add_argument("target", help="Profile name or #hashtag (e.g. durabulk or #durabulk)")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--max", type=int, default=100, help="Max images to download (default: 100)")
    parser.add_argument("--login", default=os.environ.get("INSTA_USERNAME"), help="Instagram username (default: $INSTA_USERNAME env var)")
    parser.add_argument("--output", default="images", help="Output directory (default: images)")
    args = parser.parse_args()

    is_hashtag = args.target.startswith("#")
    name = args.target.lstrip("@#")
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    L = instaloader.Instaloader(
        download_pictures=True,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
    )

    if is_hashtag and not args.login:
        print("Error: hashtag scraping requires a login. Either:")
        print(f"  1. Set the env var:  export INSTA_USERNAME=your_username")
        print(f"  2. Pass --login:     python3 local_download_images.py \"#{name}\" --login YOUR_USERNAME")
        return

    if args.login:
        try:
            L.load_session_from_file(args.login)
        except FileNotFoundError:
            print(f"ERROR: No session file found for '{args.login}'.")
            print(f"You need to create one first by running:\n")
            print(f"  python3 -m instaloader --login {args.login}\n")
            print(f"This will ask for your password and save a session file.")
            return
        # Verify the session is still valid
        try:
            test_user = L.test_login()
            if test_user:
                print(f"Logged in as @{test_user}")
            else:
                print(f"WARNING: Session for '{args.login}' has expired.")
                print(f"Please create a new one by running:\n")
                print(f"  python3 -m instaloader --login {args.login}\n")
                return
        except Exception:
            print(f"WARNING: Could not verify session for '{args.login}'. Continuing anyway...")

    count = 0

    if is_hashtag:
        print(f"Fetching posts from #{name}...")
        hashtag = instaloader.Hashtag.from_name(L.context, name)
        posts = hashtag.get_posts()
    else:
        print(f"Fetching posts from @{name}...")
        profile = instaloader.Profile.from_username(L.context, name)
        posts = profile.get_posts()

    for post in posts:
        if count >= args.max:
            break

        post_date = post.date_local
        if post_date.date() < start_dt.date() or post_date.date() > end_dt.date():
            continue

        if post.is_video:
            continue

        date_str = post_date.strftime("%Y%m%d")
        owner = post.owner_username if is_hashtag else name
        filename = f"{date_str}_{owner}_{count:03d}.jpg"
        filepath = out_dir / filename

        try:
            L.download_pic(filename=str(filepath.with_suffix("")), url=post.url, mtime=post_date)
            # instaloader adds its own extension, rename if needed
            downloaded = filepath.with_suffix(".jpg")
            if not downloaded.exists():
                # Check for other extensions instaloader might have used
                for ext in [".jpeg", ".png", ".webp"]:
                    alt = filepath.with_suffix(ext)
                    if alt.exists():
                        alt.rename(downloaded)
                        break
            if downloaded.exists():
                count += 1
                print(f"  [{count}/{args.max}] {filename}")
            else:
                print(f"  Skipped (download failed): {post.shortcode}")
        except Exception as e:
            print(f"  Skipped: {e}")
            continue

    print(f"\nDone! {count} images saved to {out_dir}/")


if __name__ == "__main__":
    main()

# Dura Bulk Detector

Detect Instagram images of the bulk carrier ship "Dura Bulk" using object detection + OCR.

## Option 1: Use the Web App (no installation needed)

Go to **https://durabulk.onrender.com/** in your browser.

You can:
- **Download from Instagram** — enter a profile or #hashtag, date range, and number of images. The server will scrape, detect boats, and check for "Dura Bulk" text automatically.
- **Upload Your Own** — drag and drop images from your computer and analyze them in the browser.

## Option 2: Download Images Locally

If you want to download Instagram images to your own computer, follow the steps below.

### Step 1: Install Python

You need Python installed on your computer.

**Mac:**
1. Open **Terminal** (search for "Terminal" in Spotlight, or find it in Applications > Utilities)
2. Type this and press Enter:
   ```
   python3 --version
   ```
3. If you see a version number (e.g. `Python 3.11.5`), you're good — skip to Step 2
4. If not, install it from https://www.python.org/downloads/ — download the installer, open it, and follow the prompts

**Windows:**
1. Open **Command Prompt** (search for "cmd" in the Start menu)
2. Type this and press Enter:
   ```
   python --version
   ```
3. If you see a version number, skip to Step 2
4. If not, install it from https://www.python.org/downloads/ — download the installer, open it, and **check the box "Add Python to PATH"** before clicking Install

### Step 2: Download this project

1. Go to https://github.com/artificialnouveau/durabulk
2. Click the green **Code** button, then click **Download ZIP**
3. Unzip the downloaded file and open the folder

### Step 3: Install the required package

Open **Terminal** (Mac) or **Command Prompt** (Windows), then navigate to the folder you just unzipped. For example:

```
cd ~/Downloads/durabulk-main
```

Then install the required package:

**Mac:**
```
pip3 install -r requirements-local.txt
```

**Windows:**
```
pip install -r requirements-local.txt
```

If `pip` is not found, try:
```
python3 -m pip install -r requirements-local.txt
```

### Step 4: Download images

Run the download script. Replace `durabulk` with the profile or hashtag you want:

```
python3 local_download_images.py durabulk
```

**More examples:**

```
# Download up to 50 images from a profile, within a date range
python3 local_download_images.py durabulk --start 2025-01-01 --end 2025-06-30 --max 50

# Download from a hashtag (requires login — see below)
python3 local_download_images.py "#durabulk" --login your_username --max 50
```

On Windows, use `python` instead of `python3`.

Images are saved to the `images/` folder as `YYYYMMDD_profilename_NNN.jpg` (e.g. `20250601_durabulk_003.jpg`).

### Hashtag downloads (requires login)

Instagram requires a login to search hashtags. First, create a session:

```
pip3 install instaloader
instaloader --login your_username
```

Enter your password when prompted. Then use `--login your_username` when running the download script.

### Step 5: Analyze the images

**Option A: Analyze in your browser (no extra install needed)**

Go to **https://durabulk.onrender.com/** and use the **Upload Your Own** tab to drag and drop your downloaded images. The OCR analysis runs entirely in your browser.

**Option B: Analyze locally with Python**

This uses EasyOCR to scan each image for "Dura Bulk" text (including partial/obscured matches) and sorts them into `results/dura_bulk/` and `results/non_dura_bulk/` folders.

```
python3 local_analyze.py
```

By default it reads from `images/` and writes to `results/`. You can change this:

```
python3 local_analyze.py --input my_images --output my_results
```

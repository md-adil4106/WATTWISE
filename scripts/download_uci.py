"""Optional helper script to download the UCI Individual Household Electric Power Consumption dataset.

Dataset Citation:
- Title: Individual Household Electric Power Consumption
- Repository: UCI Machine Learning Repository (ID: 235)
- DOI: 10.24432/C58K54
- URL: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

This script fails gracefully if no internet access is available, and skips downloading
if the dataset is already present locally.
"""

from pathlib import Path
import sys
import urllib.error
import urllib.request
import zipfile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET_FILE = DATA_DIR / "household_power_consumption.txt"
ZIP_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"


def download_and_extract() -> bool:
    """Download and extract the UCI dataset if missing, failing gracefully on error."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 10_000_000:
        print(f"[INFO] Dataset already exists at '{TARGET_FILE}' ({TARGET_FILE.stat().st_size:,} bytes). Skipping download.")
        return True

    temp_zip = DATA_DIR / "temp_uci_dataset.zip"
    print(f"[INFO] Attempting to download UCI dataset (DOI 10.24432/C58K54, ID 235) from:\n       {ZIP_URL}")

    try:
        req = urllib.request.Request(
            ZIP_URL,
            headers={"User-Agent": "WATTWISE-Data-Downloader/1.0 (Educational Academic Project)"},
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(temp_zip, "wb") as out_file:
            total_size = response.length
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB blocks
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"       Downloading: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="\r")

        print("\n[INFO] Download finished. Extracting archive...")
        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extract("household_power_consumption.txt", path=DATA_DIR)

        if temp_zip.exists():
            temp_zip.unlink()

        print(f"[SUCCESS] Dataset extracted to '{TARGET_FILE}' ({TARGET_FILE.stat().st_size:,} bytes).")
        return True

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as err:
        print(f"[WARNING] Optional automatic download could not be completed: {err}")
        print("[NOTICE] To use the real UCI dataset manually:")
        print("         1. Visit: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption")
        print("         2. Download 'individual+household+electric+power+consumption.zip'")
        print("         3. Extract 'household_power_consumption.txt' into the 'data/' directory.")
        print("[NOTICE] WATTWISE will automatically fall back to 'SYNTHETIC DEMO' mode if the file is absent.")
        if temp_zip.exists():
            temp_zip.unlink()
        return False


if __name__ == "__main__":
    success = download_and_extract()
    sys.exit(0 if success else 1)

# =================================================================================
# DOWNLOAD_OGLE_TRAINING_DATA.PY - Download Real Labeled Light Curves from OGLE
# For a conceptual explanation of how this fits into the project, see:
# walkthrough_guide.md (at the repository or workspace root)
# =================================================================================

import os
import time
import requests
import numpy as np

BASE_URL = "https://www.astrouw.edu.pl/ogle/ogle4/OCVS"
OUTPUT_DIR = "data/ogle_training"
N_PER_CLASS = 1000   # How many light curves to download per class

CATALOGS = [
    ("cepheid", "lmc", "cep", "OGLE-LMC-CEP-", 2, 4620, "lmc/cep/phot/I"),
    ("rrlyr", "blg", "rrlyr", "OGLE-BLG-RRLYR-", 1, 38000, "blg/rrlyr/phot/I"),
    ("eclipsing_binary", "lmc", "ecl", "OGLE-LMC-ECL-", 1, 26121, "lmc/ecl/phot/I"),
    # We will use OGLE-III for LPV since it is reliable and active
    ("lpv", "lmc", "lpv3", "OGLE-LMC-LPV-", 1, 79000, "lmc/lpv3"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (academic research project)"
}

def fetch_light_curve(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200 and len(resp.text.strip()) > 0:
                return resp.text
            elif resp.status_code == 404:
                return None
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None

def validate_lightcurve(text: str, min_points: int = 20) -> bool:
    rows = [l for l in text.strip().splitlines() if l and not l.startswith("#")]
    return len(rows) >= min_points

def download_catalog(class_label, id_prefix, id_start, id_end, phot_subdir,
                     n_target, id_digits, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_ids = list(range(id_start, id_end + 1))
    np.random.shuffle(all_ids)
    sampled_ids = all_ids[:n_target * 4]

    downloaded = 0
    tried = 0

    for star_id in sampled_ids:
        if downloaded >= n_target:
            break

        tried += 1
        id_str = str(star_id).zfill(id_digits)
        filename = f"{id_prefix}{id_str}.dat"
        
        # Adjust URL path for OGLE-III LPV vs OGLE-IV others
        if class_label == "lpv":
            # OGLE-III LPV: https://www.astrouw.edu.pl/ogle/ogle3/OIII-CVS/lmc/lpv/phot/I/OGLE-LMC-LPV-00001.dat
            url = f"https://www.astrouw.edu.pl/ogle/ogle3/OIII-CVS/lmc/lpv/phot/I/{filename}"
        else:
            url = f"{BASE_URL}/{phot_subdir}/{filename}"
            
        out_path = os.path.join(out_dir, filename)

        if os.path.exists(out_path):
            downloaded += 1
            continue

        content = fetch_light_curve(url)
        if content and validate_lightcurve(content):
            with open(out_path, "w") as f:
                f.write(content)
            downloaded += 1
            if downloaded % 25 == 0:
                print(f"  [{class_label}] {downloaded}/{n_target} downloaded...")

        time.sleep(0.4)

    print(f"  [{class_label}] Done: {downloaded} saved to {out_dir}")
    return downloaded

def main():
    np.random.seed(42)
    print("=" * 60)
    print("OGLE Real Light Curve Downloader")
    print("=" * 60)

    summary = {}

    for class_label, region, var_type, id_prefix, id_start, id_end, phot_subdir in CATALOGS:
        out_dir = os.path.join(OUTPUT_DIR, class_label)
        id_digits = 5 if region in ("blg",) or var_type in ("ecl", "lpv3", "rrlyr") else 4

        n = download_catalog(
            class_label, id_prefix, id_start, id_end, phot_subdir,
            N_PER_CLASS, id_digits, out_dir
        )
        summary[class_label] = n

    print("\nDOWNLOAD COMPLETE")
    for cls, n in summary.items():
        print(f"  {cls:<22}: {n} light curves")


if __name__ == "__main__":
    main()

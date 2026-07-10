# =================================================================================
# DOWNLOAD_ANDROMEDA_HCV.PY - Hubble M31 Variable Stars Downloader
# For a conceptual explanation of how this fits into the project, see:
# walkthrough_guide.md (at the repository or workspace root)
# =================================================================================
import os
import requests
import json
import pandas as pd
import numpy as np

def download_hcv_variables(ra=10.68471, dec=41.268749, radius=0.2, output_dir="data/andromeda_real_hcv", max_stars=10):
    """
    Queries the MAST Hubble Catalog of Variables (HCV) API to download actual resolved variable star
    light curves inside the Andromeda Galaxy (M31).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Querying HCV summary catalog around M31 (RA={ra}, Dec={dec}, Radius={radius} deg)...")
    
    # Base URL for the Hubble Source Catalog (HSC) API v0.1
    base_url = "https://catalogs.mast.stsci.edu/api/v0.1/hsc/v3/hcvsummary.json"
    
    params = {
        'ra': ra,
        'dec': dec,
        'radius': radius
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error: Failed to query MAST API (status code {response.status_code})")
        return []
        
    res_data = response.json()
    all_rows = res_data.get('data', [])
    
    # Column mapping from metadata:
    # 0: MatchID, 1: Filter, 4: RA, 5: Dec, 6: AutoClass (0=constant, 1=SFVC, 2=MFVC), 10: NumLC (Epoch count)
    variables = []
    for row in all_rows:
        match_id = row[0]
        filt = row[1]
        star_ra = row[4]
        star_dec = row[5]
        auto_class = row[6]
        num_lc = row[10]
        
        # We only want actual variable candidates (AutoClass > 0)
        if auto_class > 0:
            variables.append({
                'MatchID': match_id,
                'Filter': filt,
                'RA': star_ra,
                'Dec': star_dec,
                'NumLC': num_lc
            })
            
    print(f"Found {len(variables)} resolved variable candidates in M31.")
    if len(variables) == 0:
        return []
        
    # Sort by number of epochs descending to get the best sampled light curves
    variables.sort(key=lambda x: x['NumLC'], reverse=True)
    
    selected_variables = variables[:max_stars]
    print(f"Downloading detailed light curves for the top {len(selected_variables)} best-sampled variables...")
    
    detailed_base_url = "https://catalogs.mast.stsci.edu/api/v0.1/hsc/v3/hcv.json"
    downloaded_files = []
    
    for idx, var in enumerate(selected_variables):
        match_id = var['MatchID']
        filt = var['Filter']
        print(f"[{idx+1}/{len(selected_variables)}] Downloading MatchID {match_id} (Filter: {filt}, Epochs: {var['NumLC']})...")
        
        det_params = {
            'MatchID': match_id,
            'Filter': filt
        }
        
        det_response = requests.get(detailed_base_url, params=det_params)
        if det_response.status_code != 200:
            print(f"  Skipping MatchID {match_id} due to API error.")
            continue
            
        det_data = det_response.json()
        det_rows = det_data.get('data', [])
        
        if len(det_rows) == 0:
            print(f"  No time-series rows returned for MatchID {match_id}.")
            continue
            
        # Detailed columns mapping:
        # 0: MatchID, 1: Filter, 2: MJD (Time), 4: Mag, 5: CorrMag, 6: MagErr
        time_series = []
        for r in det_rows:
            time_series.append({
                'mjd': r[2],
                'mag': r[5] if r[5] is not None else r[4],  # Use CorrMag if available, else raw Mag
                'mag_err': r[6]
            })
            
        df = pd.DataFrame(time_series)
        # Sort by time
        df.sort_values(by='mjd', inplace=True)
        
        # Save to CSV file along with metadata in headers
        filename = f"hcv_{match_id}_{filt}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Write metadata as comments at the top of the file
        with open(filepath, 'w') as f:
            f.write(f"# MatchID: {match_id}\n")
            f.write(f"# Filter: {filt}\n")
            f.write(f"# RA: {var['RA']}\n")
            f.write(f"# Dec: {var['Dec']}\n")
            f.write(f"# NumLC: {var['NumLC']}\n")
            df.to_csv(f, index=False)
            
        print(f"  Saved to: {filepath}")
        downloaded_files.append(filepath)
        
    print(f"Completed! Downloaded {len(downloaded_files)} light curves to '{output_dir}'.")
    return downloaded_files

if __name__ == "__main__":
    download_hcv_variables(max_stars=15)

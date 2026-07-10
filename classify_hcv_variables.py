# =================================================================================
# CLASSIFY_HCV_VARIABLES.PY - Hubble M31 Variable Stars Classifier & Distance Modulus Estimator
# For a conceptual explanation of how this fits into the project, see:
# walkthrough_guide.md (at the repository or workspace root)
# =================================================================================
import os
import glob
import numpy as np
import pandas as pd
import torch
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from src.model import LightCurveCNN

CLASS_NAMES = {
    0: "Cepheid Variable",
    1: "RR Lyrae",
    2: "Eclipsing Binary",
    3: "Long-Period Variable (LPV)",
    4: "Non-Variable / Noise"
}

def load_hcv_star(file_path):
    """
    Parses metadata comments and reads MJD/Mag columns from HCV CSV file.
    """
    metadata = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                parts = line[1:].strip().split(":")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    val = ":".join(parts[1:]).strip()
                    metadata[key] = val
            else:
                break
                
    # Load time series
    df = pd.read_csv(file_path, comment='#')
    times = df['mjd'].values.astype(np.float32)
    mags = df['mag'].values.astype(np.float32)
    
    # Clean NaNs
    valid = np.isfinite(times) & np.isfinite(mags)
    times = times[valid]
    mags = mags[valid]
    
    return metadata, times, mags

def classify_and_map_hcv(data_dir="data/andromeda_real_hcv", model_weights="models/star_classifier.pth", seq_len=100):
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found at {model_weights}")
        return
        
    catalog_path = "data/hcv_spatial_catalog.csv"
    map_plot_path = "hcv_stars_spatial_map.png"
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightCurveCNN(num_classes=5)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()
    
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    files.sort()
    
    if not files:
        print(f"No HCV CSV light curves found in '{data_dir}'. Please run download_andromeda_hcv.py first.")
        return
        
    results = []
    plot_ra = []
    plot_dec = []
    plot_classes = []
    plot_labels = []
    
    print(f"Starting classification and distance modulus estimation on {len(files)} Hubble variable stars...")
    
    for file_path in files:
        basename = os.path.basename(file_path)
        try:
            metadata, times, mags = load_hcv_star(file_path)
            
            ra = float(metadata.get("RA", 0.0))
            dec = float(metadata.get("Dec", 0.0))
            match_id = metadata.get("MatchID", "Unknown")
            filt = metadata.get("Filter", "Unknown")
            
            if len(times) < 5:
                print(f"  Skipping {basename}: not enough valid data points (got {len(times)}).")
                continue
                
            # Apparent magnitude of the star
            apparent_mag = np.mean(mags)
            
            # Find period
            frequency, power = LombScargle(times, mags).autopower(minimum_frequency=0.01, maximum_frequency=10.0)
            best_period = 1.0 / frequency[np.argmax(power)]
            
            # Phase fold and interpolate to 100 points
            phases = (times / best_period) % 1.0
            sort_idx = np.argsort(phases)
            grid_phases = np.linspace(0.0, 1.0, seq_len)
            interpolated_mags = np.interp(grid_phases, phases[sort_idx], mags[sort_idx], period=1.0)
            
            # Normalise
            min_v = interpolated_mags.min()
            max_v = interpolated_mags.max()
            normed_mags = (interpolated_mags - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(interpolated_mags)
            
            # Model inference
            input_tensor = torch.tensor(normed_mags, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                
            pred_idx = np.argmax(probs)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100
            
            # Distance modulus calculation
            abs_mag = "N/A"
            distance_ly = "N/A"
            
            if pred_class == "Cepheid Variable" and best_period > 0:
                # Standard Classical Cepheid PL relation: M_v = -2.43 * (log10(P) - 1.0) - 4.05
                abs_mag = -2.43 * (np.log10(best_period) - 1.0) - 4.05
            elif pred_class == "RR Lyrae":
                # Standard RR Lyrae absolute magnitude M_v ~ +0.6
                abs_mag = 0.6
                
            if abs_mag != "N/A":
                # distance in parsecs d = 10^((m - M + 5) / 5)
                # Note: F814W filter is very close to V-band, so we use it as a proxy for apparent magnitude
                distance_pc = 10**((apparent_mag - abs_mag + 5) / 5)
                distance_ly = distance_pc * 3.26156
                
            # Append results
            results.append({
                "match_id": match_id,
                "filter": filt,
                "ra": ra,
                "dec": dec,
                "class": pred_class,
                "confidence": f"{confidence:.2f}%",
                "period": f"{best_period:.4f}",
                "apparent_mag": f"{apparent_mag:.3f}",
                "absolute_mag": f"{abs_mag:.3f}" if isinstance(abs_mag, float) else abs_mag,
                "distance_ly": f"{distance_ly:.1f}" if isinstance(distance_ly, float) else distance_ly
            })
            
            plot_ra.append(ra)
            plot_dec.append(dec)
            plot_classes.append(pred_idx)
            plot_labels.append(pred_class)
            
            dist_str = f"{distance_ly:.1f} ly" if isinstance(distance_ly, float) else "N/A"
            print(f"MatchID {match_id}: Class={pred_class} ({confidence:.1f}%), Period={best_period:.2f}d, Dist={dist_str}")
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            
    # Save CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(catalog_path, index=False)
    print(f"HCV spatial catalog saved successfully to: {catalog_path}")
    
    # Plot Spatial Distribution Map
    if plot_ra:
        plt.figure(figsize=(8, 6))
        
        # Color palette for classes
        colors = {
            0: "#ff4d4d",  # Cepheid (Red)
            1: "#ffb3d9",  # RR Lyrae (Pink)
            2: "#33ccff",  # Eclipsing Binary (Light Blue)
            3: "#ffcc66",  # LPV (Orange)
            4: "#99ff99"   # Non-Variable/Noise (Green)
        }
        
        markers = {
            0: "o",  # Cepheid
            1: "s",  # RR Lyrae
            2: "^",  # Eclipsing Binary
            3: "D",  # LPV
            4: "x"   # Noise
        }
        
        # Plot each class group separately for legend matching
        unique_classes = np.unique(plot_classes)
        for cl in unique_classes:
            indices = [i for i, c in enumerate(plot_classes) if c == cl]
            plt.scatter(
                [plot_ra[i] for i in indices],
                [plot_dec[i] for i in indices],
                color=colors[cl],
                marker=markers[cl],
                label=CLASS_NAMES[cl],
                s=80,
                edgecolors="black" if markers[cl] != "x" else None,
                alpha=0.85
            )
            
        # Draw M31 center target coordinate marker
        m31_ra, m31_dec = 10.68471, 41.268749
        plt.scatter(
            m31_ra, m31_dec, 
            color="black", 
            marker="P", 
            s=150, 
            label="M31 Center Pointer"
        )
        
        plt.xlabel("Right Ascension (RA in degrees)")
        plt.ylabel("Declination (Dec in degrees)")
        plt.title("Spatial Map of Hubble Variable Stars in Andromeda (M31)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(map_plot_path, dpi=150)
        plt.close()
        print(f"HCV spatial map image saved successfully to: {map_plot_path}")

if __name__ == "__main__":
    classify_and_map_hcv()

import os
import glob
import numpy as np
import torch
from astropy.table import Table
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from src.model import LightCurveCNN

# Dict mapping class indices to human-readable names
CLASS_NAMES = {
    0: "Cepheid Variable",
    1: "RR Lyrae",
    2: "Eclipsing Binary",
    3: "Long-Period Variable (LPV)",
    4: "Non-Variable / Noise"
}

def load_star_data(file_path):
    """
    Robustly loads light curve data from CSV or FITS format.
    Automatically finds the time and magnitude/flux columns, 
    converts flux to proxy magnitude, and cleans NaN/Inf values.
    """
    if file_path.endswith(".fits"):
        table = Table.read(file_path)
    else:
        # Load CSV using astropy Table for unified column matching
        table = Table.read(file_path, format="ascii.csv")
        
    colnames_lower = [c.lower() for c in table.colnames]
    
    # 1. Resolve Time Column
    time_col = None
    for candidate in ["time", "time_mjd", "mjd", "bjd"]:
        if candidate in colnames_lower:
            time_col = table.colnames[colnames_lower.index(candidate)]
            break
    if not time_col:
        # Fallback to first column
        time_col = table.colnames[0]
        
    # 2. Resolve Magnitude or Flux Column
    mag_col = None
    is_flux = False
    
    # Check for magnitude candidates first
    for candidate in ["mag", "magnitude", "magnitude_val"]:
        if candidate in colnames_lower:
            mag_col = table.colnames[colnames_lower.index(candidate)]
            break
            
    # Check for flux candidates if no mag column found
    if not mag_col:
        for candidate in ["pdcsap_flux", "sap_flux", "flux", "flux_val"]:
            if candidate in colnames_lower:
                mag_col = table.colnames[colnames_lower.index(candidate)]
                is_flux = True
                break
                
    if not mag_col:
        # Fallback to second column
        mag_col = table.colnames[1]
        
    times = np.array(table[time_col], dtype=np.float32)
    values = np.array(table[mag_col], dtype=np.float32)
    
    # 3. Clean NaN / Inf values (common in TESS/Kepler FITS data)
    valid_mask = np.isfinite(times) & np.isfinite(values)
    times = times[valid_mask]
    values = values[valid_mask]
    
    if len(times) < 10:
        raise ValueError(f"File {file_path} does not have enough valid data points (got {len(times)}).")
        
    # Sort chronologically
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]
    
    # If flux was used, convert to magnitude proxy (-flux) so a drop in flux represents
    # a drop in brightness (higher magnitude), matching the mock training distribution
    if is_flux:
        mags = -values
    else:
        mags = values
        
    return times, mags, time_col, mag_col, is_flux

def classify_curves(data_dir="data/real_light_curves", model_weights="models/star_classifier.pth", seq_len=100):
    # Verify model weights
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found at {model_weights}")
        return
        
    # Setup output paths
    catalog_path = "data/classified_stars_catalog.csv"
    plots_dir = "data/diagnostic_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightCurveCNN(num_classes=5)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()
    
    # Find all CSV and FITS files
    files = glob.glob(os.path.join(data_dir, "*.fits")) + glob.glob(os.path.join(data_dir, "*.csv"))
    files.sort()
    
    if not files:
        print(f"No light curve files found in '{data_dir}'.")
        return
        
    print(f"Found {len(files)} files to classify. Beginning inference...")
    
    results = []
    
    for file_path in files:
        basename = os.path.basename(file_path)
        try:
            times, mags, time_col, mag_col, is_flux = load_star_data(file_path)
            
            # Lomb-Scargle period detection
            frequency, power = LombScargle(times, mags).autopower(minimum_frequency=0.01, maximum_frequency=10.0)
            best_freq = frequency[np.argmax(power)]
            best_period = 1.0 / best_freq
            
            # Phase fold
            phases = (times / best_period) % 1.0
            sort_indices = np.argsort(phases)
            sorted_phases = phases[sort_indices]
            sorted_mags = mags[sort_indices]
            
            # Interpolate to grid
            grid_phases = np.linspace(0.0, 1.0, seq_len)
            interpolated_mags = np.interp(grid_phases, sorted_phases, sorted_mags, period=1.0)
            
            # Min-max normalization
            min_v = interpolated_mags.min()
            max_v = interpolated_mags.max()
            if max_v > min_v:
                normed_mags = (interpolated_mags - min_v) / (max_v - min_v)
            else:
                normed_mags = np.zeros_like(interpolated_mags)
                
            # Predict
            input_tensor = torch.tensor(normed_mags, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                
            pred_idx = np.argmax(probs)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100
            
            # Record result
            results.append({
                "filename": basename,
                "predicted_class": pred_class,
                "confidence": f"{confidence:.2f}%",
                "period": f"{best_period:.4f}"
            })
            
            print(f"Classified {basename} -> {pred_class} ({confidence:.2f}%) with Period: {best_period:.4f}")
            
            # Generate Diagnostic Plot
            fig, axs = plt.subplots(3, 1, figsize=(8, 10))
            
            # Panel 1: Raw points
            axs[0].scatter(times, mags, color='blue', s=2, alpha=0.5)
            axs[0].set_title(f"Raw Curve: {basename} (Col: {mag_col})")
            axs[0].set_ylabel("Proxy Magnitude" if is_flux else "Magnitude")
            axs[0].invert_yaxis()
            axs[0].grid(True, alpha=0.3)
            
            # Panel 2: Periodogram
            axs[1].plot(1.0 / frequency, power, color='purple', linewidth=1)
            axs[1].axvline(best_period, color='red', linestyle='--', label=f'Best Period: {best_period:.4f}')
            axs[1].set_title("Lomb-Scargle Periodogram")
            axs[1].set_xlabel("Period")
            axs[1].set_ylabel("Power")
            axs[1].set_xscale('log')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            
            # Panel 3: Phase Folded Curve
            axs[2].scatter(phases, mags, color='gray', s=2, alpha=0.4, label='Raw Points')
            axs[2].plot(grid_phases, interpolated_mags, color='red', linewidth=1.5, label='Interpolated Profile')
            axs[2].set_title(f"Phase Folded (Class: {pred_class} | Conf: {confidence:.1f}%)")
            axs[2].set_xlabel("Phase")
            axs[2].set_ylabel("Proxy Magnitude" if is_flux else "Magnitude")
            axs[2].invert_yaxis()
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f"{os.path.splitext(basename)[0]}_diagnostic.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Error classifying {basename}: {e}")
            results.append({
                "filename": basename,
                "predicted_class": "ERROR",
                "confidence": "0.00%",
                "period": "0.0000"
            })
            
    # Write output catalog
    with open(catalog_path, 'w') as f:
        f.write("filename,predicted_class,confidence,identified_period\n")
        for res in results:
            f.write(f"{res['filename']},{res['predicted_class']},{res['confidence']},{res['period']}\n")
            
    print(f"\nCompleted catalog generation! Saved to: {catalog_path}")

if __name__ == "__main__":
    classify_curves()

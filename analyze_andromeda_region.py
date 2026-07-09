import os
import glob
import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from src.model import LightCurveCNN

# Dict mapping class indices to human-readable names
CLASS_NAMES = {
    0: "Cepheid Variable",
    1: "RR Lyrae",
    2: "Eclipsing Binary",
    3: "Long-Period Variable (LPV)"
}

def analyze_andromeda_stars(data_dir="data/real_light_curves", model_weights="models/star_classifier.pth", seq_len=100):
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found at {model_weights}")
        return
        
    catalog_path = "data/andromeda_spatial_catalog.csv"
    map_plot_path = "andromeda_stars_spatial_map.png"
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightCurveCNN(num_classes=4)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()
    
    files = glob.glob(os.path.join(data_dir, "*.fits"))
    files.sort()
    
    if not files:
        print(f"No FITS light curves found in '{data_dir}'. Please run download_light_curves.py first.")
        return
        
    results = []
    
    # Lists to store values for the final spatial plot
    plot_ra = []
    plot_dec = []
    plot_classes = []
    plot_labels = []
    
    print(f"Starting spatial mapping and distance analysis on {len(files)} files...")
    
    for file_path in files:
        basename = os.path.basename(file_path)
        try:
            # 1. Read Coordinates and Catalog Magnitude from FITS Header
            hdul = fits.open(file_path)
            header = hdul[0].header
            
            ra = float(header.get("RA_OBJ", 0.0))
            dec = float(header.get("DEC_OBJ", 0.0))
            tess_mag = float(header.get("TESSMAG", 15.0))
            hdul.close()
            
            # 2. Load Light Curve Table data
            table = Table.read(file_path)
            colnames_lower = [c.lower() for c in table.colnames]
            
            # Match Time
            time_col = None
            for c in ["time", "time_mjd", "mjd", "bjd"]:
                if c in colnames_lower:
                    time_col = table.colnames[colnames_lower.index(c)]
                    break
            if not time_col:
                time_col = table.colnames[0]
                
            # Match Magnitude/Flux
            mag_col = None
            is_flux = False
            for c in ["mag", "magnitude", "pdcsap_flux", "sap_flux", "flux"]:
                if c in colnames_lower:
                    mag_col = table.colnames[colnames_lower.index(c)]
                    if "flux" in c:
                        is_flux = True
                    break
            if not mag_col:
                mag_col = table.colnames[1]
                
            times = np.array(table[time_col], dtype=np.float32)
            values = np.array(table[mag_col], dtype=np.float32)
            
            # Clean data
            valid = np.isfinite(times) & np.isfinite(values)
            times = times[valid]
            values = values[valid]
            
            if len(times) < 10:
                continue
                
            # Convert flux to magnitude proxy if necessary
            mags = -values if is_flux else values
            
            # 3. Calculate Pulsation Period
            frequency, power = LombScargle(times, mags).autopower(minimum_frequency=0.01, maximum_frequency=10.0)
            best_period = 1.0 / frequency[np.argmax(power)]
            
            # 4. Phase fold and interpolate to 100 points
            phases = (times / best_period) % 1.0
            sort_idx = np.argsort(phases)
            grid_phases = np.linspace(0.0, 1.0, seq_len)
            interpolated_mags = np.interp(grid_phases, phases[sort_idx], mags[sort_idx], period=1.0)
            
            # Min-max normalization
            min_v = interpolated_mags.min()
            max_v = interpolated_mags.max()
            normed_mags = (interpolated_mags - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(interpolated_mags)
            
            # 5. Model Inference
            input_tensor = torch.tensor(normed_mags, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                
            pred_idx = np.argmax(probs)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100
            
            # 6. Distance modulus calculation (only for Cepheid & RR Lyrae)
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
                distance_pc = 10**((tess_mag - abs_mag + 5) / 5)
                # Convert parsecs to light-years
                distance_ly = distance_pc * 3.26156
                
            # Append results
            results.append({
                "filename": basename,
                "ra": ra,
                "dec": dec,
                "class": pred_class,
                "confidence": f"{confidence:.2f}%",
                "period": f"{best_period:.4f}",
                "apparent_mag": f"{tess_mag:.3f}",
                "absolute_mag": f"{abs_mag:.3f}" if isinstance(abs_mag, float) else abs_mag,
                "distance_ly": f"{distance_ly:.1f}" if isinstance(distance_ly, float) else distance_ly
            })
            
            plot_ra.append(ra)
            plot_dec.append(dec)
            plot_classes.append(pred_idx)
            plot_labels.append(pred_class)
            
            print(f"Processed {basename}: Class={pred_class}, Dist={distance_ly} ly")
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            
    # Save CSV
    with open(catalog_path, 'w') as f:
        f.write("filename,ra,dec,predicted_class,confidence,period_days,apparent_mag,absolute_mag,distance_ly\n")
        for res in results:
            f.write(f"{res['filename']},{res['ra']},{res['dec']},{res['class']},{res['confidence']},{res['period']},{res['apparent_mag']},{res['absolute_mag']},{res['distance_ly']}\n")
            
    print(f"Spatial catalog saved successfully to: {catalog_path}")
    
    # 7. Plot Spatial Distribution Map
    if plot_ra:
        plt.figure(figsize=(8, 6))
        
        # Color palette for classes
        colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
        markers = {0: 'o', 1: 's', 2: '^', 3: 'd'}
        
        # Plot each star category individually for a clean legend
        for class_idx in range(4):
            indices = [i for i, c in enumerate(plot_classes) if c == class_idx]
            if indices:
                xs = [plot_ra[i] for i in indices]
                ys = [plot_dec[i] for i in indices]
                plt.scatter(
                    xs, ys, 
                    color=colors[class_idx], 
                    marker=markers[class_idx],
                    s=100, 
                    label=CLASS_NAMES[class_idx],
                    edgecolors='black',
                    alpha=0.8
                )
                
        plt.title("Spatial Map of Classified Stars in Andromeda region")
        plt.xlabel("Right Ascension (RA, degrees)")
        plt.ylabel("Declination (Dec, degrees)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Add background metadata note about foreground stars
        plt.figtext(
            0.5, 0.01, 
            "*Note: Calculated distances map foreground stars in the Milky Way along the line of sight to Andromeda.",
            wrap=True, horizontalalignment='center', fontsize=8, style='italic'
        )
        
        plt.tight_layout()
        plt.savefig(map_plot_path, dpi=300)
        plt.close()
        print(f"Spatial map image saved successfully to: {map_plot_path}")
    else:
        print("No valid stars with spatial coordinates processed.")

if __name__ == "__main__":
    analyze_andromeda_stars()

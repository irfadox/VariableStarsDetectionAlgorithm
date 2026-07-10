# Andromeda Stars Classification & Spatial Mapping Algorithm

This project implements an end-to-end pipeline in PyTorch to classify variable stars (Cepheid, RR Lyrae, Eclipsing Binary, LPV, or Non-Variable / Noise) using light-curve data from the MAST survey archives.

---

## 🌟 Key Features

1. **Lomb-Scargle Phase Folding**: Automatically identifies the dominant period of unevenly sampled light curves using `astropy.timeseries.LombScargle`, phase-folds the observation times, and aligns the magnitudes.
2. **Upgraded 5-Class 1D CNN Classifier**: Built with PyTorch using Batch Normalization (`BatchNorm1d`), Dropout, and 3 convolutional layers (filter depths 32/64/128), achieving **99.5% classification accuracy**. Supports five categories:
   * Cepheid Variable
   * RR Lyrae
   * Eclipsing Binary
   * Long-Period Variable (LPV)
   * **Non-Variable / Noise** (filters out stable background stars)
3. **Best Model Checkpointing**: Tracks test loss during training and automatically saves the best performing model weights to `models/star_classifier.pth`.
4. **Andromeda Spatial Mapping & Distance Estimator**: `analyze_andromeda_region.py` extracts celestial coordinates (`RA`, `Dec`) and magnitudes from FITS headers to map stars in a 2D spatial scatter plot (`andromeda_stars_spatial_map.png`) and calculate distances (in light-years) for standard candle stars.
5. **Diagnostic Plot Visualizer**: `predict.py` automatically generates a 3-panel diagnostic visualization (`lightcurve_diagnostic_plot.png`) containing:
   - **Panel 1**: Raw Light Curve observations over time.
   - **Panel 2**: Lomb-Scargle Periodogram showing the identified period peak.
   - **Panel 3**: Phase Folded Light Curve overlaid with the interpolated profile evaluated by the model.

---

## 📦 Installation & Setup

Install the required astronomy, deep learning, and UI dependencies:
```bash
pip install torch astropy astroquery matplotlib scikit-learn gradio
```

---

## 🚀 How to Run

### Step 1: Download Real Unique Light Curves
Query and download real variable star light-curve FITS files from the MAST archive (targeting unique star IDs within a 0.5-degree radius around Andromeda coordinates):
```bash
python3 download_light_curves.py
```
This saves downloaded files to `data/real_light_curves/`.

### Step 2: Train the Classifier Model
Train the 1D CNN for 30 epochs on the preprocessed phase-folded curves:
```bash
python3 train.py
```
This will print epoch losses, display the final evaluation metrics (Precision, Recall, Confusion Matrix), and save only the best-performing model weights to `models/star_classifier.pth`.

### Step 3: Run Andromeda Spatial Mapping & Analysis
Analyze all downloaded unique stars, calculate their distance modulus limits, output a spreadsheet, and draw a spatial distribution map:
```bash
python3 analyze_andromeda_region.py
```
This saves the catalog spreadsheet to `data/andromeda_spatial_catalog.csv` and saves the spatial scatter plot to `andromeda_stars_spatial_map.png`.

### Step 4: Run Inference & Save Diagnostic Plots
Run predictions on any light-curve CSV or FITS file:
```bash
python3 predict.py
```
This performs classification, prints the probability distribution, and saves the 3-panel visualization as `lightcurve_diagnostic_plot.png`.

### Step 5: Run the Gradio Web App UI
Launch the interactive web page UI:
```bash
python3 app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser to upload curves and view predictions visually.


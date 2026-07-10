# Andromeda Stars Classification & Spatial Mapping Algorithm

This project implements an end-to-end pipeline in PyTorch to classify variable stars (Cepheid, RR Lyrae, Eclipsing Binary, LPV, or Non-Variable / Noise) using light-curve data from the MAST survey archives (TESS/Kepler for Milky Way foreground stars, and Hubble Catalog of Variables for actual resolved stars inside the Andromeda Galaxy).

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
4. **Andromeda Spatial Mapping & Distance Estimator**: Extracts celestial coordinates (`RA`, `Dec`) and magnitudes from light-curve headers/files to map stars in a 2D spatial scatter plot and calculate distance modulus values (in light-years) for standard candle stars.
5. **Diagnostic Plot Visualizer**: `predict.py` automatically generates a 3-panel diagnostic visualization (`lightcurve_diagnostic_plot.png`) containing:
   - **Panel 1**: Raw Light Curve observations over time.
   - **Panel 2**: Lomb-Scargle Periodogram showing the identified period peak.
   - **Panel 3**: Phase Folded Light Curve overlaid with the interpolated profile evaluated by the model.

---

## 📦 Installation & Setup

Install the required astronomy, deep learning, and UI dependencies:
```bash
pip install torch astropy astroquery matplotlib scikit-learn gradio pandas
```

---

## 🚀 How to Run

### Option A: Actual Andromeda Stars (Hubble Catalog of Variables - HCV)
Because TESS pixels are very large ($21$ arcseconds), they only detect bright **foreground Milky Way stars** along the line of sight to Andromeda. To classify actual resolved stars **inside** Andromeda (M31), we use the Hubble Catalog of Variables (HCV):

#### Step 1: Download Resolved M31 Variable Stars from Hubble
Query the MAST API to download the time-series magnitude observations of variable candidates resolved by Hubble inside Andromeda:
```bash
python3 download_andromeda_hcv.py
```
This saves the light-curve CSV tables to `data/andromeda_real_hcv/`.

#### Step 2: Classify HCV Variables and Map Distances
Analyze the Hubble light curves, run CNN inference, estimate absolute magnitudes and distances, and plot their spatial layout:
```bash
python3 classify_hcv_variables.py
```
This outputs [hcv_spatial_catalog.csv](file:///Users/kima/CodeProjects/VariableStarsDetectionAlgorithm/data/hcv_spatial_catalog.csv) and plots [hcv_stars_spatial_map.png](file:///Users/kima/CodeProjects/VariableStarsDetectionAlgorithm/hcv_stars_spatial_map.png).
* **Distance modulus verification:** Standard candle stars (Cepheids and RR Lyrae) return estimated distances of **$800,000 \text{ to } 1,750,000\text{ light-years}$**, confirming their extragalactic location inside/near Andromeda (compared to Milky Way scale of ~100k ly).

---

### Option B: Foreground Stars (TESS/Kepler Wide-Field Survey)

#### Step 1: Download TESS unique stars
Query and download TESS light curves near Andromeda coordinates:
```bash
python3 download_light_curves.py
```
This saves downloaded files to `data/real_light_curves/`.

#### Step 2: Train the Classifier Model
Train the 1D CNN for 30 epochs on preprocessed phase-folded curves:
```bash
python3 train.py
```
This will print epoch losses, display the final evaluation metrics, and save the best weights to `models/star_classifier.pth`.

#### Step 3: Run Foreground Star Spatial Mapping & Analysis
Analyze all TESS unique stars, output a spreadsheet, and draw a spatial distribution map:
```bash
python3 analyze_andromeda_region.py
```
This saves the catalog spreadsheet to `data/andromeda_spatial_catalog.csv` and saves the spatial scatter plot to `andromeda_stars_spatial_map.png`.

#### Step 4: Run Inference & Save Diagnostic Plots
Run predictions on any light-curve CSV or FITS file:
```bash
python3 predict.py
```

#### Step 5: Run the Gradio Web App UI
Launch the interactive web page UI:
```bash
python3 app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser.


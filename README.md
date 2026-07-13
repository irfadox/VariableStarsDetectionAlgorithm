# Andromeda Variable Stars Classification Algorithm

This project implements an end-to-end deep learning pipeline in PyTorch to **detect and classify variable stars inside the Andromeda Galaxy (M31)** using light-curve data from the **Hubble Catalog of Variables (HCV)** — real, high-resolution observations made by the Hubble Space Telescope.

> **Why Hubble and not TESS?** Wide-field surveys like TESS have pixels spanning 21 arcseconds of sky — large enough to blend thousands of M31 stars together. Only Hubble's sharp resolution can isolate and measure individual stars inside Andromeda. See [walkthrough_guide.md](walkthrough_guide.md) for a full explanation.

---

## 🌟 Key Features

1. **Lomb-Scargle Phase Folding**: Automatically identifies the dominant period of unevenly-sampled Hubble light curves using `astropy.timeseries.LombScargle`, phase-folds the observations, and aligns the magnitude profile.
2. **5-Class 1D CNN Classifier**: Built with PyTorch using Batch Normalization, Dropout, and 3 convolutional layers (32/64/128 filters), achieving **99.5% classification accuracy** on five categories:
   - Cepheid Variable
   - RR Lyrae
   - Eclipsing Binary
   - Long-Period Variable (LPV)
   - Non-Variable / Noise
3. **Synthetic & Real-World Labeled Training (OGLE)**: Supports training either on fast synthetic light curves or on **real expert-labeled time-series photometry** downloaded from the Optical Gravitational Lensing Experiment (OGLE) database.
4. **Real-Data Prototype Validation**: Includes a validation script that downloads known prototype stars (like RR Lyrae and Algol) from TESS using NASA's `lightkurve` API to test how well the CNN generalizes to real photometry.
5. **Andromeda Spatial Map & Distance Estimator**: Uses standard candle period-luminosity relations (Cepheid PL, RR Lyrae absolute magnitude) to calculate distances in light-years, and plots a 2D spatial map of classified stars overlaid on M31's coordinates.
6. **Diagnostic Visualizer**: `predict.py` generates a 3-panel diagnostic plot (`lightcurve_diagnostic_plot.png`) showing raw light curves, Lomb-Scargle power spectra, and phase-folded profiles.

---

## 📦 Installation & Setup

```bash
pip install torch astropy astroquery matplotlib scikit-learn gradio pandas requests lightkurve
```

---

## 🚀 How to Run

### Option A: Standard Synthetic Training & Hubble Inference

#### Step 1: Train the Classifier Model (Synthetic)
Train the 1D CNN for 30 epochs on synthetically generated phase-folded light curves:
```bash
python3 train.py
```

#### Step 2: Download M31 Variable Stars from Hubble
Query MAST for variable candidates resolved by Hubble inside Andromeda:
```bash
python3 download_andromeda_hcv.py
```

#### Step 3: Classify HCV Variables and Map Distances
Classify Hubble stars, estimate distances via distance modulus, and plot a spatial map:
```bash
python3 classify_hcv_variables.py
```

---

### Option B: Real Labeled Data Training & Validation (OGLE + TESS)

To train on real-world stellar observations and validate against known astronomical prototypes:

#### Step 1: Download Real Labeled Light Curves from OGLE
Download 300 real time-series photometry records per class from the OGLE database:
```bash
python3 download_ogle_training_data.py
```
This saves raw light curve records under `data/ogle_training/`.

#### Step 2: Retrain the CNN on Real Data
Process the OGLE light curves and retrain the model weights:
```bash
python3 train_on_ogle.py
```
This overwrites the saved model in `models/star_classifier.pth` with weights trained on real stellar physics.

#### Step 3: Validate Against Known Prototype Stars
Download TESS light curves for famous benchmark stars (e.g. RR Lyrae, Zeta Geminorum, Algol) and evaluate the model's accuracy on real data:
```bash
python3 validate_on_known_stars.py
```
This outputs classification reports and generates diagnostic plots in `data/validation_plots/`.

---

### Step 4: Run Inference on a Single Star
Predict the class of any light-curve file (CSV or FITS) and generate a diagnostic plot:
```bash
python3 predict.py
```

### Step 5: Launch the Interactive Web App
Start the Gradio UI to upload and classify stars through a browser:
```bash
python3 app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 📁 Project Structure

```
.
├── train.py                    # Trains the CNN model on synthetic light curves
├── train_on_ogle.py            # Trains the CNN on real OGLE light curves
├── download_ogle_training_data.py # Downloads real labeled light curves from OGLE
├── validate_on_known_stars.py  # Evaluates the model on TESS benchmark prototype stars
├── predict.py                  # Classifies a single star file and generates diagnostic plots
├── app.py                      # Gradio web app for interactive classification
├── download_andromeda_hcv.py   # Downloads Hubble HCV variable star light curves for M31
├── classify_hcv_variables.py   # Classifies HCV stars, estimates distances, creates spatial map
├── src/
│   ├── model.py                # 1D CNN architecture definition
│   ├── data_setup.py           # Synthetic dataset generator and Dataset setup
│   └── engine.py               # Training and evaluation epoch loops
├── models/
│   └── star_classifier.pth     # Saved best model weights
├── data/
│   ├── andromeda_real_hcv/     # Downloaded Hubble HCV light curves (git-ignored)
│   ├── ogle_training/          # Labeled light curves from the OGLE database (git-ignored)
│   ├── validation_plots/       # Saved diagnostic validation plots (git-ignored)
│   └── hcv_spatial_catalog.csv # Classification results catalog
└── requirements.txt
```

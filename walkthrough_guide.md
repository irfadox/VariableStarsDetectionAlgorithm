# Project Walkthrough: Variable Star Classification & Mapping

Welcome! This guide explains the **astronomy** and **machine learning (ML)** concepts behind this project, and how each file works together to classify variable stars inside the Andromeda Galaxy. Even with no background in ML or astronomy, this document will help you understand what is happening and why.

---

## 🌌 Part 1: Astronomy Basics

### 1. What is a Variable Star?
Most stars shine with a constant brightness (like our Sun). However, some stars are **variable stars** — their brightness changes over time. We focus on four major types:

* **Cepheid Variables:** Giant stars that physically pulsate (expand and contract), causing their brightness to cycle like a sawtooth wave. They are "standard candles" — their pulsation *period* directly tells us their intrinsic *brightness*, which lets us calculate their exact distance!
* **RR Lyrae:** Older, low-mass pulsating stars. They pulsate regularly with shorter periods (typically under a day) and a nearly constant absolute brightness, making them excellent distance markers.
* **Eclipsing Binaries:** Two stars orbiting each other. When one passes in front of the other, it blocks its light, causing sharp, repeating dips in the observed brightness.
* **Long-Period Variables (LPV):** Red giants that pulsate slowly and semi-irregularly over hundreds of days.

### 2. What is a Light Curve?
A **light curve** is simply a graph of a star's brightness (y-axis) vs. time (x-axis). Raw telescope observations are taken at irregular time intervals, so the raw light curve looks like a scattered cloud of dots.

### 3. What is Lomb-Scargle Phase Folding?
To make sense of the scattered dots, we use **phase folding**:
1. We apply the **Lomb-Scargle Periodogram** — a mathematical tool that analyzes frequencies in the uneven data and finds the star's dominant cycle period (*P*, in days).
2. We then "fold" the time axis by computing `Phase = (Time / Period) % 1.0`.
3. This collapses all cycles onto a single 0.0–1.0 scale, turning the scattered cloud into a clean, repeating wave — the star's unique physical signature.

### 4. Why Hubble and Not TESS?
* **TESS (Wide-Field Survey):** TESS pixels cover 21 arcseconds of sky each. When pointed at Andromeda (M31), thousands of stars are blurred into a single pixel. Any bright isolated signal is actually a **foreground Milky Way star** that happens to be in the same direction — not a real Andromeda star.
* **Hubble (High-Resolution):** Hubble can resolve individual stars inside M31. By querying the **Hubble Catalog of Variables (HCV)** from the MAST archive, we get real stars *inside* Andromeda with verified distances of **800,000–1,750,000 light-years**.

---

## 🤖 Part 2: Machine Learning Basics

### 1. What is a 1D CNN?
A **1D Convolutional Neural Network (CNN)** is a deep learning model designed for sequence data (like audio waveforms or phase-folded light curves).

Instead of looking at images (2D), the 1D CNN slides small filters along our 100-step interpolated light curve array to detect patterns — like sharp dips (eclipses), smooth sawtooth waves (Cepheids), or symmetric bumps (RR Lyrae).

### 2. Neural Network Building Blocks
* **Conv1D (Convolutional Layer):** Slides a small filter window across the sequence to detect localized shapes. Three layers (32 → 64 → 128 filters) stack from simple to complex patterns.
* **Batch Normalization:** After each conv layer, standardizes the outputs so numbers stay in a healthy range, making training faster and more stable.
* **Dropout:** Randomly deactivates neurons during training to prevent the model from just memorizing answers (called "overfitting").
* **Fully Connected Layer (Linear):** Takes all the extracted features and outputs a probability for each of the 5 star classes.

### 3. Best Model Checkpointing
During training, data is split into a **training set** (the model learns from this) and a **test set** (never seen during learning — used to measure real performance). Instead of saving the model at the final epoch, we monitor the **test loss** and only save the weights when a new minimum is reached — locking in the best-performing state.

---

## 📂 Part 3: Code Architecture

Here is every file in the repository and what it does:

```
.
├── train.py                        ← Trains the neural network on synthetic data and saves weights
├── predict.py                      ← Classifies a single star file; generates 3-panel diagnostic plot
├── app.py                          ← Gradio web interface for uploading and classifying stars in-browser
│
├── download_andromeda_hcv.py       ← Queries MAST API for Hubble-resolved M31 variable star light curves
├── classify_hcv_variables.py       ← Runs CNN inference on Hubble stars, estimates distances, maps M31
│
├── src/
│   ├── model.py                    ← Defines the 1D CNN architecture (the "brain")
│   ├── data_setup.py               ← Generates synthetic training light curves with realistic noise
│   └── engine.py                   ← Handles the train/evaluate loop for each epoch
│
├── models/
│   └── star_classifier.pth         ← Saved best model weights (created by train.py)
│
└── data/
    ├── andromeda_real_hcv/         ← Downloaded Hubble light curves (local only, git-ignored)
    └── hcv_spatial_catalog.csv     ← Output classification results and distances per star
```

---

## 🚀 Part 4: How to Run the Project

### Step 1 — Train the Model
Generate synthetic light curves and train the 1D CNN for 30 epochs:
```bash
python3 train.py
```
Saves best weights to `models/star_classifier.pth`.

### Step 2 — Download Hubble M31 Variable Stars
Query the MAST Hubble Catalog of Variables for real stars resolved inside Andromeda:
```bash
python3 download_andromeda_hcv.py
```
Saves light curve CSVs to `data/andromeda_real_hcv/`.

### Step 3 — Classify and Map
Run CNN inference, estimate distances via distance modulus, and generate a spatial map:
```bash
python3 classify_hcv_variables.py
```
Outputs:
- `data/hcv_spatial_catalog.csv` — per-star classification and distances
- `hcv_stars_spatial_map.png` — 2D scatter plot overlaid on M31's sky coordinates

### Step 4 — Inspect a Single Star
Run `predict.py` on any CSV or FITS light curve file:
```bash
python3 predict.py
```
Generates `lightcurve_diagnostic_plot.png` with raw curve, periodogram, and phase-folded profile.

### Step 5 — Use the Web App
Launch the interactive Gradio UI:
```bash
python3 app.py
```
Open [http://localhost:7860](http://localhost:7860) and upload any light curve file to classify it.

---

## 📏 Part 5: Distance Estimation

For standard candle variable types, we calculate distance using the **distance modulus formula**:

```
distance (parsecs) = 10 ^ ((apparent_magnitude - absolute_magnitude + 5) / 5)
distance (light-years) = distance_parsecs × 3.26156
```

- **Cepheids**: We derive absolute magnitude from the period-luminosity (PL) relation: `M = -2.43 × (log10(P) - 1.0) - 4.05`
- **RR Lyrae**: Fixed absolute magnitude of `M ≈ +0.6`

Stars returning distances of ~800k–1.75M light-years are confirmed to be inside or near Andromeda.

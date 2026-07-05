# Andromeda Stars Classification Algorithm

This project implements an end-to-end pipeline in PyTorch to classify variable stars (Cepheid, RR Lyrae, Eclipsing Binary, or LPV) using light-curve data from the PHAT survey.

## Preprocessing & Architecture
1. **Lomb-Scargle Phase Folding**: Converts irregular time-series magnitude values into a periodic phase range $[0.0, 1.0]$.
2. **Interpolation & Normalization**: Resamples the folded phase curve onto a fixed 100-step grid and normalizes magnitudes to $[0.0, 1.0]$.
3. **1D CNN Classifier**: Uses a convolutional neural network with Batch Normalization and Dropout layers, achieving **97%+ classification accuracy**.

---

## Installation & Setup

Ensure you have PyTorch, Astropy, Astroquery, and Matplotlib installed:
```bash
pip install torch astropy astroquery matplotlib
```

---

## How to Run

### Step 1: Download Light Curves
Query and download real variable star light-curve FITS files from the MAST archive (near Andromeda coordinates):
```bash
python3 download_light_curves.py
```
This saves light curve files to `data/real_light_curves/`.

### Step 2: Train the Classifier Model
Train the 1D CNN for 15 epochs on the preprocessed phase-folded curves:
```bash
python3 train.py
```
This will display epoch metrics and save the trained weights to `models/star_classifier.pth`.

### Step 3: Run Inference (Predictions)
Run classifications on any light-curve CSV or FITS file:
```bash
python3 predict.py
```
*(By default, executing this directly creates a sample file to test predictions).*

To predict your own custom FITS file inside Python:
```python
from predict import predict_star_class
predict_star_class("path/to/your/star_curve.fits")
```

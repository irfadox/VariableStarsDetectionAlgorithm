# Variable Star Classifier — Andromeda (M31)

A deep learning pipeline that automatically **detects, classifies, and maps variable stars inside the Andromeda Galaxy** using Hubble Space Telescope photometry from the Hubble Catalog of Variables (HCV).

> 📖 **Detailed explanation of every design decision and line of code:** [walkthrough_guide.md](walkthrough_guide.md)

---

## What This Project Does

1. **Downloads** real Hubble light curves for variable stars in M31 from NASA's MAST archive
2. **Trains** a 1D Convolutional Neural Network on 1,000 expert-labeled light curves per class from the OGLE survey
3. **Classifies** each star into one of five categories using Lomb-Scargle period detection + CNN
4. **Estimates distances** using Cepheid Period-Luminosity relations and RR Lyrae standard candles
5. **Maps** all classified stars in sky coordinates overlaid on Andromeda's position

---

## Star Classes

| Class | Physical Mechanism | Typical Period |
|---|---|---|
| **Cepheid Variable** | Internal radial pulsation | 1–100 days |
| **RR Lyrae** | Short-period horizontal-branch pulsation | 0.2–1 day |
| **Eclipsing Binary** | Orbital geometry — one star blocks the other | 0.1–10 days |
| **Long-Period Variable (LPV)** | Thermal pulsations in giant red stars | 100–1000 days |
| **Non-Variable / Noise** | No periodicity | — |

---

## Why Hubble?

TESS pixels span 21 arcseconds of sky. At Andromeda's distance (2.5 million light-years), that's **thousands of stars per pixel**. Hubble's ACS camera at 0.05 arcseconds/pixel is 420× sharper and can resolve individual stars inside M31. There is no other way to classify individual variable stars in Andromeda.

---

## Model Performance

Trained on 1,000 real OGLE light curves per class (4,000 total) + 500 synthetic non-variable curves:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Cepheid | 85% | 85% | 85% |
| RR Lyrae | 84% | 79% | 81% |
| Eclipsing Binary | 86% | 90% | 88% |
| LPV | 83% | 85% | 84% |
| Non-Variable / Noise | 100% | 100% | 100% |
| **Overall Macro Avg** | **88%** | **88%** | **88%** |

**TESS prototype validation** (3/4 known stars correctly classified at >84% confidence):
- ✅ Zeta Geminorum → Cepheid (97.5%)
- ✅ RR Lyrae → RR Lyrae (84.2%)
- ✅ Algol (β Persei) → Eclipsing Binary (100%)
- ❌ R Lyrae → misclassified due to single-sector TESS aliasing (460-day period not detectable in 27-day window)

---

## Installation

```bash
pip install torch astropy astroquery matplotlib scikit-learn gradio pandas requests lightkurve
```

---

## Quick Start

### Option A — Train on Real OGLE Data (Recommended)

```bash
# 1. Download 1,000 labeled light curves per class from OGLE (~15 min)
python3 download_ogle_training_data.py

# 2. Train the CNN on real photometry (50 epochs, ~10 min on CPU)
python3 train_on_ogle.py

# 3. Validate against famous prototype stars via TESS
python3 validate_on_known_stars.py
```

### Option B — Synthetic Training + Hubble Inference

```bash
# 1. Train on synthetically generated light curves (fast, ~2 min)
python3 train.py

# 2. Download Hubble HCV variable stars from M31
python3 download_andromeda_hcv.py

# 3. Classify stars, estimate distances, and generate spatial map
python3 classify_hcv_variables.py
```

### Interactive Web App

```bash
python3 app.py
# Open http://localhost:7860 — upload any CSV/FITS light curve for classification
```

### Single-Star Inference

```bash
python3 predict.py
# Generates a 3-panel diagnostic plot: raw curve + Lomb-Scargle spectrum + phase fold
```

---

## Project Structure

```
.
├── train.py                       # Synthetic-data CNN training
├── train_on_ogle.py               # Real OGLE-data CNN training (recommended)
├── download_ogle_training_data.py # Downloads 1,000 labeled OGLE curves per class
├── validate_on_known_stars.py     # TESS prototype star validation (Algol, RR Lyrae, etc.)
├── download_andromeda_hcv.py      # Downloads Hubble HCV variable stars in M31
├── classify_hcv_variables.py      # Classifies + maps + distance-estimates M31 stars
├── predict.py                     # Single-star inference with diagnostic plot
├── app.py                         # Gradio web UI
│
├── src/
│   ├── model.py                   # LightCurveCNN: 3×Conv(32→64→128) + BN + Pool + Dropout
│   ├── data_setup.py              # Synthetic dataset generator
│   └── engine.py                  # train_epoch() and test_epoch() loops
│
├── models/
│   └── star_classifier.pth        # Saved best model weights (git-ignored if large)
│
├── data/
│   ├── andromeda_real_hcv/        # Downloaded Hubble light curves (git-ignored)
│   ├── ogle_training/             # OGLE labeled data: cepheid/ rrlyr/ eclipsing_binary/ lpv/
│   ├── validation_plots/          # Diagnostic PNGs from validate_on_known_stars.py
│   ├── validation_results.csv     # Per-star prototype validation table
│   └── hcv_spatial_catalog.csv   # M31 classification results with RA/Dec/distance
│
├── walkthrough_guide.md           # Full beginner-to-expert explanation of the entire codebase
├── requirements.txt
└── hcv_stars_spatial_map.png      # 2D sky map of classified Andromeda stars
```

---

## Technical Architecture

### Pipeline Summary

```
Raw light curve (time, magnitude)
        ↓
  Lomb-Scargle periodogram
  [astropy.timeseries.LombScargle]
        ↓
  Best period identified
        ↓
  Phase folding: phase = (t / P) mod 1
        ↓
  Linear interpolation → 100-point fixed profile
        ↓
  Min-max normalisation → [0, 1]
        ↓
  LightCurveCNN forward pass
  [Conv→BN→Pool→ReLU] × 3 → AdaptivePool → Dropout → Linear
        ↓
  Softmax → 5 class probabilities
        ↓
  Argmax → Classification
        ↓ (if Cepheid or RR Lyrae)
  Period-Luminosity / Standard Candle
        ↓
  Distance estimate (light-years)
```

### CNN Architecture Detail

```
Input:  (1, 1, 100)  — batch=1, channels=1, seq=100

Conv1d(1→32,  k=5, p=2)  → (1, 32, 100)
BatchNorm1d(32)
MaxPool1d(2)              → (1, 32, 50)
ReLU

Conv1d(32→64, k=5, p=2)  → (1, 64, 50)
BatchNorm1d(64)
MaxPool1d(2)              → (1, 64, 25)
ReLU

Conv1d(64→128, k=5, p=2) → (1, 128, 25)
BatchNorm1d(128)
MaxPool1d(2)              → (1, 128, 12)
ReLU

AdaptiveAvgPool1d(8)      → (1, 128, 8)
Flatten                   → (1, 1024)
Dropout(p=0.3)
Linear(1024→5)            → (1, 5)   [logits]
Softmax                   → [probabilities]
```

### Training Data Sources

| Class | Source | Catalog | N |
|---|---|---|---|
| Cepheid | OGLE-IV LMC fundamental-mode | `ogle4/OCVS/lmc/cep/phot/I/` | 1,000 |
| RR Lyrae | OGLE-IV Galactic Bulge RRab | `ogle4/OCVS/blg/rrlyr/phot/I/` | 1,000 |
| Eclipsing Binary | OGLE-IV LMC eclipsing | `ogle4/OCVS/lmc/ecl/phot/I/` | 1,000 |
| LPV | OGLE-III LMC Miras/SRVs | `ogle3/OIII-CVS/lmc/lpv/phot/I/` | 1,000 |
| Non-Variable | Synthetic Gaussian noise | Generated | 500 |

---

## Outputs

| File | Description |
|---|---|
| `models/star_classifier.pth` | Trained model weights |
| `data/hcv_spatial_catalog.csv` | Per-star CSV: class, confidence, period, distance |
| `hcv_stars_spatial_map.png` | Sky coordinate plot of all classified M31 stars |
| `data/validation_plots/*.png` | 3-panel diagnostic plots for each validation star |
| `data/validation_results.csv` | Prototype star validation summary table |

---

## References

- **OGLE Catalog**: Udalski et al. (1992–2015), [astrouw.edu.pl/ogle](https://www.astrouw.edu.pl/ogle/)
- **Hubble Catalog of Variables (HCV)**: Sokolovsky et al. (2017), [MAST HCV](https://archive.stsci.edu/hst/hcv.html)
- **Leavitt Law**: Leavitt & Pickering (1912), Harvard College Observatory Circular
- **lightkurve**: Lightkurve Collaboration (2018), [lightkurve.org](https://lightkurve.org)
- **PyTorch**: Paszke et al. (2019), NeurIPS

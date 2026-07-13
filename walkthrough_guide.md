# Complete Beginner's Walkthrough: Variable Star Classification & Andromeda Mapping

**Who this is for:** Anyone who is curious about this project but has no prior background in astronomy or machine learning. No equations, no jargon left unexplained. By the end, you should be able to look at every output file and plot in this project and know *exactly* what you're looking at and *why* it was done that way.

---

## 📖 Table of Contents
1. [What Is This Project Doing?](#1-what-is-this-project-doing)
2. [Astronomy Foundations](#2-astronomy-foundations)
3. [Why Hubble? The Resolution Problem](#3-why-hubble-the-resolution-problem)
4. [How We Download the Data](#4-how-we-download-the-data)
5. [Processing the Light Curves (Signal to Shape)](#5-processing-the-light-curves-signal-to-shape)
6. [Machine Learning: Teaching a Computer to Recognise Star Types](#6-machine-learning-teaching-a-computer-to-recognise-star-types)
7. [Reading the Outputs: The CSV Catalog](#7-reading-the-outputs-the-csv-catalog)
8. [Reading the Outputs: The Spatial Map Plot](#8-reading-the-outputs-the-spatial-map-plot)
9. [Reading the Outputs: The Diagnostic Plot](#9-reading-the-outputs-the-diagnostic-plot)
10. [Distance Estimation: How Far Away Are These Stars?](#10-distance-estimation-how-far-away-are-these-stars)
11. [Why Did We Choose Everything We Did?](#11-why-did-we-choose-everything-we-did)
12. [File-by-File Code Guide](#12-file-by-file-code-guide)

---

## 1. What Is This Project Doing?

The Andromeda Galaxy (also called M31) is our closest galactic neighbour — a spiral galaxy about 2.5 million light-years away. Inside it are hundreds of billions of stars.

This project answers the question: **can we automatically identify which types of variable stars (stars whose brightness changes over time) exist inside Andromeda, and calculate how far away they are — all from raw telescope data?**

To do that, we:
1. Download real Hubble Space Telescope brightness measurements for individual stars inside Andromeda.
2. Process those measurements into a clean waveform that represents the star's pulsation pattern.
3. Feed that waveform to a neural network (an AI) that we trained to recognise five star types.
4. Use the classification to estimate the distance to each star using physics equations.
5. Plot all of them on a sky map so you can see where they physically sit inside Andromeda.

To make the AI highly robust, we can train it in two ways:
* **Synthetically**: Generate simulated clean light curves to learn basic shapes.
* **On Real Data**: Download real, expert-labeled variable star light curves from the Optical Gravitational Lensing Experiment (**OGLE**) database.
We also validate our AI against real benchmark prototype stars (like RR Lyrae and Algol) using NASA's **TESS** observations.

---

## 2. Astronomy Foundations

### 2.1 What Is a Variable Star?

Imagine a lightbulb that steadily dims and brightens in a regular cycle. That is essentially what a variable star does — except the mechanism behind the flickering is driven by physics happening *inside* the star itself.

Our Sun is *not* a variable star. Its brightness is almost perfectly constant. Variable stars are special, and they come in different varieties, each with a different physical cause:

---

**Cepheid Variables** *(think: the giant golden heartbeat)*

A Cepheid is a supergiant star (far bigger than our Sun) that literally expands and contracts over and over — like a slow, rhythmic breathing cycle. As it expands it becomes brighter; as it contracts it gets dimmer. A single cycle typically takes **1 to 100 days**.

The famous thing about Cepheids is the **Period–Luminosity Relation**: the *longer* the pulsation period, the *intrinsically brighter* the star is. This is a physical law of nature, not a coincidence. It means if you time the pulsation, you can calculate exactly how intrinsically bright the star is — and therefore how far away it must be. This is why Cepheids are called **"standard candles"** — astronomers have used them since 1912 to measure distances across the universe.

---

**RR Lyrae** *(think: the steady little blue strobe)*

RR Lyrae stars are much older, lower-mass pulsating stars. They pulsate faster than Cepheids — periods are typically **less than one day** (often 0.2–0.9 days). Unlike Cepheids, they all have approximately the **same intrinsic brightness** (around absolute magnitude +0.6), regardless of their period. This consistency makes them excellent secondary distance markers. They are commonly found in globular clusters (dense star-ball formations) around galaxies like Andromeda.

---

**Eclipsing Binaries** *(think: two stars photobombing each other)*

Most stars in the universe are not alone — they travel in pairs. Eclipsing binaries are two stars orbiting so close together that from our perspective, one periodically passes in *front* of the other, blocking some of its light. This creates a very characteristic light curve: mostly flat, then a sudden sharp dip, then flat again, then another dip (the second eclipse when the other star passes). The shape is unmistakeable. The period is determined by the orbital period of the two stars.

---

**Long-Period Variables (LPV)** *(think: giant old red stars that slowly throb)*

LPVs are enormous red giant stars near the end of their lives. They pulsate much more slowly — periods can range from **100 to over 1000 days**. They are not as clockwork-regular as Cepheids or RR Lyrae; their brightness changes can be messy and semi-irregular. They are very common inside galaxies like Andromeda but less useful for precise distance measurement.

---

**Non-Variable / Noise** *(think: a boring constant star, or just measurement error)*

Not every source the telescope detects will be a genuine variable star. Some will be constant stars where the brightness wiggles are just random measurement noise (the detector always has a bit of error). Our classifier includes this as a fifth class so it can filter these out rather than mislabelling noise as a Cepheid.

---

### 2.2 What Is a Light Curve?

A **light curve** is simply a **graph of brightness over time**.

The y-axis is **magnitude** — an astronomy unit for brightness. **Important**: magnitude works *backwards* from what you might expect. A *higher* magnitude number means a *dimmer* star. Magnitude 22 is much fainter than magnitude 20. (This is a historical quirk from ancient Greek astronomers that astronomers never fixed.)

The x-axis is **MJD (Modified Julian Date)** — a way of expressing time as a single decimal number of days since midnight on November 17, 1858. We use it because it makes arithmetic across months and years easy.

When you plot a raw light curve from Hubble, it looks like a scatter of dots: the brightness measurement at each moment the telescope pointed at this star. With only 5–15 observations (Hubble doesn't stare at one spot continuously), it looks very sparse.

---

### 2.3 The Sky's Address System: RA and Dec

To specify where something is in the sky, astronomers use two coordinates:

- **Right Ascension (RA)**: Like longitude on Earth, but for the sky. Measured in degrees (0° to 360°).
- **Declination (Dec)**: Like latitude on Earth, but for the sky. Measured in degrees (-90° to +90°).

The centre of the Andromeda Galaxy sits at approximately **RA = 10.68°, Dec = 41.27°**. This is the black cross you see on our spatial map plot. All the variable stars we found are clustered nearby within a ~0.2-degree radius search window.

---

## 3. Why Hubble? The Resolution Problem

This is the single most important design decision in the entire project.

**The question:** Why didn't we use TESS, the NASA survey satellite that monitors millions of stars?

**The answer:** TESS pixels are enormous.

Each TESS detector pixel covers **21 arcseconds** of sky. One arcsecond is 1/3600th of a degree. At the distance of Andromeda (2.5 million light-years), a 21-arcsecond pixel covers a region containing **thousands of individual M31 stars** — they are all blurred together into a single brightness reading. TESS cannot separate them.

What *does* TESS detect when pointed at Andromeda? Any bright signal it picks up comes from **foreground stars** — stars in our own Milky Way that happen to be sitting directly in front of Andromeda from our perspective. These are physically unrelated to Andromeda; they just happen to be in the same direction in the sky.

**Hubble solves this.** The Hubble Space Telescope has a mirror diameter of 2.4 metres and is above the Earth's atmosphere (which blurs ground-based telescopes). Its resolution is fine enough to isolate and measure individual stars even inside crowded regions of Andromeda. The **Hubble Catalog of Variables (HCV)** is a published dataset of ~158,000 variable star candidates that Hubble has resolved inside various galaxies and clusters — including Andromeda. These are real M31 stars.

---

## 4. How We Download the Data

`download_andromeda_hcv.py` talks to the **MAST API** — the Mikulski Archive for Space Telescopes, which is NASA's central database for Hubble data.

We send a query asking: *"Give me all variable star candidates recorded near RA=10.68, Dec=41.27, within a radius of 0.2 degrees."*

The API returns a summary list. Each entry tells us:
- The star's unique ID (`MatchID`)
- Which Hubble camera filter was used (`Filter`)
- The star's coordinates (`RA`, `Dec`)
- Its classification flag (`AutoClass`: 0 = probably constant, 1 = single-filter variable candidate, 2 = multi-filter variable candidate)
- How many brightness measurements exist (`NumLC`)

We **filter to only AutoClass > 0** (genuine variable candidates) and **sort by most measurements first** (more data = better quality for period finding). We then download the detailed time-series for each selected star, saving each one as a CSV file in `data/andromeda_real_hcv/`.

**Why ACS_F814W?** The Hubble Advanced Camera for Surveys F814W is a near-infrared filter (central wavelength ~814 nanometres). We chose this because the HCV data for M31 has the most observations in this filter. For Cepheids and RR Lyrae, the F814W band is also close to the standard I-band, which is well-calibrated for period-luminosity relations.

---

## 5. Processing the Light Curves (Signal to Shape)

Raw light curves are scattered dots in time. A neural network cannot learn from scattered dots — it needs a clean, fixed-size input. We do three things:

### 5.1 Lomb-Scargle Period Finding

The **Lomb-Scargle Periodogram** is a mathematical algorithm designed specifically for unevenly-sampled time series. You give it a list of measurement times and a list of corresponding brightness values, and it scans hundreds of possible frequencies, measuring how well a simple sine wave at each frequency would explain the observed data.

The output is a **power spectrum**: a graph of "how well does each period explain my data." The period with the tallest spike is the star's dominant pulsation period.

**Why Lomb-Scargle and not a standard FFT (Fast Fourier Transform)?** Because standard FFTs require data sampled at equal time intervals. Hubble observations are not equally spaced — sometimes there are gaps of weeks. Lomb-Scargle is designed to handle this.

### 5.2 Phase Folding

Once we have the period *P* (in days), we transform every observation time *t* into a **phase**:

```
phase = (t / P) mod 1.0
```

This collapses the entire time series onto a single repeating cycle from 0.0 to 1.0. Instead of "this dip happened on day 5, and again on day 10, and again on day 15", all three dips now line up at roughly the same phase value — say 0.3. Suddenly, all cycles stack on top of each other and the repeating shape becomes clearly visible.

### 5.3 Interpolation to 100 Fixed Points

After phase folding, we have dots scattered across 0.0–1.0. But a neural network needs inputs of a *fixed, consistent size*. We use **numpy linear interpolation** to evaluate the phase-folded curve at exactly **100 equally spaced phase points** from 0.0 to 1.0. This converts every star — no matter how many observations it has — into a uniform 100-number array.

**Why 100 points?** It is a balance. Too few (say 20 points) and you lose the fine shape details that distinguish an eclipsing binary's sharp dip from a Cepheid's gradual sawtooth. Too many (say 500 points) and you are just creating fake precision from sparse data — you would be interpolating between only 5–13 real observations. 100 gives good shape resolution while staying honest to the sparse Hubble sampling.

### 5.4 Normalisation

We then scale the 100-point array so its values range from 0.0 to 1.0. This removes the absolute brightness of the star (which varies by distance and has nothing to do with its type) and preserves only the *shape* of the variability.

---

## 6. Machine Learning: Teaching a Computer to Recognise Star Types

### 6.1 Why a Neural Network?

The shape of a phase-folded light curve is highly characteristic:
- A Cepheid looks like a slow, asymmetric sawtooth — quick rise, gradual fall.
- An RR Lyrae looks like a faster, more symmetric bump.
- An eclipsing binary has a mostly flat line with sudden narrow valleys.
- An LPV looks lumpy and irregular.

Traditional pattern matching (hand-coded rules like "if the curve has a narrow dip, it is an eclipsing binary") breaks down quickly with real noisy data. A neural network learns the rules automatically by seeing thousands of examples.

### 6.2 Why a 1D CNN Specifically?

A **1D Convolutional Neural Network** is ideal for sequential data. Our phase-folded curve is a sequence of 100 numbers — basically a waveform, like an audio signal.

A convolutional layer works by sliding a small **filter** (e.g., a window of 5 numbers) across the sequence and at each position asking: "does this local shape match my filter?" This is perfect for detecting local features like dips, peaks, and slopes anywhere in the sequence — even if the shape appears at a slightly different phase position in different stars.

### 6.3 Real Training Data: The OGLE Database

While synthetic data is a great baseline, real telescope data is messy. By using `download_ogle_training_data.py`, we retrieve **real time-series photometry** from the OGLE project. These are genuine light curves of variable stars classified manually by expert astronomers.
* **Cepheids**: 300 real fundamental-mode variables in the LMC.
* **RR Lyrae**: 300 real RRab variables in the Galactic Bulge.
* **Eclipsing Binaries**: 300 real binaries in the LMC.
* **LPVs**: 300 real Mira / semi-regular variables in the LMC.

By running `train_on_ogle.py`, the CNN learns to tolerate real gaps, multi-periodic features, and natural cosmic noise patterns, vastly improving its performance on real Hubble targets.

### 6.4 CrossEntropyLoss & Optimizers
* **Cross-Entropy Loss**: Standard for multi-class classifiers. It measures how close the predicted probability distribution is to the target one-hot label.
* **Adam Optimizer**: An adaptive gradient descent optimizer that adjusts individual parameter learning rates dynamically during training.

---

## 7. Reading the Outputs: The CSV Catalog

The file `data/hcv_spatial_catalog.csv` is the final result table. Here is what every column means:

| Column | What it is | Example value |
|---|---|---|
| `match_id` | Hubble's internal ID for this star in the HCV database | `62636726` |
| `filter` | Which Hubble camera filter was used | `ACS_F814W` |
| `ra` | Right Ascension — the star's east-west position in the sky (degrees) | `10.783` |
| `dec` | Declination — the star's north-south position in the sky (degrees) | `41.362` |
| `class` | What our neural network classified the star as | `Cepheid Variable` |
| `confidence` | How certain the model was — the probability of the top prediction | `100.00%` |
| `period` | The dominant pulsation period found by Lomb-Scargle (days) | `0.3028` |
| `apparent_mag` | Average observed brightness as seen from Earth (higher = dimmer) | `22.108` |
| `absolute_mag` | Calculated intrinsic brightness (how bright it truly is) | `-0.359` |
| `distance_ly` | Estimated distance from Earth in light-years | `1,016,027` |

---

## 8. Reading the Outputs: The Spatial Map Plot

The file `hcv_stars_spatial_map.png` shows where all the classified variable stars sit in the sky relative to Andromeda's centre.

* **Axes**: X-axis is Right Ascension (RA), Y-axis is Declination (Dec).
* **Markers**: Color-coded nodes (Red circle = Cepheid, Pink square = RR Lyrae, Gold diamond = LPV) with a black cross pointing to Andromeda's astronomical center (RA = 10.685°, Dec = 41.269°).

---

## 9. Reading the Outputs: The Diagnostic Plot

When you run `predict.py` on any individual light curve, it generates a 3-panel figure called `lightcurve_diagnostic_plot.png`.

* **Panel 1 — Raw Light Curve**: Magnitude plotted against Modified Julian Date (MJD).
* **Panel 2 — Lomb-Scargle Periodogram**: Scans frequencies to locate the tallest power peak representing the period.
* **Panel 3 — Phase-Folded Light Curve**: The folded data points overlaid with the solid 100-point interpolated profile evaluated by the model.

---

## 10. Distance Estimation: How Far Away Are These Stars?

We use the **distance modulus formula**:
```
distance (parsecs) = 10 ^ ((apparent_magnitude - absolute_magnitude + 5) / 5)
distance (light-years) = distance_parsecs × 3.26
```
* **Cepheids**: Derived from the period-luminosity relation: `M = -2.43 * (log10(P) - 1.0) - 4.05`.
* **RR Lyrae**: Assumed fixed absolute magnitude of `M ≈ +0.6`.

---

## 11. Why Did We Choose Everything We Did?

| Decision | What We Chose | Why |
|---|---|---|
| **Telescope source** | Hubble HCV via MAST API | TESS pixels are too large (21") and only capture foreground Milky Way stars |
| **Filter** | ACS_F814W (near-infrared) | Best data coverage for M31; close to standard I-band used in PL relations |
| **Real Training Data** | OGLE database | Synthetic curves lack real astronomical noise, aliases, and cycle variations |
| **1D CNN architecture** | Three conv layers (32→64→128 filters) | Efficient feature extractor for 1D sequences and waveforms |

---

## 12. File-by-File Code Guide

```
train.py
│ Trains the CNN model on synthetic light curves.

train_on_ogle.py
│ Trains the CNN on real processed OGLE light curves.

download_ogle_training_data.py
│ Downloads real labeled light curves from the OGLE database.

validate_on_known_stars.py
│ Downloads benchmark prototype stars from TESS to validate CNN generalization.

download_andromeda_hcv.py
│ Downloads actual M31 variable star light curves from Hubble.

classify_hcv_variables.py
│ Classifies HCV stars, estimates distances, and maps Andromeda.

predict.py
│ Runs inference on a single file and outputs a diagnostic plot.

app.py
│ Interactive Gradio web interface.

src/model.py
│ Defines the 1D CNN structure.

src/data_setup.py
│ Preprocesses and generates synthetic training data.

src/engine.py
│ Handles train/evaluation epoch loops.
```

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

### 6.3 What Is Synthetic Training Data?

A real problem: we need *thousands* of labelled examples to train a neural network, but the HCV data for M31 has only ~15 good light curves. How do we train?

We **synthesise** training data. `src/data_setup.py` generates mathematically correct simulated light curves:
- For Cepheids: a sawtooth-like sine wave with the right asymmetry, plus random Gaussian noise.
- For RR Lyrae: a faster, more symmetric bump, with added noise.
- For Eclipsing Binaries: mostly flat line with one or two sharp dips at specific phases, plus noise.
- For LPVs: a slow, broad, somewhat messy wave.
- For Non-Variable/Noise: a flat line with pure noise.

We generate **5,000 training examples** and **1,000 test examples**. Because these are mathematical simulations of known wave types, they are perfectly labelled (we know exactly what type each one is). The model learns the general *shapes* of each class — and then we apply it to real Hubble data.

### 6.4 What Is CrossEntropyLoss?

After the model makes a prediction, we need to measure how wrong it was. **Cross-Entropy Loss** is the standard "wrongness score" for classification problems.

The model outputs a probability for each of the 5 classes (they must add up to 100%). If the correct answer is "Cepheid" and the model says 95% Cepheid, the loss is very low. If the model says 2% Cepheid, the loss is very high. The goal during training is to minimise this loss.

### 6.5 What Is the Adam Optimiser?

After computing the loss, we need to adjust the model's internal numbers (called **weights**) to make it less wrong next time. The **Adam optimiser** is the algorithm that does this adjustment. It calculates which direction to nudge each weight, and by how much (controlled by the **learning rate** of `0.001`). Adam is preferred over simpler optimisers because it adapts the step size individually for each weight — it learns how to learn.

### 6.6 What Are Epochs?

One **epoch** means the model has seen all 5,000 training examples exactly once. We train for **30 epochs** — so the model sees each example 30 times, getting progressively smarter each pass. After each epoch, we evaluate it on the 1,000 test examples it has never trained on (to make sure it is learning general patterns, not just memorising the training data).

### 6.7 Best Model Checkpointing

After each epoch, if the test loss is lower than any previous epoch, we save the model weights to `models/star_classifier.pth`. This means even if the model starts to slightly overfit in the final few epochs, we have preserved the best-ever version. This is the model we then use on real Hubble data.

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

**Why are some `absolute_mag` and `distance_ly` values N/A?**

Distance estimation only works for *standard candles* — stars whose intrinsic brightness we know from their period. We have reliable period-luminosity relations for:
- **Cepheids** (period tells us intrinsic brightness)
- **RR Lyrae** (fixed absolute magnitude ≈ +0.6)

For **LPVs**, there is no simple standard-candle relation — their brightness depends on mass, age, and composition in complex ways. So those rows correctly show N/A.

**Sanity check on the distances:** The Andromeda Galaxy is known to be approximately **2.5 million light-years** away. Our RR Lyrae stars come back with distances of ~685,000–1,066,000 ly and our Cepheids at ~844,000–1,755,000 ly. These are all within the right order of magnitude for stars in or near M31. The variation exists partly because the F814W filter is not perfectly calibrated for the standard V-band relations we used, introducing a systematic offset — but the fact that they are all clearly extragalactic (vs Milky Way scale of ~100,000 ly) confirms these are genuinely M31 stars.

---

## 8. Reading the Outputs: The Spatial Map Plot

The file `hcv_stars_spatial_map.png` shows where all the classified variable stars sit in the sky relative to Andromeda's centre.

**What the axes mean:**
- **X-axis (Right Ascension):** Moving right = further east in the sky. Ranges from about 10.68° to 10.92°.
- **Y-axis (Declination):** Moving up = further north in the sky. Ranges from about 41.17° to 41.37°.

**What the markers mean:**
- 🔴 Red circle = **Cepheid Variable**
- 🟥 Pink square = **RR Lyrae**
- 🟡 Gold diamond = **Long-Period Variable (LPV)**
- ✚ Black cross = **M31 Galaxy Centre** (RA = 10.685°, Dec = 41.269°)

**What to observe in the plot:**
- The M31 centre cross is at the lower-left. All our detected stars are scattered within a 0.2° radius of that centre — they are genuinely *inside* Andromeda's footprint on the sky.
- The stars cluster in two loose groups: one near RA ≈ 10.78° (upper-left cluster) and one near RA ≈ 10.91° (lower-right cluster). These correspond to specific Hubble observation fields (PHAT survey footprints) where dense multi-epoch data was collected.
- There are no Eclipsing Binaries in this particular sample — that is expected since Hubble's sparse sampling (5–15 observations per star) makes it hard to catch the precise timing of eclipses, which are brief and infrequent.

---

## 9. Reading the Outputs: The Diagnostic Plot

When you run `predict.py` on any individual light curve, it generates a 3-panel figure called `lightcurve_diagnostic_plot.png`.

**Panel 1 — Raw Light Curve:**
Shows the raw brightness measurements as individual dots, plotted against time (MJD). This is the "messy" original data — what Hubble actually recorded. You will notice the dots do not fall on a smooth line. This is expected: real measurements always have noise (detector imperfections, atmospheric effects for ground telescopes, etc.).

**Panel 2 — Lomb-Scargle Periodogram:**
Shows the power (y-axis) at each tested frequency (x-axis, in cycles per day). The tallest spike tells us the star's pulsation frequency. The period is `1 / frequency_at_peak`. If you see a very clean, sharp spike, the star has a very regular period. If the spectrum is messy with several comparable peaks, the variability is less well-constrained (common for LPVs).

**Panel 3 — Phase-Folded Light Curve:**
Shows the same data after phase folding — all cycles collapsed onto 0.0 to 1.0. The raw phase-folded dots are plotted, and overlaid in a solid curve is the **100-point interpolated profile** that was actually fed into the neural network. This is the cleanest view of the star's characteristic shape. You should be able to see which type it is just by looking at the shape here.

---

## 10. Distance Estimation: How Far Away Are These Stars?

### The Core Idea: Standard Candles

If you know how bright a lightbulb *intrinsically* is (say, 100 watts), and you see how bright it *appears* from a distance, you can calculate the distance. The brighter it appears, the closer it is. This is the principle of **standard candles** in astronomy.

### Step 1: Get the Intrinsic Brightness (Absolute Magnitude)

For Cepheids, we use the **Period-Luminosity relation**:
```
Absolute Magnitude (M) = -2.43 × (log10(Period_in_days) - 1.0) - 4.05
```
A Cepheid with a period of 0.3 days gives: `M = -2.43 × (log10(0.3) - 1.0) - 4.05 = -2.43 × (-0.523 - 1.0) - 4.05 = -2.43 × (-1.523) - 4.05 = 3.70 - 4.05 = -0.35`

For RR Lyrae, we simply use the fixed value `M = +0.6` (all RR Lyrae have approximately this brightness).

### Step 2: Use the Distance Modulus Formula

The **distance modulus** converts the difference between apparent brightness (what we see) and absolute brightness (what it actually is) into a distance:

```
distance (parsecs) = 10 ^ ((apparent_magnitude - absolute_magnitude + 5) / 5)
distance (light-years) = distance_parsecs × 3.26
```

**Worked example (star #62636726 from the catalog):**
- Apparent magnitude: `22.108`
- Absolute magnitude (from period 0.3028 days): `-0.359`
- Distance modulus: `22.108 - (-0.359) + 5 = 27.467`
- Distance: `10^(27.467 / 5) = 10^5.493 = 311,300 parsecs`
- In light-years: `311,300 × 3.26 = 1,015,000 light-years`

That is roughly **1 million light-years** — well within the Andromeda system.

---

## 11. Why Did We Choose Everything We Did?

| Decision | What We Chose | Why |
|---|---|---|
| **Telescope source** | Hubble HCV via MAST API | Only Hubble resolves individual stars inside M31. TESS pixels are too large and only capture Milky Way foreground stars |
| **Filter** | ACS_F814W (near-infrared) | Best data coverage for M31 in the HCV catalog; close to the I-band calibration used in standard PL relations |
| **Minimum 5 observations** | 5 epochs required | The Lomb-Scargle algorithm needs at least a few points to find a real period. Fewer than 5 is pure noise |
| **100-point sequence** | Fixed interpolation to 100 values | Balances shape resolution vs. over-interpolation from sparse 5–15-point Hubble data |
| **1D CNN architecture** | Three conv layers (32→64→128 filters) | Efficient for 1D waveforms; increasing filter depth detects progressively complex shapes |
| **Synthetic training data** | 5,000 synthetic + 1,000 test curves | Real M31 HCV data is too sparse to train on directly; synthetic data of the correct mathematical form generalises well |
| **5 classes** | Cepheid, RR Lyrae, EB, LPV, Non-Variable | Covers all major variable types expected in Andromeda; Non-Variable class ensures the model can reject noise |
| **CrossEntropyLoss** | Standard for multi-class classification | Penalises confident wrong answers more heavily than uncertain wrong answers — ideal for 5-class output |
| **Adam optimiser, lr=0.001** | Adaptive moment estimation | Self-adjusting learning rate; faster convergence than plain SGD; widely trusted default for classification tasks |
| **30 epochs** | Full training pass × 30 | Enough for the synthetic data to converge (accuracy hits 99%+ by epoch 15); not so many that overfitting occurs |
| **Batch size 32** | 32 stars per gradient step | Standard batch size; fits comfortably in RAM on any modern laptop |
| **Best-model checkpointing** | Save at lowest test loss | Prevents saving an overfit late-epoch model; always captures the generalising peak |

---

## 12. File-by-File Code Guide

```
train.py
│ Generates 5,000 synthetic light curves, trains the 1D CNN for 30 epochs,
│ saves the best weights to models/star_classifier.pth.
│ Run this FIRST before anything else.

download_andromeda_hcv.py
│ Queries NASA's MAST API for Hubble-observed variable stars near M31's
│ centre coordinates. Downloads their time-series brightness data and saves
│ each star as a CSV file in data/andromeda_real_hcv/.

classify_hcv_variables.py
│ Loads each CSV, runs Lomb-Scargle, phase-folds, interpolates to 100 points,
│ and feeds it through the trained model. Estimates distances for standard
│ candles and saves results to data/hcv_spatial_catalog.csv and
│ hcv_stars_spatial_map.png.

predict.py
│ Takes any single light-curve file (CSV or FITS) from the command line,
│ runs the full pipeline on that one star, and saves a 3-panel diagnostic
│ plot to lightcurve_diagnostic_plot.png.

app.py
│ Wraps predict.py inside a Gradio web interface. You upload a file in
│ your browser and see the classification and diagnostic plot instantly.

src/model.py
│ Defines the LightCurveCNN neural network: 3 convolutional blocks
│ (Conv1D + BatchNorm + ReLU + MaxPool + Dropout) followed by a
│ fully-connected output layer producing 5 class probabilities.

src/data_setup.py
│ Generates synthetic phase-folded light curves for each of the 5 star
│ classes with mathematically correct wave shapes and added Gaussian noise.
│ Also contains the PyTorch Dataset wrapper for use with DataLoader.

src/engine.py
│ train_epoch(): one full pass over training data — forward pass, loss
│   computation, backward pass, weight update.
│ test_epoch(): one full pass over test data — forward pass only, no
│   weight updates. Optionally prints a classification report.

models/star_classifier.pth
│ The saved weights of the best-performing trained model. This is what the
│ classifier scripts load to make predictions. Not committed to Git
│ (you generate it by running train.py).

data/andromeda_real_hcv/
│ Raw CSV files downloaded from the Hubble Catalog of Variables.
│ One file per star. Not committed to Git (downloaded locally by
│ download_andromeda_hcv.py).

data/hcv_spatial_catalog.csv
│ Final output table: one row per classified star, with coordinates,
│ star type, confidence, period, apparent magnitude, absolute magnitude,
│ and estimated distance in light-years.

hcv_stars_spatial_map.png
│ Scatter plot of all classified stars plotted at their RA/Dec sky
│ coordinates, colour-coded and shaped by star type, with Andromeda's
│ centre marked by a black cross.
```

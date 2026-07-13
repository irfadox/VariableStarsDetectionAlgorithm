# Complete Beginner's Walkthrough: Variable Star Classification & Andromeda Mapping

**Who this is for:** Anyone curious about this project with no prior background in astronomy or machine learning. No equations left unexplained. No design choices left unjustified. By the end, you should understand every file, every line of code, and exactly *why* each decision was made.

---

## 📖 Table of Contents
1. [What Is This Project Doing?](#1-what-is-this-project-doing)
2. [Astronomy Foundations](#2-astronomy-foundations)
3. [Why Hubble? The Resolution Problem](#3-why-hubble-the-resolution-problem)
4. [Step 1 — Downloading Training Data (`download_ogle_training_data.py`)](#4-step-1--downloading-training-data)
5. [Step 2 — The Lomb-Scargle Algorithm (How We Find Periods)](#5-step-2--the-lomb-scargle-algorithm)
6. [Step 3 — Phase Folding (Turning a Mess Into a Shape)](#6-step-3--phase-folding)
7. [Step 4 — The Neural Network (`src/model.py`)](#7-step-4--the-neural-network)
8. [Step 5 — Training the Model (`train_on_ogle.py`)](#8-step-5--training-the-model)
9. [Step 6 — Validating Against Real Stars (`validate_on_known_stars.py`)](#9-step-6--validating-against-real-stars)
10. [Step 7 — Classifying Andromeda (`classify_hcv_variables.py`)](#10-step-7--classifying-andromeda)
11. [Step 8 — Distance Estimation: Standard Candles](#11-step-8--distance-estimation-standard-candles)
12. [Reading the Outputs: CSV, Map, and Diagnostic Plots](#12-reading-the-outputs)
13. [Why Did We Choose Everything We Did?](#13-why-did-we-choose-everything-we-did)
14. [Known Limitations and Failure Cases](#14-known-limitations-and-failure-cases)

---

## 1. What Is This Project Doing?

The Andromeda Galaxy (also called M31) is our closest galactic neighbour — a spiral galaxy about 2.5 million light-years away. Inside it are hundreds of billions of stars.

This project answers the question: **can we automatically identify which types of variable stars exist inside Andromeda, and how far away they are — all from raw telescope data?**

To do that, we:
1. Download real Hubble Space Telescope brightness measurements for individual stars inside Andromeda.
2. Use a mathematical algorithm (Lomb-Scargle) to find how often each star pulses.
3. Convert that pulsation into a clean shape (phase-folded light curve).
4. Feed that shape to an AI (a 1D Convolutional Neural Network) trained on real expert-labeled data from the OGLE survey.
5. Use the classification to estimate distance using physics equations called **standard candles**.
6. Plot all classified stars on a sky map showing where they sit inside Andromeda.

We also validate the AI's reliability by testing it against famous "prototype" stars (e.g., the star literally named "RR Lyrae" — the star that gave the entire RR Lyrae class its name) observed by NASA's TESS satellite.

---

## 2. Astronomy Foundations

### 2.1 What Is a Variable Star?

Our Sun's brightness is almost perfectly constant. Variable stars are different — their brightness changes over time. The variation can be caused by:

- **Internal pulsations**: The star physically expands and contracts like a heartbeat. As it grows, the outer layers cool and dim; as it shrinks, they heat up and brighten.
- **Geometry**: Two stars orbiting each other can periodically block each other's light as seen from Earth.
- **Chemical instability**: Giant aged stars lose and re-accumulate their outer envelopes in long irregular cycles.

### 2.2 The Five Star Classes We Classify

| Class | Physical Cause | Typical Period | Shape of Light Curve |
|---|---|---|---|
| **Cepheid Variable** | Internal pulsation (4th-overtone instability strip) | 1–100 days | Smooth asymmetric sine — fast rise, slow fall |
| **RR Lyrae** | Short-period pulsation (horizontal-branch stars) | 0.2–1 day | Sharp asymmetric sawtooth |
| **Eclipsing Binary** | Two stars orbiting each other, one blocking the other | 0.1–10 days | Flat baseline with sharp V-shaped dips |
| **Long-Period Variable (LPV)** | Thermal pulsations in giant red stars | 100–1000 days | Broad, slow, irregular humps |
| **Non-Variable / Noise** | No periodicity — just measurement noise | — | Random scatter; no pattern |

### 2.3 Why Variable Stars Matter

**Cepheids and RR Lyrae are cosmic rulers.** There is a precise physical relationship between a Cepheid's pulsation period and its intrinsic brightness (how bright it truly is). By comparing intrinsic brightness to observed brightness (how bright it looks from Earth), we can calculate exactly how far away it is — no matter how distant. This is called the **Period-Luminosity relation** (also called the Leavitt Law after Henrietta Swan Leavitt who discovered it in 1908). This is how we know the distance to Andromeda to begin with.

### 2.4 Light Curves

A **light curve** is a time series of brightness measurements. Astronomers measure brightness in **magnitudes** — an ancient logarithmic scale where **brighter = smaller number** (yes, backwards from intuition). The Sun has magnitude −26.7. Andromeda stars observed by Hubble have magnitudes around 20–26. The logarithmic nature means a 5-magnitude difference = factor of 100× in brightness.

---

## 3. Why Hubble? The Resolution Problem

You might ask: why not use TESS (NASA's planet-hunting satellite) or another survey? The answer is **angular resolution**.

Telescopes have a fundamental limit: a pixel covers a certain angle of sky. TESS pixels each cover **21 arcseconds × 21 arcseconds** of sky. At the distance of Andromeda (2.5 million light-years), 21 arcseconds corresponds to thousands of light-years across — meaning every TESS pixel contains **thousands of Andromeda stars mixed together**. You cannot separate them.

Hubble's camera (ACS — Advanced Camera for Surveys) has a resolution of **0.05 arcseconds per pixel** — 420× sharper than TESS. Only Hubble can isolate and measure individual stars inside Andromeda.

The **Hubble Catalog of Variables (HCV)** is a systematic survey of all variable stars Hubble detected across 150 fields including Andromeda. It is hosted on the **MAST** (Mikulski Archive for Space Telescopes) at STScI and is freely queryable.

We use the **ACS_F814W** filter (near-infrared, around 814 nm wavelength). Why this filter specifically?
- It has the most data coverage in the M31 HCV fields.
- It is closest to the standard **I-band** used in the Leavitt Law calibrations, which minimises systematic errors in distance estimates.

---

## 4. Step 1 — Downloading Training Data

**File: `download_ogle_training_data.py`**

Before training a neural network, we need labeled examples. We use the **Optical Gravitational Lensing Experiment (OGLE)** — a long-running Polish astronomical survey based at the University of Warsaw that has been systematically discovering and cataloging variable stars since 1992.

OGLE is ideal because:
- It has manually classified hundreds of thousands of stars.
- Data is freely accessible via direct HTTP at `https://www.astrouw.edu.pl/ogle/`.
- The photometry spans 5–15 years, making period detection reliable.

### What the Code Does, Line by Line

```python
BASE_URL = "https://www.astrouw.edu.pl/ogle/ogle4/OCVS"
N_PER_CLASS = 1000
```
We download up to 1000 light curves per class. The URL points to OGLE's Online Catalog of Variable Stars.

```python
CATALOGS = [
    ("cepheid", "lmc", "cep", "OGLE-LMC-CEP-", 2, 4620, "lmc/cep/phot/I"),
    ("rrlyr", "blg", "rrlyr", "OGLE-BLG-RRLYR-", 1, 38000, "blg/rrlyr/phot/I"),
    ("eclipsing_binary", "lmc", "ecl", "OGLE-LMC-ECL-", 1, 26121, "lmc/ecl/phot/I"),
    ("lpv", "lmc", "lpv3", "OGLE-LMC-LPV-", 1, 79000, "lmc/lpv3"),
]
```
Each row is a tuple: `(class_name, sky_region, catalog_type, file_prefix, id_start, id_end, url_subpath)`.
- **lmc** = Large Magellanic Cloud (a nearby dwarf galaxy visible from the southern hemisphere, also contains these star types).
- **blg** = Galactic Bulge (the dense central region of the Milky Way — used for RR Lyrae because the Bulge has billions of them).
- IDs go from `id_start` to `id_end`, padded with zeros. E.g., star #14 becomes `OGLE-LMC-CEP-0014`.

```python
all_ids = list(range(id_start, id_end + 1))
np.random.shuffle(all_ids)
sampled_ids = all_ids[:n_target * 4]
```
We shuffle the IDs randomly (with `np.random.seed(42)` for reproducibility — "42" is convention) and take 4× our target. We take 4× because not every numbered star exists in the catalog (gaps due to data quality cuts). We try them one by one until we have 1000 successful downloads.

```python
time.sleep(0.4)
```
**Rate limiting.** We pause 0.4 seconds between each download request. Without this, we would flood the OGLE server with thousands of requests per minute and either get blocked or crash their server. This is basic scientific etiquette when using public research databases.

```python
if os.path.exists(out_path):
    downloaded += 1
    continue
```
**Resume capability.** If a file already exists on disk (from a previous run), we count it and skip re-downloading. This means you can safely interrupt and resume the downloader without losing progress.

### What Each Downloaded File Looks Like

Each `.dat` file is a plain text table with 3 columns: `HJD_time  magnitude  magnitude_error`. Example:
```
5262.52161 15.078 0.005
5264.52644 15.107 0.005
5265.57806 14.931 0.005
```
The time column is in HJD (Heliocentric Julian Date) — a continuous count of days since noon January 1, 4713 BC, corrected to the position of the Sun (so that the Earth's orbit doesn't introduce a 16-minute wobble in timing). The magnitude column is the star's brightness at that moment.

---

## 5. Step 2 — The Lomb-Scargle Algorithm

**Used in: `train_on_ogle.py`, `classify_hcv_variables.py`, `validate_on_known_stars.py`, `predict.py`**

### The Problem: Unevenly Sampled Data

We can't just run a simple FFT (Fast Fourier Transform) on a light curve because the measurements are taken at **irregular intervals** — some nights are cloudy, some fields are only observed in certain seasons, and Hubble only observes a field occasionally. We need an algorithm that handles gaps and irregular spacing.

### What Lomb-Scargle Does

The **Lomb-Scargle periodogram** tests every possible period and asks: "if I fold the data at this period, how well does it align into a repeating shape?" The mathematical score for each trial period is called **power**.

More precisely, at each trial frequency `f`, it fits a sinusoid of the form `y(t) = A·sin(2πft) + B·cos(2πft) + C` to the data using least-squares regression. The better the fit, the higher the power. The period with the highest power is the best candidate for the true period.

### The Code

```python
from astropy.timeseries import LombScargle

baseline = times[-1] - times[0]        # total observation span in days
min_freq = max(1.0 / baseline, 1.0 / 2000.0)   # lowest frequency to test
max_freq = 25.0                         # highest frequency (= period of 0.04 days)

frequency, power = LombScargle(times, mags).autopower(
    minimum_frequency=min_freq,
    maximum_frequency=max_freq,
    samples_per_peak=2,
    nyquist_factor=1
)
best_period = 1.0 / frequency[np.argmax(power)]
```

**Why `min_freq = max(1/baseline, 1/2000.0)`?**
The lowest meaningful frequency is one cycle per total observation span. You can't detect a period longer than your total observation baseline. The floor of `1/2000` prevents astronomically slow signals (periods > 2000 days) which are beyond LPV-class and would be measurement drift, not real periodicity.

**Why `max_freq = 25.0` (period > 0.04 days = ~1 hour)?**
Below about 1 hour, we're in the realm of p-mode oscillations (asteroseismology), not classical variable star classification. Also, Hubble's HCV data has typical cadences of days-to-weeks, making short-period detection unreliable.

**`samples_per_peak=2` and `nyquist_factor=1`:**
These control how dense the frequency grid is. The default values (`samples_per_peak=5, nyquist_factor=5`) create grids with ~100,000+ points for a 10-year OGLE baseline, which takes minutes per star. Reducing to `samples_per_peak=2, nyquist_factor=1` reduces the grid by ~12.5× with minimal impact on which period gets identified as the peak.

**`np.argmax(power)`** finds the index of the tallest spike. `frequency[that_index]` gives the frequency. `1.0 / frequency[...]` converts frequency (cycles per day) to period (days per cycle).

### Aliases: When Lomb-Scargle Gets Confused

A known failure mode: if your observation baseline is much shorter than the true period, Lomb-Scargle finds a **harmonic** or **alias** instead. For example, R Lyrae (true period ~460 days) observed in a single 27-day TESS window will show a ~18-day alias because the algorithm sees the beginning of one rise and nothing else — it fits the best sinusoid it can within 27 days.

This is why we stitch multiple TESS sectors (spanning 2019–2024) for long-period stars in `validate_on_known_stars.py`.

---

## 6. Step 3 — Phase Folding

**Used in: `train_on_ogle.py`, `classify_hcv_variables.py`, `validate_on_known_stars.py`, `predict.py`**

### The Intuition

Imagine a star that pulses every 10 days. You observe it on day 1, day 3, day 7, day 12, day 19... The raw light curve looks like a wiggly line spanning years. **Phase folding** "stacks" all the observations on top of each other as if we're watching exactly one cycle.

Every observation is assigned a **phase** value between 0 and 1:
```
phase = (time / period) % 1.0
```

The `%` operator is the modulo — it gives the fractional part of `time / period`. So if a star has period 10 days:
- An observation at day 3 → phase = 0.3 (30% through the cycle)
- An observation at day 23 → phase = 2.3 % 1 = 0.3 (same phase! Same point in the cycle)
- An observation at day 43 → phase = 4.3 % 1 = 0.3 (again the same)

All three observations are at the same physical state of the star, just different cycles.

### The Code

```python
phases   = (times / best_period) % 1.0
sort_idx = np.argsort(phases)
grid     = np.linspace(0.0, 1.0, 100)      # 100 evenly-spaced phase points
interp   = np.interp(grid, phases[sort_idx], mags[sort_idx])
```

- `np.argsort(phases)` gives us the indices that would sort the phases from 0 to 1.
- `np.interp(grid, phases[sort_idx], mags[sort_idx])` does linear interpolation: for each of the 100 evenly-spaced grid points (0.00, 0.01, 0.02... 1.00), it finds the two nearest observed phase-magnitude pairs and linearly interpolates between them.

The result is always exactly 100 numbers — regardless of whether the original light curve had 15 or 15,000 observations. This **fixed-length representation** is what the CNN requires.

### Why 100 Points?

100 was chosen as a balance:
- **Too few** (e.g. 20): the shape is too coarse to distinguish Cepheid from RR Lyrae when both look like broad bumps.
- **Too many** (e.g. 500): for Hubble stars with only 5–15 observations, the interpolated curve has long stretches of pure linear extrapolation between sparse points. This introduces false "shape" that isn't really there.

100 points captures the gross morphological differences (sharp dips vs. smooth humps vs. sawtooth) without over-interpolating.

### Normalisation

```python
mn, mx = interp.min(), interp.max()
normed = ((interp - mn) / (mx - mn)).astype(np.float32)
```

We rescale all 100 values to the range [0, 1]. This is critical because:
- Different stars have very different actual brightnesses (magnitude 15 vs. magnitude 25).
- We don't care about the absolute brightness — we care about the **shape** of the variation.
- If we didn't normalise, the CNN would learn "bright stars are Cepheids" based on magnitude value, which is meaningless — a Cepheid in a nearby galaxy looks bright; a Cepheid in a distant galaxy looks faint, but they're the same physical type.

---

## 7. Step 4 — The Neural Network

**File: `src/model.py`**

### What a Neural Network Is

A neural network is a mathematical function with millions of adjustable parameters (called **weights**). During training, we feed it examples with known answers and adjust the weights until its outputs match reality. After training, it can generalise to new examples it has never seen.

Our CNN takes 100 numbers (the normalised phase-folded light curve) and outputs 5 numbers (one confidence score per class). The class with the highest score is the prediction.

### Why a Convolutional Neural Network (1D CNN)?

A regular neural network (called a "dense" or "fully connected" network) would learn a separate weight for each of the 100 input positions. That means it learns "position 43 being bright means Cepheid" — but if the phase is slightly shifted, position 43 might not be bright at all. CNNs instead learn **local patterns that can appear anywhere**.

A 1D convolutional filter is a small window (say, 5 positions wide) that slides across the entire sequence and detects a local feature at every position. This is **translation invariant** — it doesn't matter where in the phase the sharp rise occurs, the filter will detect it.

### The Architecture, Line by Line

```python
self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
```
- `in_channels=1`: we have 1 "channel" of input (just the brightness values — like a grayscale image vs. an RGB image which has 3 channels).
- `out_channels=32`: we learn **32 different filters** simultaneously. Each filter looks for a different local pattern (e.g., one might activate on sharp dips, another on gradual slopes, another on flat plateaus).
- `kernel_size=5`: each filter looks at 5 consecutive phase points at a time.
- `padding=2`: we pad 2 zeros at each end of the sequence so the filter can slide over all 100 positions without shrinking the output length. Output length = 100.

```python
self.bn1 = nn.BatchNorm1d(32)
```
**Batch Normalisation** rescales the activations coming out of the convolution to have zero mean and unit variance across the training batch. Why? Without it, the activations can grow or shrink arbitrarily deep in the network, making training slow or unstable (called "vanishing/exploding gradients"). BatchNorm forces the values to stay in a manageable range throughout training. It also acts as a mild regulariser.

```python
self.pool1 = nn.MaxPool1d(kernel_size=2)
```
**Max pooling** slides a window of size 2 across the sequence and keeps only the maximum value in each window. This halves the sequence length (100 → 50 after pool1) and makes the representation more **position-tolerant** — a feature detected at position 22 or 23 both produce the same output after pooling.

```python
self.relu1 = nn.ReLU()
```
**Rectified Linear Unit**: `f(x) = max(0, x)`. Any negative activation becomes 0. This is the standard non-linear activation in modern CNNs. Without non-linearities, stacking layers is mathematically equivalent to one layer — the network can only learn linear functions and can't model complex shapes.

The three convolutional blocks follow the same pattern with increasing filter counts (32 → 64 → 128), building progressively more abstract representations:
- **Block 1 (32 filters)**: detects simple local features — single peaks, dips, slopes.
- **Block 2 (64 filters)**: detects combinations of the block-1 features — "a peak followed by a flat plateau" or "two peaks close together".
- **Block 3 (128 filters)**: detects the full gross morphology — "entire right half is flat with a sharp spike on the left" = eclipsing binary.

After the three blocks, the sequence length is 100 → 50 → 25 → 12 (each MaxPool halves it).

```python
self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
```
Regardless of the input sequence length, this reduces the representation to exactly 8 positions. It averages each chunk. This makes the model robust to slight variations in sequence length if ever used with non-100-point inputs.

```python
self.dropout = nn.Dropout(p=0.3)
```
**Dropout** randomly sets 30% of activations to zero during each training step. The network is forced to never rely on any single activation — it must build **redundant** representations. This is the single most effective regularisation technique in modern deep learning. It is disabled during inference (prediction), where all activations are used.

```python
self.fc = nn.Linear(in_features=128 * 8, out_features=num_classes)
```
The final **fully connected** (linear) layer. The 128-channel × 8-length tensor is flattened to 1024 numbers and mapped to 5 output numbers (one per class). These raw outputs are called **logits** — they can be any value, positive or negative. Applying `torch.softmax()` converts them to probabilities summing to 1.

### The forward() Pass

```python
def forward(self, x):
    x = self.relu1(self.pool1(self.bn1(self.conv1(x))))
    x = self.relu2(self.pool2(self.bn2(self.conv2(x))))
    x = self.relu3(self.pool3(self.bn3(self.conv3(x))))
    x = self.adaptive_pool(x)
    x = x.view(x.size(0), -1)   # flatten: (batch, 128, 8) → (batch, 1024)
    x = self.dropout(x)
    x = self.fc(x)
    return x
```

The `x.view(x.size(0), -1)` line: `x.size(0)` is the batch dimension (how many stars we're processing at once). `-1` means "infer this dimension automatically" — PyTorch calculates that 128 × 8 = 1024.

---

## 8. Step 5 — Training the Model

**File: `train_on_ogle.py`**

### The Training Loop Concept

Training a neural network is an iterative optimisation process:
1. Feed a **batch** of examples through the network (forward pass).
2. Compare the network's predictions to the true labels using a **loss function**.
3. Compute how much each weight contributed to the loss (**backpropagation**).
4. Nudge each weight slightly in the direction that reduces loss (**gradient descent**).
5. Repeat for all batches (one full pass = one **epoch**).

### Loss Function: CrossEntropyLoss

```python
criterion = nn.CrossEntropyLoss()
```

For multi-class classification, **Cross-Entropy Loss** is the standard. For a single example, it is:

`Loss = −log(probability of the correct class)`

If the model assigns 90% probability to the right class → Loss = −log(0.9) = 0.105 (small, good)
If the model assigns 5% probability to the right class → Loss = −log(0.05) = 3.0 (large, bad)

The logarithm heavily penalises confident wrong answers. This is ideal because it teaches the model to be both correct *and* calibrated (not overconfident).

### Optimizer: Adam

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

**Adam** (Adaptive Moment Estimation) is a gradient descent optimiser that automatically adjusts the learning rate for each individual parameter. Parameters that have been oscillating (uncertain direction) get smaller updates. Parameters that consistently move in the same direction get larger updates. This makes training faster and more stable than naive gradient descent.

`lr=1e-3` (0.001) is the base learning rate — the maximum step size for any weight update.

### Learning Rate Scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)
```

If the validation loss doesn't improve for 5 consecutive epochs (`patience=5`), the learning rate is halved (`factor=0.5`). This prevents overshooting: late in training when the model is close to the optimum, large weight updates can jump past the best solution. A smaller learning rate makes more precise adjustments.

### Best-Model Checkpointing

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("  ✓ saved")
```

We save the model **only when validation loss improves**. This is critical: without it, the final saved model would be from epoch 50, which might have slightly overfit (memorised training quirks). The best-checkpoint approach always saves the model that generalised best to unseen data.

`model.state_dict()` is a Python dictionary of all learnable parameters (all the conv filter weights, BatchNorm scales, linear layer weights, etc.). Saving this to a `.pth` file allows loading the model later without re-training.

### The Synthetic Non-Variable Class

```python
def generate_noise_curves(n=500, seq_len=100):
    for _ in range(n):
        noise_level = np.random.uniform(0.01, 0.05)
        flat_curve   = np.random.normal(0.5, noise_level, seq_len)
        flat_curve   = np.clip(flat_curve, 0.0, 1.0)
```

We generate 500 synthetic "non-variable" curves: Gaussian noise centred at 0.5 with small standard deviation (1–5%). Why synthetic? Because non-variable stars have no periodicity — the Lomb-Scargle algorithm will still find *some* peak by chance (noise has many small peaks), and the resulting phase-folded curve will look like a noisy flat line. Generating these synthetically means we can create as many as we want without downloading anything. Non-variable stars are the most common type in Andromeda, so having a robust "noise" class prevents misclassifying blank noise as a Cepheid.

### Stratified Train/Val Split

```python
X_train, X_val, y_train, y_val = train_test_split(
    all_sequences, all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)
```

`stratify=all_labels` ensures each split (train and val) has the same class proportions. Without stratification, a random split might accidentally put all 1000 Cepheids in the training set and none in validation — or vice versa. Stratification guarantees each class is 80% train / 20% val.

### Training Results (1,200 real OGLE stars + 500 synthetic noise)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Cepheid | 82% | 82% | 82% |
| RR Lyrae | 79% | 73% | 76% |
| Eclipsing Binary | 90% | 92% | 91% |
| LPV | 84% | 88% | 86% |
| Non-Variable / Noise | 100% | 100% | 100% |
| **Overall** | — | — | **89%** |

**What precision and recall mean:**
- **Precision**: Of all the stars the model *predicted* as Cepheids, 82% actually were Cepheids.
- **Recall**: Of all actual Cepheids in the test set, the model correctly identified 82% of them.
- **F1**: The harmonic mean of precision and recall — balances both metrics in one number.

The confusion matrix showed that most misclassifications are Cepheid ↔ RR Lyrae (both have smooth, asymmetric profiles) and LPV ↔ Eclipsing Binary (both can show irregular broad structures when the period is an alias).

---

## 9. Step 6 — Validating Against Real Stars

**File: `validate_on_known_stars.py`**

### Why We Need an Independent Validation Set

The OGLE validation accuracy (89%) uses stars from the same survey as the training data. While these are genuinely unseen stars, they share the same noise characteristics, cadence patterns, and data quality pipeline. A model that works on OGLE might still fail on Hubble data (different telescope, different noise, different filters).

To test **cross-survey generalisation**, we use TESS observations of four famous "prototype" stars — stars whose type is known with absolute certainty from 100+ years of study.

### The Four Prototype Stars

| Star | True Type | Why Famous |
|---|---|---|
| Zeta Geminorum | Cepheid | Bright, nearby Cepheid studied since the early 1900s |
| RR Lyrae | RR Lyrae | **The** star that defined the entire class in 1899 |
| Algol (β Persei) | Eclipsing Binary | Known since antiquity — "the Demon Star" — as a periodic dimmer |
| R Lyrae | LPV | Bright semi-regular red giant with ~460-day period |

### Downloading TESS Data: The lightkurve Library

```python
import lightkurve as lk
results = lk.search_lightcurve(search_name, mission="TESS")
lc = results[0].download(download_dir=tmpdir, show_progress_bar=False)
```

`lightkurve` is a NASA-sponsored Python library that queries the **MAST** archive for TESS data and returns light curve objects. `search_lightcurve()` returns a table of all available data products for that target. `download()` fetches the FITS file from MAST to a local temporary directory.

We use a **fresh temp directory** per star (via `tempfile.mkdtemp()`) rather than the default lightkurve cache. This prevents corrupt-file issues: if a download is interrupted mid-way, lightkurve caches the partial file and re-uses it on the next run (causing a FITS read error). Temp directories are unique every time, so there's no stale data to accidentally reuse.

### Multi-Sector Stitching for LPVs

```python
if known_period_days >= 100:
    qlp_mask = ["QLP" in str(a) for a in results.author]
    qlp_results = results[qlp_mask] if any(qlp_mask) else results
    lc_collection = qlp_results.download_all(download_dir=tmpdir)
    lc = lc_collection.stitch(corrector_func=None)
```

R Lyrae has a true period of ~460 days. A single TESS sector covers only ~27 days — just 6% of one cycle. It is physically impossible to detect the true period from one sector.

**Why QLP-only?** There are three TESS data pipelines: SPOC, QLP, and TASOC. SPOC uses one time reference epoch (BTJD), QLP uses another (also BTJD but with a slightly different offset in some sectors), and TASOC uses yet another format. When stitching light curves from different pipelines, the time axis can become non-monotonic (times running backwards between sectors), causing Lomb-Scargle to compute a negative baseline and fail. We use QLP exclusively — it provides a consistent time system across all sectors.

**`corrector_func=None`** in the stitch call: by default, lightkurve normalises each sector to the same flux scale before stitching. This normalisation divides by the sector's median flux, which fails when the median is close to zero (as happens with detrended photometry). We disable it and handle our own magnitude conversion.

### Flux to Magnitude Conversion

```python
mags = -2.5 * np.log10(np.abs(flux_vals) + 1e-10)
```

TESS provides **flux** (electrons per second, proportional to brightness), but our model was trained on **magnitudes** (logarithmic brightness). This converts flux to a magnitude proxy using Pogson's equation: `m = −2.5 × log₁₀(flux)`. The `+ 1e-10` prevents a divide-by-zero crash on exactly-zero flux values.

### Validation Results (Current Model, 300 OGLE Training Stars)

| Star | LS Period Found | Known Period | Predicted | Correct? |
|---|---|---|---|---|
| Zeta Geminorum | 9.90 d | 10.15 d | Cepheid (97.5%) | ✅ |
| RR Lyrae | 0.566 d | 0.567 d | RR Lyrae (84.2%) | ✅ |
| Algol (β Persei) | 2.880 d | 2.867 d | Eclipsing Binary (100%) | ✅ |
| R Lyrae | 18.2 d | ~460 d | Eclipsing Binary (88%) | ❌ |

**Result: 3/4 = 75%** (up from 1/4 = 25% with the original synthetic-only model).

**Why did R Lyrae fail?** The Lomb-Scargle algorithm found an 18-day alias instead of the 460-day true period. When we folded the data at 18 days, the profile looked like a sharp asymmetric dip — visually similar to an eclipsing binary transit. The CNN correctly identified that shape as eclipsing binary; the error was in the period detection step, not the classification step. This is a physical limitation of short-baseline TESS data for ultra-long-period variables, not a model bug.

---

## 10. Step 7 — Classifying Andromeda

**File: `classify_hcv_variables.py`**

This script loads the downloaded Hubble light curves from `data/andromeda_real_hcv/`, runs each through the full pipeline (Lomb-Scargle → phase fold → CNN), and saves a results CSV + spatial map.

### Loading an HCV File

```python
df = pd.read_csv(file_path, comment='#')
times = df['mjd'].values.astype(np.float32)
mags  = df['mag'].values.astype(np.float32)
```

HCV files use Modified Julian Date (MJD = JD − 2,400,000.5) as the time axis. The `comment='#'` tells pandas to skip comment lines (which contain metadata: MatchID, Filter, RA, Dec). We then cast to `float32` (single-precision) rather than `float64` (double-precision) to save memory and because CNN inputs don't need sub-nanomagnitude precision.

### Minimum Data Threshold

```python
if len(times) < 5:
    print(f"  Skipping {basename}: not enough valid data points.")
    continue
```

Lomb-Scargle needs at least a few cycles sampled to identify a period. With fewer than 5 data points, the "best period" is essentially a random guess. We skip those stars to avoid polluting the catalog with meaningless classifications.

### CNN Inference

```python
input_tensor = torch.tensor(normed_mags, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(input_tensor)
    probs  = torch.softmax(output, dim=1).squeeze().cpu().numpy()
```

- `.unsqueeze(0)` adds a batch dimension (the CNN expects `(batch_size, channels, length)`, not `(channels, length)`).
- `.unsqueeze(0)` again adds the channel dimension — we have 1 channel.
- Final shape: `(1, 1, 100)` — batch_size=1, channels=1, length=100.
- `torch.no_grad()` disables gradient computation during inference. Since we're not training, we don't need to track how each weight contributed to the output. Disabling this saves ~50% memory and speeds up inference.
- `torch.softmax(output, dim=1)` converts raw logits to probabilities summing to 1. The most confident class is the prediction.

---

## 11. Step 8 — Distance Estimation: Standard Candles

The term "standard candle" refers to any object whose **intrinsic luminosity** (absolute brightness) is known. By comparing how bright it looks from Earth to how bright it truly is, we can calculate how far away it must be.

### The Distance Modulus Formula

```python
distance_pc = 10 ** ((apparent_mag - abs_mag + 5) / 5)
distance_ly = distance_pc * 3.26156
```

This comes from the inverse-square law of light. If a candle moves twice as far away, it appears 4× dimmer (brightness ∝ 1/distance²). In the magnitude system:
`μ = m − M = 5 × log₁₀(d) − 5`
Rearranging: `d = 10^((m − M + 5)/5)` where d is in parsecs (1 parsec = 3.26 light-years).

### Cepheid Period-Luminosity Relation

```python
if pred_class == "Cepheid Variable" and best_period > 0:
    abs_mag = -2.43 * (np.log10(best_period) - 1.0) - 4.05
```

This is the **Leavitt Law** calibrated for the I-band (approximately matching our ACS_F814W Hubble filter). A Cepheid with a 10-day period (`log₁₀(10) = 1.0`) has `M = −2.43 × 0 − 4.05 = −4.05`. A 100-day Cepheid has `M = −2.43 × 1 − 4.05 = −6.48` (intrinsically brighter by ~2.4 magnitudes = ~9× in flux). The coefficients `−2.43` and `−4.05` come from calibration using Cepheids in the Large Magellanic Cloud whose distance is independently known.

### RR Lyrae Fixed Absolute Magnitude

```python
elif pred_class == "RR Lyrae":
    abs_mag = 0.6
```

All RR Lyrae stars have nearly identical intrinsic brightness: `M_V ≈ +0.6` magnitudes (approximately). This is because they occupy a narrow strip in the HR diagram (called the "instability strip") where their core physics constrains their luminosity to a tight range. A 0.6-magnitude scatter in this value corresponds to about ±30% in distance uncertainty — not perfect, but remarkably precise for stars at 2.5 million light-years.

---

## 12. Reading the Outputs

### The CSV Catalog (`data/hcv_spatial_catalog.csv`)

| Column | What it means | Example |
|---|---|---|
| `match_id` | Hubble's unique ID for this star | `62636726` |
| `filter` | Hubble camera filter used | `ACS_F814W` |
| `ra` | Right Ascension: east-west sky position in degrees | `10.783` |
| `dec` | Declination: north-south sky position in degrees | `41.362` |
| `class` | Model's classification | `Cepheid Variable` |
| `confidence` | Probability of the top prediction | `100.00%` |
| `period` | Dominant period found by Lomb-Scargle (days) | `0.3028` |
| `apparent_mag` | Average observed brightness (higher = dimmer) | `22.108` |
| `absolute_mag` | Calculated intrinsic brightness | `−0.359` |
| `distance_ly` | Estimated distance from Earth in light-years | `1,016,027` |

### The Spatial Map (`hcv_stars_spatial_map.png`)

- X-axis: Right Ascension (RA) — east-west position in the sky (in degrees)
- Y-axis: Declination (Dec) — north-south position in the sky (in degrees)
- Markers: colour and shape coded by class. The black cross marks Andromeda's centre at RA=10.685°, Dec=41.269°.

Andromeda spans about 3.2° × 1.0° of sky. Stars plotted close together on the map may actually be tens of thousands of light-years apart in 3D space — the map shows only the projected 2D positions on the sky.

### Diagnostic Plots (`data/validation_plots/*.png`)

Each star gets a 3-panel plot:
- **Panel 1 — Raw Light Curve**: Brightness (magnitude proxy) vs. time. The y-axis is inverted (astronomy convention: up = brighter = smaller magnitude number).
- **Panel 2 — Phase-Folded Curve**: The same data, but each observation's x-position is its phase (0–1 = one full cycle). The solid purple line is the interpolated 100-point profile the CNN actually used.
- **Panel 3 — Classification Probabilities**: Bar chart of all 5 class probabilities. A healthy classification shows one dominant bar; uncertainty shows multiple bars of similar height.

---

## 13. Why Did We Choose Everything We Did?

| Decision | What We Chose | Why |
|---|---|---|
| **Telescope** | Hubble HCV via MAST | 0.05" resolution isolates individual M31 stars; TESS (21") blends thousands per pixel |
| **Filter** | ACS_F814W | Most data coverage in M31 HCV; closest to standard I-band for PL relations |
| **Training data** | OGLE real photometry | Synthetic curves miss real noise, gaps, aliasing, and period drift |
| **OGLE region — Cepheids** | LMC fundamental-mode | LMC has the most densely characterised Cepheid population; fundamental-mode are cleanest |
| **OGLE region — RR Lyrae** | Galactic Bulge RRab | Bulge has enormous density of RR Lyrae; RRab (type a) are the cleanest sawtooth type |
| **N = 1000 per class** | 1000 labeled examples | More data → better generalisation; OGLE has millions available so 1000 is a cheap fraction |
| **Synthetic Non-Variable** | 500 Gaussian noise curves | Non-variable stars are the most common but have no obvious labeled source; synthetic is adequate for flat-line detection |
| **seq_len = 100** | 100-point interpolation | Enough shape resolution for classification; not over-interpolated for sparse 5-15 point Hubble curves |
| **CNN architecture** | 3× Conv(32→64→128) + AdaptivePool + Dropout | Standard effective architecture for 1D sequence classification; deeper than 3 layers shows diminishing returns on this data volume |
| **kernel_size = 5** | 5-point sliding window | Captures local features spanning ~5% of a cycle; small enough to detect sharp dips, large enough to avoid pixel-level noise |
| **BatchNorm** | After every conv | Stabilises training and acts as regulariser; almost always beneficial in modern CNNs |
| **Dropout p=0.3** | Drop 30% during training | Empirically effective regularisation; prevents overfitting on the ~1700 training samples |
| **Adam, lr=0.001** | Adaptive moment optimiser | Self-tuning per-parameter learning rates; fast convergence; standard default for classification |
| **ReduceLROnPlateau** | Halve LR after 5 bad epochs | Prevents late-training oscillation without requiring manual learning rate scheduling |
| **50 epochs** | 50 training passes | Real data requires more epochs to converge than synthetic; training loss stabilises around epoch 25 |
| **Batch size = 32** | 32 stars per update | Fits easily in RAM; small enough for good gradient noise (regularisation); large enough for stable BatchNorm statistics |
| **Best checkpoint** | Save at lowest val loss | Prevents saving an overfit late-epoch model; val loss is a better model selector than final epoch |
| **TESS for validation** | NASA MAST lightkurve | Free, well-maintained API for known bright prototype stars |
| **QLP-only for LPV TESS** | QLP pipeline sectors only | Consistent time reference system; mixing SPOC + QLP + TASOC causes non-monotonic time axis → negative LS baseline → crash |

---

## 14. Known Limitations and Failure Cases

### LPV Classification with Short-Baseline Data
Long-Period Variables (periods 100–1000 days) require an observation baseline longer than their period to be detected correctly. A single TESS 27-day window for an LPV star will always produce an alias period, leading to misclassification. For Andromeda data, Hubble's HCV has baselines of 5–10 years, so this is much less of a problem than for TESS validation.

### Hubble's Sparse Sampling
Many Hubble HCV stars have only 5–15 brightness measurements spanning years. The interpolated 100-point phase-folded curve for such a star is mostly linear extrapolation between 5 points — the shape is inferred, not measured. Classification confidence will be lower for these stars. We mitigate this by requiring at least 5 observations before attempting classification.

### Period Aliasing in General
Lomb-Scargle finds the single most significant period. Some stars have multiple significant periods (beat frequencies in binary star systems, or semi-regular LPVs). The algorithm will pick one — not necessarily the physically meaningful one.

### F814W ≠ Standard I-band
The Cepheid PL relation coefficients (−2.43 and −4.05) were calibrated in the standard Cousins I-band. ACS_F814W is close but not identical. This introduces a systematic offset in distance estimates of approximately 0.05–0.1 magnitudes, corresponding to 2–5% in distance — acceptable for a survey-level analysis.

### Model Not Retrained on Hubble Data
The CNN was trained on OGLE photometry (ground-based, southern-hemisphere, multi-year baseline). Hubble photometry has different noise characteristics, a different time sampling pattern, and different filter profiles. Cross-survey generalisation introduces uncertainty we cannot fully quantify without a labeled Hubble test set.

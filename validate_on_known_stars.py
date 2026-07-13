# =================================================================================
# VALIDATE_ON_KNOWN_STARS.PY - Real-Data Model Validation Against Prototype Variable Stars
# For a conceptual explanation of how this fits into the project, see:
# walkthrough_guide.md (at the repository or workspace root)
# =================================================================================
#
# PURPOSE:
# Our CNN was trained entirely on *synthetic* (mathematically generated) light curves.
# A fair question is: does the model actually work on REAL telescope data?
#
# This script answers that by downloading TESS light curves for a handful of
# famous, undisputed prototype variable stars — stars whose type is so well
# established by a century of astronomical study that they literally gave their
# names to their class (e.g., "RR Lyrae" is THE RR Lyrae star). We then run
# each one through our Lomb-Scargle → phase-fold → CNN pipeline and report
# whether the model gets the right answer.
#
# STAR SELECTION RATIONALE:
# We pick one or two of the brightest and most-observed example of each class
# because TESS is most likely to have high-quality data for bright, well-known
# targets. Period values below are the astronomically accepted reference periods.
#
#   Star             | Type              | Period (days) | Why chosen
#   -----------------+-------------------+---------------+----------------------------
#   Zeta Geminorum   | Cepheid Variable  | 10.148        | Bright classical Cepheid
#   RR Lyrae         | RR Lyrae          | 0.5669        | THE prototype RR Lyrae star
#   Algol (β Persei) | Eclipsing Binary  | 2.8673        | THE prototype eclipsing binary
#   R Lyrae          | LPV               | ~460          | Bright semi-regular red giant
#
# WHAT THIS TELLS US:
# If the model correctly classifies 3 or 4 of these, the synthetic training
# approach is validated for real-world use.
# If it fails on several, it suggests the synthetic waves don't capture the
# messiness of real photometry — and we should train on real labeled catalogs
# such as OGLE (see note at end of file).

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# ── Import our model ──────────────────────────────────────────────────────────
try:
    from src.model import LightCurveCNN
except ImportError:
    print("ERROR: Could not import src.model. Run this script from the repo root.")
    sys.exit(1)

# ── Try importing lightkurve ───────────────────────────────────────────────────
try:
    import lightkurve as lk
except ImportError:
    print("ERROR: lightkurve is not installed.")
    print("Install it with: pip install lightkurve")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS   = "models/star_classifier.pth"
SEQ_LEN         = 100
OUTPUT_DIR      = "data/validation_plots"
RESULTS_CSV     = "data/validation_results.csv"

CLASS_NAMES = {
    0: "Cepheid Variable",
    1: "RR Lyrae",
    2: "Eclipsing Binary",
    3: "Long-Period Variable (LPV)",
    4: "Non-Variable / Noise"
}

# Prototype stars we will test
# Each entry: (human_name, TESS_search_name, true_class_index, known_period_days)
PROTOTYPE_STARS = [
    ("Zeta Geminorum",  "Zeta Geminorum",  0,  10.148),  # Cepheid
    ("RR Lyrae",        "RR Lyrae",        1,  0.5669),  # RR Lyrae
    ("Algol (β Persei)","Algol",           2,  2.8673),  # Eclipsing Binary
    ("R Lyrae",         "R Lyrae",         3,  460.0),   # LPV (semi-regular)
]


def download_tess_lightcurve(search_name: str, known_period_days: float):
    """
    Download and stitch TESS light curves for `search_name`.
    Returns (times_days, magnitudes) as numpy arrays, or (None, None) on failure.
    """
    print(f"  Searching MAST for TESS data on '{search_name}'...")
    try:
        results = lk.search_lightcurve(search_name, mission="TESS")
        if len(results) == 0:
            print(f"  WARNING: No TESS light curves found for '{search_name}'.")
            return None, None

        # Download the first available sector (fastest; usually enough for period finding)
        lc = results[0].download()
        if lc is None:
            print(f"  WARNING: Download returned None for '{search_name}'.")
            return None, None

        # Use SAP flux if available, fall back to whatever flux column exists
        flux_col = "sap_flux" if "sap_flux" in lc.columns else "flux"
        lc = lc.select_flux(flux_col)

        # Remove NaN and obvious outliers
        lc = lc.remove_nans()
        lc = lc.remove_outliers(sigma=5)

        times_days = lc.time.value
        flux_vals  = lc[flux_col].value

        # Convert flux to magnitude proxy so it matches our training data format:
        # magnitude ∝ −2.5 * log10(flux).  We normalise so only shape matters.
        with np.errstate(divide="ignore", invalid="ignore"):
            mags = -2.5 * np.log10(np.abs(flux_vals) + 1e-10)

        # Remove non-finite values after log
        mask = np.isfinite(times_days) & np.isfinite(mags)
        return times_days[mask], mags[mask]

    except Exception as exc:
        print(f"  ERROR downloading '{search_name}': {exc}")
        return None, None


def phase_fold_and_classify(times, mags, known_period, device, model):
    """
    Run the full Lomb-Scargle → phase-fold → CNN pipeline on (times, mags).
    Returns (predicted_class_idx, confidence_pct, ls_period, all_probs).
    """
    # ── Lomb-Scargle ──────────────────────────────────────────────────────────
    min_freq = 1.0 / min(500.0, 0.8 * (times[-1] - times[0]))
    max_freq = 20.0   # cycles / day  (period down to 0.05 days)
    frequency, power = LombScargle(times, mags).autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq
    )
    ls_period = 1.0 / frequency[np.argmax(power)]

    # ── Phase fold using Lomb-Scargle period ──────────────────────────────────
    phases = (times / ls_period) % 1.0
    sort_idx = np.argsort(phases)
    grid_phases = np.linspace(0.0, 1.0, SEQ_LEN)
    interp_mags = np.interp(grid_phases, phases[sort_idx], mags[sort_idx])

    # ── Normalise 0→1 ─────────────────────────────────────────────────────────
    mn, mx = interp_mags.min(), interp_mags.max()
    normed = (interp_mags - mn) / (mx - mn) if mx > mn else np.zeros(SEQ_LEN)

    # ── CNN inference ─────────────────────────────────────────────────────────
    tensor = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    confidence = probs[pred_idx] * 100.0
    return pred_idx, confidence, ls_period, probs, phases, sort_idx, mags, grid_phases, normed


def save_diagnostic(star_name, times, mags, phases, sort_idx, grid_phases,
                    normed, probs, ls_period, known_period, pred_class, true_class,
                    output_path):
    """Save a 3-panel diagnostic plot identical in structure to the main project's predict.py."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"{star_name}  |  True: {true_class}  |  Predicted: {pred_class}\n"
        f"Known period: {known_period:.4f} d   |   LS period found: {ls_period:.4f} d",
        fontsize=11
    )

    # Panel 1 — Raw light curve
    axes[0].scatter(times, mags, s=2, alpha=0.4, color="#4fc3f7")
    axes[0].set_xlabel("Time (days, BJD)")
    axes[0].set_ylabel("Magnitude proxy")
    axes[0].set_title("Raw Light Curve")
    axes[0].invert_yaxis()  # Astronomy convention: brighter = lower number = plotted higher

    # Panel 2 — Phase-folded raw dots
    axes[1].scatter(phases[sort_idx], mags[sort_idx], s=2, alpha=0.4, color="#ce93d8")
    axes[1].plot(grid_phases, normed * (mags.max() - mags.min()) + mags.min(),
                 color="#ab47bc", lw=1.5, label="Interpolated profile")
    axes[1].set_xlabel("Phase (0–1)")
    axes[1].set_ylabel("Magnitude proxy")
    axes[1].set_title(f"Phase-Folded  (P={ls_period:.4f} d)")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)

    # Panel 3 — Class probability bar chart
    class_labels = [CLASS_NAMES[i].replace(" Variable", "").replace(" / Noise", "") for i in range(5)]
    bar_colors = ["#ef5350", "#f48fb1", "#42a5f5", "#ffa726", "#66bb6a"]
    bars = axes[2].bar(class_labels, probs * 100, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[2].set_ylim(0, 110)
    axes[2].set_ylabel("Confidence (%)")
    axes[2].set_title("Classification Probabilities")
    axes[2].tick_params(axis="x", rotation=30)
    # Annotate bars
    for bar, p in zip(bars, probs):
        if p > 0.02:
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{p*100:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"ERROR: Model weights not found at '{MODEL_WEIGHTS}'.")
        print("Run 'python3 train.py' first to generate the model.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LightCurveCNN(num_classes=5)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from '{MODEL_WEIGHTS}' (device: {device})\n")

    results = []

    for human_name, search_name, true_idx, known_period in PROTOTYPE_STARS:
        print(f"{'='*60}")
        print(f"Testing: {human_name}  (True class: {CLASS_NAMES[true_idx]})")

        times, mags = download_tess_lightcurve(search_name, known_period)
        if times is None or len(times) < 20:
            print(f"  SKIP: insufficient data ({len(times) if times is not None else 0} points).\n")
            results.append({
                "star": human_name,
                "true_class": CLASS_NAMES[true_idx],
                "predicted_class": "SKIPPED",
                "confidence": "N/A",
                "ls_period": "N/A",
                "known_period": known_period,
                "correct": False
            })
            continue

        print(f"  Downloaded {len(times)} data points from TESS.")

        try:
            pred_idx, confidence, ls_period, probs, phases, sort_idx, mags_orig, grid_phases, normed = \
                phase_fold_and_classify(times, mags, known_period, device, model)
        except Exception as exc:
            print(f"  ERROR during pipeline: {exc}")
            results.append({
                "star": human_name,
                "true_class": CLASS_NAMES[true_idx],
                "predicted_class": "ERROR",
                "confidence": "N/A",
                "ls_period": "N/A",
                "known_period": known_period,
                "correct": False
            })
            continue

        pred_class = CLASS_NAMES[pred_idx]
        is_correct = (pred_idx == true_idx)
        tick = "✅" if is_correct else "❌"

        print(f"  LS period found:  {ls_period:.4f} d   (known: {known_period:.4f} d)")
        print(f"  Prediction:       {pred_class}  ({confidence:.1f}%)  {tick}")
        print(f"  All class probs:  { {CLASS_NAMES[i]: f'{probs[i]*100:.1f}%' for i in range(5)} }\n")

        # Save diagnostic plot
        plot_name = human_name.replace(" ", "_").replace("(", "").replace(")", "").replace("β", "beta") + ".png"
        plot_path = os.path.join(OUTPUT_DIR, plot_name)
        save_diagnostic(
            human_name, times, mags, phases, sort_idx, grid_phases,
            normed, probs, ls_period, known_period, pred_class, CLASS_NAMES[true_idx],
            plot_path
        )
        print(f"  Diagnostic plot saved to: {plot_path}")

        results.append({
            "star": human_name,
            "true_class": CLASS_NAMES[true_idx],
            "predicted_class": pred_class,
            "confidence": f"{confidence:.1f}%",
            "ls_period": f"{ls_period:.4f}",
            "known_period": known_period,
            "correct": is_correct
        })

    # ── Summary report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Star':<25} {'True Class':<28} {'Predicted':<28} {'OK?'}")
    print("-" * 90)
    n_correct = 0
    n_attempted = 0
    for r in results:
        if r["predicted_class"] not in ("SKIPPED", "ERROR"):
            n_attempted += 1
            if r["correct"]:
                n_correct += 1
        tick = "✅" if r["correct"] else ("—" if r["predicted_class"] in ("SKIPPED", "ERROR") else "❌")
        print(f"{r['star']:<25} {r['true_class']:<28} {r['predicted_class']:<28} {tick}")

    print("-" * 90)
    if n_attempted > 0:
        print(f"\nAccuracy on {n_attempted} attempted stars: {n_correct}/{n_attempted} ({n_correct/n_attempted*100:.0f}%)")

    # Save CSV
    import csv
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["star", "true_class", "predicted_class",
                                                "confidence", "ls_period", "known_period", "correct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {RESULTS_CSV}")
    print(f"Diagnostic plots saved to: {OUTPUT_DIR}/\n")

    print("NOTE: If the model misclassifies several stars above, consider retraining")
    print("on real labeled light curves. The best public source is the OGLE catalog:")
    print("  https://ogle.astrouw.edu.pl/ogle4/OCVS/")


if __name__ == "__main__":
    main()

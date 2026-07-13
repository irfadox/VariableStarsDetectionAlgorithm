# =================================================================================
# TRAIN_ON_OGLE.PY - Retrain CNN on Real OGLE Labeled Light Curves
# For a conceptual explanation of how this fits into the project, see:
# walkthrough_guide.md (at the repository or workspace root)
# =================================================================================

import os
import glob
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from astropy.timeseries import LombScargle
from sklearn.model_selection import train_test_split

try:
    from src.model import LightCurveCNN
    from src.engine import train_epoch, test_epoch
except ImportError:
    print("ERROR: Run this script from the repository root directory.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OGLE_DATA_DIR   = "data/ogle_training"
MODEL_SAVE_PATH = "models/star_classifier.pth"
SEQ_LEN         = 100
BATCH_SIZE      = 32
EPOCHS          = 50
LR              = 1e-3
N_SYNTHETIC_NOISE = 500
MIN_POINTS      = 15

LABEL_MAP = {
    "cepheid":          0,
    "rrlyr":            1,
    "eclipsing_binary": 2,
    "lpv":              3,
    "noise":            4,
}
CLASS_NAMES = {v: k for k, v in LABEL_MAP.items()}

def load_ogle_file(filepath: str):
    try:
        data = np.loadtxt(filepath, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 2:
            return None, None
        times = data[:, 0].astype(np.float32)
        mags  = data[:, 1].astype(np.float32)
        valid = np.isfinite(times) & np.isfinite(mags)
        return times[valid], mags[valid]
    except Exception:
        return None, None

def process_lightcurve(times, mags, seq_len=SEQ_LEN):
    if times is None or len(times) < MIN_POINTS:
        return None

    try:
        baseline = times[-1] - times[0]
        min_freq = max(1.0 / baseline, 1.0 / 2000.0)
        max_freq = 25.0

        frequency, power = LombScargle(times, mags).autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=2,
            nyquist_factor=1
        )

        if len(power) == 0:
            return None

        best_period = 1.0 / frequency[np.argmax(power)]

        phases   = (times / best_period) % 1.0
        sort_idx = np.argsort(phases)
        grid     = np.linspace(0.0, 1.0, seq_len)
        interp   = np.interp(grid, phases[sort_idx], mags[sort_idx])

        mn, mx = interp.min(), interp.max()
        if mx - mn < 1e-6:
            normed = np.zeros(seq_len, dtype=np.float32)
        else:
            normed = ((interp - mn) / (mx - mn)).astype(np.float32)

        return normed
    except Exception:
        return None

def generate_noise_curves(n: int, seq_len: int = SEQ_LEN) -> list:
    curves = []
    for _ in range(n):
        noise_level = np.random.uniform(0.01, 0.05)
        flat_curve  = np.random.normal(0.5, noise_level, seq_len).astype(np.float32)
        flat_curve  = np.clip(flat_curve, 0.0, 1.0)
        curves.append(flat_curve)
    return curves

class OGLEDataset(Dataset):
    def __init__(self, sequences: list, labels: list):
        self.sequences = sequences
        self.labels    = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq   = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx],    dtype=torch.long)
        return seq, label

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_sequences = []
    all_labels    = []
    class_counts  = {}

    for class_label, class_idx in LABEL_MAP.items():
        if class_label == "noise":
            continue

        class_dir = os.path.join(OGLE_DATA_DIR, class_label)
        if not os.path.isdir(class_dir):
            print(f"WARNING: No directory found for class '{class_label}' at {class_dir}")
            continue

        files = glob.glob(os.path.join(class_dir, "*.dat"))
        print(f"\nProcessing {len(files)} files for class: {class_label}...")

        n_ok = 0
        for fpath in files:
            times, mags = load_ogle_file(fpath)
            seq = process_lightcurve(times, mags)
            if seq is not None:
                all_sequences.append(seq)
                all_labels.append(class_idx)
                n_ok += 1

        class_counts[class_label] = n_ok
        print(f"  Successfully processed: {n_ok}/{len(files)}")

    print(f"\nGenerating {N_SYNTHETIC_NOISE} synthetic Non-Variable curves...")
    noise_curves = generate_noise_curves(N_SYNTHETIC_NOISE)
    all_sequences.extend(noise_curves)
    all_labels.extend([LABEL_MAP["noise"]] * N_SYNTHETIC_NOISE)
    class_counts["noise"] = N_SYNTHETIC_NOISE

    X_train, X_val, y_train, y_val = train_test_split(
        all_sequences, all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )
    print(f"\nTrain: {len(X_train)} samples  |  Val: {len(X_val)} samples")

    train_ds = OGLEDataset(X_train, y_train)
    val_ds   = OGLEDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = LightCurveCNN(num_classes=5)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")

    print(f"\nTraining for {EPOCHS} epochs on real OGLE data...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        is_final = (epoch == EPOCHS)
        val_loss,  val_acc   = test_epoch(model, val_loader, criterion, device, print_metrics=is_final)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS}  Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.1f}%  |  Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  ✓ saved")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()

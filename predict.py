# Import standard os library to handle path checks
import os
# Import numpy for math operations and sorting
import numpy as np
# Import PyTorch library
import torch
# Import astropy Table for reading FITS tables
from astropy.table import Table
# Import Lomb-Scargle period finder
from astropy.timeseries import LombScargle
# Import the CNN model architecture
from src.model import LightCurveCNN

# Dictionary mapping class indices to human-readable names
CLASS_NAMES = {
    0: "Cepheid Variable",
    1: "RR Lyrae",
    2: "Eclipsing Binary",
    3: "Long-Period Variable (LPV)"
}

# Function to run inference on a single light curve file
def predict_star_class(file_path, model_weights_path="models/star_classifier.pth", seq_len=100):
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"Error: Light curve file '{file_path}' does not exist.")
        return None
        
    # Verify model weights exist
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights '{model_weights_path}' not found. Please train the model first.")
        return None

    # Load data from file
    print(f"Reading light curve: {file_path}...")
    if file_path.endswith(".fits"):
        table = Table.read(file_path)
        table.sort('time')
        times = np.array(table['time'], dtype=np.float32)
        mags = np.array(table['mag'], dtype=np.float32)
    else:
        # Load CSV
        data_array = np.genfromtxt(file_path, delimiter=',', names=True)
        times = data_array['time']
        mags = data_array['mag']
        
    # Lomb-Scargle period search
    print("Finding dominant period using Lomb-Scargle...")
    try:
        frequency, power = LombScargle(times, mags).autopower(minimum_frequency=0.01, maximum_frequency=10.0)
        best_freq = frequency[np.argmax(power)]
        best_period = 1.0 / best_freq
        print(f"Identified Period: {best_period:.4f} time units")
    except Exception as e:
        print(f"Lomb-Scargle failed ({e}). Falling back to period = 1.0")
        best_period = 1.0

    # Phase-fold
    phases = (times / best_period) % 1.0
    sort_indices = np.argsort(phases)
    sorted_phases = phases[sort_indices]
    sorted_mags = mags[sort_indices]
    
    # Interpolate to fixed sequence length
    grid_phases = np.linspace(0.0, 1.0, seq_len)
    interpolated_mags = np.interp(grid_phases, sorted_phases, sorted_mags, period=1.0)
    
    # Normalize magnitudes to [0.0, 1.0]
    min_v = interpolated_mags.min()
    max_v = interpolated_mags.max()
    if max_v > min_v:
        normed_mags = (interpolated_mags - min_v) / (max_v - min_v)
    else:
        normed_mags = np.zeros_like(interpolated_mags)
        
    # Format into float32 PyTorch tensor shape [1, 1, seq_len] (Batch=1, Channel=1, Length=100)
    input_tensor = torch.tensor(normed_mags, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Initialize model architecture
    model = LightCurveCNN(num_classes=4)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Run prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        # Apply Softmax to get class probabilities
        probabilities = torch.softmax(outputs, dim=1).squeeze().numpy()
        
    # Get highest probability index
    pred_class_idx = np.argmax(probabilities)
    pred_label = CLASS_NAMES[pred_class_idx]
    confidence = probabilities[pred_class_idx] * 100
    
    print("\n--- Prediction Results ---")
    print(f"Predicted Category: {pred_label}")
    print(f"Confidence score:   {confidence:.2f}%")
    
    # Print out distribution
    print("\nProbability Distribution:")
    for idx, name in CLASS_NAMES.items():
        print(f"  {name}: {probabilities[idx] * 100:.2f}%")
        
    return pred_label, confidence

# Main execution entry for debugging
if __name__ == "__main__":
    # Create a quick mock CSV star curve to test inference script locally
    mock_file = "test_star_curve.csv"
    print("Creating mock star light curve for testing...")
    
    # Generate mock Cepheid variable curve (sawtooth profile)
    np.random.seed(42)
    t = np.sort(np.random.uniform(0, 50, 80))
    # Sawtooth approximation
    m = 15.0 + 0.8 * (np.sin(t * 0.2) + 0.3 * np.sin(t * 0.4)) + np.random.normal(0, 0.02, 80)
    
    # Save CSV
    with open(mock_file, 'w') as f:
        f.write("time,mag\n")
        for times, mags in zip(t, m):
            f.write(f"{times},{mags}\n")
            
    # Run prediction test
    predict_star_class(mock_file)
    
    # Clean up test file
    if os.path.exists(mock_file):
        os.remove(mock_file)

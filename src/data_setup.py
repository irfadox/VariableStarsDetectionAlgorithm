# Import standard os library to handle paths
import os
# Import numpy for math operations and interpolation
import numpy as np
# Import PyTorch library
import torch
# Import standard PyTorch Dataset class
from torch.utils.data import Dataset
# Import astropy Table for reading astronomical catalogs
from astropy.table import Table

# Helper function showing how we parse real FITS/VOTable files with astropy
def parse_real_light_curve(file_path, time_col='time', mag_col='mag'):
    # Load the table dynamically from FITS/VOTable/ASCII format
    table = Table.read(file_path)
    
    # Sort the table rows chronologically by the time column
    table.sort(time_col)
    
    # Extract times and magnitudes as raw NumPy float arrays
    times = np.array(table[time_col], dtype=np.float32)
    mags = np.array(table[mag_col], dtype=np.float32)
    
    # Return sorted time and magnitude coordinates
    return times, mags

# Generate synthetic light curve coordinates for testing pipeline
def generate_mock_light_curve(star_type, length=80):
    # Set up random phase shift
    phase = np.random.uniform(0, 2 * np.pi)
    
    # Set up an irregular sequence of time observations
    times = np.sort(np.random.uniform(0, 50, length))
    
    # Model target curves depending on the variable star class
    if star_type == 0:  # Cepheid: asymmetric sawtooth-like curve
        mags = 15.0 + 0.8 * (np.sin(times * 0.2 + phase) + 0.3 * np.sin(times * 0.4 + phase))
    elif star_type == 1:  # RR Lyrae: faster pulsation period
        mags = 19.0 + 0.5 * (np.sin(times * 0.8 + phase) + 0.2 * np.sin(times * 1.6 + phase))
    elif star_type == 2:  # Eclipsing Binary: flat profile with primary/secondary dips
        mags = 16.0 - 0.6 * np.abs(np.sin(times * 0.1 + phase))**8
    elif star_type == 3:  # Long-Period Variable (LPV): slow, high-amplitude sine wave
        mags = 12.0 + 3.0 * np.sin(times * 0.03 + phase)
    else:  # Non-Variable / Noise: flat line with white noise
        mags = np.full(length, 15.0, dtype=np.float32)
        
    # Add minor Gaussian noise to simulate telescope observation uncertainty
    noise_std = 0.15 if star_type == 4 else 0.05
    mags += np.random.normal(0, noise_std, length)
    
    # Return time coordinates and simulated magnitudes
    return times.astype(np.float32), mags.astype(np.float32)
 
# Custom PyTorch Dataset to handle light curve sequences
class LightCurveDataset(Dataset):
    # Initialize the dataset with parameters
    def __init__(self, num_samples=100, seq_len=100, is_mock=True, data_dir=None):
        # Store fixed target sequence length
        self.seq_len = seq_len
        # Store mode option
        self.is_mock = is_mock
        # Store data files directory
        self.data_dir = data_dir
        # Store total sample count
        self.num_samples = num_samples
        
        # If mock mode, pre-generate simulated curves and labels
        if self.is_mock:
            self.data = []
            self.labels = []
            for _ in range(num_samples):
                # Pick a random variable star class [0, 1, 2, 3, 4]
                label = np.random.randint(0, 5)
                # Generate irregular curve
                times, mags = generate_mock_light_curve(label)
                # Append data and class labels
                self.data.append((times, mags))
                self.labels.append(label)
        else:
            # Look for fits/csv tables in the specified data folder
            import glob
            self.file_paths = glob.glob(os.path.join(self.data_dir, "*.fits")) + glob.glob(os.path.join(self.data_dir, "*.csv"))
            self.file_paths.sort()
            self.num_samples = len(self.file_paths)

    # Return total length of dataset
    def __len__(self):
        # Return total samples size
        return self.num_samples

    # Load, interpolate, normalize and return a single data item
    def __getitem__(self, idx):
        # Fetch raw times and magnitude coordinates
        if self.is_mock:
            times, mags = self.data[idx]
            label = self.labels[idx]
        else:
            file_path = self.file_paths[idx]
            # Parse real light curve values from table
            if file_path.endswith(".fits"):
                times, mags = parse_real_light_curve(file_path, time_col='time', mag_col='mag')
            else:
                # Load CSV files using standard numpy reader
                data_array = np.genfromtxt(file_path, delimiter=',', names=True)
                times = data_array['time']
                mags = data_array['mag']
                
            # Classify label depending on filename prefix convention (e.g. cepheid_star_01.csv)
            filename = os.path.basename(file_path).lower()
            if "cepheid" in filename:
                label = 0
            elif "rrlyrae" in filename:
                label = 1
            elif "eb" in filename or "binary" in filename:
                label = 2
            elif "lpv" in filename or "longperiod" in filename:
                label = 3
            else:
                label = 4

            
        # Use Lomb-Scargle periodogram to find the dominant frequency (period) of the light curve
        from astropy.timeseries import LombScargle
        
        try:
            # Run autopower to compute periodogram
            frequency, power = LombScargle(times, mags).autopower(minimum_frequency=0.01, maximum_frequency=10.0)
            # Find the best frequency corresponding to highest power peak
            best_freq = frequency[np.argmax(power)]
            best_period = 1.0 / best_freq
        except Exception:
            # Fallback to period of 1.0 if calculation fails
            best_period = 1.0

        # Phase-fold the irregular times: Phase = (Time / Period) mod 1
        phases = (times / best_period) % 1.0
        
        # Sort both phases and magnitudes based on the phases order
        # This aligns the points in phase-space, revealing the clean pulsation curve
        sort_indices = np.argsort(phases)
        sorted_phases = phases[sort_indices]
        sorted_mags = mags[sort_indices]
        
        # Define a uniform phase grid strictly from 0.0 to 1.0
        grid_phases = np.linspace(0.0, 1.0, self.seq_len)
        
        # Interpolate the phase-folded magnitude profile onto the uniform phase grid
        # Periodicity boundary handling: wrap around using period=1.0 behavior
        interpolated_mags = np.interp(grid_phases, sorted_phases, sorted_mags, period=1.0)
        
        # Min-max normalize magnitude values per curve to range [0.0, 1.0]
        min_v = interpolated_mags.min()
        max_v = interpolated_mags.max()
        if max_v > min_v:
            normed_mags = (interpolated_mags - min_v) / (max_v - min_v)
        else:
            normed_mags = np.zeros_like(interpolated_mags)
            
        # Convert preprocessed sequence to PyTorch float32 tensor
        # Reshape to [1, seq_len] representing [Channels, Sequence_Length] for Conv1D
        tensor_data = torch.tensor(normed_mags, dtype=torch.float32).unsqueeze(0)
        # Convert label index to long integer tensor
        tensor_label = torch.tensor(label, dtype=torch.long)
        
        # Return processed curve data and category label
        return tensor_data, tensor_label


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
    else:  # Long-Period Variable (LPV): slow, high-amplitude sine wave
        mags = 12.0 + 3.0 * np.sin(times * 0.03 + phase)
        
    # Add minor Gaussian noise to simulate telescope observation uncertainty
    mags += np.random.normal(0, 0.05, length)
    
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
                # Pick a random variable star class [0, 1, 2, 3]
                label = np.random.randint(0, 4)
                # Generate irregular curve
                times, mags = generate_mock_light_curve(label)
                # Append data and class labels
                self.data.append((times, mags))
                self.labels.append(label)

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
            # Placeholder illustrating how files would load from folder directory
            # list files, read target index, fetch label from name structure
            raise NotImplementedError("Real file path ingestion requires local catalog folder.")
            
        # Interpolate irregular time measurements onto a fixed grids
        # Define 100 evenly spaced points from first to last observation
        grid_times = np.linspace(times[0], times[-1], self.seq_len)
        # Interpolate magnitudes
        interpolated_mags = np.interp(grid_times, times, mags)
        
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

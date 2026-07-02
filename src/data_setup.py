# Import standard operating system interface module
import os
# Import glob for pattern-matching file paths
import glob
# Import PyTorch library
import torch
# Import the base Dataset class from PyTorch
from torch.utils.data import Dataset
# Import NumPy for array operations
import numpy as np
# Import fits module from astropy to read FITS format space files
from astropy.io import fits

# Define a custom PyTorch dataset class for handling PHAT FITS images
class PHATFitsDataset(Dataset):
    # Initialize the dataset with directory path and optional file extension
    def __init__(self, data_dir, extension="*.fits"):
        # Store the target data directory
        self.data_dir = data_dir
        # Find all files matching the extension in the directory
        self.file_paths = glob.glob(os.path.join(data_dir, extension))
        # Ensure we sort the files to maintain a consistent index order
        self.file_paths.sort()

    # Define the method to get the total number of items in the dataset
    def __len__(self):
        # Return the count of matching FITS file paths found
        return len(self.file_paths)

    # Define the method to load, preprocess, and return a single data item
    def __getitem__(self, idx):
        # Retrieve the specific file path corresponding to the given index
        file_path = self.file_paths[idx]
        
        # Open the FITS file safely using astropy.io.fits context manager
        with fits.open(file_path) as hdul:
            # Extract the raw image data from the primary HDU (usually index 0)
            # Cast the numpy array data to float32 immediately to prepare for PyTorch
            raw_data = hdul[0].data.astype(np.float32)
        
        # Replace all NaN (Not a Number) values in the image array with 0.0
        # This prevents bad pixels from causing NaN loss or gradients during training
        processed_data = np.nan_to_num(raw_data, nan=0.0)
        
        # Determine the 99.9th percentile value to use as an outlier clipping threshold
        # This effectively removes exceptionally bright pixels caused by cosmic rays
        threshold = np.percentile(processed_data, 99.9)
        
        # Clip/clamp the pixel values at the 99.9th percentile threshold
        # Values below 0.0 are kept as-is, and any value exceeding the threshold is capped
        clipped_data = np.clip(processed_data, a_min=None, a_max=threshold)
        
        # Find the minimum pixel value in the clipped array
        min_val = clipped_data.min()
        # Find the maximum pixel value in the clipped array
        max_val = clipped_data.max()
        
        # Normalize the clipped array strictly between 0.0 and 1.0
        # Check if max and min are different to prevent division by zero in flat images
        if max_val > min_val:
            # Perform min-max scaling to shift and scale the range to [0.0, 1.0]
            normalized_data = (clipped_data - min_val) / (max_val - min_val)
        else:
            # If the image is completely flat, fill it with zeros
            normalized_data = np.zeros_like(clipped_data)
            
        # Convert the preprocessed NumPy array into a PyTorch tensor
        tensor_data = torch.from_numpy(normalized_data)
        
        # Add a channel dimension at the beginning, producing shape [1, Height, Width]
        tensor_data = tensor_data.unsqueeze(0)
        
        # Determine the target label for classification based on the filename
        # If 'variable' is in the filename, assign class 1 (variable star)
        # Otherwise, assign class 0 (background noise / non-variable star)
        if "variable" in os.path.basename(file_path).lower():
            # Create a label tensor of value 1.0 (binary classification target)
            label = torch.tensor(1.0, dtype=torch.float32)
        else:
            # Create a label tensor of value 0.0 (binary classification target)
            label = torch.tensor(0.0, dtype=torch.float32)
            
        # Return both the preprocessed image tensor and its corresponding label
        return tensor_data, label

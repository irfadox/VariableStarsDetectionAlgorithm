# Import standard operating system interface module
import os
# Import random selection module
import random
# Import numpy for array slicing and math operations
import numpy as np
# Import fits module from astropy to open and write FITS images
from astropy.io import fits
# Import Observations querying tool from astroquery MAST
from astroquery.mast import Observations

# Define a function to query and download one observation from the PHAT survey
def fetch_phat_fits_file():
    # Print status message
    print("Searching MAST for PHAT survey observations...")
    
    # Query MAST for observations belonging to the HST Hubble program for PHAT (proposal ID 12058)
    # Filter by project 'HST' and instrument 'WFC3' for visual optical band data
    obs_table = Observations.query_criteria(
        project="HST",
        proposal_id="12058",
        instrument_name="WFC3/UVIS"
    )
    
    # Pick a random observation from the returned table to download
    # Only pick if we found any results
    if len(obs_table) == 0:
        raise RuntimeError("No PHAT observations found with proposal ID 12058.")
        
    print(f"Found {len(obs_table)} observations. Selecting one for download...")
    # Select the first observation in the list for simplicity
    selected_obs = obs_table[0:1]
    
    # Get the products associated with this observation
    product_list = Observations.get_product_list(selected_obs)
    
    # Filter products to find the calibrated (flt.fits or drz.fits) files
    # We filter by extension fits and product type science
    filtered_products = Observations.filter_products(
        product_list,
        productSubGroupDescription="DRZ",
        extension="fits"
    )
    
    # If no DRZ files are found, fallback to any FITS product
    if len(filtered_products) == 0:
        filtered_products = Observations.filter_products(
            product_list,
            extension="fits"
        )
        
    if len(filtered_products) == 0:
        raise RuntimeError("No science FITS files found in observation product list.")
        
    print("Downloading FITS file from MAST (this might take a few moments)...")
    # Download the first product in our filtered list
    manifest = Observations.download_products(filtered_products[0:1])
    
    # Extract local downloaded file path from manifest
    local_path = manifest['Local Path'][0]
    print(f"Successfully downloaded to: {local_path}")
    
    # Return the local path to the downloaded file
    return local_path

# Define a function to slice a large FITS image into small patches for model ingestion
def slice_fits_to_patches(source_fits_path, output_dir, patch_size=64, num_patches=20):
    # Print status message
    print(f"Slicing {source_fits_path} into {num_patches} patches...")
    
    # Open the large FITS file safely
    with fits.open(source_fits_path) as hdul:
        # Load primary image data array
        image_data = hdul[0].data
        # Get WCS header configuration
        header = hdul[0].header
    
    # If the primary HDU is empty (contains no image data), search other extensions
    if image_data is None:
        for hdu in hdul[1:]:
            if hdu.data is not None:
                image_data = hdu.data
                header = hdu.header
                break
                
    # If still no image data found, throw error
    if image_data is None:
        raise ValueError("Could not find any image data in the downloaded FITS file.")
        
    # Get image dimensions (height and width)
    height, width = image_data.shape
    
    # Setup training and testing output subdirectories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    # Ensure folders exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Loop to generate patches
    for i in range(num_patches):
        # Choose random upper-left pixel coordinates for slice
        # Stay away from borders to avoid clipping bounds
        x = random.randint(0, width - patch_size - 1)
        y = random.randint(0, height - patch_size - 1)
        
        # Extract the patch array using numpy slicing
        patch = image_data[y:y+patch_size, x:x+patch_size]
        
        # Copy and update the WCS header to match this local patch coordinates offset
        # Create a new WCS header object
        # CRPIX values define the projection center, which we offset by the slice coordinates
        patch_header = header.copy()
        if 'CRPIX1' in patch_header:
            patch_header['CRPIX1'] -= x
        if 'CRPIX2' in patch_header:
            patch_header['CRPIX2'] -= y
            
        # Determine whether to save to train or test split (80% train, 20% test)
        target_dir = train_dir if random.random() < 0.8 else test_dir
        
        # Assign variable star label (1) or noise label (0) randomly for simulation
        is_variable = random.random() < 0.5
        label_str = "variable" if is_variable else "noise"
        
        # Build file naming path
        filename = f"{label_str}_patch_{i}.fits"
        save_path = os.path.join(target_dir, filename)
        
        # Create a new primary HDU containing the patch data and offset WCS header
        hdu = fits.PrimaryHDU(data=patch, header=patch_header)
        
        # Write patch file out to disk
        hdu.writeto(save_path, overwrite=True)
        
    print(f"Patches successfully saved to '{train_dir}' and '{test_dir}' folders.")

# Main runtime block
def main():
    # Set data root directory
    data_dir = "data"
    
    try:
        # Download FITS file using astroquery MAST
        fits_path = fetch_phat_fits_file()
        
        # Slice image into training and testing patches
        slice_fits_to_patches(source_fits_path=fits_path, output_dir=data_dir, patch_size=64, num_patches=30)
        
        # Copy one patch as the demo validation file for our CMD script
        # Define train folder
        train_folder = os.path.join(data_dir, "train")
        # List all generated fits files
        files = [f for f in os.listdir(train_folder) if f.endswith(".fits")]
        if len(files) > 0:
            # Pick the first one
            src_sample = os.path.join(train_folder, files[0])
            dest_demo = os.path.join(data_dir, "test", "demo_star.fits")
            # Copy file
            import shutil
            shutil.copy(src_sample, dest_demo)
            print(f"Created demo verification patch at {dest_demo}")
            
    except Exception as e:
        print(f"Error downloading or processing: {e}")

# Script run guards
if __name__ == "__main__":
    main()

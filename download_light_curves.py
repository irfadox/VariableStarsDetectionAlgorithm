# Import standard os library to handle directory creation
import os
# Import astropy units to define coordinate query search radii
import astropy.units as u
# Import SkyCoord for astronomical coordinate translation
from astropy.coordinates import SkyCoord
# Import Observations module from astroquery MAST
from astroquery.mast import Observations

# Define a function to download real light curves from MAST archive around a target coordinate
def download_mast_light_curves(target_name="M31", output_dir="data/real_light_curves", max_files=5):
    # Print status message
    print(f"Querying MAST for variable star/time-series observations near target: {target_name}...")
    
    # Ensure target destination folder exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Search for MAST observations based on target coordinates (M31 Andromeda coordinates)
    # We look for time-series projects like K2, Kepler, TESS, or HST catalogs
    obs_table = Observations.query_object(objectname=target_name, radius="0.1 deg")
    
    # Filter observations to find time-series data collections
    # This filters specifically for missions producing stellar light curves
    timeseries_obs = obs_table[
        (obs_table['obs_collection'] == 'TESS') | 
        (obs_table['obs_collection'] == 'Kepler') | 
        (obs_table['obs_collection'] == 'K2') |
        (obs_table['project'] == 'HST')
    ]
    
    if len(timeseries_obs) == 0:
        # Fallback to downloading raw table search criteria if no direct matches
        print("No direct TESS/Kepler time-series matches. Downloading standard observations...")
        timeseries_obs = obs_table[0:max_files]
    else:
        print(f"Found {len(timeseries_obs)} time-series observations in target region.")
        timeseries_obs = timeseries_obs[0:max_files]
        
    # Get associated products list
    products = Observations.get_product_list(timeseries_obs)
    
    # Filter data products specifically for light curve files (typically ending in _lc.fits)
    filtered_products = Observations.filter_products(
        products,
        productSubGroupDescription="LC",
        extension="fits"
    )
    
    if len(filtered_products) == 0:
        # Fallback to any FITS product in observation list
        filtered_products = Observations.filter_products(
            products,
            extension="fits"
        )
        
    if len(filtered_products) == 0:
        print("No FITS data products found to download.")
        return []
        
    # Cap downloads to max_files parameter
    filtered_products = filtered_products[0:max_files]
    
    downloaded_paths = []
    print(f"Downloading up to {len(filtered_products)} light curve FITS files...")
    
    # Download each target product safely
    for idx, product in enumerate(filtered_products):
        try:
            # Download using astroquery MAST
            manifest = Observations.download_products(product)
            local_path = manifest['Local Path'][0]
            
            # Move/copy file to target output folder with standard names
            filename = f"star_lightcurve_{idx}.fits"
            dest_path = os.path.join(output_dir, filename)
            
            import shutil
            shutil.move(local_path, dest_path)
            print(f"Downloaded and saved: {dest_path}")
            downloaded_paths.append(dest_path)
        except Exception as e:
            print(f"Failed to download product {idx}: {e}")
            
    print(f"Completed! Downloaded {len(downloaded_paths)} files to '{output_dir}'.")
    return downloaded_paths

# Execution guards
if __name__ == "__main__":
    # Run download for 3 files as validation test
    download_mast_light_curves(target_name="M31", max_files=3)

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
    # We look for time-series projects like K2, Kepler, or TESS catalogs within a wider 0.5 degree radius
    obs_table = Observations.query_object(objectname=target_name, radius="0.5 deg")
    
    # Filter observations to find time-series data collections for distinct target stars
    timeseries_obs = obs_table[
        ((obs_table['obs_collection'] == 'TESS') | 
         (obs_table['obs_collection'] == 'Kepler') | 
         (obs_table['obs_collection'] == 'K2')) &
        (obs_table['target_name'] != 'TESS FFI') &
        (obs_table['target_name'] != '')
    ]
    
    # Filter out duplicate target stars to ensure distinct spatial coordinates on our map!
    unique_obs = []
    seen_targets = set()
    for row in timeseries_obs:
        target = row['target_name']
        if target not in seen_targets:
            seen_targets.add(target)
            unique_obs.append(row)
            
    if len(unique_obs) == 0:
        print("No direct TESS/Kepler time-series matches. Using standard observations...")
        unique_obs = obs_table
    else:
        print(f"Found {len(unique_obs)} unique time-series stars in target region.")
        
    downloaded_paths = []
    print(f"Searching and downloading up to {max_files} distinct light curve FITS files...")
    
    for obs in unique_obs:
        if len(downloaded_paths) >= max_files:
            break
        try:
            # Query products for this specific unique star
            products = Observations.get_product_list(obs)
            if len(products) == 0:
                continue
                
            # Filter specifically for light curves (LC)
            filtered_products = Observations.filter_products(
                products,
                productSubGroupDescription="LC",
                extension="fits"
            )
            
            if len(filtered_products) == 0:
                # Fallback to any FITS product
                filtered_products = Observations.filter_products(
                    products,
                    extension="fits"
                )
                
            if len(filtered_products) == 0:
                continue
                
            # Download the first available product for this observation
            product = filtered_products[0]
            manifest = Observations.download_products(product)
            if len(manifest) == 0 or 'Local Path' not in manifest.colnames:
                continue
            local_path = manifest['Local Path'][0]
            
            # Save file
            filename = f"star_lightcurve_{len(downloaded_paths)}.fits"
            dest_path = os.path.join(output_dir, filename)
            
            import shutil
            shutil.move(local_path, dest_path)
            print(f"Downloaded and saved: {dest_path}")
            downloaded_paths.append(dest_path)
        except Exception as e:
            print(f"Skipping observation due to error: {e}")
            continue
            
    print(f"Completed! Downloaded {len(downloaded_paths)} files to '{output_dir}'.")
    return downloaded_paths

# Execution guards
if __name__ == "__main__":
    # Run download for up to 8 files to build a rich spatial map of distinct stars
    download_mast_light_curves(target_name="M31", max_files=8)

# Import standard os library to handle paths
import os
# Import astropy fits module to read FITS headers
from astropy.io import fits
# Import WCS module to map pixel coordinates to sky coordinates (RA/Dec)
from astropy.wcs import WCS
# Import units module to specify angular radii (e.g., degrees or arcseconds)
import astropy.units as u
# Import coordinates module to handle sky coordinates objects
from astropy.coordinates import SkyCoord
# Import the Gaia catalog querying class from astroquery
from astroquery.gaia import Gaia
# Import PyTorch library
import torch
# Import standard plotting module
import matplotlib.pyplot as plt
# Import the CNN model definition
from src.model import VariableStarCNN

# Helper function to query Gaia data around specific sky coordinates
def query_gaia_catalog(ra, dec, radius_arcsec=10.0):
    # Print progress information to console
    print(f"Querying Gaia Catalog around RA: {ra:.5f}, Dec: {dec:.5f}...")
    
    # Create an astropy SkyCoord coordinate object representing target position
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    
    # Define search radius in arcseconds
    radius = radius_arcsec * u.arcsec
    
    # Perform a local cone search querying the remote Gaia database server
    job = Gaia.cone_search(coordinate=coord, radius=radius)
    
    # Retrieve the resulting table from the search job
    results = job.get_results()
    
    # Return table of matching Gaia sources
    return results

# Helper function to plot Color-Magnitude Diagram
def plot_cmd(gaia_table, match_idx=None, output_path="color_magnitude_diagram.png"):
    # Print starting diagram generation status
    print("Generating Color-Magnitude Diagram (CMD)...")
    
    # Extract BP minus RP color from the result table rows
    bp_rp = gaia_table['bp_rp']
    # Extract G-band mean magnitude values from the result table rows
    phot_g_mean_mag = gaia_table['phot_g_mean_mag']
    
    # Initialize a new matplotlib figure and subplots
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Scatter plot all retrieved stars in this field as semi-transparent grey dots
    ax.scatter(bp_rp, phot_g_mean_mag, color='gray', alpha=0.6, label='Field Stars (Gaia)', edgecolors='none')
    
    # Highlight the specific matched variable star index if it is provided
    if match_idx is not None and match_idx < len(gaia_table):
        # Draw a larger red star marker over the matched catalog entry
        ax.scatter(bp_rp[match_idx], phot_g_mean_mag[match_idx], color='red', marker='*', s=250, label='Identified Variable Star')
    
    # Set standard Color-Magnitude Diagram labels
    # Label the X-axis as Color index (BP - RP magnitude differences)
    ax.set_xlabel('Color (BP - RP)')
    # Label the Y-axis as Apparent G Magnitude
    ax.set_ylabel('Brightness (G-Mag)')
    
    # Invert the Y-axis since lower magnitudes denote brighter stars
    ax.invert_yaxis()
    
    # Display the grid lines for plotting assistance
    ax.grid(True, linestyle='--', alpha=0.5)
    # Enable plot legend placement
    ax.legend()
    # Add title description
    ax.set_title("Color-Magnitude Diagram (CMD)")
    
    # Save the output diagram file to target path
    plt.savefig(output_path, dpi=300)
    # Close figures to conserve memory usage
    plt.close()
    
    # Print status indicating image is written
    print(f"Color-Magnitude Diagram saved to {output_path}")

# Main execution loop
def main():
    # Define a demo placeholder FITS file path
    fits_file = "data/test/demo_star.fits"
    # Define local path where weights are saved
    model_weights_path = "models/variable_star_model.pth"
    
    # Output message if demo assets are missing
    if not os.path.exists(fits_file):
        print(f"Please place a valid FITS file at '{fits_file}' to run validation.")
        return
        
    # Open FITS file to load header information
    with fits.open(fits_file) as hdul:
        # Load the main image header
        header = hdul[0].header
        # Load raw image numpy array
        raw_data = hdul[0].data
        
    # Read spatial pixel width and height dimensions
    h, w = raw_data.shape
    # Set central star pixel location coordinates
    star_x = w / 2.0
    star_y = h / 2.0
    
    # Initialize Astropy WCS module from file header information
    wcs = WCS(header)
    
    # Convert pixel coords (star_x, star_y) into physical Sky coordinates (RA, Dec)
    # origin=0 is used since numpy dimensions are 0-indexed
    ra, dec = wcs.all_pix2world([[star_x, star_y]], 0)[0]
    
    # Print coordinates mapped via WCS
    print(f"Mapped Star Pixel Coordinates ({star_x}, {star_y}) to RA: {ra:.5f}, Dec: {dec:.5f}")
    
    # Load the CNN Model Architecture definition
    model = VariableStarCNN()
    
    # If saved model weights file is available, load them
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
        print("Loaded trained model weights.")
    else:
        print("Model weights not found. Running with randomly initialized weights for validation check.")
        
    # Prepare image array data (identical to data_setup preprocessing pipeline steps)
    processed = np.nan_to_num(raw_data.astype(np.float32), nan=0.0)
    # Clip cosmic ray pixel outliers
    threshold = np.percentile(processed, 99.9)
    clipped = np.clip(processed, None, threshold)
    # Min-max normalize
    min_v, max_v = clipped.min(), clipped.max()
    normed = (clipped - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(clipped)
    # Format into [Batch, Channels, Height, Width] Float32 tensor for CNN ingestion
    input_tensor = torch.from_numpy(normed).unsqueeze(0).unsqueeze(0)
    
    # Evaluate model predictions
    model.eval()
    with torch.no_grad():
        # Retrieve raw prediction logit output
        output = model(input_tensor)
        # Apply sigmoid to calculate probability
        probability = torch.sigmoid(output).item()
        
    print(f"Model Variable Star Probability: {probability * 100:.2f}%")
    
    # Check if probability qualifies star as variable
    if probability >= 0.5:
        print("AI Predicts: Variable Star detected!")
        
        # Query Gaia catalog database
        gaia_table = query_gaia_catalog(ra=ra, dec=dec, radius_arcsec=15.0)
        
        # If stars are retrieved from the query table, locate closest catalog matches
        if len(gaia_table) > 0:
            print(f"Found {len(gaia_table)} Gaia sources in target field.")
            # Set target closest match index as first entry (index 0)
            target_idx = 0
            # Retrieve matching target star properties
            star_g = gaia_table['phot_g_mean_mag'][target_idx]
            star_color = gaia_table['bp_rp'][target_idx]
            print(f"Matching Gaia Star: G-Mag={star_g:.2f}, Color={star_color:.2f}")
            
            # Generate the plot diagram
            plot_cmd(gaia_table, match_idx=target_idx, output_path="color_magnitude_diagram.png")
        else:
            print("No Gaia catalog entries found within search radius.")
    else:
        print("AI Predicts: Background noise / non-variable star. Skipping catalog match.")

# Run main script execution guards
if __name__ == "__main__":
    # Call validation script execution handler
    main()

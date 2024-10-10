import rasterio
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import argparse

"""
Script to plot generated quantiles of the displacements.

$ python src/displacements/plot_displacements.py generated/messina_001_20241010_134621 0 0.5 0.8 0.9
"""
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot displacements.")

    # Add arguments
    parser.add_argument(
        'rundir', 
        type=str, 
        help="Path to the rundir (must be a valid directory)."
    )
    parser.add_argument(
        'shakemap', 
        type=int, 
        help="Integer representing the selected shakemap."
    )
    parser.add_argument(
        'quantiles', 
        type=float, 
        nargs='+',  # Accept one or more quantiles
        help="List of quantiles (must be floats)."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Check if rundir is a valid directory
    if not os.path.isdir(args.rundir):
        parser.error(f"'{args.rundir}' is not a valid directory.")
    
    # Validate quantiles are between 0 and 1
    for q in args.quantiles:
        if q < 0 or q > 1:
            parser.error(f"Quantile {q} is out of bounds. Must be between 0 and 1.")

    return args


def plot(rundir, shakemap=0, quantiles=[0.9, 0.8, 0.5]):
    
    working_dir = os.path.join(rundir, "displacements")
    # load json file
    with open(os.path.join(working_dir, "content.json"),'r') as f:
        content = json.load(f)

    # Select quantiles and shakemap_sample to plot.
    content = [e for e in content if e["quantile"] in quantiles and e["shakemap_sample"] in [0]]

    raster_paths = [os.path.join(working_dir, e["file"]) for e in content]
    titles = ["{}-quantile of {}".format(e["quantile"], e["value"]) for e in content]

    # Initialize lists to store raster data and no-data values
    rasters = []
    nodata_vals = []

    # Read all rasters and store the data
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            nodata_vals.append(src.nodata)
            rasters.append(raster_data)

    # Get the global min and max across all rasters, ignoring no-data values
    global_min = min([np.nanmin(r) for r in rasters])
    global_max = min([np.nanmax(r) for r in rasters])

    # Create a figure with subplots, one for each raster
    fig, axes = plt.subplots(1, len(raster_paths), figsize=(16, 6), layout='compressed')

    # Loop through each raster and plot
    for i, raster_data in enumerate(rasters):
        # Mask no-data values
        raster_data = np.ma.masked_equal(raster_data, nodata_vals[i])
        
        # Plot the raster with a shared color scale (global_min, global_max)
        img = axes[i].imshow(raster_data, cmap='tab20b_r', vmin=global_min, vmax=global_max)
        cnts = axes[i].contour(raster_data, colors='k', levels=[np.log(5.)], origin='lower')
        #img = axes[i].imshow(raster_data, cmap='tab20b')
        
        # Add title
        axes[i].set_title(titles[i], loc="left")

    # Add a single colorbar for the entire figure
    cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.08, pad=0.01)
    cbar.set_label('Displacements')

    # Adjust layout and show the plot
    #plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "displacements.png"))


if __name__ == "__main__":
    # Parse and retrieve the arguments
    args = parse_args()
    plot(args.rundir, shakemap=args.shakemap, quantiles=args.quantiles)
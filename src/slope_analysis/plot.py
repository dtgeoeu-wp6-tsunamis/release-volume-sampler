import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import argparse
import logging


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('plot')

"""
Script to plot generated quantiles of the yield acelleration or fos.

Description of plot:
Different quantiles for the slope aligned yield-aceleration as calculated using inifinite slope analaysis for the Messina strait. 
The uncertainty stems from uncertainty of the friction angle, cohesion, density and the thickness of the sliding surface.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Plot yield accellerations.")

    # Add arguments
    parser.add_argument(
        'rundir', 
        type=str, 
        help="Path to the rundir (must be a valid directory)."
    )
    parser.add_argument(
        'quantiles', 
        type=float, 
        nargs='+',  # Accept one or more quantiles
        help="List of quantiles (must be floats)."
    )
    parser.add_argument(
        '--value',
        type=str,
        help="Type of output to plot.",
        default="yield_acceleration"
    )
    parser.add_argument(
        '--logscale',
        action=argparse.BooleanOptionalAction, 
        type=bool,
        help="Plot values in logscale.",
        default=False
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


def plot(rundir, value, quantiles, logscale):    
    working_dir = os.path.join(rundir, "slope_analysis")
    
    # load json file
    with open(os.path.join(working_dir, "content.json"),'r') as f:
        content = json.load(f)

    # Select quantiles and value to plot.
    content = [e for e in content if e["quantile"] in quantiles and e["value"] == value]
    titles = ["{}-quantile of {}".format(e["quantile"], e["value"]) for e in content]
    
    logger.info(f"plot titles: {titles}")

    # Initialize lists to store raster data and no-data values
    rasters = []
    nodata_vals = []

    # Read all rasters and store the data
    for e in content:
        raster_path = os.path.join(working_dir, e["file"])
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            if not logscale and e["scale"] == "log10":
                raster_data = 10**raster_data
            nodata_vals.append(src.nodata)
            rasters.append(raster_data)

    # Get the global min and max across all rasters, ignoring no-data values
    global_min = min([np.nanmin(r) for r in rasters])
    global_max = min([np.nanmax(r) for r in rasters])

    # Create a figure with subplots, one for each raster
    fig, axes = plt.subplots(1, len(rasters), figsize=(16, 6), layout='compressed')

    # Loop through each raster and plot
    for i, raster_data in enumerate(rasters):
        # Mask no-data values
        raster_data = np.ma.masked_equal(raster_data, nodata_vals[i])
        
        # Plot the raster with a shared color scale (global_min, global_max)
        img = axes[i].imshow(raster_data, cmap='tab20b_r', vmin=global_min, vmax=global_max)
        
        # Add title
        axes[i].set_title(titles[i], loc="left")

    # Add a single colorbar for the entire figure
    cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.08, pad=0.01)
    cbar.set_label('Yield Acceleration')

    # Adjust layout and show the plot
    plt.savefig(os.path.join(working_dir, f"{value}_plot.png"))
    
    
if __name__ == "__main__":
    # Parse and retrieve the arguments
    args = parse_args()
    logger.info(f"args: {args}")
    plot(rundir=args.rundir, value=args.value, quantiles=args.quantiles, logscale=args.logscale)
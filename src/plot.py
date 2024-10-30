
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import numpy as np
import json
import argparse
import logging
from utils.utils import read_tif, create_dir

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('plot')

"""
Script to plot content of the generated folder.

Description of plot:
Different quantiles for the slope aligned yield-aceleration as calculated using inifinite slope analaysis for the Messina strait. 
The uncertainty stems from uncertainty of the friction angle, cohesion, density and the thickness of the sliding surface.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Plot all rasters in a given generated subfolder in a \
        unified colorscale using matplotlib.")

    # Add arguments
    parser.add_argument(
        'folder', 
        type=str, 
        help="Path to subfolder of generated directory."
    )
    
    parser.add_argument(
        '--m', 
        type=float, 
        default=-np.inf,
        help="Max value of colorbar."
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

    # Check if folder is a valid directory
    if not os.path.isdir(args.folder):
        parser.error(f"No such directory {args.folder}.")
    if not os.path.isfile(os.path.join(args.folder, "content.json")):
        parser.error(f"No content.json file in {args.folder}.")
        
    return args


def plot(working_dir, logscale, m):    
    # Create output folder
    plot_dir = os.path.join(working_dir, "plots")
    create_dir(plot_dir)
    
    # load json file
    with open(os.path.join(working_dir, "content.json"),'r') as f:
        content = json.load(f)

    # Initialize lists to store raster data and no-data values
    rasters = []
    nodata_vals = []

    # Read all rasters and store the data
    for e in content:
        raster_path = os.path.join(working_dir, e["file"])
        raster_data, msk, profile = read_tif(raster_path)
        
        
        if not logscale and e["scale"] == "log10":
            logger.info("Transform output from log10 scale.")
            rasters.append(10**raster_data)
        else:
            rasters.append(raster_data)
            

    # Get the global min and max across all rasters, ignoring no-data values
    global_min = min([np.nanmin(r) for r in rasters])
    global_max = max(max([np.nanmax(r) for r in rasters]), m)
    logger.info(f"global_min: {global_min}, global_max: {global_max}")

    # Loop through each raster and plot
    for i, e in enumerate(content):
        fig, ax = plt.subplots() 
        # Plot the raster with a shared color scale (global_min, global_max)
        img = ax.imshow(rasters[i], vmin=global_min, vmax=global_max, cmap='tab20b_r')
        #img = ax.imshow(rasters[i], norm=LogNorm())
        
        # Add title
        if "quantile" in e.keys():
            quantile = e["quantile"]
            ax.set_title(f"Quantile: {quantile}.", loc="left")
        if "sample" in e.keys():
            sample = e["sample"]
            ax.set_title(f"Sample: {sample}.", loc="left")
        if "threshold" in e.keys():
            threshold = e["threshold"]
            ax.set_title(f"Threshold: {threshold}.", loc="left")
    
        # Add a single colorbar for the entire figure
        #cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.08, pad=0.01)
        filename, value, unit = e["file"].replace(".tif",".png"), e["value"], e["unit"]
        fig.colorbar(img, ax=ax, label=f"{value}")

        # Adjust layout and show the plot
        fig.savefig(os.path.join(plot_dir, filename))
        plt.close()
    
    
if __name__ == "__main__":
    # Parse and retrieve the arguments
    args = parse_args()
    logger.info(f"args: {args}")
    plot(args.folder, logscale=args.logscale, m=args.m)
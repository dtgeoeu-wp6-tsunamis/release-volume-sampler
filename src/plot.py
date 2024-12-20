
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import numpy as np
import json
import argparse
from rvsampler.utils import read_tif, create_dir
from rvsampler.set_logg import setup_logger
import rasterio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter

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
        default=np.inf,
        help="Max value of colorbar."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Check if folder is a valid directory
    if not os.path.isdir(args.folder):
        parser.error(f"No such directory {args.folder}.")
    if not os.path.isfile(os.path.join(args.folder, "content.json")):
        parser.error(f"No content.json file in {args.folder}.")
        
    return args


def plot(working_dir, m):    
    # Create output folder
    plot_dir = os.path.join(working_dir, "plots")
    create_dir(plot_dir)
    
    logger = setup_logger("plot", plot_dir)
    
    # load json file
    with open(os.path.join(working_dir, "content.json"),'r') as f:
        content = json.load(f)

    global_min, global_max = get_global_maxmin(working_dir, content, m, logger)

    # Loop through each raster and plot
    for i, e in enumerate(content):
        # Set title
        if "quantile" in e.keys():
            quantile = e["quantile"]
            title = f"Quantile: {quantile}."
        if "sample" in e.keys():
            sample = e["sample"]
            title = f"Sample: {sample}."
        if "threshold" in e.keys():
            threshold = e["threshold"]
            title = f"Threshold: {threshold}."
        
        # Make plot
        plot_raster(working_dir, plot_dir, e["file"], global_min, global_max, e["value"], e["unit"], title, logger)


def get_global_maxmin(working_dir, content, m, logger):
    # Initialize lists to store raster data and no-data values
    mins, maxs =[], []

    # Read all rasters and store the data
    for e in content:
        raster_path = os.path.join(working_dir, e["file"])
        raster_data, msk, profile = read_tif(raster_path)
        mins.append(np.nanmin(raster_data))
        maxs.append(np.nanmax(raster_data))
        

    # Get the global min and max across all rasters, ignoring no-data values
    global_min, global_max = min(mins), min(max(maxs), m)
    logger.info(f"global_min: {global_min}, global_max: {global_max}")
    return(global_min, global_max)


def plot_raster(working_dir, plot_dir, file, global_min, global_max, value, unit, title, logger):
    
    # Load raster data
    with rasterio.open(os.path.join(working_dir, file)) as src:
        data = src.read(1)
        bounds = src.bounds
        crs = src.crs

    # Convert bounds to extent
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # Create a map with the raster CRS
    if crs.is_geographic:
        projection = ccrs.PlateCarree()
    else:
        projection = ccrs.epsg(crs.to_epsg())
         

    # Plot raster with Cartopy
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection':projection})
    img = ax.imshow(
        data,
        origin='upper',
        vmin=global_min,
        vmax=global_max,
        extent=extent,
        transform=projection,
        cmap='tab20b_r'
    )
    ax.set_title(title, loc="left")
    
    # Add map features
    #ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Calculate tick values dynamically based on raster bounds
    lon_ticks = np.linspace(bounds.left, bounds.right, num=5)  # 5 ticks along longitude
    lat_ticks = np.linspace(bounds.bottom, bounds.top, num=5)  # 5 ticks along latitude
    
    # Add gridlines
    gridlines = ax.gridlines(
        draw_labels=True,  # Show labels on gridlines
        xlocs=lon_ticks,  # X ticks (longitudes)
        ylocs=lat_ticks,  # Y ticks (latitudes)
        linewidth=0.5,
        color='gray',
        linestyle='--'
    )

    # Customize tick labels
    gridlines.xlabel_style = {'size': 10, 'color': 'black'}
    gridlines.ylabel_style = {'size': 10, 'color': 'black'}

    gridlines.right_labels = False

    # Add a single colorbar for the entire figure
    #cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.08, pad=0.01)
    fig.colorbar(img, ax=ax, label=f"{value}")

    # Adjust layout and show the plot
    filename = os.path.join(plot_dir, file.replace(".tif",".png"))
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title 
    fig.savefig(filename)
    logger.info(f"Created figure: {filename}")
    plt.close()
    

    
if __name__ == "__main__":
    # Parse and retrieve the arguments
    args = parse_args()
    plot(args.folder, m=args.m)
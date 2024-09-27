import rasterio
import matplotlib.pyplot as plt
import os
import numpy as np
import json

"""
Script to plot generated quantiles of the displacements.
"""

# Filepaths for your rasters
working_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001_20240923_121008/release_volumes"

# load json file
with open(os.path.join(working_dir, "content.json"),'r') as f:
    content = json.load(f)

# Select quantiles to plot.
content = [e for e in content if e["quantile"] in [0.99, 0.9, 0.8]]


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
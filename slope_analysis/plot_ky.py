import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import os
import numpy as np

"""
Script to plot generated quantiles of the yield acelleration.

Description of plot:
Different quantiles for the slope aligned yield-aceleration as calculated using inifinite slope analaysis for the Messina strait. 
The uncertainty stems from uncertainty of the friction angle, cohesion, density and the thickness of the sliding surface.
"""


# Filepaths for your rasters
working_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001_20240909_130842/slope_analysis"
raster_paths = [os.path.join(working_dir, file) for file in ['ky_0.1.tif', 'ky_0.2.tif', 'ky_0.5.tif']]
titles = ["0.1-quantile of $k_y$", "0.2-quantile of $k_y$", "0.5-quantile of k_y"]

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
    #img = axes[i].imshow(raster_data, cmap='tab20b')
    
    # Add title
    axes[i].set_title(titles[i], loc="left")

# Add a single colorbar for the entire figure
cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.08, pad=0.01)
cbar.set_label('Yield Acceleration')

# Adjust layout and show the plot
#plt.tight_layout()
plt.savefig(os.path.join(working_dir, "ky_plot.png"))
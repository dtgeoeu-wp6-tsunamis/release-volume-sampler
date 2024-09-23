import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
import math

# Filepaths for your rasters
working_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001_20240923_121008/cluster_analysis"
number_of_patches = [2000, 1000, 500, 200]
raster_paths = [os.path.join(working_dir,f'patches_{n_patches}.tif') for n_patches in number_of_patches]

# Create a figure with subplots, one for each raster
fig, axes = plt.subplots(2, math.ceil(len(raster_paths)/2), figsize=(12, 10))

# Loop through each raster and plot
for i, raster_path in enumerate(raster_paths):
    # Open the raster
    with rasterio.open(raster_path) as src:
        # Read the first band
        raster_data = src.read(1)
        
        # Mask out any no-data values
        raster_data = np.ma.masked_equal(raster_data, src.nodata)
        position = (math.floor(i/2), i%2)

        img = axes[position[0], position[1]].imshow(raster_data, cmap='flag')
        
        # Add title
        axes[position[0], position[1]].set_title(f'{number_of_patches[i]} patches', loc="left")
        
        # Add a colorbar
        #cbar = fig.colorbar(img, ax=axes[i], orientation='vertical')
        #cbar.set_label('Value')

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(os.path.join(working_dir,"grid.png"))

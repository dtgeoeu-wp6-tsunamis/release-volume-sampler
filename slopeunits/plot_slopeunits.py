import rasterio
from pyproj import CRS, Transformer
import numpy as np
import os
#import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#from matplotlib import ticker
import plotly.graph_objects as go
import argparse


parser = argparse.ArgumentParser(description="Release volume sampler")
parser.add_argument('--bathy_file', required=True, help='Bathymetri file, geotiff, lat/lon coordaintes')
parser.add_argument('--run_path', required=True, help='Slopeunits folder.')
args = parser.parse_args()
bathy_file = args.bathy_file
run_path = args.run_path

bathy_file = bathy_file[:-4]+"_projected_truncated.tif"
#bathy_file = r'/home/sfr/release-volume-sampler/input/bathy/localMessinaBathy.tif'
#run_path = r'/home/sfr/release-volume-sampler/slopeunits'

with rasterio.open(bathy_file) as src:
    # Read the elevation or data values (assume a single band)
    data = src.read(1)
    
    data[data == 0] = np.nan

    height, width = src.height, src.width
    transform = src.transform
    src_crs = src.crs  # should be UTM
    dst_crs = "EPSG:4326"  # WGS84

    # Create arrays of pixel coordinates
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')

    # Flatten for transformation
    xs_flat = np.array(xs).flatten()
    ys_flat = np.array(ys).flatten()

    # Create transformer from UTM to lat/lon
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    lons, lats = transformer.transform(xs_flat, ys_flat)

    # Reshape back to 2D arrays
    lon = np.array(lons).reshape((height, width))
    lat = np.array(lats).reshape((height, width))
    
#Res_dir = r'slopeunits/Results'
Res_dir = os.path.join(run_path,'Results')
Res_dir_out = Res_dir
Res_dirs = os.listdir(Res_dir)


for rd in Res_dirs:
    # Read result data
    dd = rasterio.open(os.path.join(Res_dir,rd,'slumap_clean.tif'))
    dataC = dd.read(1)
    # Remove nan data
    dataC[dataC > 60000] = 0

    # Split into 50 categories that each represent a color
    numC = 50
    a = np.arange(1,numC+1)
    c = 0;
    for i in range(np.max(dataC)):
        dataC[dataC == i+1] = a[c]
        c += 1
        if c > numC-1:
            c = 0

    # Plot the figure
    # Step 2: Create the 3D plot with Plotly
    fig = go.Figure(data=[go.Surface(z=data, x=lon, y=lat, surfacecolor = dataC, colorscale='Viridis')])
    
    # Customize layout
    fig.update_layout(
        title='3D GeoTIFF Elevation Plot',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='data'
        )
    )

    os.makedirs(os.path.join(Res_dir_out,rd), exist_ok=True)
    
    # Step 3: Save the plot as an interactive HTML file
    output_file = os.path.join(Res_dir_out,rd,'interactive_3d_geotiff_plot.html')
    fig.write_html(output_file)
    
    print(f'Interactive 3D plot saved as {output_file}')
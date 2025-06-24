import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from pyproj import CRS as PyCRS
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Release volume sampler")
parser.add_argument('--bathy_file', required=True, help='Bathymetri file, geotiff, lat/lon coordaintes')
args = parser.parse_args()
bathy_file = args.bathy_file

def get_utm_crs_from_latlon(lon, lat):
    """Return appropriate UTM CRS for given lon/lat."""
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    # For some reason it has to be 3065, Erlend might know why.
    epsg_code = 3065#32600 + utm_zone if is_northern else 32700 + utm_zone
    return CRS.from_epsg(epsg_code)

# Input/output paths
input_path = "/home/sfr/release-volume-sampler/input/bathy/localMessinaBathy.tif"
output_path = input_path[:-4]+'_projected_truncated.tif'
#output_path = "test.tif"

# Open the source GeoTIFF
with rasterio.open(input_path) as src:
    # Get center coordinates in lon/lat
    center_row = src.height // 2
    center_col = src.width // 2
    lon, lat = src.xy(center_row, center_col)
    
    # Auto-determine UTM CRS
    dst_crs = get_utm_crs_from_latlon(lon, lat)

    # Compute transform and new shape
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)

    # Update metadata
    kwargs = src.meta.copy()
    kwargs.update({
    'crs': dst_crs,
    'transform': transform,
    'width': width,
    'height': height,
    'nodata': 0  # 👈 Set nodata to 0
    })

    # Write the reprojected GeoTIFF
    # Write the reprojected GeoTIFF
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            dst_array = np.empty((height, width), dtype=src.meta['dtype'])

            reproject(
                source=rasterio.band(src, i),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

            # Truncate positive values
            dst_array[dst_array > 0] = 0
            #dst_array[dst_array < -5000] = 0

            # Ensure nodata values are set to 0
            if src.nodata is not None:
                dst_array[dst_array == src.nodata] = 0
            dst_array = np.nan_to_num(dst_array, nan=0)

            dst.write(dst_array, i)

print(f"✅ Reprojected GeoTIFF written to: {output_path}")

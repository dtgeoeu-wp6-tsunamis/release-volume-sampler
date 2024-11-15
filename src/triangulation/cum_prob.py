import rasterio
import json
import os
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("cumprob.py")

def main():
    """ Creates lookup table for the cummulative probability of the fos by triangle.
    """
    cummulative_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001/fos/cummulative"
    tri_mask_path = "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/triangulation_raster.tif"
    triangulation_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation"
    

    logger.info("loading triangulation mask")
    with rasterio.open(tri_mask_path) as src:
        tri_mask = src.read(1)

    n_triangles = int(tri_mask.max())
    logger.info(f"Number of triangles: {n_triangles}.")
    
    logger.info(f"Loads cummulative probabilities from {cummulative_dir}")
    with open(os.path.join(cummulative_dir, "content.json"),'r') as f:
        content = json.load(f)

    # Initialize lists to store raster data
    #rasters = []
    # Read all rasters and store the data
    #for e in content:
    #    raster_path = os.path.join(cummulative_dir, e["file"])
    #    raster_data, msk, profile = read_tif(raster_path)
    #    rasters.append(raster_data)

    triangle_cummulative_probs = np.empty((n_triangles, len(content)))
    thresholds = np.empty(len(content))

    #for raster_index, e in enumerate(content):
    #    thresholds.append(e['threshold'])
    #    for tri_index in range(n_triangles):
    #        triangle_cummulative_probs[tri_index, raster_index] = np.nanmin(rasters[raster_index][tri_mask == tri_index], initial=9999)
    
    # Parallel execution
    def process_raster(raster_index, e):
        raster_path = os.path.join(cummulative_dir, e["file"])
        logger.info(f"Process raster: {raster_path}")
        raster_data, msk, profile = read_tif(raster_path)
        
        # Prepare the output for a single raster
        cummulative_probs = np.full(n_triangles, np.nan)
        for tri_index in range(n_triangles):
            cummulative_probs[tri_index] = np.nanmin(raster_data[tri_mask == tri_index], initial=9999)
        
        return raster_index, e['threshold'], cummulative_probs
    
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_raster, raster_index, e): raster_index for raster_index, e in enumerate(content)
        }

        for future in as_completed(futures):
            raster_index, threshold, cummulative_probs = future.result()
            thresholds[raster_index] = threshold
            triangle_cummulative_probs[:, raster_index] = cummulative_probs
            logger.info(f"Processing raster {raster_index} is complete.")

    np.savez(os.path.join(triangulation_dir,"cummulative_fos.npz"), thresholds=thresholds, cummulative_probs=triangle_cummulative_probs) 



def read_tif(fname):
    "Read .tif data and profile using rasterio."
    #logger.info(f"Read file: {fname}")
    with rasterio.open(fname) as src:
        #data = np.ma.masked_equal(src.read(1), src.nodata)
        data = src.read(1)
        msk = np.where(src.read_masks(1) == src.nodata, False, True)
        profile = src.profile.copy()
    return data, msk, profile


if __name__ == "__main__":
    main()
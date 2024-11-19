import rasterio
import json
import os
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.utils import read_tif, setup_logger

def main():
    """ To ensure that modules are imports works, run the script as a module.
    
    release-volume-sampler$ python -m src.triangulation.cumprobs_by_triangle
    """
    
    rundir = "/home/ebr/projects/release-volume-sampler/generated/messina_001"
    caclulate_cummulative_logfos_probabilities(rundir)


def caclulate_cummulative_logfos_probabilities(rundir):
    """ Creates lookup table for the cummulative probability of the fos by triangle.
    """
    cummulative_dir = os.path.join(rundir, "slope_analysis", "fos", "cummulative")
    triangulation_dir = os.path.join(rundir, "triangulation")
    tri_mask_path = os.path.join(triangulation_dir, "triangulation.tif")
    
    logger = setup_logger("cumprob", triangulation_dir)

    logger.info("loading triangulation mask")
    with rasterio.open(tri_mask_path) as src:
        tri_mask = src.read(1)

    n_triangles = int(tri_mask.max())
    logger.info(f"Number of triangles: {n_triangles}.")
    
    logger.info(f"Loads cummulative probabilities from {cummulative_dir}")
    with open(os.path.join(cummulative_dir, "content.json"),'r') as f:
        content = json.load(f)

    triangle_cummulative_probs = np.empty((n_triangles, len(content)))
    thresholds = np.empty(len(content))
    
    # Parallel execution
    def process_raster(raster_index, e):
        raster_path = os.path.join(cummulative_dir, e["file"])
        logger.info(f"Process raster: {raster_path}")
        raster_data, msk, profile = read_tif(raster_path, logger)
        
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


if __name__ == "__main__":
    main()
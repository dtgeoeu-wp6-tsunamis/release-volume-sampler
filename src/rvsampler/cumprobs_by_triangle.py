import rasterio
import json
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import read_tif
from logging import setup_logger

def main():
    """ To ensure that modules are imports works, run the script as a module.
    
    release-volume-sampler$ python -m src.triangulation.cumprobs_by_triangle
    """
    # Cummulative probabilities for FOS
    rundir = "/home/ebr/projects/release-volume-sampler/generated/messina_001"
    params = {
        "rundir": rundir,
        "cummulative_dir": os.path.join(rundir, "slope_analysis", "fos", "cummulative"),
        "outfile_name": "cummulative_fos.npz"
    }
    caclulate_cummulative_probabilities(**params)
    
    # Exceedance probabilities for displacement.
    params = {
        "rundir": rundir,
        "cummulative_dir": os.path.join(rundir, "displacements"),
        "outfile_name": "exceedance_displacement.npz"
    }
    
    #caclulate_cummulative_probabilities(**params)


def caclulate_cummulative_probabilities(rundir, cummulative_dir, outfile_name):
    """ Creates lookup table.
    """
    
    triangulation_dir = os.path.join(rundir, "triangulation")
    outfile = os.path.join(triangulation_dir, outfile_name)
    tri_mask_path = os.path.join(triangulation_dir, "triangulation.tif")
    
    logger = setup_logger("cumprob", triangulation_dir)

    logger.info("loading triangulation mask")
    with rasterio.open(tri_mask_path) as src:
        tri_mask = src.read(1)

    n_triangles = int(tri_mask.max()) + 1
    logger.info(f"Number of triangles: {n_triangles}.")
    
    logger.info(f"Loads cummulative probabilities from {cummulative_dir}")
    with open(os.path.join(cummulative_dir, "content.json"),'r') as f:
        content = json.load(f)

    triangle_probs = np.empty((n_triangles, len(content)))
    thresholds = np.empty(len(content))
    
    # Parallel execution
    def process_raster(raster_index, e):
        raster_path = os.path.join(cummulative_dir, e["file"])
        logger.info(f"Process raster: {raster_path}")
        raster_data, msk, profile = read_tif(raster_path, logger)
        
        # Prepare the output for a single raster
        probs = np.full(n_triangles, np.nan)
        for tri_index in range(n_triangles):
            probs[tri_index] = np.nanmin(raster_data[tri_mask == tri_index], initial=9999)
        
        return raster_index, e['threshold'], probs
    
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_raster, raster_index, e): raster_index for raster_index, e in enumerate(content)
        }

        for future in as_completed(futures):
            raster_index, threshold, probs = future.result()
            thresholds[raster_index] = threshold
            triangle_probs[:, raster_index] = probs
            logger.info(f"Processing raster {raster_index} is complete.")

    np.savez(outfile, thresholds=thresholds, probs=triangle_probs) 


if __name__ == "__main__":
    main()
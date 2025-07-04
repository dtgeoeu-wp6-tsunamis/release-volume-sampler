import os
import numpy as np
import pandas as pd
import rasterio
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

from rvsampler.displacements import DisplacementProbabilityAggregator
from rvsampler.shakemaps_reader import ShakemapsReader
from rvsampler.utils import create_dir
from rvsampler.set_logg import setup_logger
from rvsampler.database_handler import VolumeDatabaseHandler

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize



def main():
    """This script assigns release probabilities to the release volumes in the database.
    """
    # Rundir tas inn som input
    parser = argparse.ArgumentParser(description="Release volume sampler")
    parser.add_argument('--rundir', required=True, help='Path to the run directory')
    parser.add_argument('--rootdir', required=True, help='Path to the rvsampler root directory')
    args = parser.parse_args()
    rundir = args.rundir
    rootdir = args.rootdir
    
    logger = setup_logger("preparational", rundir)
    shakemaps_params = {
        #"shakemaps_filename": os.path.join(rundir, "input/shakemaps/messina_1908/predicted_data_NN_Messina_1908.json"),
        "shakemaps_filename": os.path.join(rootdir, "input/shakemaps/PGA_data/H_Z_pda_data_log10_G.json"),
        "source_parameters_filename": os.path.join(rootdir, "input/shakemaps/messina_1908/source_parameters.csv")
    }
    displacements_exceedance_params = {
        "cumulative_dir": os.path.join(rundir, "displacements"),
        "outfile_name": "exceedance_displacement.npz"
    }
    
    if os.path.exists(os.path.join(rundir, "shakemaps")):
        logger.info("Preprocessing of shakemaps already done, skipping.")
    else:
        preprocess_shakemaps(rundir, **shakemaps_params)
    if os.path.exists(os.path.join(rundir, "displacements")):
        logger.info("Displacement allready calculated, skipping.")
    else:
        calculate_displacement_probabilities(rundir)
    
    calculate_release_volume_probabilities(rundir)

def preprocess_shakemaps(rundir, shakemaps_filename, source_parameters_filename, cumulative=False):
    """ Method to read shakemap and: 
    1. Compute cumulative probabilities (cumulative=True).
    2. Interpolate over domain and write to rasters.
    """
    
    shakemaps_reader = ShakemapsReader(
        shakemaps_filename=shakemaps_filename,
        source_parameters_filename=source_parameters_filename,
        rundir=rundir,
        thresholds=np.linspace(-3,0,40), 
        aggregate=False,
        samples=[0,3]
    )
    
    # Interpolate over computational region and write to files
    with rasterio.open(os.path.join(rundir, "bathy_truncated.tif")) as src:
        bounds = src.bounds
        profile = src.profile.copy()
    
    shakemaps_reader.write_shakemaps_to_rasters(
        profile=profile, 
        bounds=bounds,
        interpolation_method='linear', # “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”
    )

def calculate_displacement_probabilities(rundir, cumulative=False):
    """Displacement probabilities.
    """
    thresholds = np.arange(1, 10, step=1.) # Displacement thresholds in cm.
    dpa = DisplacementProbabilityAggregator(rundir, thresholds, magnitude=7)
    if cumulative:
        dpa.compute_aggregated_probabilities()
    else:
        dpa.compute_probabilities_by_sample(nr_of_pga_thresholds=100)

def calculate_release_volume_probabilities(rundir):
    with VolumeDatabaseHandler(rundir) as volumes_db:
        volumes_db.assign_probabilities_to_seed_triangles(
            displacement_threshold=5.,
            displacement_dir=os.path.join(rundir, "displacements/sample_0"),
            table_filename="exceedance_displacement.npz", 
            column_name = "p_shake"
        )
        volumes_db.compute_release_probabilities()

if __name__ == "__main__":
    main()
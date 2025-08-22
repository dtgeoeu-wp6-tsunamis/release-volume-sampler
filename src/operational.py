import os
import numpy as np
import rasterio
import argparse

from rvsampler.displacements import DisplacementProbabilityAggregator
from rvsampler.shakemaps_reader import ShakemapsReader
from rvsampler.utils import create_dir
from rvsampler.set_logg import setup_logger
from rvsampler.database_handler import VolumeDatabaseHandler
from rvsampler.aggregate import ProbabilityAggregator


def main():
    """This script assigns release probabilities to the release volumes in the database.
    """
    # Rundir as input
    parser = argparse.ArgumentParser(description="Release volume sampler")
    parser.add_argument('--rundir', required=True, help='Path to the run directory')
    parser.add_argument('--rootdir', required=True, help='Path to the rvsampler root directory')
    args = parser.parse_args()
    rundir = args.rundir
    rootdir = args.rootdir
    
    logger = setup_logger("operational", rundir)
    shakemaps_params = {
        #"shakemaps_filename": os.path.join(rundir, "input/shakemaps/messina_1908/predicted_data_NN_Messina_1908.json"),
        "shakemaps_filename": os.path.join(rootdir, "input/shakemaps/PGA_data/H_Z_pda_data_log10_G.json"),
        "source_parameters_filename": os.path.join(rootdir, "input/shakemaps/messina_1908/source_parameters.csv")
    }
    if os.path.exists(os.path.join(rundir, "shakemaps", "completed")):
        logger.info("Preprocessing of shakemaps already done, skipping.")
    else:
        preprocess_shakemaps(rundir, **shakemaps_params)
    if os.path.exists(os.path.join(rundir, "displacements", "completed")):
        logger.info("Displacement exceedance probabilities allready calculated, skipping.")
    else:
        calculate_displacement_probabilities(rundir, cumulative=False)
    if os.path.exists(os.path.join(rundir, "aggregation", "completed")):
        logger.info("Aggregation already done, skipping.")
    else:
        aggregate_cluster_release_probabilities(rundir)

def preprocess_shakemaps(rundir, shakemaps_filename, source_parameters_filename, cumulative=False):
    """ Method to read shakemap and: 
    1. Compute cumulative probabilities (cumulative=True).
    2. Interpolate over domain and write to rasters.
    """
    
    shakemaps_reader = ShakemapsReader(
        shakemaps_filename=shakemaps_filename,
        source_parameters_filename=source_parameters_filename,
        rundir=rundir,
    )

    with rasterio.open(os.path.join(rundir, "bathy_truncated.tif")) as src:
        bounds = src.bounds
        profile = src.profile.copy()

    if cumulative:
        shakemaps_reader.write_cumulative_distribution(
            profile=profile, 
            bounds=bounds,
            interpolation_method='linear', # “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”
            thresholds=np.linspace(-3, 0, 40), # Thresholds for cumulative distribution
        )
    else:
        shakemaps_reader.write_shakemaps_to_rasters(
            profile=profile, 
            bounds=bounds,
            samples=None, # None (reads all samples) or specify a list of samples to read, e.g., [0,3]
            interpolation_method='linear', # “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”
        )
    shakemaps_reader.completed()
    
def calculate_displacement_probabilities(rundir, cumulative=False):
    """Displacement probabilities.
    """
    threshold = 5.0 # Displacement thresholds in cm.
    dpa = DisplacementProbabilityAggregator(rundir, magnitude=7)
    if cumulative:
        dpa.compute_aggregated_probabilities(threshold=threshold)
    else:
        dpa.compute_probabilities_by_sample(displacement_threshold=threshold, nr_of_pga_thresholds=100)
    dpa.completed()
    
    # Write probabilities to database
    if cumulative:
        with VolumeDatabaseHandler(rundir) as volumes_db:
            volumes_db.assign_probabilities_to_seed_triangles(
                displacement_dir=os.path.join(rundir,"displacements","cumulative"),
                table_filename="exceedance_displacement.npz",
                column_name="p_shake_cum"
    )
    else:
        displacements_dir = os.path.join(rundir, "displacements")
        with VolumeDatabaseHandler(rundir) as volumes_db:
            for fname in os.listdir(displacements_dir):
                if fname.startswith("sample_") and os.path.isdir(os.path.join(displacements_dir, fname)):
                    sample_nr = fname.split("_")[-1]
                    column_name = f"p_shake_{sample_nr}"
                    volumes_db.assign_probabilities_to_seed_triangles(
                        displacement_dir=os.path.join(displacements_dir, fname),
                        table_filename="exceedance_displacement.npz",
                        column_name=column_name
                    )

def aggregate_cluster_release_probabilities(rundir):
    """Aggregate cluster release probabilities.
    """
    pag = ProbabilityAggregator(rundir)
    pag.compute_cluster_release_probabilities()
    pag.plot_cluster_probability_heatmap(save_fig=True)
    pag.completed()


if __name__ == "__main__":
    main()
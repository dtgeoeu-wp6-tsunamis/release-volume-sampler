import os
import numpy as np
import rasterio

from rvsampler.displacements import DisplacementProbabilityAggregator
from rvsampler.shakemaps_reader import ShakemapsAggregator
from rvsampler.utils import create_dir
from rvsampler.cumprobs_by_triangle import caclulate_cumulative_probabilities
from rvsampler.database_handler import VolumeDatabaseHandler


def main():
    """This script assigns release probabilities to the release volumes in the database.
    """
    project_dir = "/home/ebr/projects/release-volume-sampler"
    rundir =  os.path.join(project_dir, "generated/messina_002")
    shakemaps_params = {
        "shakemaps_filename": os.path.join(project_dir, "input/shakemaps/messina_1908/predicted_data_NN_Messina_1908.json"),
        "source_parameters_filename": os.path.join(project_dir, "input/shakemaps/messina_1908/source_parameters.csv")
    }
    displacements_exceedance_params = {
        "cumulative_dir": os.path.join(rundir, "displacements"),
        "outfile_name": "exceedance_displacement.npz"
    }
    
    aggregate_shakemaps(rundir, **shakemaps_params)
    calculate_displacement_probabilities(rundir)
    caclulate_cumulative_probabilities(rundir, **displacements_exceedance_params)
    
    filter_config = {
        "tsunami_potential_ratio_threshold": 1.,
        "max_rasters": 1000,
    }
     
    with VolumeDatabaseHandler(rundir) as volumes_db:
        volumes_db.load_probabilities_from_shakemap(displacement_threshold=5., 
                                                    table_filename="exceedance_displacement.npz", 
                                                    column_name = "p_shake")
    
        volumes_db.write_volumes_to_csv()
        volumes_db.write_volumes_to_rasters(**filter_config)
        
        #volumes_db.plot_distribution()
        #volumes_db.plot_release_density_plots()
        
        volumes_db.plot_distribution(seed_prob="p_shake")
        volumes_db.plot_release_density_plots(seed_prob="p_shake")


def aggregate_shakemaps(rundir, shakemaps_filename, source_parameters_filename):
    """ Method to compute cumulative distribution of the shakemaps.
    """
    def shake_value(point):
        # Pullback function to extract value from shakemap.
        Z, H = [10**np.array(point[shake_param]) for shake_param in ["Z_pga", "H_pga"]]
        return np.log10(np.sqrt(Z**2 + H**2))
    
    shakemaps_aggregator = ShakemapsAggregator(
        shakemaps_filename=shakemaps_filename,
        source_parameters_filename=source_parameters_filename,
        rundir=rundir)
    
    # Interpolate cumulative over computational region and write to files
    with rasterio.open(os.path.join(rundir, "bathy_truncated.tif")) as src:
        bounds = src.bounds
        profile = src.profile.copy()
    
    shakemaps_aggregator.compute_cumulative(profile=profile, 
                                             bounds=bounds, 
                                             interpolation_method='linear', # “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”
                                             thresholds=np.linspace(-3,0,40), 
                                             shake_value=shake_value)


def calculate_displacement_probabilities(rundir):
    """Displacement probabilities.
    """
    thresholds = np.arange(1, 10, step=1.) # Displacement thresholds in cm.
    dpa = DisplacementProbabilityAggregator(rundir, thresholds, magnitude=7)
    dpa.compute_probabilities()
    

if __name__ == "__main__":
    main()
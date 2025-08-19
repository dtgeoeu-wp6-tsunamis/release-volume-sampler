import os
import numpy as np
import shutil
import rasterio
import argparse
import sys

# Import from modules.
from rvsampler.preprocess import truncate_positive_values, slope, aspect
from rvsampler.slope_analysis import SlopeAnalysis
from rvsampler.triangulate import Triangulate, Triangulation
from rvsampler.release_volume_sampler import RecursiveReleaseAnalysis
from rvsampler.database_handler import VolumeDatabaseHandler
from rvsampler.cluster import ClusterAnalysis
from rvsampler.cumprobs_by_triangle import caclulate_cumulative_probabilities

from rvsampler.utils import create_dir
from rvsampler.set_logg import setup_logger


def main():
    """
    This script creates the database with potential release volumes.
    """
    # Rundir tas inn som input
    parser = argparse.ArgumentParser(description="Release volume sampler")
    parser.add_argument('--region', required=True, help='Name of region to process with additional numbering (e.g., "messina_001")')
    parser.add_argument('--rootdir', required=True, help='Path to the base directory of the release volume sampler')
    args = parser.parse_args()
    region = args.region
    rootdir = args.rootdir
    rundir = os.path.join(rootdir, 'generated', region)
    
    logger = setup_logger("preparational", rundir)
    create_dir(rundir, logger)
    # --rundir /home/sfr/release-volume-sampler
    #rundir = r'/home/sfr/release-volume-sampler'
    logger.info(f"Running preparational script for region {region} in {rundir}")
    
    #inputfolder = rundir + "input"
    config = {
        "rundir": rundir,
        "singularity_image": os.path.join(rootdir,"images", "grass.sif"),
        "bathyfile": os.path.join(rootdir,'input', "bathy", "messina_001/bathy_truncated.tif"),
        "soilregions_filename": os.path.join(rootdir,'input', 'soilparams','regions.tif'),
        "soil_parameters_filename": os.path.join(rootdir,'input', 'soilparams','params.json'),
        "logger": logger,
        "slopeunitfile": None, #os.path.join(rootdir, 'input', 'slopeunits', 'slumap.tif'),
    }
    logger.info(f"Configuration: {config}")
    # Run steps if not already done.
    if os.path.exists(os.path.join(rundir, "completed")):
        logger.info("Initialization already done, skipping.")
    else:
        initialize(**config)
    if os.path.exists(os.path.join(rundir, "slope_analysis", "completed")):
        logger.info("Slope analysis already done, skipping.")
    else:
        execute_slope_analysis(rundir)
    if os.path.exists(os.path.join(rundir, "triangulation", "completed")):
        logger.info("Triangulation already done, skipping.")
    else:
        triangulate_domain(rundir)
    if os.path.exists(os.path.join(rundir, "volumes", "completed")):
        logger.info("Release volume sampling already done, skipping.")
    else:
        # Clear volumes directory if exists
        volumes_dir = os.path.join(rundir, "volumes")
        if os.path.exists(volumes_dir):
            shutil.rmtree(volumes_dir)
        
        sample_release_volumes(rundir)
        assign_volume_features(rundir)
        cluster_release_volumes(rundir)
        with open(os.path.join(rundir, "volumes", "completed"), "w") as f:
            pass
    if os.path.exists(os.path.join(rundir, "volumes", "volumes.csv")):
        logger.info("Release volumes already written to csv and rasters, skipping.")
    else:
        write_volumes(rundir)
   
def initialize(rundir, bathyfile, soilregions_filename, soil_parameters_filename, singularity_image, logger, slopeunitfile=None):
    
    logfile = os.path.join(rundir, "preparational_external_software.txt")
    
    # Copy soilparameterfiles to rundir
    shutil.copy(soilregions_filename, os.path.join(rundir, "soilregions.tif"))
    shutil.copy(soil_parameters_filename, os.path.join(rundir, "soilparams.json"))
    if slopeunitfile is not None:
        shutil.copy(slopeunitfile, os.path.join(rundir, "slumap.tif"))

    # Assert that the raster is in a geographic (lon-lat) coordinate system
    logger.info("Verifying that input bathymetri is logitude-latitude.")
    with rasterio.open(bathyfile) as src:
        assert src.crs.is_geographic, "The raster is not in a longitude-latitude coordinate system (geographic CRS)."

    logger.info(f"Copy {bathyfile} to {rundir}.")
    
    outfile = "bathy"    
    shutil.copy(bathyfile, os.path.join(rundir, f"{outfile}.tif"))
    outfile = truncate_positive_values(rundir, singularity_image, outfile, logfile) 
    
    slope(bathyfile, output_dir=rundir, logfile=logfile)
    aspect(bathyfile, output_dir=rundir, logfile=logfile)
    
    with open(os.path.join(rundir, "completed"), "w") as f:
        pass  # Just create an empty file 
    
    return rundir


def execute_slope_analysis(rundir):
    """
    Parameters defined as discrete distributions or constants.
    May supply functional relations. Order according to dependencies.
    physical_parameters = {
        "distributions": {
            "friction_angle": [(18, 0.248), (22, 0.504), (25, 0.248)], # [(value, weight),...]
            "cohesion": [(10, 0.248),(15,0.504),(20, 0.248)],
            "thickness": [(2, 0.248), (6, 0.504),(10, 0.248)],
            "density": [(1600, 0.248),(2000, 0.504),(2400, 0.248)],
        },
        "constants": {
            "density_of_water": 1020,
            "gravity": 9.81,
            "excess_pore_pressure": 0.
        }
    }
    """
    sa = SlopeAnalysis(rundir, slopefile="slope.tif")
    
    quantiles = [0.01, 0.1, 0.5, 0.9, 0.99]
    sa.compute_quantiles(quantiles, write_fos=True, write_ky=True)
    
    fos_thresholds = np.linspace(0, 2, num=50)
    sa.compute_cumulative(fos_thresholds, feature_name="logfos", write=True)
    
    ky_thresholds = np.linspace(-3,1, num=50)
    sa.compute_cumulative(ky_thresholds, feature_name="logky", write=True)

    sa.completed()
    
def triangulate_domain(rundir):
    config = {
        "rundir": rundir,
        "bathyfile": "bathy_truncated.tif",
        "utm_epsg_code": 32633,     # Messina strait
        "resolution": (110, 110),   # Mesh of nodes in unfitted triangulation.
        "slopeunitfile": os.path.join(rundir, "slumap.tif"), # Used for assigning slopeunit to triangles.
    }
    #optimization_params = {
    #    "num_iterations": 2000,
    #    "batch_size": 3000,
    #    "shape_weight": 5e1,
    #    "area_weight": 5e-11,
    #    "elevation_weight": 1e-2
    #}
    optimization_params = {
        "num_iterations": 200,
        "batch_size": 3000,
        "shape_weight": 5e1,
        "area_weight": 5e-11,
        "elevation_weight": 1e-2
    }
    triang = Triangulate(**config)
    triang.fit(**optimization_params)
    triang.plot_triangulation()
    triang.write_to_file() # Includes assignment of slopeunits if given.
    
    # Applied for assigning extent of grid for simulation of landslides.
    triang = Triangulation(rundir)
    triang.initialize_triangle_properties()
    triang.poly_slopes(filename="poly_slopes.npy")
    triang.slopeunits_to_triangles(filename="slopeunits_to_triangles.npy")
    
    # Assign fos cumulative probabilities lookuptable by triangle
    cumulative_dir = os.path.join(rundir, "slope_analysis", "fos", "cumulative")
    outfile_name = "cumulative_fos.npz" # Writes to triangulation dir..
    triang.create_lookuptable(cumulative_dir, outfile_name)
    triang.completed()

def sample_release_volumes(rundir):

    # Initialize the database.
    with VolumeDatabaseHandler(rundir) as db:
        db.initialize_db()
    
    config = {
        "rundir": rundir,
        #"mesh_path": os.path.join(rundir, "triangulation", "triangulation.vtk"),
        "cumprob_logfos_path": os.path.join(rundir, "slope_analysis", "fos", "cumulative",\
            "cumulative_fos.npz"),
        "utm_epsg_code": 32633, # Messina strait
    }
    
    #run_config = {
    #    "fos_threshold": 1.6,
    #    "recursive_probability_threshold": 0.001,
    #    "seed_triangle_probability_threshold": 0.005,
    #    "max_workers":10,
    #    "max_n_seed_triangles": 10000,
    #    "use_slopeunits": True,
    #    "max_n_slopeunits": 10000,
    #    "max_n_simultaneous": 2,
    #}
    run_config = {
        "fos_threshold": 1.6,
        "recursive_probability_threshold": 1.e-4,
        "seed_triangle_probability_threshold": 0.05,
        "max_workers":10,
        "max_n_seed_triangles": 50,
        
        # Slopeunit parameters.
        "use_slopeunits": False, 
        "max_n_slopeunits": 10000, 
        "max_n_simultaneous": 2,    # Number of simultaneous seed triangles when using slopeunits.
        
        # Distance threshold for sampling pairs of seed triangles when not applying slopeuints.
        "pairs_distance_threshold": 1000,
    }
    # Execute analysis.
    analysis = RecursiveReleaseAnalysis(**config)
    analysis.run(**run_config)
    
    # Verify that no volumes contain duplicate triangles.
    with VolumeDatabaseHandler(rundir) as db:
        assert db.test_no_duplicate_triangles_in_released()
    
    
def assign_volume_features(rundir):
    """
    Assign volume features to the release volumes.
    This includes volume, thickness, tsunami potential ratio and no2d.
    """
    # volume_factor: Factor to scale thickness of the release volumes.
    with VolumeDatabaseHandler(rundir) as db:
        db.assign_volume_features(volume_factor=2.0)
    
def cluster_release_volumes(rundir):
    
    config = {
        "rundir": rundir,
        "n_clusters": 100,
        "random_state": 0,
        "batch_size": 1000,
        "feature_columns": ['area', 'no2d', 'mean_elevation',
                            'mean_northing', 'mean_easting'],
        "columns_to_scale": ['area', 'no2d', 'mean_elevation',
                             'mean_northing', 'mean_easting'],
        "weights": {
            'area': 1.0,
            'no2d': 1.0,
            'mean_elevation': 1.0,
            'mean_northing': 10.0,
            'mean_easting': 10.0
        },
    }
    # Initialize ClusterAnalysis object
    cluster_analysis = ClusterAnalysis(**config)    
    # Fit the clustering model
    cluster_analysis.fit()
    # Write cluster label database
    cluster_analysis.write_to_database()
    # Find representatives and write to database
    cluster_analysis.find_representatives()
    # Close the database connection
    cluster_analysis.close()

def write_volumes(rundir):
    """ Writes the release volumes to csv and rasters.
    This is used for further analysis and visualization.
    """
    filter_config = {
        "tsunami_potential_ratio_threshold": 0.,
        "max_rasters": 100,
        "raster_driver": 'GTiff', # GTiff/AAIGrid
    }
    
    # Write the volumes to csv
    with VolumeDatabaseHandler(rundir) as volumes_db:
    
        volumes_db.write_volumes_to_csv(max_rasters=filter_config['max_rasters'])
        volumes_db.write_volumes_to_rasters(**filter_config)
        
        #volumes_db.plot_distribution()
        #volumes_db.plot_release_density_plots()
        
        #volumes_db.plot_distribution(seed_prob="p_shake")
        #volumes_db.plot_release_density_plots(seed_prob="p_shake")


if __name__ == "__main__":
    main()
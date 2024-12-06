import os, sys
import numpy as np
from datetime import datetime
import shutil
import json

# Import from modules.
from src.rvsampler.preprocess import preprocess, slope, aspect, compute_pixel_areas
from src.rvsampler.slope_analysis import SlopeAnalysis
from src.rvsampler.triangulate import Triangulation
from src.rvsampler.cumprobs_by_triangle import caclulate_cummulative_probabilities
from src.rvsampler.release_volume_sampler import RecursiveReleaseAnalysis
from src.rvsampler.volume_writer import VolumeWriter
from src.rvsampler.displacements import displacement
from src.rvsampler.utils import create_dir
from src.rvsampler.logging import setup_logger


def main():
    """
    This script performs stability analysis in the sought region. Based on this analysis a selection
    of possible release volumes are generated.
    
    Outline:
        - Create analysis dir.
        - Copy bathymetry into analysis dir (bathy.tif)
        
        - Calculate slope/aspect using whitebox tools.
        (- Calculate slopeunits using r.slopeunits https://doi.org/10.5194/gmd-9-3975-2016)
        
        - Find yield acceleration threshold values from displacement and pga range.
        - Select volumes by intersecting yield acceleration thresholds for each slopeunit.
        
        # Preparation for scenario selection
        - Calculate cummulative probabilities of yield acceleration. 
    """
    # Settings
    project_dir = "/home/ebr/projects/release-volume-sampler"
    # Bathy
    bathyfile = os.path.join(project_dir, "input/bathy/messina_001/localMessinaBathy.tif")
    
    
    # Soilparams.
    soilregions_filename = os.path.join(project_dir, "input/soilparams/regions.tif")
    soil_parameters_filename = os.path.join(project_dir, "input/soilparams/params.json")
    
    #map_projection_epsg = 3065 # EPSG code of the mapprojection applied for computations. Must have units metre! 6709
    singularity_image = os.path.join(project_dir, "images/grass.sif")
    generated = os.path.join(project_dir, "generated")
    
    #slopeunitsfile = os.path.join(generated, "slopeunits/slumap_clean.tif")
    scenario = "messina_001"
    
    rundir =  os.path.join(generated, scenario)
    logfile = os.path.join(rundir, "log.txt")
    logger = setup_logger("preprocess", rundir)
    create_dir(rundir, logger)    
    
    # preprocess_bathy:
    run_preprocess(bathyfile, soilregions_filename, soil_parameters_filename, singularity_image, rundir, logfile, logger)
    
    #run_slope_analysis:
    execute_slope_analysis(rundir)
    
    # make triangulation.
    triangulate_domain(rundir)
    
    # Run volume sampler
    sample_release_volumes(rundir) 
    
    # Calculate volume statistics and write volumes to file
    write(rundir)
    

def run_preprocess(bathyfile, soilparamsfile, soilregionsfile, singularity_image, rundir, logfile, logger):
    bathyfile = preprocess(bathyfile, singularity_image, logger=logger, map_projection_epsg=None, output_dir=rundir, logfile=logfile)
    
    # Copy soilparameterfiles to rundir
    shutil.copy(soilparamsfile, os.path.join(rundir, "soilregions.tif"))
    shutil.copy(soilregionsfile, os.path.join(rundir, "soilparams.json"))
    
    # calculate_slopes_and_aspect and pixel areas
    slope_file = slope(bathyfile, output_dir=rundir, logfile=logfile)
    aspect(bathyfile, output_dir=rundir, logfile=logfile)
    compute_pixel_areas(rundir, bathy_file=bathyfile, slope_file=slope_file, logger=logger)


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
    sa.compute_cummulative(fos_thresholds, feature_name="logfos", write=True)
    
    ky_thresholds = np.linspace(-3,1, num=50)
    sa.compute_cummulative(ky_thresholds, feature_name="logky", write=True)


def triangulate_domain(rundir):
    config = {
        "rundir": rundir,
        "bathyfile": "bathy_truncated.tif",
        "utm_epsg_code": 32633, #Messina strait
        "resolution": (110, 110)
    }
    optimization_params = {
        "num_iterations": 2000,
        "batch_size": 3000,
        "shape_weight": 5e1,
        "area_weight": 5e-11,
        "elevation_weight": 1e-2
    }
    triang = Triangulation(**config)
    triang.fit(**optimization_params)
    triang.plot_triangulation()
    triang.write_to_file()
    
    # Calculate cumulative probabilities lookup table by triangle
    cummulative_dir = os.path.join(rundir, "slope_analysis", "fos", "cummulative")
    outfile_name = "cummulative_fos.npz" # Writes to triangulation dir..
    caclulate_cummulative_probabilities(rundir, cummulative_dir, outfile_name)


def sample_release_volumes(rundir):
    
    config = {
        "rundir": rundir,
        "mesh_path": os.path.join(rundir, "triangulation", "triangulation.vtk"),
        "cumprob_logfos_path": os.path.join(rundir, "triangulation", "cummulative_fos.npz"),
        "utm_epsg_code": 32633, # Messina strait
    }
    
    run_config = {
        "fos_threshold": 1.6,
        "recursive_probability_threshold": 0.001,
        "seed_triangle_probability_threshold": 0.005,
    }
    # Execute analysis.
    analysis = RecursiveReleaseAnalysis(**config)
    analysis.run(**run_config)


def write(rundir):
    """ To ensure that module imports works, run the script as a module.
    release-volume-sampler$ python -m src.volume_sampler.volume_writer
    """
    # Usage example
    config = {
        "rundir": rundir,
    }
    
    filter_config = {
        "tsunami_potential_ratio_threshold": 1.,
        "max_rasters": 1000,
    }
    
    
    # Execute
    writer = VolumeWriter(**config)
    writer.write_volumes_to_csv()
    writer.write_volumes_to_rasters(**filter_config)
    
    #writer.plot_distribution()
    #writer.plot_release_density_plots()
    
    #writer.plot_distribution(seed_prob="p_shake")
    #writer.plot_release_density_plots(seed_prob="p_shake")
if __name__ == "__main__":
    main()
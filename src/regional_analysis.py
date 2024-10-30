import os, sys
import logging
import numpy as np
from datetime import datetime
from slope_analysis.slope_analysis import SlopeAnalysis
from preprocess.preprocess import preprocess, slope, aspect
from slopeunits.slopeunits import run_grassjob
from displacements.displacements import displacement
from utils.utils import create_dir
from collections import namedtuple
import shutil
import json

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()


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
    slopeunitsfile = os.path.join(generated, "slopeunits/slumap_clean.tif")
    scenario = "messina_001"
    #run = None #"messina_001_20241023_071936"
    
    #if run is None:
        # Create time stamped rundir
    #    formatted_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    #    run = f"{scenario}_{formatted_datetime}"
    
    rundir =  os.path.join(generated, scenario)
    logfile = os.path.join(rundir, "log.txt")
    create_dir(rundir)    
    
    #logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logFormatter = logging.Formatter("'%(levelname)s:%(message)s'")

    fileHandler = logging.FileHandler(os.path.join(rundir, "run.log"))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    #consoleHandler = logging.StreamHandler()
    #consoleHandler.setFormatter(logFormatter)
    #logger.addHandler(consoleHandler)
   
    
    # preprocess_bathy:
    run_preprocess(bathyfile, soilregions_filename, soil_parameters_filename, singularity_image, rundir, logfile)
    
    #run_slope_analysis:
    #execute_slope_analysis(rundir)
    
    # Preselection of volumes.
    preselect_volumes(rundir, slopeunitsfile)
    

def run_preprocess(bathyfile, soilparamsfile, soilregionsfile, singularity_image, rundir, logfile):
    bathyfile = preprocess(bathyfile, singularity_image, map_projection_epsg=None, output_dir=rundir, logfile=logfile)
    
    # Copy soilparameterfiles to rundir
    shutil.copy(soilparamsfile, os.path.join(rundir, "soilregions.tif"))
    shutil.copy(soilregionsfile, os.path.join(rundir, "soilparams.json"))
    
    # calculate_slopes_and_aspect:
    slope(bathyfile, output_dir=rundir, logfile=logfile)
    aspect(bathyfile, output_dir=rundir, logfile=logfile)


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
    
    fos_thresholds = np.linspace(0, 2, num=20)
    sa.compute_cummulative(fos_thresholds, feature_name="logfos", write=True)
    
    ky_thresholds = np.linspace(-3,1, num=20)
    sa.compute_cummulative(ky_thresholds, feature_name="logky", write=True)


def preselect_volumes(rundir, slopeunitsfile):
    volume_selection_dir = os.path.join(os.path.join(rundir, "volume_selection"))
    create_dir(volume_selection_dir)
    
    logpgas = np.log10([0.3, 0.6, 1.])
    logdelta = np.log10(30.)
    magnitude = 7.
    probability_threshold = 0.5
    
    selection_criteria = []
    logky = np.linspace(-5,1,200000)
    for logpga in logpgas:
        # Extract biggest yield acelleration with displacement larger than delta.
        more_than_delta = displacement(logky=logky, logpga=logpga, M=magnitude)[0] > logdelta
        root_logky = logky[more_than_delta][-1]
        selection_criteria.append({"logky": root_logky, 
                                   "delta": 10**logdelta, 
                                   "pga": 10**logpga, 
                                   "M":magnitude})
    
    with open(os.path.join(volume_selection_dir, 'selection_criteria.json'), 'w') as f:
        json.dump(selection_criteria, f, indent=4)
    
    # Extract areas using the Cummulative distribution of logky.
    sa = SlopeAnalysis(rundir, slopefile="slope.tif")
    sa.compute_cummulative([params["logky"] for params in selection_criteria], 
                           feature_name="logky", 
                           write=True, 
                           output_dir=volume_selection_dir)
    
    # Loop through slopeunits and intersect with cumulative thresholds.


if __name__ == "__main__":
    main()
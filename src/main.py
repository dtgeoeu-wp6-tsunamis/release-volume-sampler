import os, sys
import logging
import numpy as np
from datetime import datetime
from slope_analysis.slope_analysis import SlopeAnalysis
from preprocess.preprocess import preprocess, slope, aspect
from slopeunits.slopeunits import run_grassjob
from utils.utils import create_dir
from displacements.displacements import DsiplacementProbabilityAggregator
from shakemap_reader.shakemaps_reader import ShakemapsAggregator
from collections import namedtuple
import rasterio
import shutil


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()


def main():
    """
    Outline:
        - Create scenario folder
        - Copy bathymetry into scenario folder (bathy.tif)
        
        - Calculate slope/aspect (This may be done using either grass or whitebox tools)
        - Calculate slopeunits using r.slopeunits https://doi.org/10.5194/gmd-9-3975-2016
        
        - Calculate yield accelerations.
        - Calculate displacements from yield acellerations and PGA.
        - Get volumes by thresshold.
        - Intersect with slopeunits.
        
    """
    # Settings
    project_dir = "/home/ebr/projects/release-volume-sampler"
    # Bathy
    bathyfile = os.path.join(project_dir, "input/bathy/messina_001/localMessinaBathy.tif")
    # Shakemaps
    shakemaps_filename = os.path.join(project_dir, "input/shakemaps/messina_1908/predicted_data_NN_Messina_1908.json")
    source_parameters_filename = os.path.join(project_dir, "input/shakemaps/messina_1908/source_parameters.csv")
    # Soilparams.
    soilregions_filename = os.path.join(project_dir, "input/soilparams/regions.tif")
    soil_parameters_filename = os.path.join(project_dir, "input/soilparams/params.json")
    
    #map_projection_epsg = 3065 # EPSG code of the mapprojection applied for computations. Must have units metre! 6709
    singularity_image = os.path.join(project_dir, "images/grass.sif")
    
    generated = os.path.join(project_dir, "generated")
    scenario = "messina_001"
    run = None #"messina_001_20241023_071936"
    
    if run is None:
        # Create time stamped rundir
        formatted_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = f"{scenario}_{formatted_datetime}"
    
    rundir =  os.path.join(generated, run)
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
    
    #calculate_slope_units:
    #run_grassjob(singularity_image, project_dir, rundir, logfile=logfile)
    
    #run_slope_analysis:
    execute_slope_analysis(rundir)
    
    # Parse and aggregate
    aggregate_shakemaps(rundir, shakemaps_filename, source_parameters_filename, bathyfile)
    
    #calculate displacement probabilities
    calculate_displacement_probabilities(rundir)
    

def run_preprocess(bathyfile, soilparamsfile, soilregionsfile, singularity_image, rundir, logfile):
    bathyfile = preprocess(bathyfile, singularity_image, map_projection_epsg=None, output_dir=rundir, logfile=logfile)
    
    # Copy soilparameterfiles to rundir
    shutil.copy(soilparamsfile, os.path.join(rundir, "soilregions.tif"))
    shutil.copy(soilregionsfile, os.path.join(rundir, "soilparams.json"))
    
    # calculate_slopes_and_aspect:
    slope(bathyfile, output_dir=rundir, logfile=logfile)
    aspect(bathyfile, output_dir=rundir, logfile=logfile)


def aggregate_shakemaps(rundir, shakemaps_filename, source_parameters_filename, bathyfile):
        
    def shake_value_function(point):
        # Pullback function to extract value from shakemap.
        Z, H = [10**np.array(point[shake_param]) for shake_param in ["Z_pga", "H_pga"]]
        return np.log10(np.sqrt(Z**2 + H**2))
    
    shakemaps_aggregator = ShakemapsAggregator(
        shakemaps_filename=shakemaps_filename,
        source_parameters_filename=source_parameters_filename,
        thresholds=np.linspace(-3,0,30),
        shake_value={"name": "pga", "function":shake_value_function},
        rundir=rundir)
    
    # Interpolate cummulative over computational region and write to files
    with rasterio.open(bathyfile) as src:
        bounds = src.bounds
        profile = src.profile.copy()
    
    shakemaps_aggregator.write_cummulative(profile, bounds, 'linear')
    # “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”


def execute_slope_analysis(rundir):
    #quantiles = [0.1, 0.2, 0.5]
    quantiles = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.99]
    
    # Parameters defined as discrete distributions or constants.
    # May supply functional relations. Order according to dependencies.
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
    sa = SlopeAnalysis(rundir, slopefile="slope.tif")
    
    quantiles = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.99]
    sa.compute_quantiles(quantiles, write_fos=True, write_ky=True)
    
    fos_thresholds = np.linspace(0, 2, num=20)
    sa.compute_cummulative(fos_thresholds, feature_name="logfos", write=True)
    
    ky_thresholds = np.linspace(-1,1, num=20)
    sa.compute_cummulative(ky_thresholds, feature_name="logky", write=True)


def calculate_displacement_probabilities(rundir):
    thresholds = np.arange(1, 3, step=0.2) # Displacement thresholds in cm.
    dpa = DsiplacementProbabilityAggregator(rundir, thresholds, magnitude=7)
    dpa.compute_probabilities()

if __name__ == "__main__":
    main()
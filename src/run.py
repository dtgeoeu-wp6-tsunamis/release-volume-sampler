import os, sys
import logging
import numpy as np
from datetime import datetime
from slope_analysis.slope_analysis import SlopeAnalysis
from preprocess.preprocess import preprocess, slope, aspect
from slopeunits.slopeunits import run_grassjob
from utils.utils import create_dir
from displacements.displacements import DisplacementProbabilityAggregator
from shakemap_reader.shakemaps_reader import ShakemapsAggregator
from collections import namedtuple
import rasterio
import shutil


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()


def main():
    """
    This script calculates the probability that displacements is larger than a given threshold
    in a given earthquake scenario. This is applied to sample release volumes. 
    
    Note that the script has to be executed after the regional stability analysis.
    
    Outline:
        - Calculate displacements from yield acellerations and shakemaps.
        - Sample volumes based on displacements and slopeunits.
        - Project sampled volumes onto a preselected set of scenarios.
        
    """
    # Settings
    project_dir = "/home/ebr/projects/release-volume-sampler"
    
    # Shakemaps
    shakemaps_filename = os.path.join(project_dir, "input/shakemaps/messina_1908/predicted_data_NN_Messina_1908.json")
    source_parameters_filename = os.path.join(project_dir, "input/shakemaps/messina_1908/source_parameters.csv")
    
    generated = os.path.join(project_dir, "generated")
    scenario = "messina_001"
    #run = "messina_001_20241023_071936"
    
    rundir =  os.path.join(generated, scenario)
    logfile = os.path.join(rundir, "log.txt")
    
    #logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logFormatter = logging.Formatter("'%(levelname)s:%(message)s'")

    fileHandler = logging.FileHandler(os.path.join(rundir, "run.log"))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    #consoleHandler = logging.StreamHandler()
    #consoleHandler.setFormatter(logFormatter)
    #logger.addHandler(consoleHandler)
   
    
    # Parse and aggregate
    aggregate_shakemaps(rundir, shakemaps_filename, source_parameters_filename)
    
    #calculate displacement probabilities
    calculate_displacement_probabilities(rundir)
    

def aggregate_shakemaps(rundir, shakemaps_filename, source_parameters_filename):
        
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
    with rasterio.open(os.path.join(rundir, "bathy_truncated.tif")) as src:
        bounds = src.bounds
        profile = src.profile.copy()
    
    shakemaps_aggregator.write_cummulative(profile, bounds, 'linear')
    # “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”


def calculate_displacement_probabilities(rundir):
    thresholds = np.arange(1, 10, step=1.) # Displacement thresholds in cm.
    dpa = DisplacementProbabilityAggregator(rundir, thresholds, magnitude=7)
    dpa.compute_probabilities()

if __name__ == "__main__":
    main()
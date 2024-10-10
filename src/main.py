import os, sys
import subprocess
import logging
from datetime import datetime
import shutil
from slope_analysis.slope_analysis import run_analysis
from preprocess.preprocess import preprocess, slope, aspect
from slopeunits.slopeunits import run_grassjob
from utils.utils import create_dir
from displacements.displacements import calculate_displacements

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

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
    bathyfile = os.path.join(project_dir, "input/bathy/messina_001/localMessinaBathy.tif")
    shakemaps_filename = os.path.join(project_dir, "input/shakemaps/messina_1908/predicted_data_NN_Messina_1908.json")
    #map_projection_epsg = 3065 # EPSG code of the mapprojection applied for computations. Must have units metre! 6709
    singularity_image = os.path.join(project_dir, "images/grass.sif")
    
    generated = os.path.join(project_dir, "generated")
    scenario = "messina_001"
    run = None #"messina_001_20240923_121008"
    
    if run is None:
        # Create time stamped rundir
        formatted_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = f"{scenario}_{formatted_datetime}"
    
    rundir =  os.path.join(generated, run)
    logfile = os.path.join(rundir,"log.txt")
    create_dir(rundir)    
    
    preprocess_bathy = True
    calculate_slopes_and_aspect = True
    calculate_slope_units = False
    run_slope_analysis = True
    calculate_displacements_b = True
    
    # Computations
    if preprocess_bathy:
        bathyfile = preprocess(bathyfile, singularity_image, map_projection_epsg=None, output_dir=rundir, logfile=logfile)
    if calculate_slopes_and_aspect:
        slope(bathyfile, output_dir=rundir,logfile=logfile)
        aspect(bathyfile, output_dir=rundir, logfile=logfile)
    if calculate_slope_units:
        run_grassjob(singularity_image, project_dir, rundir, logfile=logfile)
    if run_slope_analysis:
        execute_slope_analysis(rundir)
    if calculate_displacements_b:
        calculate_displacements(rundir, shakemaps_filename, write_shakemaps=True, write_displacements=True, make_plots=True)


def execute_slope_analysis(rundir):
    quantiles = [0.1, 0.2, 0.5]
    
    # Parameters defined as discrete distributions or constants.
    # May supply functional relations. Order according to dependencies.
    physical_parameters = {
        "distributions": {
            "friction_angle": [(24.3, 1)], # [(value, weight),...]
            "cohesion": [(20, 1)],
            "thickness": [(3.6, 0.248), (4, 0.504),(4.4, 0.248)],
            "density": [(1800, 0.5),(2000, 0.5)],
        },
        "constants": {
            "density_of_water": 1020,
            "gravity": 9.81,
            "excess_pore_pressure": 0.
        }
    }
    run_analysis(rundir, quantiles, physical_parameters, slopefile="slope.tif", write_fos=True, write_ky=True)

if __name__ == "__main__":
    main()
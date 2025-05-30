import os
import numpy as np
import shutil
import rasterio
import argparse

# Import from modules.
from rvsampler.preprocess import truncate_positive_values, slope, aspect
from rvsampler.slope_analysis import SlopeAnalysis
from rvsampler.triangulate import Triangulation
from rvsampler.cumprobs_by_triangle import caclulate_cumulative_probabilities
from rvsampler.release_volume_sampler import RecursiveReleaseAnalysis
from rvsampler.database_handler import VolumeDatabaseHandler
from rvsampler.utils import create_dir
from rvsampler.set_logg import setup_logger


def main():
    """
    This script creates the database with potential release volumes.
    """
    # Rundir tas inn som input
    parser = argparse.ArgumentParser(description="Release volume sampler")
    parser.add_argument('--rundir', required=True, help='Path to the run directory')
    args = parser.parse_args()
    rundir = args.rundir
    
    # --rundir /home/sfr/release-volume-sampler
    #rundir = r'/home/sfr/release-volume-sampler'
    
    #inputfolder = rundir + "input"
    config = {
        "generated": os.path.join(rundir,"generated"), 
        "scenario":"messina_001",
        "singularity_image": os.path.join(rundir,"images/grass.sif"),
        "bathyfile": os.path.join(rundir,'input', "bathy/messina_001/localMessinaBathy.tif"),
        "soilregions_filename": os.path.join(rundir,'input', 'soilparams','regions.tif'),
        "soil_parameters_filename": os.path.join(rundir,'input', 'soilparams','params.json'),
    }
    
    rundir = initialize(**config)
    
    execute_slope_analysis(rundir)
    triangulate_domain(rundir)
    sample_release_volumes(rundir) 

    
def initialize(generated, scenario, bathyfile, soilregions_filename, soil_parameters_filename, singularity_image):
    
    rundir =  os.path.join(generated, scenario)
    logfile = os.path.join(rundir, "log.txt")
    logger = setup_logger("preparational", rundir)
    create_dir(rundir, logger)
    
    # Copy soilparameterfiles to rundir
    shutil.copy(soilregions_filename, os.path.join(rundir, "soilregions.tif"))
    shutil.copy(soil_parameters_filename, os.path.join(rundir, "soilparams.json"))
    

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
    cumulative_dir = os.path.join(rundir, "slope_analysis", "fos", "cumulative")
    outfile_name = "cumulative_fos.npz" # Writes to triangulation dir..
    caclulate_cumulative_probabilities(rundir, cumulative_dir, outfile_name)


def sample_release_volumes(rundir):
    
    # Initialize the database.
    with VolumeDatabaseHandler(rundir) as db:
        db.initialize_db()
    
    config = {
        "rundir": rundir,
        "mesh_path": os.path.join(rundir, "triangulation", "triangulation.vtk"),
        "cumprob_logfos_path": os.path.join(rundir, "triangulation", "cumulative_fos.npz"),
        "utm_epsg_code": 32633, # Messina strait
    }
    
    run_config = {
        "fos_threshold": 1.6,
        "recursive_probability_threshold": 0.001,
        "seed_triangle_probability_threshold": 0.005,
    }
    # Execute analysis.
    analysis = RecursiveReleaseAnalysis(**config)
    # make slope polygons and save to file for use in operational.py - this is used for evaluating possible 
    # slide scenarios and can be used for making the computational grid for bingclaw
    poly_slopes(rundir, analysis)
    
    analysis.run(**run_config)
      
def poly_slopes(rundir, analysis):
    # list of all triangles
    utriangles = np.arange(analysis.n_triangles)
    
    upstream_dict = {}
    while len(utriangles) > 0:
        #print(len(utriangles))
        tlist = get_all_upstream(utriangles[0],-1,analysis)
        upstream_dict[utriangles[0]] = tlist
        # remove those found from the list
        for i in tlist:
            utriangles = np.delete(utriangles, np.where(utriangles == i))
    
    np.save(os.path.join(rundir, "triangulation", "poly_slopes.npy"), upstream_dict)
  
def get_all_upstream(start, last, analysis, collected=None):
    # Calculate all upstream triangles for given start triangle
    
    if collected is None:
        collected = []
    if start is not None and start not in collected and start > -1:  # avoid duplicates or infinite loops
        collected.append(start)
        #print('#######' + str(start)+'##########' + str(last))
        upstream = analysis.get_upstream_triangles(start)
        for i in upstream:
            collected = get_all_upstream(i, start, analysis, collected)

    return collected


if __name__ == "__main__":
    main()
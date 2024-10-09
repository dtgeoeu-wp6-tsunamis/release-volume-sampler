import os, sys
import subprocess
import logging
#import tempfile
from datetime import datetime
import rasterio
import shutil
from slope_analysis import slope_analysis as sa

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MAX_PROCS_PER_DEM = 8

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
    bathyfile = os.path.join(project_dir, "bathy/messina_001/localMessinaBathy.tif")
    map_projection_epsg = 3065 # EPSG code of the mapprojection applied for computations. Must have units metre! 6709
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
    
    if not os.path.exists(rundir):
        try:
            os.makedirs(rundir)
            logger.info(f"Created directory {rundir}")
        except OSError as e:
            sys.exit(f"Can't create {rundir}: {e}")
    
    preprocess = True
    calculate_slopes_and_aspect = True
    calculate_slope_units = False
    run_slope_analysis = True
    
    # Computations
    if preprocess:
        bathyfile = preprocess_bathy(bathyfile, singularity_image, map_projection_epsg=None, output_dir=rundir, logfile=logfile)
    if calculate_slopes_and_aspect:
        slope(bathyfile, output_dir=rundir,logfile=logfile)
        aspect(bathyfile, output_dir=rundir, logfile=logfile)
    if calculate_slope_units:
        run_grassjob(singularity_image, project_dir, rundir, logfile=logfile)
    if run_slope_analysis:
        execute_slope_analysis(rundir)


def preprocess_bathy(bathymetry, singularity_image, output_dir, logfile, map_projection_epsg=None, working_dir=None):
    """
    Preprocessing steps before running calculations.
    """
    infile = bathymetry
    outfile = "bathy"
    if working_dir is None: working_dir = output_dir
    logger.info(f"Copy {bathymetry} to {output_dir}.")
    shutil.copy(bathymetry, os.path.join(output_dir, f"{outfile}.tif"))

    # Reproject bathymetry
    if map_projection_epsg is not None:
        outfile = project_bathy(working_dir, outfile, singularity_image, map_projection_epsg, logfile)
    
    # Truncate values on shore.
    outfile = truncate_positive_values(working_dir, singularity_image, outfile, logfile) 
    return(f"{outfile}.tif")


def run_grassjob(singularity_image, project_dir, rundir, logfile):
    """
    Create grass project from bathy in rundir.
    run grassjob.sh: calculates slopes and slopeunits.
    
    grass -e -c $rundir/bathy.tif [rundir]/grassdata
    grass $grass_project/PERMANENT --exec sh grassjob.sh $rundir
    """
    grass_project = os.path.join(rundir, "grassdata")
    bathy = os.path.join(rundir, "bathy_projected_truncated.tif")
    if not os.path.exists(grass_project):
        logger.info(f"Creating Grass GIS project from {bathy} in {rundir}.")
        completed_proc = subprocess.run(
            ["singularity",
            "exec",
            singularity_image,
            "grass",
            "-e",
            "-c",
            bathy,
            grass_project],
            cwd=rundir,
            stdout=subprocess.PIPE
        )
        completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
        log_process(completed_proc, logfile)
    
    
    logger.info(f"Copy grassjob.sh to {rundir}.")
    shutil.copy(os.path.join(project_dir, "slopeunits", "grassjob.sh"), rundir)
    
    completed_proc = subprocess.run(
        ["singularity",
        "exec",
        singularity_image,
        "grass",
        os.path.join(grass_project,"PERMANENT"),
        "--exec",
        "sh",
        "grassjob.sh",
        bathy],
        cwd=rundir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)

def truncate_positive_values(working_dir, singularity_image, infile, logfile):
    """
    Truncate positive values using gdal_calc.
    
    $ gdal_calc.py -A bathy/localMessinaBathy.tif --outfile=truncated.tif --calc="A*(A<0)" --NoDataValue=0
    """
    outfile = f"{infile}_truncated"
    completed_proc = subprocess.run(
        ["singularity",
        "exec",
        f"{singularity_image}",
        "gdal_calc",
        "-A",
        f"{infile}.tif",
        f"--outfile={outfile}.tif",
        "--calc=A*(A<0)",
        "--NoDataValue=0"],
        cwd=working_dir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)
    return(outfile)


def slope(infile, output_dir, logfile, working_dir=None):
        """
        Slope
        Description:
        Calculates a slope raster from an input DEM.
        Toolbox: Geomorphometric Analysis
        Parameters:

        Flag               Description
        -----------------  -----------
        -i, --dem          Input raster DEM file.
        -o, --output       Output raster file.
        --zfactor          Optional multiplier for when the vertical and horizontal units are not the same.
        --units            Units of output raster; options include 'degrees', 'radians', 'percent'


        Example usage:
        >>./whitebox_tools -r=Slope -v --wd="/path/to/data/" --dem=DEM.tif -o=output.tif --units="radians"
        Documentation: https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html#Slope
        NOTE: Whitebox tools don't need a projected DEM.
        """
        if not working_dir: working_dir = output_dir
        outfile = os.path.join(output_dir, "slope.tif")
        completed_proc = subprocess.run(
            ["whitebox_tools",
            "-r=Slope",
            "--dem={}".format(infile),
            "-o={}".format(outfile),
            "--max_procs={}".format(8)],
            cwd=working_dir,
            stdout=subprocess.PIPE
        )
        completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
        log_process(completed_proc, logfile)
        return(outfile)


def aspect(infile, output_dir, logfile, working_dir = None):
    """
        Aspect
        Description:
        Calculates an aspect raster from an input DEM.
        Toolbox: Geomorphometric Analysis
        Parameters:

        Flag               Description
        -----------------  -----------
        -i, --dem          Input raster DEM file.
        -o, --output       Output raster file.
        --zfactor          Optional multiplier for when the vertical and horizontal units are not the same.


        Example usage:
        >>./whitebox_tools -r=Aspect -v --wd="/path/to/data/" --dem=DEM.tif -o=output.tif
    """
    if not working_dir: working_dir = output_dir
    outfile = os.path.join(output_dir, "aspect.tif")
    completed_proc = subprocess.run(
        ["whitebox_tools",
        "-r=Aspect",
        "--dem={}".format(infile),
        "-o={}".format(outfile),
        "--max_procs={}".format(8)],
        cwd=working_dir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)
    return(outfile)


def project_bathy(working_dir, infile, singularity_image, map_projection_epsg, logfile):
    """
    Reproject raster using gdalwarp.
    $ gdalwarp -t_srs EPSG:6709 /home/ebr/projects/release-volume-sampler/bathy/messina_001/bathy_truncated.tif bathy.tif 
    """
    outfile = f"{infile}_projected"
    completed_proc = subprocess.run(
        ["singularity",
        "exec",
        f"{singularity_image}",
        "gdalwarp",
        "-t_srs",
        f"EPSG:{map_projection_epsg}",
        "-r",
        "cubicspline",
        f"{infile}.tif",
        f"{outfile}.tif"],
        cwd=working_dir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)
    return(outfile)


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
    sa.run_analysis(rundir, quantiles, physical_parameters, slopefile="slope.tif", write_fos=True, write_ky=True)


def log_process(completed_process, log_file):
        with open(log_file, 'a') as log:
            log.write("Process args: {}\n".format(" ".join(completed_process.args)))
            # If Python version > 3.7 replace with shlex.join (better formatting in logfile).
            log.write("Process stdout: {}\n".format(completed_process.stdout.decode("utf-8")))
 

if __name__ == "__main__":
    main()
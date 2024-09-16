import os, sys
import subprocess
import logging
#import tempfile
from datetime import datetime
import rasterio
import shutil
from fos import displacements

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
        Python script
        - Create scenario folder
        - Copy bathymetry into scenario folder (bathy.tif)
        
        Prepare features in container (gdal, grass, whiteboxtools?):
        - Calculate slope/aspect (This may be done using either grass or whitebox tools)
        - Calculate slopeunits using r.slopeunits https://doi.org/10.5194/gmd-9-3975-2016
        
        
        - Calculate displacements
        
    """
    # Settings
    project_dir = "/home/ebr/projects/release-volume-sampler"
    bathyfile = os.path.join(project_dir, "bathy/messina_001/localMessinaBathy.tif")
    map_projection_epsg = 3065 # EPSG code of the mapprojection applied for computations. Must have units metre! 6709
    singularity_image = os.path.join(project_dir, "images/grass.sif")
    
    generated = os.path.join(project_dir, "generated")
    scenario = "messina_001"
    run = None #"messina_001_20240816_092455"
    
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
    
    # Computations
    if preprocess_bathy:
        bathy = preprocess_bathy(bathyfile, singularity_image, map_projection_epsg, output_dir=rundir, logfile=logfile)
        grass_project = create_grass_project(bathy, singularity_image, rundir, logfile=logfile)
        run_grassjob(grass_project, bathy, singularity_image, project_dir, rundir, logfile=logfile)
        calculate_fos(rundir)


def preprocess_bathy(bathymetry, singularity_image, map_projection_epsg, output_dir, logfile, working_dir=None):
    """
    Copying bathy to generated folder.
    TODO: Include preprocessing steps (When gdal is available).
    # To truncate positive values
    gdal_calc.py -A bathy/localMessinaBathy.tif --outfile=truncated.tif --calc="A*(A<0)" --NoDataValue=0
    
    # Transform to suitable coordinates.
    gdalwarp -t_srs EPSG:6709 /home/ebr/projects/release-volume-sampler/bathy/messina_001/bathy_truncated.tif bathy.tif 
    """
    infile = bathymetry
    outfile = "bathy"
    if working_dir is None: working_dir = output_dir
    logger.info(f"Copy {bathymetry} to {output_dir}.")
    shutil.copy(bathymetry, os.path.join(output_dir, f"{outfile}.tif"))
    
    infile = outfile
    outfile = f"{outfile}_projected"
    completed_proc = subprocess.run(
        ["singularity",
        "exec",
        f"{singularity_image}",
        "gdalwarp",
        "-t_srs",
        f"EPSG:{map_projection_epsg}",
        "-r",
        "cubicspline",
        f"{bathymetry}",
        f"{outfile}.tif"],
        cwd=working_dir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)
    
    infile = outfile
    outfile = f"{outfile}_truncated"
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
    return(f"{outfile}.tif")

def create_grass_project(bathy, singularity_image, rundir, logfile):
    """
    Create grass project from bathy in rundir.
    grass -e -c $rundir/bathy.tif [rundir]/grassdata
    """
    grass_project = os.path.join(rundir, "grassdata")
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
    return(grass_project)

def run_grassjob(grass_project, bathy, singularity_image, project_dir, rundir, logfile):
    """
    run grassjob.sh
    grass $grass_project/PERMANENT --exec sh grassjob.sh $rundir
    """
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


def calculate_fos(rundir):
    displacements.calculate_factor_of_safety(rundir)


def slope(bathy, output_dir, logfile):
    """
    Calculate slopes using grass.
    """
    if working_dir is None: working_dir = output_dir
    logger.info(f"Calculate slopes.")
    
    completed_proc = subprocess.run([],
        cwd=working_dir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)


def log_process(completed_process, log_file):
        with open(log_file, 'a') as log:
            log.write("Process args: {}\n".format(" ".join(completed_process.args)))
            # If Python version > 3.7 replace with shlex.join (better formatting in logfile).
            log.write("Process stdout: {}\n".format(completed_process.stdout.decode("utf-8")))


    

if __name__ == "__main__":
    main()
import os, sys
import subprocess
import logging
from datetime import datetime
import shutil
from utils.utils import log_process

MAX_PROCS_PER_DEM = 8

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger("preprocess")


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


def preprocess(bathymetry, singularity_image, output_dir, logfile, map_projection_epsg=None, working_dir=None):
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
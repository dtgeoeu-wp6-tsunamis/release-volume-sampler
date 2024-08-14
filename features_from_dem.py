import os, sys
import subprocess
import logging
import tempfile
from datetime import datetime

MAX_PROCS_PER_DEM = 8

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("")

def main():
    
    
    """
    Outline: 
        - Create scenario folder
        - Copy bathymetry into scenario folder (bathy.tif)
        - Calculate slope 
        - Calculate basins
    """
    project_dir = "/home/ebr/projects/release-volume-sampler"
    bathyfile = os.path.join(project_dir, "bathy/messina_001/localMessinaBathy.tif")
    generated = os.path.join(project_dir, "generated")
    scenario = "messina_001"
    
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

    slope(bathyfile, output_dir=rundir, logfile=logfile)

def slope(infile, output_dir, logfile, working_dir = None):
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
    """
    if working_dir is None: working_dir = output_dir
    outfile = "slope.tif"
    completed_proc = subprocess.run(
        ["whitebox_tools",
        "-r=Slope",
        "--dem={}".format(infile),
        "-o={}".format(outfile),
        "--max_procs={}".format(MAX_PROCS_PER_DEM)],
        cwd=working_dir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)
    return(outfile)

def basins():
    """
    Basins
    Description:
    Identifies drainage basins that drain to the DEM edge.
    Toolbox: Hydrological Analysis
    Parameters:

    Flag               Description
    -----------------  -----------
    --d8_pntr          Input raster D8 pointer file.
    -o, --output       Output raster file.
    --esri_pntr        D8 pointer uses the ESRI style scheme.


    Example usage:
    >>./whitebox_tools -r=Basins -v --wd="/path/to/data/" --d8_pntr='d8pntr.tif' -o='output.tif'

    ihttps://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#Basins
    
    
    BreachDepressions
    Description:
    Breaches all of the depressions in a DEM using Lindsay's (2016) algorithm. This should be preferred over depression filling in most cases.
    Toolbox: Hydrological Analysis
    Parameters:

    Flag               Description
    -----------------  -----------
    -i, --dem          Input raster DEM file.
    -o, --output       Output raster file.
    --max_depth        Optional maximum breach depth (default is Inf).
    --max_length       Optional maximum breach channel length (in grid cells; default is Inf).
    --flat_increment   Optional elevation increment applied to flat areas.
    --fill_pits        Optional flag indicating whether to fill single-cell pits.


    Example usage:
    >>./whitebox_tools -r=BreachDepressions -v --wd="/path/to/data/" --dem=DEM.tif -o=output.tif

    
    D8Pointer
    Description:
    Calculates a D8 flow pointer raster from an input DEM.
    Toolbox: Hydrological Analysis
    Parameters:

    Flag               Description
    -----------------  -----------
    -i, --dem          Input raster DEM file.
    -o, --output       Output raster file.
    --esri_pntr        D8 pointer uses the ESRI style scheme.


    Example usage:
    >>./whitebox_tools -r=D8Pointer -v --wd="/path/to/data/" --dem=DEM.tif -o=output.tif
    """
    
    return(None)

def log_process(completed_process, log_file):
        with open(log_file, 'a') as log:
            log.write("Process args: {}\n".format(" ".join(completed_process.args)))
            # If Python version > 3.7 replace with shlex.join (better formatting in logfile).
            log.write("Process stdout: {}\n".format(completed_process.stdout.decode("utf-8")))

if __name__ == "__main__":
    main()
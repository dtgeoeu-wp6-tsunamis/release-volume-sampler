import os, sys
import subprocess
import logging
from datetime import datetime
import shutil
import argparse

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger("slopeunits")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract slopeunits.")

    # Add arguments
    parser.add_argument(
        'rundir', 
        type=str, 
        help="Output directory."
    )
    
    parser.add_argument(
        'image', 
        type=str, 
        help="Singularity image."
    )
    
    parser.add_argument(
        'bathymetri', 
        type=str, 
        help="Bathymetri file. Projected coordinates."
    )
    
    # Parse the arguments
    args = parser.parse_args()

    # Check valid arguments
    if not os.path.isdir(args.rundir):
        parser.error(f"No such directory: {args.rundir}.")
    if not os.path.isfile(args.image):
        parser.error(f"No such file: {args.image}.")
    if not os.path.isfile(args.bathymetri):
        parser.error(f"No such file: {args.bathymetri}.")
        
    return args

def run_grassjob(rundir, singularity_image, bathy):
    """
    Create grass project from bathy in rundir.
    run grassjob.sh: calculates slopes and slopeunits.
    
    grass -e -c $rundir/bathy.tif [rundir]/grassdata
    grass $grass_project/PERMANENT --exec sh grassjob.sh $rundir
    """
    #grass_project = os.path.join(rundir, "grassdata")
    grass_project = "grassdata"
    slopeunits_dir = os.path.join(rundir,"slopeunits")
    create_dir(slopeunits_dir)
    
    logfile = os.path.join(slopeunits_dir, "log.txt")

    
    if not os.path.exists(grass_project):
        logger.info(f"Creating Grass GIS project from {bathy} in {slopeunits_dir}.")
        completed_proc = subprocess.run(
            ["singularity",
            "exec",
            singularity_image,
            "grass",
            "-e",
            "-c",
            bathy,
            "grassdata"],
            cwd=slopeunits_dir,
            stdout=subprocess.PIPE
        )
        completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
        log_process(completed_proc, logfile)
    
    logger.info(f"Copy grassjob.sh to {slopeunits_dir}.")
    # Get the directory where the script is located
    grassjob = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grassjob.sh")
    if not os.path.isfile(grassjob):
        logger.error(f"Cannot execute grassjob. No such file: {grassjob}.")
    shutil.copy(grassjob, slopeunits_dir)
    
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
        cwd=slopeunits_dir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)
    
    # Reproject output to fit bathymetri using gdalwarp
    
    _,_, bathy_profile = read_tif(bathy)
    # gdalwarp -t_srs EPSG:4326 slumap_clean_utm.tif slumap_clean.tif
    completed_proc = subprocess.run(
        ["singularity",
        "exec",
        singularity_image,
        "gdalwarp",
        "-t_srs",
        "EPSG:4326",
        "-te ",
        bathy],
        cwd=slopeunits_dir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)
    
    
    

def log_process(completed_process, log_file):
        with open(log_file, 'a') as log:
            log.write("Process args: {}\n".format(" ".join(completed_process.args)))
            # If Python version > 3.7 replace with shlex.join (better formatting in logfile).
            log.write("Process stdout: {}\n".format(completed_process.stdout.decode("utf-8")))


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            logger.info(f"Created directory {dir_name}")
        except OSError as e:
            sys.exit(f"Can't create {dir_name}: {e}")


def read_tif(fname):
    "Read .tif data and profile using rasterio."
    #logger.info(f"Read file: {fname}")
    with rasterio.open(fname) as src:
        #data = np.ma.masked_equal(src.read(1), src.nodata)
        data = src.read(1)
        msk = np.where(src.read_masks(1) == src.nodata, False, True)
        profile = src.profile.copy()
    return data, msk, profile


if __name__ == "__main__":
    # Parse and retrieve the arguments
    args = parse_args()
    logger.info(f"args: {args}")
    run_grassjob(rundir=args.rundir, singularity_image=args.image, bathy=args.bathymetri)
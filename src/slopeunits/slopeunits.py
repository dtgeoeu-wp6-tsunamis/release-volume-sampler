import os, sys
import subprocess
import logging
from datetime import datetime
import shutil
import argparse
import rasterio

logging.basicConfig(level = logging.INFO)
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
    Reproject slopeunits using gdalwarp so the raster is alligned with the bathymetri.
    
    grass -e -c $rundir/bathy.tif [rundir]/grassdata
    grass $grass_project/PERMANENT --exec sh grassjob.sh $rundir
    """
    #grass_project = os.path.join(rundir, "grassdata")
    grass_project = "grassdata"
    slopeunits_dir = os.path.join(rundir, "slopeunits")
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
    
    logger.info("Run grassjob.sh.")
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
    
    
    logger.info("Reproject slopeunits raster to fit bathymetri using gdalwarp")
    # File paths
    source_raster = os.path.join(slopeunits_dir, "slumap_clean_utm.tif") # Name is set in grassjob.sh
    output_raster = os.path.join(slopeunits_dir, "slumap.tif")           # Final slopeunits map.

    # Extract information from target raster using rasterio
    with rasterio.open(bathy) as target:
        target_crs = target.crs.to_string()  # CRS in Proj4 or EPSG format
        target_transform = target.transform
        target_resolution = (target_transform.a, -target_transform.e)  # Pixel size (x_res, y_res)
        
        # Calculate bounding box in the target CRS
        left, bottom, right, top = target.bounds

    # Define the gdalwarp command with extracted parameters
    gdalwarp_command = [
        "singularity",
        "exec",
        "/home/ebr/projects/release-volume-sampler/images/grass.sif",
        "gdalwarp",
        "-t_srs", target_crs,                                           # Target CRS
        "-tr", str(target_resolution[0]), str(target_resolution[1]),    # Target resolution
        "-te", str(left), str(bottom), str(right), str(top),            # Target extent (bounding box)
        "-r", "near",                                                   # Resampling method (e.g., bilinear)
        source_raster,                                                  # Input source raster
        output_raster                                                   # Output raster
    ]
    
    # Run gdalwarp
    completed_proc = subprocess.run(
        gdalwarp_command,
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


if __name__ == "__main__":
    # Parse and retrieve the arguments
    args = parse_args()
    logger.info(f"args: {args}")
    run_grassjob(rundir=args.rundir, singularity_image=args.image, bathy=args.bathymetri)
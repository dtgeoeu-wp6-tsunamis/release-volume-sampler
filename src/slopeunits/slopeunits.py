import os, sys
import subprocess
import logging
from datetime import datetime
import shutil
from utils.utils import log_process

logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger("slopeunits")

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
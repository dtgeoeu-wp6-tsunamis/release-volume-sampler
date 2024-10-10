import sys
import os
import logging
import rasterio

logger = logging.getLogger("utils")

def read_tif(fname):
    "Read .tif data and profile using rasterio."
    logger.info(f"Read file: {fname}")
    with rasterio.open(fname) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def write_tif(fname, data, profile):
    "Write .tif data and profile using rasterio."
    logger.info(f"Write file: {fname}")
    with rasterio.open(fname, 'w', **profile) as dst:
        dst.write(data, 1)


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            logger.info(f"Created directory {dir_name}")
        except OSError as e:
            sys.exit(f"Can't create {dir_name}: {e}")


def log_process(completed_process, log_file):
        with open(log_file, 'a') as log:
            log.write("Process args: {}\n".format(" ".join(completed_process.args)))
            # If Python version > 3.7 replace with shlex.join (better formatting in logfile).
            log.write("Process stdout: {}\n".format(completed_process.stdout.decode("utf-8")))
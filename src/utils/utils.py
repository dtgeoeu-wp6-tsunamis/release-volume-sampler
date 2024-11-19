import sys
import os
import logging
import rasterio
import json
import numpy as np
from contextlib import contextmanager


def read_tif(fname, logger=None):
    "Read .tif data and profile using rasterio."
    if logger: logger.info(f"Read file: {fname}")
    with rasterio.open(fname) as src:
        #data = np.ma.masked_equal(src.read(1), src.nodata)
        data = src.read(1)
        msk = np.where(src.read_masks(1) == src.nodata, False, True)
        profile = src.profile.copy()
    return data, msk, profile


def write_tif(fname, data, profile, logger=None):
    "Write .tif data and profile using rasterio."
    if logger: logger.info(f"Write file: {fname}")
    with rasterio.open(fname, 'w', **profile) as dst:
        dst.write(data, 1)


def create_dir(dir_name, logger = None):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            if logger: logger.info(f"Created directory {dir_name}")
        except OSError as e:
            sys.exit(f"Can't create {dir_name}: {e}")


def log_process(completed_process, log_file):
        with open(log_file, 'a') as log:
            log.write("Process args: {}\n".format(" ".join(completed_process.args)))
            # If Python version > 3.7 replace with shlex.join (better formatting in logfile).
            log.write("Process stdout: {}\n".format(completed_process.stdout.decode("utf-8")))


def write_content(content, output_dir):
    with open(os.path.join(output_dir, 'content.json'), 'w') as f:
        json.dump(content, f, indent=4)


def cummulative(samples, xs, weights=None, axis=-1):
    """
    Get the cumulative probability of a list of weighted samples.

    Params:
    
    xs: 1D array length j
        values at which to evaluate the cumulative function.
    samples: ndarray
        Sampled values stacked along axis.
    weights: 1D array
        Weights of each sample.
    axis: int
        Axis along which the samples are stacked. Defaults to -1 (last axis)
    
    Returns: ndarray of shape length of xs times number of samples.
        cumulative probabilities P(s < x).
    
    """
    if weights is None:
        n = samples.shape[axis]
        weights = np.ones(n)/n
    # First approach:
    #ind_order = np.argsort(samples, axis=axis)
    #sorted_samples = np.take_along_axis(samples, ind_order, axis=axis)
    #sorted_weights = weights[ind_order]
    # compute cumulative sums of weights and add 0 as an initial weight.
    #cumsum_weights = np.insert(np.cumsum(sorted_weights, axis=axis), 0, 0, axis=axis)
    #return(np.vstack([cumsum_weights[i, np.sum(sorted_samples < x, axis=-1)] for i,x in enumerate(xs)]))
    
    # Using np.average
    cummulative = np.vstack([np.average(samples < x, weights=weights, axis=axis) for x in xs])
    return(np.nan_to_num(cummulative))


@contextmanager
def temporary_working_directory(path):
    # Save the current working directory
    original_directory = os.getcwd()
    try:
        # Change to the new directory
        os.chdir(path)
        yield  # Control goes to the block under the with statement
    finally:
        # Return to the original directory
        os.chdir(original_directory)


def setup_logger(routine_name, log_folder):
    """
    Set up a logger for a specific routine.
    
    Args:
        routine_name (str): The name of the routine (used in the logfile name).
        log_folder (str): The folder where the logfile will be saved.

    Returns:
        logger: Configured logger object.
    """
    # Ensure the log folder exists
    os.makedirs(log_folder, exist_ok=True)
    
    # Define the log file path
    log_file = os.path.join(log_folder, f"{routine_name}.log")
    
    # Create a logger
    logger = logging.getLogger(routine_name)
    logger.setLevel(logging.INFO)  # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    # Prevent duplication of handlers if logger is already configured
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create a console (stream) handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
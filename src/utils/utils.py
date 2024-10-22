import sys
import os
import logging
import rasterio
import json
import numpy as np

logger = logging.getLogger("utils")

def read_tif(fname):
    "Read .tif data and profile using rasterio."
    #logger.info(f"Read file: {fname}")
    with rasterio.open(fname) as src:
        #data = np.ma.masked_equal(src.read(1), src.nodata)
        data = src.read(1)
        msk = np.where(src.read_masks(1) == src.nodata, False, True)
        profile = src.profile.copy()
    return data, msk, profile


def write_tif(fname, data, profile):
    "Write .tif data and profile using rasterio."
    #logger.info(f"Write file: {fname}")
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
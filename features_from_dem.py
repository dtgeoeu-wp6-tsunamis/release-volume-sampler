import os, sys
import subprocess
import logging
#import tempfile
from datetime import datetime
import rasterio
import shutil

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
        Feature extraction 
        - Create scenario folder
        - Copy bathymetry into scenario folder (bathy.tif)
        - Calculate slope 
        - Calculate aspect
        
        Segment bathymetry using K-means clustering
        - Extract features for each cell (x_center, y_center, slope, aspect)
        - Assign class: https://scikit-learn.org/stable/modules/clustering.html#k-means
        - Write class to raster.
    """
    # Settings
    project_dir = "/home/ebr/projects/release-volume-sampler"
    bathyfile = os.path.join(project_dir, "bathy/messina_001/bathy_truncated.tif")
    generated = os.path.join(project_dir, "generated")
    scenario = "messina_001"
    run = None #"messina_001_20240816_092455"
    
    calculate_geomorphometric_features = True

    segment_bathy = True
    n_labels = 200
    weights = [20,20, 1, 1, 1] # Weighting of features (x, y, slope, cos_aspect, sin_aspect)
    random_state = 0 # Random state for initialisation of Clustering.
    
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
    if calculate_geomorphometric_features:
        bathy = preprocess_bathy(bathyfile, output_dir=rundir)
        slope(bathy, output_dir=rundir, logfile=logfile)
        aspect(bathy, output_dir=rundir, logfile=logfile)

    if segment_bathy:
        X, bathy_mask, bathy_transform = load_features(bathyfile, rundir)
        labels = fit_labels(X, n_labels, weights, random_state)
        outfile = write_labels_to_file(labels, bathy_mask, bathy_transform, rundir)


def preprocess_bathy(bathymetry, output_dir):
    """
    Copying bathy to generated folder.
    TODO: Include preprocessing steps (When gdal is available).
    # To truncate positive values
    gdal_calc.py -A bathy/localMessinaBathy.tif --outfile=truncated.tif --calc="A*(A<0)" --NoDataValue=0
    """
    outfile = "bathy.tif"
    logger.info(f"Copy {outfile} to {output_dir}.")
    shutil.copy(bathymetry, os.path.join(output_dir, outfile))
    return outfile


def slope(infile, output_dir, logfile, working_dir = None):
    logger.info("Calculating slopes.")
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
        "--units=radians",
        "--max_procs={}".format(MAX_PROCS_PER_DEM)],
        cwd=working_dir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)
    return(outfile)


def aspect(infile, output_dir, logfile, working_dir = None):
    logger.info("Calculating aspects.")
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
    if working_dir is None: working_dir = output_dir
    outfile = "aspect.tif"
    completed_proc = subprocess.run(
        ["whitebox_tools",
        "-r=Aspect",
        "--dem={}".format(infile),
        "-o={}".format(outfile),
        "--max_procs={}".format(MAX_PROCS_PER_DEM)],
        cwd=working_dir,
        stdout=subprocess.PIPE
    )
    completed_proc.check_returncode()  # raise CalledProcessError if return code is non-zero.
    log_process(completed_proc, logfile)
    return(outfile)


def load_features(bathyfile, rundir):
    logger.info("Loading bathymetric features.")
    # Logg available rasters in rundir.
    rasters = []
    for filename in filter(lambda file: file.endswith(".tif"), os.listdir(rundir)): 
        with rasterio.open(os.path.join(rundir,filename)) as raster:
            rasters.append(os.path.split(raster.name)[-1])
    logger.info(f"Found rasters: {rasters} in folder: {rundir}")
    
    # Load features
    with rasterio.open(os.path.join(rundir, "slope.tif")) as ds:
        slope = ds.read(1)

    with rasterio.open(os.path.join(rundir, "aspect.tif")) as ds:
        aspect = ds.read(1)

    with rasterio.open(os.path.join(rundir, "bathy.tif")) as bathy_ds:
        logger.info(bathy_ds.crs)
        z = bathy_ds.read(1)
        bathy_transform = bathy_ds.transform

    cols, rows = np.meshgrid(np.arange(bathy_ds.width), np.arange(bathy_ds.height))
    xs, ys = bathy_transform * (rows, cols)

    sin_aspect = np.sin(aspect*np.pi/180.)
    cos_aspect = np.cos(aspect*np.pi/180.)

    mask = z<0
    X = np.vstack([xs[mask], ys[mask], slope[mask], sin_aspect[mask], cos_aspect[mask]]).T
    return(X, mask, bathy_transform)


def fit_labels(X, n_labels, weights, random_state):
    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X)
    X_trans = scaler.transform(X)
    
    # Fit
    kmeans = KMeans(n_clusters=n_labels,
                    random_state=random_state,
                    n_init="auto", 
                    algorithm="lloyd")
    
    kmeans.fit(np.matmul(X_trans,np.diag(weights)))
    return(kmeans.labels_)


def write_labels_to_file(labels, bathy_mask, bathy_transform, rundir):
    label_map = np.empty(bathy_mask.shape)
    label_map[:] = np.nan
    label_map[bathy_mask] = labels
    
    outfile = "labels.tif"
    with rasterio.open(
        os.path.join(rundir, outfile),
        'w',
        driver='GTiff',
        height=bathy_mask.shape[0],
        width=bathy_mask.shape[1],
        count=1,
        dtype=labels.dtype,
        crs='+proj=latlong',
        transform=bathy_transform,
    ) as dst:
        dst.write(label_map, 1)
    return(outfile)


def log_process(completed_process, log_file):
        with open(log_file, 'a') as log:
            log.write("Process args: {}\n".format(" ".join(completed_process.args)))
            # If Python version > 3.7 replace with shlex.join (better formatting in logfile).
            log.write("Process stdout: {}\n".format(completed_process.stdout.decode("utf-8")))


if __name__ == "__main__":
    main()

import rasterio
import numpy as np
import os
import logging
import json
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from utils.utils import create_dir, read_tif, write_tif

""" Cacluation of displacements from yield acceleration and shakemaps.

excecution: poetry run python displacements/displacements.py or call calculate_displacements from main.py.
"""

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('slope_analysis')
figure_dir = None


def displacement(ky, pga, M=None, pgv=None, model="scalar"):
    """ Calculation of ground displacements
    Implementation of the statistical model for ground displacements of natural slopes subject to earthquakes 
    given in [1]. Note: The statistical model is develpped for subaerial conditions.
    
    Parameters:
        ky: float or ndarray 
            Yield acceleration calculated using infinite slope analysis [g].
        pga: float or ndarray. Same dimension as ky.
            Peak ground acceleration of the event [g]. 
        M: float
            moment magnitude of the event.
        pgv: float or ndarray. Same dimension as ky.
            Peak ground velocity of the event [cm/s].
        model: string
            Different models. Options are "scalar" or "vector". If "scalar", then M must be supplied. 
            If "vector", then pgv must be supplied. Default is "scalar".
        
    Returns:
        ln(displacements) [cm], standard_deviation: float or ndarray, float or ndarray
        Logarithm of estimated displacements and associated standard deviation. 
        Standard deviation of lognormal multiplicative noise (as a function of ky/pga).  
        
    1. Rathje and Saygili, ‘Probabilistic Assessment of Earthquake-Induced Sliding Displacements of Natural Slopes’.
    """
    if model == "scalar":
        a = [-29.06, 42.49, - 19.64,-4.85, 4.89] # polynomial coefficients.
        ln_d = np.polyval(a, ky/pga) + 0.72*np.log(pga) + 0.89*(M-6)
        sigma_ln = np.polyval([-0.539, 0.789, 0.732], ky/pga)
    elif model == "vector":
        a = [-30.5, 44.75, -20.84, -4.58, -1.56]
        ln_d = np.polyval(a, ky/pga) - 0.64*np.log(pga) + 1.55*np.ln(pgv)
        sigma_ln = 0.405 + 0.524*(ky/pga)
    return(ln_d, sigma_ln)


def calculate_displacements(rundir, shakemaps_filename, write_shakemaps=False, write_displacements=False, make_plots=False):
    #pga=0.7
    magnitude = 7.
    shake_value = 'Z_pga'
    nr_of_shake_samples = 3
    interpolation_method = 'linear' # “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”

    # Load shakemaps from file 
    with open(shakemaps_filename, 'r') as f:
        shakemaps = json.load(f)
    
    # Create output directories
    if write_displacements:
        displacements_dir = os.path.join(rundir, "displacements")
        create_dir(displacements_dir)
    if write_shakemaps:
        shakemaps_dir = os.path.join(rundir, "shakemaps")
        create_dir(shakemaps_dir)
    if make_plots:
        figure_dir = os.path.join(rundir, "figures")
        create_dir(figure_dir)
    
    # Load yield acceleration maps.
    slope_analysis_output_folder = os.path.join(rundir, "slope_analysis")

    # load content file
    with open(os.path.join(slope_analysis_output_folder, "content.json"),'r') as f:
        slope_analysis_output = json.load(f)

    # Read yield acceleration output from slope analysis.
    yield_acceleration_quantiles = [e for e in slope_analysis_output if e["value"] == "yield_acceleration"]

    output = []
    for shake_sample in range(nr_of_shake_samples):
        for i, element in enumerate(yield_acceleration_quantiles):
            with rasterio.open(os.path.join(slope_analysis_output_folder, element["file"])) as src:
                ky = src.read(1)
                ky_profile = src.profile.copy()
                if i == 0:
                    # New shakemap.
                    pga = interpolate_shakemap(shakemaps=shakemaps,
                                        shake_value=shake_value,
                                        shake_sample=shake_sample,
                                        interpolation_method=interpolation_method,
                                        bbox=src.bounds,
                                        n_cols=ky_profile["width"],
                                        n_rows=ky_profile["height"])
                    if write_shakemaps:
                        shakemap_file = os.path.join(shakemaps_dir, "pga_{}.tif".format(shake_sample))
                        write_tif(shakemap_file, pga, ky_profile)
            
                # Calculate displacement quantiles 
                # Here we apply the fact that displacements are decreasing with ky.
                ln_d, ln_sigma = displacement(ky=ky, pga=pga, M=magnitude)
                if make_plots:
                    plot_raster(ln_d, "displacement_{}_{}.png".format(shake_sample, i))
                
                if write_displacements:
                    filename = f"d_{i}_{shake_sample}.tif"
                    output.append({
                        "file": filename, 
                        "quantile": 1. - element["quantile"],
                        "shakemaps_filename": shakemaps_filename,
                        "shakemap_sample": shake_sample,
                        "shake_value": shake_value,
                        "value": "displacement", 
                        "scaling": "ln",
                        })
                    write_tif(os.path.join(displacements_dir, filename), ln_d, ky_profile)
                
    if write_displacements:
        content_file = os.path.join(displacements_dir, 'content.json')
        logger.info("Write file: {}".format(content_file))
        with open(content_file, 'w') as f:
            json.dump(output, f, indent=4)


def plot_raster(data, filename):
    """
        Applied to debug shakemap interpolation.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation='bilinear', origin='upper')
    plt.savefig(os.path.join(figure_dir, filename))


def interpolate_shakemap(shakemaps, shake_value, shake_sample, interpolation_method, bbox, n_rows, n_cols, make_plots=False):
    # Create grid interpolator
    lon_shake, lat_shake, data = zip(*[(point['lon'], point['lat'], point[shake_value][shake_sample]) for _, point in shakemaps.items()])
    grid_shake = np.array(list(set(lon_shake))), np.array(list(set(lat_shake))) # Extract unique values 
    [g.sort() for g in grid_shake] # in ascending order.
    grid_shake_shape = grid_shake[0].shape[0], grid_shake[1].shape[0]
    data = np.reshape(data, grid_shake_shape, order='C') # Reshape data. 
    # Lon fixed, Lat change -> C order. Location of data(ij): lowerleft + (j*delta_lat, i*delta_lon) 
    if make_plots:
        plot_raster(np.flip(data.T,-1), os.path.join(figure_dir, "shakemap.png"))
    shake_interp = RegularGridInterpolator(points=grid_shake, values=np.flip(data.T,-1), method=interpolation_method) # Data need to be ij matrix format (See np.meshgrid)
    
    # verify that interpolation is exact at grid values.RSS nonzero with flipped axes...
    #rss = np.array([(shake_interp((point['lon'], point['lat'])) - point[shake_value][shake_sample])**2 for _,point in shakemaps.items()]).sum()
    #logger.debug("RSS at grid points: {}".format(rss))
    
    # Interpolate grid.
    grid_int = np.meshgrid(np.linspace(bbox.left, bbox.right, num=n_cols), np.linspace(bbox.bottom, bbox.top, num=n_rows), indexing='ij')
    shake_values = shake_interp(grid_int)
    if make_plots:
        plot_raster(shake_values, os.path.join(figure_dir, "shake_values.png"))
    return(np.flip(shake_values,-1).T)


def main():
    rundir = "/home/ebr/projects/release-volume-sampler/generated/messina_001_20241010_134621"
    shakemaps_filename = "/home/ebr/projects/release-volume-sampler/input/shakemaps/messina_1908/predicted_data_NN_Messina_1908.json"
    
    calculate_displacements(rundir, shakemaps_filename, write_shakemaps=False, write_displacements=True, make_plots=True)

if __name__ == "__main__":
    main()
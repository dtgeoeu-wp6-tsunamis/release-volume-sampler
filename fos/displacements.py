import rasterio
import numpy as np
import os
import logging

""" Cacluation of displacements

excecution: poetry run python fos/displacements.py or call from main.py

Options for feeding data:
    1. Single scalar value.
    2. Raster with geological units and dictionary linking units to values.
    3. Raster with values.

"""


#logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('displacements')


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


def get_fos(slope_data, params):
    """
    Calculation of FOS with no ground movement (s = 0).
    """
    logger.info("Calculating Factor Of Safety.")
    
    gamma = (params["density"] - params["density_of_water"])/params["density"]
    c = params["cohesion"]*1000. #in Pa
    u = params["excess_pore_pressure"]*1000. #in Pa
    mu = np.tan(np.radians(params["friction_angle"]))
    g = params["gravity"]
    rho = params["density"]
    H = params["thickness"]
    alpha = np.radians(slope_data)
    
    return (c-u*mu)/(rho*H*gamma*g*np.sin(alpha)) + mu/np.tan(alpha)


def calculate_factor_of_safety(working_dir, params, filenames):
    """
    Load files, calculate and write output to file.
    """
    slope_data, slope_profile = read_tif(os.path.join(working_dir, filenames["slope"]))
    fos = get_fos(slope_data, params)
    write_tif(fname = os.path.join(working_dir, "fos.tif"),
              data = fos, 
              profile = slope_profile)


def main():
    #logger.addHandler(logging.StreamHandler())
    working_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001_20240909_130842"

    filenames = {
        "slope": "slope.tif"
    }

    params = {
        "friction_angle": 24.3,         # [degrees]
        "cohesion": 20.,                # [kPa]
        "thickness": 4,                 # depth to slide surface measured along slope normal [m].
        "density": 2000,                # density of slide [kg/m3]
        "density_of_water": 1020,       # density of water [kg/m3]
        "gravity": 9.81,                # [m/s2]
        "excess_pore_pressure": 0.      # [kPa]
    }
    
    calculate_factor_of_safety(working_dir=working_dir, params=params, filenames=filenames)


if __name__ == "__main__":
    main()
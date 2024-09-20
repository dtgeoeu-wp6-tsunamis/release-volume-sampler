import rasterio
import numpy as np
import os
import sys
import logging

""" Cacluation of Factor of Safety and yield acceleration.

excecution: poetry run python slope_analysis/slope_analysis.py or call from main.py
"""

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('slope_analysis')


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


def infinite_slope_analysis(slope, friction_angle, cohesion, thickness, density, density_of_water = 1000, gravity = 9.81, excess_pore_pressure = 0., yield_angle = None):
    """
    Returns Factor of Safety and Yield Acelleration associated with an infinite slope analysis.
    Yield acceleration is expressed in multiples of gravity. Cyrrently drained/undrained conditions are not considered.
    
    Parameters: 
    
    slope: number or ndarray
        Slope angle in degrees.
    friction_angle:
        friction angle in degrees.
    cohesion: 
        Cohesion [kPa]
    thickness:
        depth to slide surface measured slope normal [m].
    density:
        density of slide [kg/m3]
    density_of_water:
        density of water [kg/m3] (default is 1000 kg/m3)
    gravity:
        acceleration of gravitation [m/s2] (default is 9.81)
    excess_pore_pressure:
        Excess pore pressure [kPa] (default is 0)
    yield_angle:
        Direction of shaking given in degrees with horizontal axis. If None, shaking is assumed to be parallell with slope.
    TODO: Verify correction factor for yield acelleration!
    """
    logger.info("Calculating Factor Of Safety.")
    
    gamma = (density - density_of_water)/density
    c = cohesion*1000. #in Pa
    u = excess_pore_pressure*1000. #in Pa
    mu = np.tan(np.radians(friction_angle))
    g = gravity
    rho = density
    H = thickness
    alpha = np.radians(slope)
    
    fos = (c-u*mu)/(rho*H*gamma*g*np.sin(alpha)) + mu/np.tan(alpha)
    ky = gamma*(np.sin(alpha)*(fos-1.))
    
    if yield_angle is not None:
        psi = np.radians(yield_angle)
        ky = ky/(np.cos(psi-alpha) - np.sin(psi-alpha)*mu) # Correction factor
    return fos, ky


def run_analysis(working_dir, quantiles, physical_parameters, slopefile="slope.tif", write_fos=True, write_ky=True):
    """
    Run infinite slope analysis with uncertain input parameters. Output is written to folder 
    the subdirectory [working_dir]/slope_analysis.
    
    Parameters:
    
    working_dir: str
        Path to working directory. 
    quantiles: list
        List of quantiles to output raster maps.
    slopefile: str
        name of raster with calculated slopes. Has to be located in working_directory. Defaults to slope.tif.
    write_fos: bool
        If True, calculates quantiles of Factor-of-Safety and writes to files. Defaults to True
    write_ky: bool
        If True, calculates quantiles of yield acceleration and writes to files. Defaults to True
    physical_parameters: dict
        Geotechnical parameters and sliding layer properties defined as discrete distributions or constants. 
        keys: friction_angle, cohesion, thickness, density must be included either as distribution or constants. 
        optional: density_of_water, gravity, excess_pore_pressure, yield_angle have default values (See function infinite_slope_analysis).
        Todo: May supply functional relations. Order keys according to dependencies.
        
        Example:
            physical_parameters = {
                "distributions": {
                    "friction_angle": [(24.3, 1)], # [(value, weight),...]
                    "cohesion": [(20, 1)],
                    "thickness": [(3.6, 0.248), (4, 0.504),(4.4, 0.248)],
                    "density": [(1800, 0.5),(2000, 0.5)],
                },
                "constants": {
                    "density_of_water": 1020,
                    "gravity": 9.81,            # Defaults to 9.81
                    "excess_pore_pressure": 0.  # Defaults to 0.
                    "yield_angle": 0. # Defaults to None in which case it is assumed to be parallel with slope.
                }
            }
    """
    
    # Create subdirectory for output
    slope_analysis_dir = os.path.join(working_dir, "slope_analysis")
    if not os.path.exists(slope_analysis_dir):
        try:
            os.makedirs(slope_analysis_dir)
            logger.info(f"Created directory {slope_analysis_dir}")
        except OSError as e:
            sys.exit(f"Can't create {slope_analysis_dir}: {e}")

    # Create event tree.
    root_node = Node(weight=1, distributions=physical_parameters["distributions"], params=physical_parameters["constants"], parent=None)
    
    # Traverse leaf nodes and create linear interpolations for slope variation.
    slope_data, slope_profile = read_tif(os.path.join(working_dir, slopefile))
    
    # Create slope range for interpolation
    slopes = np.linspace(np.nanmin(slope_data), np.nanmax(slope_data), num=100)
    kys, weights, foss = [], [], []
    sum_of_weights = 0.

    for count, node in enumerate(root_node.leaf_nodes):
        logger.info(f"Leaf node:{count} \n Weight:{node.weight} \n Params:{node.params} \n")
        fos, ky = infinite_slope_analysis(slopes, **node.params)
        foss.append(fos)
        kys.append(ky)
        weights.append(node.weight)
        sum_of_weights += node.weight
    
    # Compute quantiles
    fos_quantiles = np.quantile(np.stack(foss), quantiles, axis=0, weights=np.array(weights), method='inverted_cdf')
    ky_quantiles = np.quantile(np.stack(kys), quantiles, axis=0, weights=np.array(weights), method='inverted_cdf')
    logger.info(f"Sum of weights: {sum_of_weights}")
    
    # Evaluate quantiles by interpolation and write to file
    for i,quantile in enumerate(quantiles):
        fos_quantiles_raster = np.interp(slope_data, slopes, fos_quantiles[i,:])
        write_tif(fname = os.path.join(working_dir,"fos", f"fos_{quantile}.tif"), data = fos_quantiles_raster, profile = slope_profile)
        ky_quantiles_raster = np.interp(slope_data, slopes, ky_quantiles[i,:])
        write_tif(fname = os.path.join(working_dir,"fos", f"ky_{quantile}.tif"), data = ky_quantiles_raster, profile = slope_profile)


def main():
    # Example...
    
    working_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001_20240909_130842"
    quantiles = [0.1, 0.2, 0.5]
    
    # Parameters defined as discrete distributions or constants.
    # May supply functional relations. Order according to dependencies.
    physical_parameters = {
        "distributions": {
            "friction_angle": [(24.3, 1)], # [(value, weight),...]
            "cohesion": [(20, 1)],
            "thickness": [(3.6, 0.248), (4, 0.504),(4.4, 0.248)],
            "density": [(1800, 0.5),(2000, 0.5)],
        },
        "constants": {
            "density_of_water": 1020,
            "gravity": 9.81,
            "excess_pore_pressure": 0.
        }
    }

    run_analysis(working_dir, quantiles, physical_parameters, slopefile="slope.tif", write_fos=True, write_ky=True)


class Node():
    leaf_nodes = []
    list_of_parameters = None
    
    
    def __init__(self, weight, params, distributions, parent):
        self.weight = weight
        self.distributions = distributions 
        self.params = params
        self.parent = parent
        self.children = []
        
        if self.list_of_parameters is None:
            self.list_of_parameters = list(self.distributions.keys()) + list(self.params.keys())
        
        if len(self.params) < len(self.list_of_parameters):
            self.create_children()
        else:
            self.leaf_nodes.append(self)


    def create_children(self):
        # Select first parameter not already created
        parameter = list(self.distributions.keys())[0]
        
        distributions = self.distributions.copy()
        for value, weight in distributions.pop(parameter):
            #TODO: Check if value is function of params and calculate value(params)
            # One option is to apply and suply a string. https://docs.sympy.org/latest/modules/utilities/lambdify.html
            params = self.params.copy()
            params[parameter] = value
            self.children.append(Node(self.weight*weight, params, distributions, self))


if __name__ == "__main__":
    main()
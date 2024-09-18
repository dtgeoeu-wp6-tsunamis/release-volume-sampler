import rasterio
import numpy as np
import os
import logging

""" Cacluation of displacements

excecution: poetry run python fos/displacements.py or call from main.py

Options for feeding data:
    1. Single scalar value or point distributions.
    2. Raster with geological units and dictionary linking units to values.
    3. Raster with values.
    
"""


logging.basicConfig(level = logging.INFO)
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
    
    params = {
        "friction_angle": 24.3,         # [degrees]     normal distribution
        "cohesion": 20.,                # [kPa]         lognormal distribution
        "thickness": 4,                 # depth to slide surface measured along slope normal [m].
        "density": 2000,                # density of slide [kg/m3]
        "density_of_water": 1020,       # density of water [kg/m3]
        "gravity": 9.81,                # [m/s2]
        "excess_pore_pressure": 0.      # [kPa]
    }
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


def main():
    #logger.addHandler(logging.StreamHandler())
    working_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001_20240909_130842"
    
    # TODO: Create subdirectory fos

    filenames = {
        "slope": "slope.tif"
    }
   
    # Parameters defined as discrete distributions or constants.
    # May supply functional relations. Order according to dependencies.
    distributions = {
        "friction_angle": [(24.3, 1)], # [(value, weight),...]
        "cohesion": [(20, 1)],
        "thickness": [(3.6, 0.248), (4, 0.504),(4.4, 0.248)],
        "density": [(1800, 0.5),(2000, 0.5)],
    }

    params = {
        "density_of_water": 1020,
        "gravity": 9.81,
        "excess_pore_pressure": 0.
    }

    # Create event tree.
    root_node = Node(weight=1, distributions=distributions,params=params, parent=None)
    
    # Traverse leaf nodes and calculate fos
    slope_data, slope_profile = read_tif(os.path.join(working_dir, filenames["slope"]))
    
    sum_of_weights = 0.
    for count, node in enumerate(root_node.leaf_nodes):
        logger.info(f"Leaf node:{count} \n Weight:{node.weight} \n Params:{node.params} \n")
        fos = get_fos(slope_data, node.params)
        # write_tif(fname = os.path.join(working_dir,"fos", f"fos_{count}.tif"), data = fos, profile = slope_profile)
        
        # TODO aggregate (approximate) cumulative distribution
        
        sum_of_weights += node.weight
    
    logger.info(f"Sum of weights: {sum_of_weights}")
    

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
            params = self.params.copy()
            params[parameter] = value
            self.children.append(Node(self.weight*weight, params, distributions, self))


if __name__ == "__main__":
    main()
import numpy as np
import os
import logging
import json

from utils.utils import read_tif, write_tif, create_dir, cummulative


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('slope_analysis')


class SlopeAnalysis:
    
    def __init__(self, working_dir, physical_parameters, slopefile="slope.tif"):
        """
        Run infinite slope analysis with uncertain input parameters. 
        
        Parameters:
        
        working_dir: str
            Path to working directory. 
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
        slopefile: str
            name of raster with calculated slopes. Has to be located in working_directory. Defaults to slope.tif.
        """
        self.working_dir = working_dir
        self.physical_parameters = physical_parameters
        self.slopefile = slopefile
        
        # Subdirectory names for output
        self.fos_dir = os.path.join(working_dir, "fos")
        self.yield_acceleration_dir = os.path.join(working_dir, "yield_acceleration")
        
        # Load slopedata
        self.slope_data, self.slope_msk, self.slope_profile = read_tif(os.path.join(working_dir, slopefile))
        
        # Create event tree.
        self.root_node = Node(weight=1, distributions=physical_parameters["distributions"], params=physical_parameters["constants"], parent=None)
        
        # Traverse leaf nodes and create linear interpolations for slope variation.
        self.slopes = np.linspace(np.nanmin(self.slope_data), np.nanmax(self.slope_data), num=500)
        logger.info(f"Slope range for lookuptable: {np.min(self.slopes)}, {np.max(self.slopes)}")
        kys, weights, foss = [], [], []
        sum_of_weights = 0.
        
        for count, node in enumerate(self.root_node.leaf_nodes):
            logger.info(f"Leaf node:{count} \n Weight:{node.weight} \n Params:{node.params} \n")
            fos, ky = self.infinite_slope_analysis(self.slopes, **node.params)
            foss.append(fos)
            kys.append(ky)
            weights.append(node.weight)
            sum_of_weights += node.weight
        
        logger.info(f"Sum of weights: {sum_of_weights}")
        self.foss = np.stack(foss)
        logger.info(f"foss.shape: {self.foss.shape}")
        self.kys = np.stack(kys)
        logger.info(f"kys.shape: {self.kys.shape}")
        self.weights = np.array(weights)
        logger.info(f"weights.shape: {self.weights.shape}")
    
    
    def compute_quantiles(self, quantiles, write_fos=False, write_ky=False):
        """
        Compute quantiles of the factor of safety and the yield acceleration.
        Output is written to the subdirectory [working_dir]/slope_analysis.
        
        Parameters:
        
        quantiles: list
            List of quantiles to output raster maps.
        write_fos: bool
            If True, calculates quantiles of Factor-of-Safety over the topography and writes each quantile to file. Defaults to False.
        write_ky: bool
            If True, calculates quantiles of yield acceleration over the topography and writes each quantile to file. Defaults to False.
        
        """
        
        fos_output, ky_output = [], []
        
        # Create Lookup tables
        fos_quantiles = np.quantile(self.foss, quantiles, axis=0, weights=self.weights, method='inverted_cdf')
        ky_quantiles = np.quantile(self.kys, quantiles, axis=0, weights=self.weights, method='inverted_cdf')
        
        if write_fos: 
            create_dir(self.fos_dir)
            fos_quantile_dir = os.path.join(self.fos_dir, "quantiles")
            create_dir(fos_quantile_dir)
        if write_ky: 
            create_dir(self.yield_acceleration_dir)
            ky_quantile_dir = os.path.join(self.yield_acceleration_dir, "quantiles")
            create_dir(ky_quantile_dir)
        
        # Evaluate quantiles by interpolation of lookuptables over topography and write to files.
        for i, quantile in enumerate(quantiles):
            ky_quantiles_filename, fos_quantiles_filename = f"ky_quantiles_{i}.tif", f"fos_quantiles_{i}.tif"
            
            if write_fos:
                fos_quantile_flat = np.interp(self.slope_data[self.slope_msk], self.slopes, fos_quantiles[i,:])
                fos_quantile_raster = np.empty(self.slope_data.shape)
                fos_quantile_raster[self.slope_msk] = fos_quantile_flat
                fos_quantile_raster[~self.slope_msk] = np.nan
                write_tif(fname = os.path.join(fos_quantile_dir, fos_quantiles_filename), data = fos_quantile_raster, profile = self.slope_profile)
                fos_output.append({"file": fos_quantiles_filename, "quantile": quantile, "value": "log factor_of_safety", "scale": "log10", "unit":""})
        
            if write_ky: 
                ky_quantile_flat = np.interp(self.slope_data[self.slope_msk], self.slopes, ky_quantiles[i,:])
                ky_quantile_raster = np.empty(self.slope_data.shape)
                ky_quantile_raster[self.slope_msk] = ky_quantile_flat
                ky_quantile_raster[~self.slope_msk] = np.nan
                write_tif(fname = os.path.join(ky_quantile_dir, ky_quantiles_filename), data = ky_quantile_raster, profile = self.slope_profile)
                ky_output.append({"file": ky_quantiles_filename, "quantile": quantile, "value": "log yield_acceleration", "scale":"log10", "unit": "g"})
        
        if write_fos: self.write_content(fos_output, fos_quantile_dir)
        if write_ky: self.write_content(ky_output, ky_quantile_dir)
    
    
    def compute_cummulative_fos(self, thresholds, write_fos=False):
        # Lookup table
        fos_cummulative = cummulative(self.foss, thresholds, weights=self.weights, axis=0)
        logger.info(f"fos_cummulative shape: {fos_cummulative.shape}")
        
        if write_fos:
            output = []
            create_dir(self.fos_dir)
            fos_cum_dir = os.path.join(self.fos_dir, "cummulative")
            create_dir(fos_cum_dir)
        
            # Evaluate cummulative by interpolation and write to files.
            for i, threshold in enumerate(thresholds):
                fos_cummulative_filename = f"fos_cum_{i}.tif"
                fos_cummulative_flat = np.interp(self.slope_data[self.slope_msk], self.slopes, fos_cummulative[i,:])
                fos_cummulative_raster = np.empty(self.slope_data.shape)
                fos_cummulative_raster[self.slope_msk] = fos_cummulative_flat
                fos_cummulative_raster[~self.slope_msk] = np.nan
                write_tif(fname = os.path.join(fos_cum_dir, fos_cummulative_filename), data = fos_cummulative_raster, profile = self.slope_profile)
                output.append({"file": fos_cummulative_filename, "threshold":threshold, "value": "cummulative of logfos", "scale": "", "unit": ""})
            
            self.write_content(output, fos_cum_dir)
        return(fos_cummulative)
    
    
    def compute_cummulative_ky(self, thresholds, write_ky=True):
        # Lookup table
        ky_cummulative = cummulative(self.kys, thresholds, weights=self.weights, axis=0)
        
        if write_ky:
            ky_output = []
            create_dir(self.yield_acceleration_dir)
            ky_cum_dir = os.path.join(self.yield_acceleration_dir, "cummulative")
            create_dir(ky_cum_dir)
            
            # Evaluate cummulative by interpolation and write to files.
            for i, threshold in enumerate(thresholds):
                ky_cummulative_filename = f"ky_cum_{i}.tif"
                ky_cummulative_flat = np.interp(self.slope_data[self.slope_msk], self.slopes, ky_cummulative[i,:])
                ky_cummulative_raster = np.empty(self.slope_data.shape)
                ky_cummulative_raster[self.slope_msk] = ky_cummulative_flat
                ky_cummulative_raster[~self.slope_msk] = np.nan
                write_tif(fname = os.path.join(ky_cum_dir, ky_cummulative_filename), data = ky_cummulative_raster, profile = self.slope_profile)
                ky_output.append({"file": ky_cummulative_filename, "threshold":threshold, "value": "probability", "scale": "", "unit": "g"})
            
            self.write_content(ky_output, ky_cum_dir)
        return(ky_cummulative)
        
    
    def write_content(self, content, output_dir):
        with open(os.path.join(output_dir, 'content.json'), 'w') as f:
            json.dump(content, f, indent=4)
        
    
    @staticmethod
    def infinite_slope_analysis(slope, friction_angle, cohesion, thickness, density, density_of_water = 1000, gravity = 9.81, excess_pore_pressure = 0., yield_angle = None, eps=0.001):
        """
        Returns Factor of Safety and Yield Acelleration associated with an infinite slope analysis.
        Yield acceleration is expressed in log10 scale of multiples of gravity. Cyrrently drained/undrained conditions are not considered.
        
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
        eps: float (default is 0.001)
            Truncation thresshold. Set fos and ky to NAN when sin of slope is less than eps. 
        TODO: Verify correction factor for yield acelleration!
        
        Returns:
        fos, ky: (ndarray, ndarray)
            Base 10 logarithm of the factor of safety and the yield acelleration.
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
        eps = 1e-6 # truncation threshold.
        
        fos = np.where(np.sin(alpha) > eps, (c-u*mu)/(rho*H*gamma*g*np.sin(alpha)) + mu/np.tan(alpha), np.nan)
        #fos = (c-u*mu)/(rho*H*gamma*g*np.sin(alpha)) + mu/np.tan(alpha)
        if np.sum(fos < 1) > 0:
            logger.warning(f"FOS less than 1. Replace with 1.")
            fos = np.where(fos < 1, 1., fos)
        
        #ky = np.where(np.sin(alpha) > eps, gamma*np.sin(alpha)*(fos-1.), np.nan) 
        #ky = gamma*np.sin(alpha)*(fos-1.)
        
        fos_times_sin = (c-u*mu)/(rho*H*gamma*g) + mu*np.cos(alpha)
        ky = gamma*(fos_times_sin-np.sin(alpha))
        
        if yield_angle is not None:
            psi = np.radians(yield_angle)
            ky = ky/(np.cos(psi-alpha) - np.sin(psi-alpha)*mu) # Correction factor
        
        return np.log10(fos), np.log10(ky)


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

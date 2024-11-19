import numpy as np
import os
import logging
import json

from src.utils.utils import read_tif, write_tif, create_dir, cummulative, setup_logger


class SlopeAnalysis:
    
    def __init__(self, rundir, regionfile="soilregions.tif", soilparamsfile="soilparams.json", slopefile="slope.tif"):
        """
        Run infinite slope analysis with uncertain input parameters. 
        
        Parameters:
        
        rundir: str
            Path to working directory.
        soilregionsfile: str
            Name of regionsfile contained in wrking_dir. Partition of the region into units with different soilparameters. 
        soilparamsfile: str
            jsonfile with list of geotechnical parameters and sliding layer properties defined as discrete distributions or constants for each region. 
            Each set of parameters ordered according to the values of the regionfile. 
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
            name of raster with calculated slopes. Has to be located in rundirectory. Defaults to slope.tif.
        """
        self.out_dir = os.path.join(rundir, "slope_analysis")
        create_dir(self.out_dir)
        self.logger = setup_logger("slopeanalysis", self.out_dir)
        #self.physical_parameters = physical_parameters
        self.slopefile = slopefile
        self.regionfile = regionfile
        self.soilparamsfile = soilparamsfile
        
        # Subdirectory names for output
        self.fos_dir = os.path.join(self.out_dir, "fos")
        self.yield_acceleration_dir = os.path.join(self.out_dir, "yield_acceleration")
        
        # Load data
        self.slope_data, self.slope_msk, self.slope_profile = read_tif(os.path.join(rundir, slopefile), self.logger)
        
        # load soil parameters file.
        soilparameter_file_path = os.path.join(rundir, self.soilparamsfile)
        with open(soilparameter_file_path,'r') as f:
            self.logger.info(f"Loads soilparameters file: {soilparameter_file_path}")
            self.soilparameters = json.load(f)
        
        # load regionfile
        regionfile_path = os.path.join(rundir, self.regionfile)
        self.regions,_,_ = read_tif(regionfile_path, self.logger)
        
        self.slopes = []
        self.foss = []
        self.kys = []
        self.weights = []
        self.region_masks = []
        
        # Create event tree, one for each region
        for region, physical_parameters in enumerate(self.soilparameters):
            self.logger.info(f"Calculating fos and yield acceleration tables for region {region}")
            self.logger.info(f"parameters: {physical_parameters}")
            root_node = Node(weight=1, distributions=physical_parameters["distributions"], params=physical_parameters["constants"], parent=None)
            region_mask = np.logical_and(np.where(self.regions == region, True, False), self.slope_msk)
            
            
            # Traverse leaf nodes and create linear interpolations for slope variation.
            slopes = np.linspace(np.nanmin(self.slope_data[region_mask]), np.nanmax(self.slope_data[region_mask]), num=100)
            self.logger.info(f"Slope range for lookuptable (region {region}), {np.min(slopes)}, {np.max(slopes)}")
            kys, weights, foss = [], [], []
            sum_of_weights = 0.
            self.logger.info(f"sum_of_weights: {sum_of_weights}")
            
            for count, node in enumerate(root_node.leaf_nodes):
                self.logger.info(f"Leaf node:{count} \n Weight:{node.weight} \n Params:{node.params} \n")
                fos, ky = self.infinite_slope_analysis(slopes, **node.params)
                foss.append(fos)
                kys.append(ky)
                weights.append(node.weight)
                sum_of_weights += node.weight
            
            self.logger.info(f"Sum of weights: {sum_of_weights}")
            self.slopes.append(slopes)
            self.foss.append(np.stack(foss))
            self.kys.append(np.stack(kys))
            self.weights.append(np.array(weights))
            self.region_masks.append(region_mask)
        
    
    def compute_quantiles(self, quantiles, write_fos=False, write_ky=False):
        """
        Compute quantiles of the factor of safety and the yield acceleration.
        Output is written to the subdirectory [rundir]/slope_analysis.
        
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
        fos_quantiles = [np.quantile(self.foss[region], quantiles, axis=0, weights=self.weights[region], method='inverted_cdf') for region,_ in enumerate(self.soilparameters)]
        ky_quantiles = [np.quantile(self.kys[region], quantiles, axis=0, weights=self.weights[region], method='inverted_cdf') for region,_ in enumerate(self.soilparameters)]
        
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
                fos_quantile_raster = np.empty(self.slope_data.shape)
                fos_quantile_raster[~self.slope_msk] = np.nan
                for region,_ in enumerate(self.soilparameters):
                    fos_quantile_raster[self.region_masks[region]] =np.interp(self.slope_data[self.region_masks[region]], self.slopes[region], fos_quantiles[region][i,:])
                write_tif(fname = os.path.join(fos_quantile_dir, fos_quantiles_filename), data = fos_quantile_raster, profile = self.slope_profile, logger=self.logger)
                fos_output.append({"file": fos_quantiles_filename, "quantile": quantile, "value": "log factor_of_safety", "scale": "log10", "unit":""})
        
            if write_ky: 
                ky_quantile_raster = np.empty(self.slope_data.shape)
                ky_quantile_raster[~self.slope_msk] = np.nan                
                for region,_ in enumerate(self.soilparameters):
                    ky_quantile_raster[self.region_masks[region]] = np.interp(self.slope_data[self.region_masks[region]], self.slopes[region], ky_quantiles[region][i,:])
                write_tif(fname = os.path.join(ky_quantile_dir, ky_quantiles_filename), data = ky_quantile_raster, profile = self.slope_profile, logger=self.logger)
                ky_output.append({"file": ky_quantiles_filename, "quantile": quantile, "value": "log yield_acceleration", "scale":"log10", "unit": "g"})
        
        if write_fos: self.write_content(fos_output, fos_quantile_dir)
        if write_ky: self.write_content(ky_output, ky_quantile_dir)
    
    
    def compute_cummulative(self, thresholds, feature_name="logfos", write=False, output_dir=None):
        if feature_name == "logfos":
            feature = self.foss
            dir = self.fos_dir if output_dir is None else output_dir
        elif feature_name == "logky":
            feature = self.kys
            dir = self.yield_acceleration_dir if output_dir is None else output_dir
        else:
            raise ValueError("feature name must be either logfos of logky.")
        
        feature_cummulative = [cummulative(feature[region], thresholds, weights=self.weights[region], axis=0) for region,_ in enumerate(self.soilparameters)]
        
        if write:
            output = []
            create_dir(dir, self.logger) # dir
            cum_dir = os.path.join(dir, "cummulative")
            create_dir(cum_dir, self.logger)
        
            # Evaluate cummulative by interpolation and write to files.
            for i, threshold in enumerate(thresholds):
                cummulative_filename = f"{feature_name}_cum_{i}.tif"
                cummulative_raster = np.empty(self.slope_data.shape)
                cummulative_raster[~self.slope_msk] = np.nan
                for region,_ in enumerate(self.soilparameters):
                    cummulative_raster[self.region_masks[region]] = np.interp(self.slope_data[self.region_masks[region]], self.slopes[region], feature_cummulative[region][i,:])
                write_tif(fname = os.path.join(cum_dir, cummulative_filename), data = cummulative_raster, profile = self.slope_profile, logger=self.logger)
                output.append({"file": cummulative_filename, "threshold":threshold, "value": f"cummulative of {feature_name}", "scale": "", "unit": ""})
            
            self.write_content(output, cum_dir)
        return(feature_cummulative)
    
    """
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
    """
    
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
        
        gamma = (density - density_of_water)/density
        c = cohesion*1000. #in Pa
        u = excess_pore_pressure*1000. #in Pa
        mu = np.tan(np.radians(friction_angle))
        g = gravity
        rho = density
        H = thickness
        alpha = np.radians(slope)
        eps = 1e-6 # truncation threshold.
        fos_replacement_value = 1000
        
        #fos = np.where(np.sin(alpha) > eps, (c-u*mu)/(rho*H*gamma*g*np.sin(alpha)) + mu/np.tan(alpha), np.nan)
        
        #ky = np.where(np.sin(alpha) > eps, gamma*np.sin(alpha)*(fos-1.), np.nan) 
        #ky = gamma*np.sin(alpha)*(fos-1.)
        
        fos = (c-u*mu)/(rho*H*gamma*g*np.sin(alpha)) + mu/np.tan(alpha)
        fos_times_sin = (c-u*mu)/(rho*H*gamma*g) + mu*np.cos(alpha)
        ky = gamma*(fos_times_sin-np.sin(alpha))
        
        # Replace values where not stable.
        ky = np.where(fos<1, gamma*np.sin(alpha)*(fos_replacement_value-1.), gamma*(fos_times_sin-np.sin(alpha))) 
        fos = np.where(fos<1, fos_replacement_value, fos)
        
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

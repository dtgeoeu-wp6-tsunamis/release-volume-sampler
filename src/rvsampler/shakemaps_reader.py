import numpy as np
import os
import json
from scipy.interpolate import RegularGridInterpolator
import csv

from rvsampler.utils import create_dir, write_tif, cumulative, write_content
from rvsampler.set_logg import setup_logger

class ShakemapsReader:
    
    def __init__(self, shakemaps_filename, source_parameters_filename, rundir, thresholds=None, weights=None, aggregate=False):
        
        self.shakemaps_filename = shakemaps_filename
        self.source_parameters_filename = source_parameters_filename
        self.shakemaps_out_dir = os.path.join(rundir, "shakemaps")
        self.logger = setup_logger("shakemaps_reader", self.shakemaps_out_dir)
        create_dir(self.shakemaps_out_dir)
        with open(self.shakemaps_filename, 'r') as f:
            self.shakemaps = json.load(f)
        self.compute_logpga()
        self.weights = weights
        self.aggregate = False
        self.nr_of_maps = len(self.shakemaps[0]["Z_pga"])
        
        
        if aggregate:
            self.aggregate = True
            assert thresholds is not None, "Thresholds must be provided for cumulative shakemaps."
            self.thresholds = thresholds
            self.weights = weights
            self.compute_cumulative()
       
        self.logger.info(self.shakemaps[0].keys()) 
        self.logger.info(f"Number of lon-lat points in shakemaps: {len(self.shakemaps)}")
        self.logger.info(f"Number of shakemaps {self.nr_of_maps}")
        # Use with old messina data, but not really neccessary 
        #self.source_parameters = []
        #with open(self.source_parameters_filename, newline='') as csvfile:
        #    reader = csv.DictReader(csvfile)
        #    for row in reader:
        #        self.source_parameters.append(row)
        if self.weights is None:
            self.weights = np.ones(len(self.source_parameters))/len(self.nr_of_maps)
    
    @staticmethod
    def logpga(point):
        # Function to extract relevant value from shakemap.
        Z, H = [10**np.array(point[shake_param]) for shake_param in ["Z_pga", "H_pga"]]
        return np.log10(np.sqrt(Z**2 + H**2))
    
    def write_shakemaps_to_rasters(self, profile, bounds, interpolation_method):
        
        if self.aggregate:
            shakemap_content = []
            for i, threshold in enumerate(self.thresholds):
                cumulative = self.interpolate_shakemap(
                                                value="cumulative", 
                                                sample=i, 
                                                bbox=bounds, 
                                                n_cols=profile["width"], 
                                                n_rows=profile["height"],
                                                interpolation_method=interpolation_method)
                filename = f"pga_cum_{i}.tif"
                shakemap_content.append({
                    "file": filename,
                    "threshold":threshold,
                    "value":"Probability",
                    "scale":"log10",
                    "unit":"g"
                })
                write_tif(os.path.join(self.shakemaps_out_dir, filename), cumulative, profile, self.logger)
        else:
            shakemap_content = []
            for i in range(self.nr_of_maps):
                logpga = self.interpolate_shakemap(
                                                value="logpga", 
                                                sample=i, 
                                                bbox=bounds, 
                                                n_cols=profile["width"], 
                                                n_rows=profile["height"],
                                                interpolation_method=interpolation_method)
                filename = f"pga_{i}.tif"
                shakemap_content.append({
                    "file": filename,
                    "value": "logpga",
                    "scale": "log10",
                    "unit": "g"
                })
                write_tif(os.path.join(self.shakemaps_out_dir, filename), logpga, profile, self.logger)
        write_content(shakemap_content, self.shakemaps_out_dir)
    
    def compute_logpga(self):
        # Apply the shake value function to each point in the shakemaps
        self.logger.info(f"Computing shake value of shakemaps")
        for point in self.shakemaps:
            point["logpga"] = ShakemapsReader.logpga(point)
    
    def compute_cumulative(self):
        # Compute cumulative probability of given value over entire shakemap grid. Save in same structure as input file.
        # Load shakemaps from file 
        self.logger.info("Computing cumulative shakemaps")
        for point in self.shakemaps:
            point["cumulative"] =  cumulative(point["logpga"], self.thresholds, self.weights)
        
        # This is for the old shakemap file
        #for _,point in shakemaps.items():
        #    shakemaps_cumulative.append(
        #        {
        #            "lon": point["lon"],
        #            "lat": point["lat"],
        #            "cumulative": cumulative(shake_value(point), thresholds, self.weights)
        #        }
        #    )
    
    def interpolate_shakemap(self, value, sample, bbox, n_rows, n_cols, interpolation_method):
        # Create grid interpolator
        self.logger.info(f"Interpolating shakemap - sample:{sample}, value: {value}, method: {interpolation_method}")
        lon_shake, lat_shake, data = zip(*[(point['lon'], point['lat'], point[value][sample]) for point in self.shakemaps])
        grid_shake = np.array(list(set(lon_shake))), np.array(list(set(lat_shake))) # Extract unique values 
        [g.sort() for g in grid_shake] # in ascending order.
        grid_shake_shape = grid_shake[0].shape[0], grid_shake[1].shape[0]
        data = np.reshape(data, grid_shake_shape, order='C') # Reshape data. 
        # Lon fixed, Lat change -> C order. Location of data(ij): lowerleft + (j*delta_lat, i*delta_lon) 
        shake_interp = RegularGridInterpolator(points=grid_shake, values=np.flip(data.T,-1), method=interpolation_method) # Data need to be ij matrix format (See np.meshgrid)
        
        # Interpolate grid.
        grid_int = np.meshgrid(np.linspace(bbox.left, bbox.right, num=n_cols), np.linspace(bbox.bottom, bbox.top, num=n_rows), indexing='ij')
        shake_values = shake_interp(grid_int)
        return(np.flip(shake_values,-1).T)
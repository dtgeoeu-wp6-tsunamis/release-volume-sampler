import numpy as np
import os
import json
from scipy.interpolate import RegularGridInterpolator
import csv

from utils import create_dir, read_tif, write_tif, cummulative, write_content
from logging import setup_logger

class ShakemapsAggregator:
    
    def __init__(self, shakemaps_filename, source_parameters_filename, thresholds, rundir, shake_value):
        
        # Parameters
        self.shakemaps_filename = shakemaps_filename
        self.source_parameters_filename = source_parameters_filename
        self.thresholds = thresholds
        self.shakemaps_out_dir = os.path.join(rundir, "shakemaps")
        self.logger = setup_logger("shakemaps_aggregator", self.shakemaps_out_dir)
        self.shake_value = shake_value
         
        # Load source parameters
        self.source_parameters = []
        with open(self.source_parameters_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.source_parameters.append(row)
        
        self.weights = np.ones(len(self.source_parameters))/len(self.source_parameters)
        
        # Compute cummulative probabilities.
        self.shakemap_cummulative = self.compute_cummulative()


    def write_cummulative(self, profile, bounds, interpolation_method):
        create_dir(self.shakemaps_out_dir)
        
        shakemap_content = []
        for i, threshold in enumerate(self.thresholds):
            logpga = self.interpolate_shakemap(shakemaps=self.shakemap_cummulative,
                                               shake_value="cummulative", 
                                               shake_sample=i, 
                                               bbox=bounds, 
                                               n_cols=profile["width"], 
                                               n_rows=profile["height"],
                                               interpolation_method=interpolation_method)
            filename = "pga_cum_{}.tif".format(i)
            shakemap_content.append({
                "file": filename,
                "threshold":threshold,
                "value":"Probability",
                "scale":"log10",
                "unit":"g"
            })
            write_tif(os.path.join(self.shakemaps_out_dir, filename), logpga, profile, self.logger)
        write_content(shakemap_content, self.shakemaps_out_dir)
    
    
    def compute_cummulative(self):
        # Compute cumulative probability of given value over entire shakemap grid. Save in same structure as input file.
        # Load shakemaps from file 
        shakemaps_cummulative =  []
        with open(self.shakemaps_filename, 'r') as f:
            shakemaps = json.load(f)
        
        for _,point in shakemaps.items():
            shakemaps_cummulative.append(
                {
                    "lon": point["lon"],
                    "lat": point["lat"],
                    "cummulative": cummulative(self.shake_value["function"](point), self.thresholds, self.weights)
                }
            )
        return(shakemaps_cummulative)

    
    def interpolate_shakemap(self, shakemaps, shake_value, shake_sample, bbox, n_rows, n_cols, interpolation_method):
        # Create grid interpolator
        self.logger.info(f"Interpolating shakemap - sample:{shake_sample}, value: {shake_value}, method: {interpolation_method}")
        lon_shake, lat_shake, data = zip(*[(point['lon'], point['lat'], point[shake_value][shake_sample]) for point in shakemaps])
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
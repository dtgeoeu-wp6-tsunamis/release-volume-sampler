import numpy as np
import os
import logging
import json

from utils.utils import create_dir, read_tif, write_tif, write_content

""" Cacluation of displacements probabilities from yield acceleration and shakemaps.
"""

#logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('displacements')

class DsiplacementProbabilityAggregator:
    
    def __init__(self, rundir, displacement_thresholds):
        self.rundir = rundir
        self.displacement_thresholds = displacement_thresholds
        
        self.ky_dir = os.path.join(rundir, "yield_acceleration/cummulative")
        self.pga_dir = os.path.join(rundir, "shakemaps")


    def compute_probabilities(self):
        logger.info("Calculating displacement probabilities.")
        # Create output dir
        output_dir = os.path.join(self.rundir, "displacements")
        create_dir(output_dir)
        
        # Load and compute probability densities.
        cummulative_ky, ky_thresholds, profile = self.load_cummulative(self.ky_dir)
        cummulative_pga, pga_thresholds, profile = self.load_cummulative(self.pga_dir)
        ky_density = np.diff(cummulative_ky, axis=0)
        pga_density = np.diff(cummulative_pga, axis=0)
        
        M = 7 # Need to embed also magnitude as a distribution in the grid..

        # Compute displacement at grid centers
        ky_centers = 0.5*(ky_thresholds[1:] + ky_thresholds[:-1])
        pga_centers = 0.5*(pga_thresholds[1:] + pga_thresholds[:-1])
        kys, pgas = np.meshgrid(ky_centers, pga_centers)
        log_d, log_sigma = self.displacement(kys, pgas, M)

        # Compute probabilities and write to files
        content = []
        for i,delta in enumerate(self.displacement_thresholds):
            logger.info(f"Delta: {delta}")
            d_is_bigger = log_d > np.log10(delta)
            probs = np.sum((pga_density[:,np.newaxis]*ky_density[np.newaxis,:])[d_is_bigger,:], axis=0)
            
            # Non vectorized implementation (slow).
            # probs = np.zeros(pga_density.shape[1:])
            # for i in range(pga_density.shape[1]):
            #     for j in range(pga_density.shape[2]):
            #         probs[i,j] = np.sum(np.outer(pga_density[:,i,j], ky_density[:,i,j])[d_is_bigger])
            
            # Write to file
            filename = f"exceedance_prob_{i}.tif"
            content.append({"file": filename, "threshold": delta, "value": "exceedance prob.", "unit":"", "scale":""})
            write_tif(os.path.join(output_dir, filename), probs, profile)
        write_content(content, output_dir)


    @staticmethod
    def load_cummulative(dir):
        # load json file
        with open(os.path.join(dir, "content.json"),'r') as f:
            content = json.load(f)

        # Initialize lists to store raster data and no-data values
        rasters = []
        thresholds = []
        #nodata_vals = []

        # Read all rasters and store the data
        for e in content:
            thresholds.append(e["threshold"])
            raster_path = os.path.join(dir, e["file"])
            raster_data, msk, profile = read_tif(raster_path)
            rasters.append(raster_data)
        return(np.stack(rasters), np.array(thresholds), profile)
    
    
    @staticmethod
    def displacement(logky, logpga, M=None, logpgv=None, model="scalar"):
        """ Calculation of ground displacements
        Implementation of the statistical model for ground displacements of natural slopes subject to earthquakes 
        given in [1]. Note: The statistical model is develpped for subaerial conditions.
        
        Parameters:
            logky: float or ndarray 
                Log (Base 10) of Yield acceleration calculated using infinite slope analysis [g].
            logpga: float or ndarray. Same dimension as ky.
                Log (Base 10) of Peak ground acceleration of the event [g]. 
            M: float
                moment magnitude of the event.
            logpgv: float or ndarray. Same dimension as ky.
                Log (Base 10) of Peak ground velocity of the event [cm/s].
            model: string
                Different models. Options are "scalar" or "vector". If "scalar", then M must be supplied. 
                If "vector", then pgv must be supplied. Default is "scalar".
            
        Returns:
            Log of displacements [cm], Log of standard_deviation: float or ndarray, float or ndarray
            Logarithm of estimated displacements and associated standard deviation. 
            Standard deviation of lognormal multiplicative noise (as a function of ky/pga).  
            
        1. Rathje and Saygili, ‘Probabilistic Assessment of Earthquake-Induced Sliding Displacements of Natural Slopes’.
        """
        ky_pga_ratio = 10**(logky - logpga)
        logscale_factor = np.log10(np.exp(1))
        lnpga = logpga/logscale_factor
        
        if model == "scalar":
            a = [-29.06, 42.49, - 19.64,-4.85, 4.89] # polynomial coefficients.
            ln_d = np.where(ky_pga_ratio < 1., np.polyval(a, ky_pga_ratio) + 0.72*lnpga + 0.89*(M-6), 0.)
            sigma_ln = np.where(ky_pga_ratio < 1., np.polyval([-0.539, 0.789, 0.732], ky_pga_ratio), 0.)
        elif model == "vector":
            lnpgv = logpgv/np.log10(np.exp(1))
            a = [-30.5, 44.75, -20.84, -4.58, -1.56]
            ln_d = np.where(ky_pga_ratio < 1., np.polyval(a, ky_pga_ratio) - 0.64*lnpga + 1.55*lnpgv, 0)
            sigma_ln = np.where(ky_pga_ratio, 0.405 + 0.524*(ky_pga_ratio), 0.)
        return(ln_d*logscale_factor, sigma_ln*logscale_factor)

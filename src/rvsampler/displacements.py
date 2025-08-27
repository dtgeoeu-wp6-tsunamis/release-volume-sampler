import numpy as np
import os
import json
import rasterio
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import ndimage

from concurrent.futures import ThreadPoolExecutor
from rvsampler.utils import create_dir, read_tif, write_tif, write_content
from rvsampler.triangulate import Triangulation 
from rvsampler.set_logg import setup_logger, close_logger

""" Cacluation of displacements probabilities from yield acceleration and shakemaps.
"""

def displacement(logky, logpga, M=None, logpgv=None, model="scalar"):
    """ Calculation of ground displacements
    Implementation of the statistical model for ground displacements of natural slopes subject to 
    earthquakes given in [1]. Note: The statistical model is develpped for subaerial conditions.
    
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
        
    1. Rathje and Saygili, ‘Probabilistic Assessment of Earthquake-Induced Sliding Displacements of 
    Natural Slopes’.
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


class DisplacementProbabilityAggregator:
    
    def __init__(self, rundir, magnitude):
        self.rundir = rundir
        self.ky_dir = os.path.join(rundir, "slope_analysis", "yield_acceleration", "cumulative")
        self.pga_dir = os.path.join(rundir, "shakemaps")
        
        # Used for calculation of displacement. Ideally be embedded as a distribution, but not very 
        # sensitive.
        self.magnitude = magnitude 

        self.output_dir = os.path.join(self.rundir, "displacements")
        create_dir(self.output_dir)
        self.logger = setup_logger("displacements", self.output_dir)
        
        # May be convenient to compute displacements.. 
        #self.source_parameters = []
        #if self.source_parameters_filename is not None:
        #    self.logger.info(f"Load source parameter file: {self.source_parameters_filename}")
        #    with open(self.source_parameters_filename, newline='') as csvfile:
        #        reader = csv.DictReader(csvfile)
        #        for row in reader:
        #            self.source_parameters.append(row)

    def compute_aggregated_probabilities(self, displacement_threshold, create_lookup_table=True):
        """
        Computation of exceedance probability P(displacement > delta) given the distribution
        of PGA values represented by the cumulative in the shakemaps dir.
        """
        # Create output dir
        self.logger.info("Calculating displacement probabilities for aggregated shakemaps.")
        cumulative_out_dir = os.path.join(self.output_dir, "cumulative")
        create_dir(cumulative_out_dir, self.logger, clear=True)

        # Load and compute probability densities.
        self.logger.info(f"Loads cumulative probabilities: {self.ky_dir}")
        cumulative_ky, ky_thresholds, profile = self.load_cumulative(self.ky_dir)
        self.logger.info(f"Loads cumulative probabilities: {self.pga_dir}")
        cumulative_pga, pga_thresholds, profile = self.load_cumulative(self.pga_dir)
        ky_density = np.diff(cumulative_ky, axis=0)
        pga_density = np.diff(cumulative_pga, axis=0)
        
        # Compute displacement at grid centers
        ky_centers = 0.5*(ky_thresholds[1:] + ky_thresholds[:-1])
        pga_centers = 0.5*(pga_thresholds[1:] + pga_thresholds[:-1])
        kys, pgas = np.meshgrid(ky_centers, pga_centers)
        log_d, log_sigma = displacement(kys, pgas, self.magnitude)

        # Compute probabilities and write to files
        content = []
        d_is_bigger = log_d > np.log10(displacement_threshold)
        probs = np.sum((pga_density[:,np.newaxis]*ky_density[np.newaxis,:])[d_is_bigger,:], axis=0)
        
        # Non vectorized implementation (slow).
        # probs = np.zeros(pga_density.shape[1:])
        # for i in range(pga_density.shape[1]):
        #     for j in range(pga_density.shape[2]):
        #         probs[i,j] = np.sum(np.outer(pga_density[:,i,j], ky_density[:,i,j])[d_is_bigger])
        
        # Write to file
        filename = f"exceedance_prob.tif"
        content.append({"file": filename, "threshold": displacement_threshold, "value": "exceedance prob.", 
                        "unit":"", "scale":""})
        write_tif(os.path.join(cumulative_out_dir, filename), probs, profile, self.logger)
        write_content(content, cumulative_out_dir)
        if create_lookup_table:
            triangulation = Triangulation(self.rundir)
            triangulation.create_lookuptable(cumulative_out_dir, outfile_name="exceedance_displacement.npz")

    def compute_probabilities_by_sample(self, displacement_threshold, nr_of_pga_thresholds=100, create_lookup_table=True):
        """
        Estimation of conditional probabilities P(displacement > delta | PGA). This is done by 
        calculting displacement on a grid of threshold values for yield acceleration (k_y) and PGA 
        values. The marginal probability associated with each fixed pga value is calculated applying
        the cumulative distribution of k_y. This is applied as a lookuptable by binning each pga 
        values of a given shakemap.
        
        Parameters:
            nr_of_pga_thresholds: Number of thresholds used for binning of pga values.
            displacement_threshold: Threshold for exceedance probability.
            
        Note: Uncertainty in displacement is currently not taken into account.
        """
        self.logger.info("Calculating displacement probabilities by sample.")
        
        if create_lookup_table: 
            triangulation = Triangulation(self.rundir)

        # Load and compute probability densities.
        self.logger.info(f"Loads cumulative probabilities: {self.ky_dir}")
        cumulative_ky, ky_thresholds, profile = self.load_cumulative(self.ky_dir)
        self.logger.info(f"Load pga samples: {self.pga_dir}")
        pga_rasters, pga_thresholds, profile, sample_numbers = self.load_samples(
            self.pga_dir, 
            nr_of_thresholds=nr_of_pga_thresholds
        )
        ky_density = np.diff(cumulative_ky, axis=0)
        
        # Compute displacement at grid centers
        ky_centers = 0.5*(ky_thresholds[1:] + ky_thresholds[:-1])
        pga_centers = 0.5*(pga_thresholds[1:] + pga_thresholds[:-1])
        kys, pgas = np.meshgrid(ky_centers, pga_centers)
        log_d, log_sigma = displacement(kys, pgas, self.magnitude) 
        
        d_is_bigger = log_d > np.log10(displacement_threshold)
        probs_by_threshold = np.tensordot(ky_density, d_is_bigger, axes=[0, 1])  # shape: (n_pga_bins, y, x)
        
        # Compute probabilities and write to files
        for j, pga_raster in enumerate(pga_rasters):
            sample_out_dir = os.path.join(self.output_dir, f"sample_{sample_numbers[j]}")
            create_dir(sample_out_dir, self.logger, clear=True)
            
            indices = np.searchsorted(pga_thresholds, pga_raster, side="right") - 1
            indices = np.clip(indices, 0, probs_by_threshold.shape[2] - 1)
            probs = np.take_along_axis(probs_by_threshold, indices[..., None], axis=2)[..., 0]
            filename = f"exceedance_prob.tif"
            content = [{"file": filename, "threshold": displacement_threshold, "sample": sample_numbers[j],
                        "value": "exceedance prob.", "unit": "", "scale": ""}]
            write_tif(os.path.join(sample_out_dir, filename), probs, profile, self.logger)
            write_content(content, sample_out_dir)
            if create_lookup_table:
                triangulation.create_lookuptable(sample_out_dir, outfile_name="exceedance_displacement.npz")
        
            
    def load_cumulative(self, dir):
        # load json file
        with open(os.path.join(dir, "content.json"),'r') as f:
            content = json.load(f)

        # Initialize lists to store raster data.
        rasters = []
        thresholds = []

        # Read all rasters and store the data
        for e in content:
            thresholds.append(e["threshold"])
            raster_path = os.path.join(dir, e["file"])
            raster_data, msk, profile = read_tif(raster_path, self.logger)
            rasters.append(raster_data)
        return(np.stack(rasters), np.array(thresholds), profile)
    
    def load_samples(self, dir, nr_of_thresholds=500):
        # load json file
        with open(os.path.join(dir, "content.json"), 'r') as f:
            pga_content = json.load(f)
        
        # initialize list to store values
        samples = []        
        sample_numbers = []
        
        # Read all rasters
        for e in pga_content:
            raster_path = os.path.join(dir, e["file"])
            sample_numbers.append(e["sample"])
            raster_data, msk, profile = read_tif(raster_path, None)
            samples.append(raster_data)
        samples = np.stack(samples)
        
        # Create thresholds
        thresholds = np.linspace(start=samples.min(), stop=samples.max(), num=nr_of_thresholds)
        
        return(samples, thresholds, profile, sample_numbers)

    def completed(self):
        self.logger.info("Displacement probability processing completed.")
        with open(os.path.join(self.output_dir, "completed"), "w") as f:
            pass
        close_logger(self.logger)
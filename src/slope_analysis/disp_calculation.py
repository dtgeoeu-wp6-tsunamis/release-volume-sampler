#This code computes the seismic displacements at grid cell level

#by Laide Ojomo, 2023, UT Austin

import os  # Provides functions to interact with the operating system, like handling file paths.
import rasterio as rio  # Used for reading and writing raster data (like .tif files), and handling geospatial data.
import numpy as np  # Used for numerical operations, especially on arrays and matrices.
from logging import disable

def read_tif(fname, dtype):
    """
    Read .tif data and profile using rasterio.
    
    Parameters:
    fname (str): File name of the .tif file.
    dtype: Data type to which the array is cast.
    
    Returns:
    tuple: Tuple containing the data and profile from the .tif file.
    """
    with rio.open(fname) as src:
        data = src.read(1).astype(dtype)
        profile = src.profile.copy()
    return data, profile

def read_tif2(fname, dtype):
    "Read .tif data and profile all bands using rasterio ."
    with rio.open(fname) as src:
        data = src.read().astype(dtype)
        profile = src.profile.copy()
    return data, profile

def output_file(fname, data, profile):
    """
    Write data to a .tiff file using rasterio.
    
    Parameters:
    fname (str): Output file name.
    data: Data to be written.
    profile: Profile information for the output file.
    """
    with rio.open(fname, 'w', **profile, encoding='utf-8') as dst:
        dst.write(data)

def calc_disp(ky, pga, pgv):
    """
    Perform calculations to generate displacement data based on input parameters.
    
    Parameters:
    ky_7: Array of ky values.
    pga: Peak Ground Acceleration values.
    pgv: Peak Ground Velocity values.
    
    Returns:
    tuple: Tuple containing displacement data and associated weights.
    """
    w_ky = np.array([0.054866, 0.135893, 0.198097, 0.222288, 0.198097, 0.135893, 0.054866], dtype=np.single) # M & R(1983) 7 prob points
    w_d = np.array([0.3, 0.3, 0.4], dtype=np.single) #displacement model weights
    a1 = 4.89; a2 = -4.85; a3 = -19.64; a4 = 42.49;  a5 = -29.06; a6 = 0.72; a7 = 0.89
    b1 = -2.710; b2 = 2.335; b3 = -1.478
    c1 = -1.56; c2 = -4.58; c3 = -20.84; c4 = 44.75;  c5 = -30.50; c6 = -0.64; c7 = 1.55
    mag = 8.019897 # obtained from the metadata for scenario 1

    Disp = np.empty((7, 3), dtype=np.single)
    sigma = np.empty((7, 3), dtype=np.single)
    w_total = np.empty((7, 3), dtype=np.single)
    final_disp = np.zeros((21, 5), dtype=np.single)
    final_wt = np.zeros((21, 5), dtype=np.single)
    binned_disp = np.zeros((6), dtype=np.single)

    for a,n in np.ndindex((w_ky.shape[0],w_d.shape[0])):
        x = ky[a]/pga
        w_total[a, n] = w_ky[a]*w_d[n]

        if x < 1:
            if n == 0: # displacement model RS-PGA,M
                Disp[a, n] = max(np.exp(a1 + a2*x + a3*x**2 + a4*x**3 + a5*x**4 + a6*np.log(pga) + a7*(mag-6)), 0.1)
                sigma[a, n] = 0.732 + 0.789*x -0.539*(x**2) 
            if n == 1: # displacement model J-PGA,M
                Disp[a, n] = max(np.exp(np.log(10**(b1 + (np.log10(((1-x)**b2) * (x**b3))) + 0.424*mag))), 0.1)
                sigma[a, n] = np.log(10**0.454)
            if n == 2: # displacement model RS-PGA,PGV
                Disp[a, n] = max(np.exp(c1 + c2*x + c3*x**2 + c4*x**3 + c5*x**4 + c6*np.log(pga) + c7*np.log(pgv)), 0.1)
                sigma[a, n] = 0.405 + 0.524*x
        else: 
            Disp[a, n] = 0.1
            sigma[a, n] = 0.01
            
    Disp = Disp.reshape((21));  sigma = sigma.reshape((21)); w_total = w_total.reshape((21))
    
    cdf_approx5 = [0.034893, 0.211702, 0.50000, 0.788298, 0.965107] # Miller and Rice (1983) 5 CDF points
    wt_5 = np.array([0.101080, 0.244290, 0.309260, 0.244290, 0.101080], dtype='float32') # M & R(1983) 5 prob points
    eps5 = np.array([-1.81329717, -0.80052964, 0,  0.80052964,  1.81329717]) #z-score of probabilities
    
    for dp,ep in np.ndindex((Disp.shape[0],eps5.shape[0])):
        final_disp[dp,ep] = max(np.exp(np.log(Disp[dp])+(sigma[dp]*eps5[ep])),0.1) # exp(ln(disp)+sigma*epsilon)
        final_wt[dp,ep] = w_total[dp]*wt_5[ep]
      
    final_disp = final_disp.reshape((105))    
    D_test = final_disp.reshape((105));  D_w_test = final_wt.reshape((105))
         
    D_w_combo = np.array(list(zip(D_test,D_w_test))) # combine disp and corresponding weights
    D_w_sorted = sorted(D_w_combo, key = lambda x:x[0])     # sort the data in ascending order  
    D_test,D_w_test = zip(*D_w_sorted)         # combine sorted data
    cdf_disp = np.cumsum(D_w_test)
    
    cdf_50th = [0.5]
    data_disp = np.interp(cdf_50th, cdf_disp, D_test)
    
    bins = [0, 5, 30, 10000]
    heights,bins = np.histogram(final_disp,bins=bins)
    heights = heights/sum(heights) 
    
    p = 0.4
    p_disp = np.array([0.14718, 0.10407198, 0.08061381, 0.0681311])
    bin3 = heights[2]
    bin6 = (bin3/p*p_disp).round(3)
    del_bin = np.delete(heights, 2)
    binned_disp = np.concatenate((del_bin, bin6))

    return data_disp, binned_disp

data_geol_id, geol_profile = read_tif("loc1_geol.tif", int)
data_geol_id = np.array(data_geol_id, str)
data_slope, _ = read_tif("loc1_slope.tif", np.single)
data_ky, _ = read_tif2("loc1_ky7.tif", np.single)
data_ky =  np.moveaxis(data_ky, 0, 2)
data_pga, _ = read_tif("loc1_pga.tif", np.single)
data_pgv, _ = read_tif("loc1_pgv.tif", np.single)

# Flatten to simplify iteration
data_geol_id_flat = data_geol_id.flatten()
data_slope_flat = data_slope.flatten()
data_ky_flat = data_ky.reshape(-1, data_ky.shape[-1])
data_pga_flat = data_pga.flatten()
data_pgv_flat = data_pgv.flatten()

ntasks = len(data_geol_id_flat)
print(f"ntasks: {ntasks}")

# Allocated output array.
data_disp = np.empty((data_geol_id.size), dtype=np.single)
binned_disp = np.empty((data_geol_id.size, 6), dtype=np.single)
_nan1 = np.full(1, np.nan, dtype=np.single)
_nan6 = np.full(6, np.nan, dtype=np.single)


for idx in range(ntasks):
    if (data_slope_flat[idx] > 10) and (data_geol_id_flat[idx] in [str(x) for x in [25, 27, 29, 31, 32, 33, 30, 22, 21, 20, 8, 10, 12, 14, 15, 16, 5, 3]]):
        data_disp[idx], binned_disp[idx] = calc_disp(data_ky_flat[idx], data_pga_flat[idx], data_pgv_flat[idx])
    else:
        data_disp[idx], binned_disp[idx] = _nan1, _nan6

data_disp = data_disp.reshape((data_geol_id.shape))
binned_disp = binned_disp.reshape((*data_geol_id.shape, 6))

geol_profile_6 = dict(geol_profile)
geol_profile_6["count"] = 6

new_binned_disp = np.array([binned_disp[:, :, idx] for idx in range(binned_disp.shape[-1])])

output_file('loc1_disp.tif', data_disp.reshape(1, *data_disp.shape), geol_profile) 
output_file('loc1_binned_disp.tif', new_binned_disp, geol_profile_6)

# Reclassify rasters
dispgt5 = data_disp.copy()
dispgt5[np.where(dispgt5 < 5)] = 1 # no landslide
dispgt5[np.where(dispgt5 >= 5)] = 2 # landslide
output_file('loc1_dispgt5.tif', dispgt5.reshape(1, *data_disp.shape), geol_profile) 

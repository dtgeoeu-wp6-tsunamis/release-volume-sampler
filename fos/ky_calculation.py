#This code computes full distribution of yield acceleration, ky at grid cell 
#level, and resamples to 7 CDF points using Miller and Rice (1983)

#by Laide Ojomo, 2023, UT Austin

from logging import disable
import rasterio as rio
import numpy as np
from numba import njit

def read_tif(fname, dtype):
    "Read .tif data and profile using rasterio."
    with rio.open(fname) as src:
        data = src.read(1).astype(dtype)
        profile = src.profile.copy()
    return data, profile

def output_file(fname, data, profile):
    "Write .tiff data and profile using rasterio."
    with rio.open(fname, 'w', **profile, encoding='utf-8') as dst:
        dst.write(data)

def reader(fname):
    "Read .csv to dict."
    with open(fname, "r") as f:
        lines = f.readlines()
    _dict = {}
    _, *keys = lines[0].split(",")
    for line in lines[1:]:
        line = line
        row_key, *values = line.split(",")
        _dict[row_key.strip()] = {key.strip():float(value.strip()) for key, value in zip(keys, values) if key not in ["geology"]}
    return _dict

def read_strength_data(fname):
    "Read strength data from .csv"
    return reader(fname)

def read_weights(fname):
    "Read weight from .csv."
    return reader(fname)

@njit() #speed up python with numba

def task_b(slope, eps_phi, eps_c, data_c, gam_s, data_t, data_phi, data_m, gam_w, w_t, w_m, w_phi, w_c):
    # Compute fs and ky                 
    ky = np.empty((3, 3, 3, 3), dtype=np.single)
    w_init = np.empty((3, 3, 3, 3), dtype=np.single)
    
    slope_rad = np.radians(slope)
    for i, j, k, l in np.ndindex((eps_phi.shape[0], eps_c.shape[0], w_t.shape[0], w_m.shape[0])):
        fs = (np.exp(data_c[j+3*i])/(gam_s*data_t[k]*np.sin(slope_rad))) + (np.tan(np.radians(data_phi[i]))/np.tan(slope_rad)*(1-data_m[l]*(gam_w/gam_s)))
        fs = max(1.1, fs)
        ky[i,j,k,l] = (fs - 1)/((np.cos(slope_rad)*np.tan(np.radians(data_phi[i])))+(1/np.tan(slope_rad)))
        w_init[i,j,k,l] = w_phi[i]*w_c[j]*w_t[k]*w_m[l]
    
    return ky, w_init

def calc_ky(geology, slope, strength_data, w_t, w_m, w_phi, w_c):   
    # rho to assume correl or none btw c & phi
    eps_phi = np.array([-1.4, 0, 1.4], dtype=np.single)
    rho = -0.5
    eps_c = np.array([-1.4, 0, 1.4], dtype=np.single)

    _strength_data = strength_data[geology]
    c = _strength_data['lnc']
    sig_c = _strength_data['sigma_lnc']
    phi = _strength_data['frict_angle']
    sig_phi = _strength_data['sigma_phi']
    tl = _strength_data['thick_low']
    tb = _strength_data['thick_best']
    th = _strength_data['thick_high']
    gl = _strength_data['gwt_low']
    gb = _strength_data['gwt_best']
    gh = _strength_data['gwt_high']
    gam_w = _strength_data['gam_w']
    gam_s = _strength_data['gam_s']

    data_t = np.array((tl, tb, th), dtype=np.single)
    data_m = np.array((1-(gl/tl), 1-(gb/tb), 1-(gh/th)), dtype=np.single)
    data_m[data_m<0] = 0
    data_m[data_m>1] = 1

    data_c = np.zeros((9), dtype=np.single)
    data_phi = np.zeros((3), dtype=np.single)

    cnt = 0
    for i, _eps_phi in enumerate(eps_phi):
        data_phi[i] = phi + _eps_phi*sig_phi
        for _eps_c in eps_c:
            data_c[cnt] = (c + (rho*_eps_phi*sig_c)) + _eps_c*sig_c*np.sqrt(1-(rho**2))
            cnt += 1

    ky, w_init = task_b(slope, eps_phi, eps_c, data_c, gam_s, data_t, data_phi, data_m, 
                                       gam_w, w_t, w_m, w_phi, w_c)

    # Compute CDF of ky
    ky_test = ky.reshape((81));  ky_w_test = w_init.reshape((81))

    ## Note: Change sort, should be faster.
    sort_ids = np.argsort(ky_test, kind="stable")
    ky_test = ky_test[sort_ids]
    ky_w_test = ky_w_test[sort_ids]
    cdf = np.cumsum(ky_w_test)
    
    cdf_approx7 = [0.019106, 0.115498, 0.285336, 0.50000, 0.714664, 0.884502, 0.980894] # Miller and Rice (1983) 7 CDF points
    _ky_7 = np.interp(cdf_approx7, cdf, ky_test)
    return _ky_7

if __name__ == "__main__":
    # Load inputs
    data_geol_id, geol_profile = read_tif("loc1_geol.tif", int)
    data_geol_id = np.array(data_geol_id, str)
    data_slope, _ = read_tif("loc1_slope.tif", np.single)
    data_strength = read_strength_data("strength_data.csv")
    data_weight = read_weights("weights.csv")

    w_t = np.array([data_weight["w_t"][key] for key in ["low", "best", "high"]])
    w_m = np.array([data_weight["w_m"][key] for key in ["low", "best", "high"]])
    w_phi = np.array([data_weight["w_phi"][key] for key in ["low", "best", "high"]])
    w_c = np.array([data_weight["w_c"][key] for key in ["low", "best", "high"]])
    
    # Flatten to simplify iteration
    data_geol_id_flat = data_geol_id.flatten()
    data_slope_flat = data_slope.flatten()

    # Allocated output array.
    ky_7 = np.empty((data_geol_id.size, 7), dtype=np.single)
    _nan7 = np.full(7, np.nan, dtype=np.single)

    ntasks = len(data_geol_id_flat)
    print(f"ntasks: {ntasks}")
    for idx in range(ntasks):
        if (data_slope_flat[idx] > 10) and (data_geol_id_flat[idx] in [str(x) for x in [25, 27, 29, 31, 32, 33, 30, 22, 21, 20, 8, 10, 12, 14, 15, 16, 5, 3]]):
            ky_7[idx] = calc_ky(data_geol_id_flat[idx], data_slope_flat[idx], data_strength, w_t, w_m, w_phi, w_c)
        else:
            ky_7[idx] = _nan7

    ky_7 = ky_7.reshape((*data_geol_id.shape, 7))
    
    geol_profile_7 = dict(geol_profile)
    geol_profile_7["count"] = 7

    new_ky_7 = np.array([ky_7[:, :, idx] for idx in range(ky_7.shape[-1])])

    np.save("loc1_ky7.npy", ky_7, allow_pickle=True)
    output_file('loc1_ky7.tif', new_ky_7, geol_profile_7)
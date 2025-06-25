import os
import numpy as np
import pandas as pd
import rasterio
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

from rvsampler.displacements import DisplacementProbabilityAggregator
from rvsampler.shakemaps_reader import ShakemapsReader
from rvsampler.utils import create_dir
from rvsampler.cumprobs_by_triangle import caclulate_cumulative_probabilities
from rvsampler.database_handler import VolumeDatabaseHandler

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize



def main():
    """This script assigns release probabilities to the release volumes in the database.
    """
    # Rundir tas inn som input
    parser = argparse.ArgumentParser(description="Release volume sampler")
    parser.add_argument('--rundir', required=True, help='Path to the run directory')
    parser.add_argument('--rootdir', required=True, help='Path to the rvsampler root directory')
    args = parser.parse_args()
    rundir = args.rundir
    rootdir = args.rootdir
    
    shakemaps_params = {
        #"shakemaps_filename": os.path.join(rundir, "input/shakemaps/messina_1908/predicted_data_NN_Messina_1908.json"),
        "shakemaps_filename": os.path.join(rootdir, "input/shakemaps/PGA_data/H_Z_pda_data_G.json"),
        "source_parameters_filename": os.path.join(rootdir, "input/shakemaps/messina_1908/source_parameters.csv")
    }
    displacements_exceedance_params = {
        "cumulative_dir": os.path.join(rundir, "displacements"),
        "outfile_name": "exceedance_displacement.npz"
    }
    
    aggregate_shakemaps(rundir, **shakemaps_params)
    #calculate_displacement_probabilities(rundir)
    #caclulate_cumulative_probabilities(rundir, **displacements_exceedance_params)
    
    """
    filter_config = {
        "tsunami_potential_ratio_threshold": 1.,
        "max_rasters": 10,
        "raster_driver": 'Gtiff', # Gtiff/AAIGrid
    }
    
    # Lage csv med shakemap probabilities og regne ut sannsynelighet for alle
    # volumer og clustere for hvert shakemap
    prob_filename = os.path.join(resdir, 'triangulation', "exceedance_displacement.npz")
    probs = load_probabilities(displacement_threshold=5., table_filename=prob_filename)
    
    volumes_file = os.path.join(resdir, 'volumes','Volumes2.csv')
    cluster_file = os.path.join(resdir, 'volumes','Clusters.csv')
    bath_file = os.path.join(rundir, 'input', 'bathy', 'messina_001', "bathy_truncated.tif")
   """
    
    
    
    

def load_probabilities(displacement_threshold=5., table_filename="exceedance_displacement.npz"):
    diplacement_exceedance = np.load(table_filename)
    thresholds, exceedance_probs = diplacement_exceedance["thresholds"], diplacement_exceedance["probs"]
    interpolator = interp1d(x=thresholds, y=exceedance_probs, fill_value=(1.,0.), bounds_error=True)
    probabilities = interpolator(displacement_threshold)
    # Remove nans
    probabilities[probabilities == 9999] = 0
    
    return probabilities

def aggregate_shakemaps(rundir, shakemaps_filename, source_parameters_filename, cumulative=False):
    """ Method to read shakemap and: 
    1. Compute cumulative probabilities if wanted.
    2. Interpolate over domain and write to rasters.
    """
    
    shakemaps_reader = ShakemapsReader(
        shakemaps_filename=shakemaps_filename,
        source_parameters_filename=source_parameters_filename,
        rundir=rundir,
        thresholds=np.linspace(-3,0,40), 
        aggregate=False
    )
    
    # Interpolate over computational region and write to files
    with rasterio.open(os.path.join(rundir, "bathy_truncated.tif")) as src:
        bounds = src.bounds
        profile = src.profile.copy()
    
    shakemaps_reader.write_shakemaps_to_rasters(
        profile=profile, 
        bounds=bounds,
        interpolation_method='linear', # “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”
    )


def calculate_displacement_probabilities(rundir):
    """Displacement probabilities.
    """
    thresholds = np.arange(1, 10, step=1.) # Displacement thresholds in cm.
    dpa = DisplacementProbabilityAggregator(rundir, thresholds, magnitude=7)
    dpa.compute_probabilities()
    
     
    # Plot the distribtuions
    # Er nok enklere å gjøre dette samtidig som man laster alt i cluster_probabilities
    # Litt usikker på om det faktisk blir riktig nå også!
    #with VolumeDatabaseHandler(resdir) as volumes_db:
    #    volumes_db.load_probabilities_from_shakemap(displacement_threshold=5., 
    #                                                table_filename="exceedance_displacement.npz", 
    #                                                column_name = "p_shake")
   ## Move this to preparational script 
   #     volumes_db.write_volumes_to_csv(max_rasters=filter_config['max_rasters'])
    #    volumes_db.write_volumes_to_rasters(**filter_config)
    #    
    #    volumes_db.plot_distribution()
    #    volumes_db.plot_release_density_plots()
    #    
    #    volumes_db.plot_distribution(seed_prob="p_shake")
    #    volumes_db.plot_release_density_plots(seed_prob="p_shake")
        
    # Cluster the volumes into n_clusters, use the csv for now, but might be faster to use the already
    # opened db?
    #volumes_csv = os.path.join(resdir, 'volumes', "volumes.csv")
    #tri_tif = os.path.join(resdir, 'triangulation', "triangulation.tif")
    #bath_tif = os.path.join(rundir, 'input', 'bathy', 'messina_001', "bathy_truncated.tif")
    #upstream_dict_path = os.path.join(resdir, 'triangulation',"poly_slopes.npy")

    #cluster_volumes(volumes_csv, tri_tif, bath_tif, resdir, upstream_dict_path)

def cluster_probabilities(probabilities, volumes_file, cluster_file, resdir, bath_file):
    df_full = pd.read_csv(volumes_file)
    df_cluster = pd.read_csv(cluster_file)
    
    # This method uses only the volumes with single seed!!!!!!!!!!#######
    # For the other method se below
    # Unique clusters and seeds from the full dataset
    all_clusters = np.arange(df_full['cluster'].max() + 1)
    all_seeds = np.unique(df_full['seed_triangle'])

    # Filter only rows with seed_triangle2 == -1
    filtered = df_full[df_full['seed_triangle2'] == -1]

    # Group by cluster and seed_triangle, then sum condprob
    grouped = filtered.groupby(['cluster', 'seed_triangle'])['condprob'].sum().reset_index()

    # Pivot to 2D array, then reindex to ensure full dimensions
    pivot = grouped.pivot(index='cluster', columns='seed_triangle', values='condprob')

    # Reindex to fill in missing clusters and seeds with zeros
    pivot = pivot.reindex(index=all_clusters, columns=all_seeds, fill_value=0)

    # Convert to NumPy array
    all_probs = pivot.to_numpy()
    cluster_ids = pivot.index.to_numpy()
    seed_ids = pivot.columns.to_numpy()
    all_probs2 = all_probs * probabilities[seed_ids][np.newaxis, :]

    df_cluster['prob1'] = np.nansum(all_probs2,axis=1)
    
    # This method assumes that all unique seed triangle combinations are their own tree
    # Pre-allocate result array
    Pc2 = np.zeros(500)

    # Extract only needed columns as arrays for speed
    seed1 = df_full['seed_triangle'].values
    seed2 = df_full['seed_triangle2'].values
    cluster = df_full['cluster'].values
    condprob = df_full['condprob'].values

    # Loop through clusters
    for cin in range(500):
        # Filter rows for this cluster
        mask = cluster == cin
        seed_matrix = np.column_stack((seed1[mask], seed2[mask]))
        condprob_c = condprob[mask]

        # Get unique combinations and inverse indices
        unique_combinations, inverse_idx = np.unique(seed_matrix, axis=0, return_inverse=True)

        # Sum condprob for each unique seed pair
        prob_sums = np.zeros(len(unique_combinations))
        np.add.at(prob_sums, inverse_idx, condprob_c)

        # Compute weighted contribution
        for i, (s1, s2) in enumerate(unique_combinations):
            p1 = probabilities[s1]
            p2 = probabilities[s2] if s2 > -1 else 1
            Pc2[cin] += prob_sums[i] * p1 * p2
    df_cluster['prob2'] = Pc2
    
    # Egentlig cluster_file, men siden jeg driver å tester trenger jeg ikke skrive over foreløpig
    cluster_file2 = os.path.join(resdir, 'volumes','Clusters2.csv')
    df_cluster.to_csv(cluster_file2, index=False)
    plot_probabilities(df_full,resdir, probabilities, bath_file)
    
def plot_probabilities(df_this,resdir, probabilities, bath_file):
    # Flatten and count seeds (excluding -1)
    allseeds_flat = pd.concat([df_this['seed_triangle'], df_this['seed_triangle2']])
    allseeds_flat = allseeds_flat[allseeds_flat != -1]
    seed_counts = allseeds_flat.value_counts()

    # Map counts back to each row (get max of both seed columns)
    def get_row_count(row):
        seeds = [row['seed_triangle'], row['seed_triangle2']]
        return max(seed_counts.get(seeds[0], 0), seed_counts.get(seeds[1], 0))

    # Compute frequencies per row
    df_this['seed_count'] = df_this.apply(get_row_count, axis=1)

    # Filter valid rows (exclude -1)
    valid_mask = (df_this['seed_triangle'] != -1)

    df_this['seed_prob'] = probabilities[df_this['seed_triangle']]
    # 1. Get unique seed_triangle values (excluding -1 if needed)
    unique_seeds = df_this['seed_triangle'].unique()
    unique_seeds = unique_seeds[unique_seeds != -1]

    # 2. For each unique seed, get the first occurrence’s coordinates
    grouped = df_this[df_this['seed_triangle'].isin(unique_seeds)].groupby('seed_triangle').first()

    # 3. Get longitude, latitude, and corresponding probability
    lons = grouped['lon']
    lats = grouped['lat']
    probs = probabilities[grouped.index]

    valid_mask = (
        df_this['seed_count'].notna() &
        df_this['lon'].notna() &
        df_this['lat'].notna()
    )

    # Apply mask to all 3 series
    lons2 = df_this.loc[valid_mask, 'lon']
    lats2 = df_this.loc[valid_mask, 'lat']
    weights2 = df_this.loc[valid_mask, 'seed_count']

    # Load the batymetri
    with rasterio.open(bath_file) as src:
        bathymetri = src.read(1)  # First band
        bathymetri[bathymetri > 60000] = 0
        bounds = src.bounds 
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]


    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # First plot: seed count with logarithmic color scale
    im0 = axes[0].imshow(bathymetri, cmap='grey', extent=extent, origin='upper')
    _, _, _, sc0 = axes[0].hist2d(
        lons2, lats2,
        bins=100,
        cmap='tab20b',
        weights=weights2,
        norm=LogNorm(vmin=1, vmax=weights2.max())  # adjust as needed
    )
    fig.colorbar(sc0, ax=axes[0], label='Volume count')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Volume locations')
    axes[0].grid(True)

    # Create masked probabilities
    probs_masked = np.where(probs > 0, probs, np.nan)

    # Plot bathymetri background
    im1 = axes[1].imshow(bathymetri, cmap='gray', extent=extent, origin='upper')

    valid_mask = (
        ~np.isnan(lats) &
        ~np.isnan(lons) &
        ~np.isnan(probs_masked)
    )

    # Apply the mask
    lats_valid = lats[valid_mask]
    lons_valid = lons[valid_mask]
    probs_valid = probs_masked[valid_mask]

    # Compute 2D histogram manually
    H, xedges, yedges = np.histogram2d(
        lats_valid, lons_valid,
        bins=100,
        weights=probs_valid
    )

    # Mask where H is zero or NaN
    H_masked = np.ma.masked_where((H == 0) | np.isnan(H), H)

    # Define meshgrid for pcolormesh
    X, Y = np.meshgrid(yedges, xedges)

    # Overlay histogram using pcolormesh (respects masking)
    pc = axes[1].pcolormesh(X, Y, H_masked, cmap='tab20b', norm=Normalize(vmin=0.001, vmax=0.3))

    # Add colorbar and labels
    fig.colorbar(pc, ax=axes[1], label='Probability')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Probability from shakemap')
    axes[1].grid(True)

    plt.tight_layout()
    filename = os.path.join(resdir,'volumes','Volumes_probs_new.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
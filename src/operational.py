import os
import numpy as np
import pandas as pd
import rasterio
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from rvsampler.displacements import DisplacementProbabilityAggregator
from rvsampler.shakemaps_reader import ShakemapsAggregator
from rvsampler.utils import create_dir
from rvsampler.cumprobs_by_triangle import caclulate_cumulative_probabilities
from rvsampler.database_handler import VolumeDatabaseHandler, write_raster, Bingclaw_gridsize



def main():
    """This script assigns release probabilities to the release volumes in the database.
    """
    # Rundir tas inn som input
    parser = argparse.ArgumentParser(description="Release volume sampler")
    parser.add_argument('--rundir', required=True, help='Path to the run directory')
    parser.add_argument('--resdir', required=True, help='Path to the results directory')
    args = parser.parse_args()
    rundir = args.rundir
    resdir = args.resdir
    
    shakemaps_params = {
        "shakemaps_filename": os.path.join(rundir, "input/shakemaps/messina_1908/predicted_data_NN_Messina_1908.json"),
        "source_parameters_filename": os.path.join(rundir, "input/shakemaps/messina_1908/source_parameters.csv")
    }
    displacements_exceedance_params = {
        "cumulative_dir": os.path.join(resdir, "displacements"),
        "outfile_name": "exceedance_displacement.npz"
    }
    
    aggregate_shakemaps(resdir, **shakemaps_params)
    calculate_displacement_probabilities(resdir)
    caclulate_cumulative_probabilities(resdir, **displacements_exceedance_params)
    
    filter_config = {
        "tsunami_potential_ratio_threshold": 1.,
        "max_rasters": 10,
        "raster_driver": 'AAIGrid',
    }
     
    with VolumeDatabaseHandler(resdir) as volumes_db:
        volumes_db.load_probabilities_from_shakemap(displacement_threshold=5., 
                                                    table_filename="exceedance_displacement.npz", 
                                                    column_name = "p_shake")
    
        volumes_db.write_volumes_to_csv(max_rasters=filter_config['max_rasters'])
        volumes_db.write_volumes_to_rasters(**filter_config)
        
        #volumes_db.plot_distribution()
        #volumes_db.plot_release_density_plots()
        
        volumes_db.plot_distribution(seed_prob="p_shake")
        volumes_db.plot_release_density_plots(seed_prob="p_shake")
        
    # Cluster the volumes into n_clusters, use the csv for now, but might be faster to use the already
    # opened db?
    volumes_csv = os.path.join(resdir, 'volumes', "volumes.csv")
    tri_tif = os.path.join(resdir, 'triangulation', "triangulation.tif")
    bath_tif = os.path.join(rundir, 'input', 'bathy', 'messina_001', "bathy_truncated.tif")
    upstream_dict_path = os.path.join(resdir, 'triangulation',"poly_slopes.npy")

    cluster_volumes(volumes_csv, tri_tif, bath_tif, resdir, upstream_dict_path)
    

def cluster_volumes(volfile, trifile, bathfile, resdir, upstream_dict_path):
    df_vol = pd.read_csv(volfile)
    # Read triangles
    with rasterio.open(trifile) as src:
        triangles = src.read(1)  # First band
        #triangles[triangles > 60000] = 0
        bounds = src.bounds  # (left, bottom, right, top)
        tri_profile = src.profile
        transform = src.transform 
        height, width = triangles.shape

        # Create a grid of pixel coordinates
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))

        # Apply the affine transform to get x and y (lon/lat or projected coords)
        lon_tri, lat_tri = transform * (cols, rows)
    # Read bathymetri
    with rasterio.open(bathfile) as src:
        bathymetri = src.read(1)  # First band
        bathymetri[bathymetri > 60000] = 0
    # Gradient
    Z = bathymetri  # shape (H, W)

    # Compute gradient along both axes
    dz_dy, dz_dx = np.gradient(Z)  # dy: rows (y-axis), dx: columns (x-axis)

    # Compute gradient magnitude (slope intensity)
    gradient_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
    
    # Build a DataFrame of triangle info
    df_tri = pd.DataFrame({
        'triangle': triangles.flatten(),
        'lon': lon_tri.flatten(),
        'lat': lat_tri.flatten(),
        'z': bathymetri.flatten(),
        'grad': gradient_magnitude.flatten()
    })

    # Step 1: Compute triangle means (same as before)
    triangle_means = df_tri.groupby('triangle')[['lon', 'lat', 'z', 'grad']].mean().reset_index()

    # Step 2: Merge means into df_vol via 'seed_triangle'
    df_vol = df_vol.merge(
        triangle_means,
        how='left',
        left_on='seed_triangle',
        right_on='triangle',
        suffixes=('', '_mean')
    )

    # Step 3: Build X matrix using column names directly
    X = df_vol[['volume', 'no2d', 'lon', 'lat', 'z']].to_numpy()

    # Step 4: Remove rows with NaNs
    X = X[~np.isnan(X).any(axis=1)]

    # Filter df_vol for rows where no2d >= 0.5
    df_filtered = df_vol[df_vol['no2d'] >= 0.5]

    # Build X_filtered matrix and remove rows with NaNs
    X_filtered = df_filtered[['volume', 'no2d', 'lon', 'lat', 'z']].to_numpy()
    X_filtered = X_filtered[~np.isnan(X_filtered).any(axis=1)]
    
    # Cluster the results
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    n_clusters = 500  # choose based on your data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    # Reverse the scaling of the centroids
    centroids_original = scaler.inverse_transform(centroids)

    df_vol = df_vol.iloc[:len(X_scaled)].copy()
    df_vol['cluster'] = labels
    
    # Initialize a dictionary to store closest member indices
    closest_members = {}

    # Loop over each cluster
    for cluster_id in range(n_clusters):
        # Get indices of points in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        # Get the points in this cluster
        cluster_points = X_scaled[cluster_indices]
        
        # Get the centroid of this cluster
        centroid = centroids[cluster_id].reshape(1, -1)
        
        # Compute distances to centroid
        distances = cdist(cluster_points, centroid).flatten()
        
        # Get the index of the closest point (relative to the cluster subset)
        min_index_in_cluster = np.argmin(distances)
        
        # Get the index in the original dataset
        original_index = cluster_indices[min_index_in_cluster]
        
        # Store the result
        closest_members[cluster_id] = original_index

    # Optionally, create a DataFrame of the results
    closest_df = df_vol.loc[list(closest_members.values())].copy()
    closest_df['cluster'] = closest_df.index.map({v: k for k, v in closest_members.items()})
    
    upstream_dict = np.load(upstream_dict_path, allow_pickle=True).item()
    # Write raster volumes
    for i in range(len(closest_df)):
        idx = closest_df.index[i]
        volume = closest_df.loc[idx]
        write_raster(volume, triangles, tri_profile, os.path.join(resdir,'volumes'))
        # Calculate the boxes
        LONLO, LONHI, LATLO, LATHI = Bingclaw_gridsize(volume, upstream_dict, lon_tri, lat_tri, triangles)
        closest_df.at[idx, 'LONLO'] = LONLO
        closest_df.at[idx, 'LONHI'] = LONHI
        closest_df.at[idx, 'LATLO'] = LATLO
        closest_df.at[idx, 'LATHI'] = LATHI
    
    # Save the clusters to csv
    closest_df.to_csv(os.path.join(resdir,'volumes','Clusters.csv'), index=False)
    
    # Save also update volume list with cluster labels
    # Overskride i fremtiden tenker jeg er likesågreit
    df_vol.to_csv(os.path.join(resdir,'volumes','Volumes2.csv'), index=False)


def aggregate_shakemaps(rundir, shakemaps_filename, source_parameters_filename):
    """ Method to compute cumulative distribution of the shakemaps.
    """
    def shake_value(point):
        # Pullback function to extract value from shakemap.
        Z, H = [10**np.array(point[shake_param]) for shake_param in ["Z_pga", "H_pga"]]
        return np.log10(np.sqrt(Z**2 + H**2))
    
    shakemaps_aggregator = ShakemapsAggregator(
        shakemaps_filename=shakemaps_filename,
        source_parameters_filename=source_parameters_filename,
        rundir=rundir)
    
    # Interpolate cumulative over computational region and write to files
    with rasterio.open(os.path.join(rundir, "bathy_truncated.tif")) as src:
        bounds = src.bounds
        profile = src.profile.copy()
    
    shakemaps_aggregator.compute_cumulative(profile=profile, 
                                             bounds=bounds, 
                                             interpolation_method='linear', # “linear”, “nearest”, “slinear”, “cubic”, “quintic” and “pchip”
                                             thresholds=np.linspace(-3,0,40), 
                                             shake_value=shake_value)


def calculate_displacement_probabilities(rundir):
    """Displacement probabilities.
    """
    thresholds = np.arange(1, 10, step=1.) # Displacement thresholds in cm.
    dpa = DisplacementProbabilityAggregator(rundir, thresholds, magnitude=7)
    dpa.compute_probabilities()
    

if __name__ == "__main__":
    main()
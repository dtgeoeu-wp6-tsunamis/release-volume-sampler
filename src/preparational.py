import os
import numpy as np
import shutil
import rasterio
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# Import from modules.
from rvsampler.preprocess import truncate_positive_values, slope, aspect
from rvsampler.slope_analysis import SlopeAnalysis
from rvsampler.triangulate import Triangulation
from rvsampler.cumprobs_by_triangle import caclulate_cumulative_probabilities
from rvsampler.release_volume_sampler import RecursiveReleaseAnalysis
from rvsampler.database_handler import VolumeDatabaseHandler, write_raster, Bingclaw_gridsize
from rvsampler.utils import create_dir
from rvsampler.set_logg import setup_logger


def main():
    """
    This script creates the database with potential release volumes.
    """
    # Rundir tas inn som input
    parser = argparse.ArgumentParser(description="Release volume sampler")
    parser.add_argument('--rundir', required=True, help='Path to the run directory')
    parser.add_argument('--resdir', required=True, help='Path to the results directory')
    args = parser.parse_args()
    rundir = args.rundir
    resdir = args.resdir
    
    # --rundir /home/sfr/release-volume-sampler
    #rundir = r'/home/sfr/release-volume-sampler'
    
    #inputfolder = rundir + "input"
    config = {
        "generated": os.path.join(rundir,"generated"), 
        "scenario":"messina_001",
        "singularity_image": os.path.join(rundir,"images/grass.sif"),
        "bathyfile": os.path.join(rundir,'input', "bathy/messina_001/localMessinaBathy.tif"),
        "soilregions_filename": os.path.join(rundir,'input', 'soilparams','regions.tif'),
        "soil_parameters_filename": os.path.join(rundir,'input', 'soilparams','params.json'),
    }
    
    filter_config = {
        "tsunami_potential_ratio_threshold": 1.,
        "max_rasters": 1000,
        "raster_driver": 'AAIGrid', # Gtiff/AAIGrid
    }
        
    execute_slope_analysis(resdir)
    triangulate_domain(resdir)
    sample_release_volumes(resdir) 
    
    # Write the volumes to csv
    with VolumeDatabaseHandler(resdir) as volumes_db:
        # This is now written in a seperate file in operational.py
        #volumes_db.load_probabilities_from_shakemap(displacement_threshold=5., 
        #                                            table_filename="exceedance_displacement.npz", 
        #                                            column_name = "p_shake")
    
        volumes_db.write_volumes_to_csv(max_rasters=filter_config['max_rasters'])
        volumes_db.write_volumes_to_rasters(**filter_config)
        
        volumes_db.plot_distribution()
        volumes_db.plot_release_density_plots()
        
        #volumes_db.plot_distribution(seed_prob="p_shake")
        #volumes_db.plot_release_density_plots(seed_prob="p_shake")
        
    # Cluster the volumes into n_clusters, use the csv for now, but might be faster to use the already
    # opened db?
    volumes_csv = os.path.join(resdir, 'volumes', "volumes.csv")
    tri_tif = os.path.join(resdir, 'triangulation', "triangulation.tif")
    bath_tif = os.path.join(rundir, 'input', 'bathy', 'messina_001', "bathy_truncated.tif")
    upstream_dict_path = os.path.join(resdir, 'triangulation',"poly_slopes.npy")
    print('Cluster_volumes')
    cluster_volumes(volumes_csv, tri_tif, bath_tif, resdir, upstream_dict_path)
    
def cluster_volumes(volfile, trifile, bathfile, resdir, upstream_dict_path):
    print('read volumes')
    df_vol = pd.read_csv(volfile)
    # Read triangles
    print('read triangles')
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
    print('read bathymetri')
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

    print('Calcualte clusters with KMEANS')
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
    print('Find nearest member to centroid')
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
    print('Write raster volumes')
    for i in range(len(closest_df)):
        idx = closest_df.index[i]
        volume = closest_df.loc[idx]
        write_raster(dict(volume), triangles, tri_profile, os.path.join(resdir,'volumes'), raster_driver='AAIGrid', crop=True)
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
    
def initialize(generated, scenario, bathyfile, soilregions_filename, soil_parameters_filename, singularity_image):
    
    rundir =  os.path.join(generated, scenario)
    logfile = os.path.join(rundir, "log.txt")
    logger = setup_logger("preparational", rundir)
    create_dir(rundir, logger)
    
    # Copy soilparameterfiles to rundir
    shutil.copy(soilregions_filename, os.path.join(rundir, "soilregions.tif"))
    shutil.copy(soil_parameters_filename, os.path.join(rundir, "soilparams.json"))
    

    # Assert that the raster is in a geographic (lon-lat) coordinate system
    logger.info("Verifying that input bathymetri is logitude-latitude.")
    with rasterio.open(bathyfile) as src:
        assert src.crs.is_geographic, "The raster is not in a longitude-latitude coordinate system (geographic CRS)."

    logger.info(f"Copy {bathyfile} to {rundir}.")
    
    outfile = "bathy"    
    shutil.copy(bathyfile, os.path.join(rundir, f"{outfile}.tif"))
    outfile = truncate_positive_values(rundir, singularity_image, outfile, logfile) 
    
    slope(bathyfile, output_dir=rundir, logfile=logfile)
    aspect(bathyfile, output_dir=rundir, logfile=logfile)
    return rundir


def execute_slope_analysis(rundir):
    """
    Parameters defined as discrete distributions or constants.
    May supply functional relations. Order according to dependencies.
    physical_parameters = {
        "distributions": {
            "friction_angle": [(18, 0.248), (22, 0.504), (25, 0.248)], # [(value, weight),...]
            "cohesion": [(10, 0.248),(15,0.504),(20, 0.248)],
            "thickness": [(2, 0.248), (6, 0.504),(10, 0.248)],
            "density": [(1600, 0.248),(2000, 0.504),(2400, 0.248)],
        },
        "constants": {
            "density_of_water": 1020,
            "gravity": 9.81,
            "excess_pore_pressure": 0.
        }
    }
    """
    sa = SlopeAnalysis(rundir, slopefile="slope.tif")
    
    quantiles = [0.01, 0.1, 0.5, 0.9, 0.99]
    sa.compute_quantiles(quantiles, write_fos=True, write_ky=True)
    
    fos_thresholds = np.linspace(0, 2, num=50)
    sa.compute_cumulative(fos_thresholds, feature_name="logfos", write=True)
    
    ky_thresholds = np.linspace(-3,1, num=50)
    sa.compute_cumulative(ky_thresholds, feature_name="logky", write=True)


def triangulate_domain(rundir):
    config = {
        "rundir": rundir,
        "bathyfile": "bathy_truncated.tif",
        "utm_epsg_code": 32633, #Messina strait
        "resolution": (110, 110)
    }
    optimization_params = {
        "num_iterations": 2000,
        "batch_size": 3000,
        "shape_weight": 5e1,
        "area_weight": 5e-11,
        "elevation_weight": 1e-2
    }
    triang = Triangulation(**config)
    triang.fit(**optimization_params)
    triang.plot_triangulation()
    triang.write_to_file()
    
    # Calculate cumulative probabilities lookup table by triangle
    cumulative_dir = os.path.join(rundir, "slope_analysis", "fos", "cumulative")
    outfile_name = "cumulative_fos.npz" # Writes to triangulation dir..
    caclulate_cumulative_probabilities(rundir, cumulative_dir, outfile_name)


def sample_release_volumes(rundir):
    
    # Initialize the database.
    with VolumeDatabaseHandler(rundir) as db:
        db.initialize_db()
    
    config = {
        "rundir": rundir,
        "mesh_path": os.path.join(rundir, "triangulation", "triangulation.vtk"),
        "cumprob_logfos_path": os.path.join(rundir, "triangulation", "cumulative_fos.npz"),
        "utm_epsg_code": 32633, # Messina strait
    }
    
    run_config = {
        "fos_threshold": 1.6,
        "recursive_probability_threshold": 0.001,
        "seed_triangle_probability_threshold": 0.005,
    }
    # Execute analysis.
    analysis = RecursiveReleaseAnalysis(**config)
    # make slope polygons and save to file for use in operational.py - this is used for evaluating possible 
    # slide scenarios and can be used for making the computational grid for bingclaw
    poly_slopes(rundir, analysis)
    
    analysis.run(**run_config)
      
def poly_slopes(rundir, analysis):
    # list of all triangles
    utriangles = np.arange(analysis.n_triangles)
    
    upstream_dict = {}
    while len(utriangles) > 0:
        #print(len(utriangles))
        tlist = get_all_upstream(utriangles[0],-1,analysis)
        upstream_dict[utriangles[0]] = tlist
        # remove those found from the list
        for i in tlist:
            utriangles = np.delete(utriangles, np.where(utriangles == i))
    
    np.save(os.path.join(rundir, "triangulation", "poly_slopes.npy"), upstream_dict)
  
def get_all_upstream(start, last, analysis, collected=None):
    # Calculate all upstream triangles for given start triangle
    
    if collected is None:
        collected = []
    if start is not None and start not in collected and start > -1:  # avoid duplicates or infinite loops
        collected.append(start)
        #print('#######' + str(start)+'##########' + str(last))
        upstream = analysis.get_upstream_triangles(start)
        for i in upstream:
            collected = get_all_upstream(i, start, analysis, collected)

    return collected


if __name__ == "__main__":
    main()
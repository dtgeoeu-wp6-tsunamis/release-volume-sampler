import json
import pandas as pd
import rasterio
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
import cartopy.crs as ccrs
import sqlite3
import csv
import ast

from rvsampler.set_logg import setup_logger
from rvsampler.utils import create_dir
from rvsampler.triangulate import Triangulation

# Map rasterio driver to file suffix.
SUFFIX_MAP = {
    "GTiff": ".tif",
    "AAIGrid": ".asc",
}

# Schema for the "volumes" table
VOLUMES_SCHEMA = {
    "id": "INTEGER PRIMARY KEY",
    "released": "JSON",
    "condprob": "REAL",
    "area": "REAL",
    "mean_elevation": "REAL",
    "mean_easting": "REAL",
    "mean_northing": "REAL",
    "mean_slope": "REAL",
    "seed_triangles": "JSON",
    "p_fos_seed": "REAL",
    "volume": "REAL",
    "thickness": "REAL",
    "tsunami_potential_ratio": "REAL",
    "no2d": "REAL",
    "slopeunit": "INTEGER",
    "cluster": "INTEGER",
    "cluster_center_dist": "REAL",
    "is_representative": "BOOLEAN",
    "p_shake": "REAL",  # Probability from shakemap
}

SEED_TRIANGLES_SCHEMA = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "triangle_id": "INTEGER",
    "slopeunit": "INTEGER",
    "p_fos": "REAL",
    "p_shake": "REAL"
}
#COLUMN_NAMES = list(VOLUMES_SCHEMA.keys())

def main():
    """ To ensure that module imports works, run the script as a module.
    release-volume-sampler$ python 
    """
    # Usage example
    config = {
        "rundir": "/home/ebr/projects/release-volume-sampler/generated/messina_001",
    }
    
    filter_config = {
        "tsunami_potential_ratio_threshold": 1.,
        "max_rasters": 100,
        "raster_driver": 'AAIGrid',
    }
    
    
    # Execute
    with VolumeDatabaseHandler(**config) as volumes_db:
        volumes_db.load_probabilities_from_shakemap(displacement_threshold=5., 
                                                    table_filename="exceedance_displacement.npz", 
                                                    column_name = "p_shake")
    

        volumes_db.write_volumes_to_csv(max_rasters=filter_config['max_rasters'])
        volumes_db.write_volumes_to_rasters(**filter_config)
        
        #volumes_db.plot_distribution()
        #volumes_db.plot_release_density_plots()
        
        volumes_db.plot_distribution(seed_prob="p_shake")
        volumes_db.plot_release_density_plots(seed_prob="p_shake")

   

class VolumeDatabaseHandler:
    """
    Class designed for interaction with the database of volumes. 
    
    Tasks:
    1. Filter volumes based on the tsunamogenic potential. Using statistical relations.
    2. Assign thickness (Statistically from total area), smooth and create raster files.
    3. Make figures to obtain an overview of the distribution.
    4. Assign probabilities from shakemaps.
    5. Write clusters/volumes to csv and rasters.
    
    """
    def __init__(self, rundir, db_file="volumes.db"):
        
        self.rundir = rundir
        self.output_dir = os.path.join(rundir, "volumes")
        self.triangulation_dir = os.path.join(rundir, "triangulation")
        self.tri_mask_path = os.path.join(self.triangulation_dir, "triangulation.tif")
        self.upstream_dict_path = os.path.join(self.triangulation_dir,"poly_slopes.npy")
        #self.slopeunits_to_triangles = np.load(os.path.join(self.triangulation_dir,\
        #    "slopeunits_to_triangles.npy"), allow_pickle=True).item()
        self.db_file = os.path.join(self.output_dir, db_file)
        
        # Create volumes dir if it does not exist
        create_dir(self.output_dir, logger = None, clear=False)
        
        with rasterio.open(self.tri_mask_path) as src:
            self.tri_mask = src.read(1)  # Read the triangle mask
            self.tri_profile = src.profile  # Copy metadata to use in output
            self.bounds = src.bounds
            self.crs = src.crs
            
            # Create lat/lon also, this might be defined somewhere else also which 
            # might make this obsolete
            transform = src.transform 
            height, width = self.tri_mask.shape
            # Create a grid of pixel coordinates
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            # Apply the affine transform to get x and y (lon/lat or projected coords)
            self.lon_tri, self.lat_tri = transform * (cols, rows)
        
        self.logger = setup_logger("db_handler", self.output_dir)
        self.conn = None
        self.cursor = None

    def initialize_db(self):
        """
        Initialize the SQLite database and table if they don't exist.
        """
        
        try:
            #Create volumes table
            columns = ", ".join(f"{name} {type_}" for name, type_ in VOLUMES_SCHEMA.items())
            create_table_sql = f"CREATE TABLE IF NOT EXISTS volumes ({columns});"
            self.cursor.execute(create_table_sql)
            
            # Create seed_triangles table
            seed_columns = ", ".join(f"{name} {type_}" for name, type_ in SEED_TRIANGLES_SCHEMA.items())
            create_seed_table_sql = f"CREATE TABLE IF NOT EXISTS seed_triangles ({seed_columns});"
            self.cursor.execute(create_seed_table_sql)
            
            self.conn.commit()
            self.logger.info("Database initialized or already exists.")
        except Exception as e:
            self.logger.info(f"Error initializing database: {e}")


    def __enter__(self):
        self.conn = sqlite3.connect(self.db_file)
        self.conn.row_factory = sqlite3.Row # Return rows as dicts.
        self.cursor = self.conn.cursor()
        self.logger.info("Database connection initialized.")
        return self


    def __exit__(self, exc_type, exc_value, traceback):
            if self.conn:
                self.conn.close()
                self.logger.info("Database connection closed.")


    def insert_volume(self, volume_data):
        """
        Insert a new volume record into the database.
        Args:
            volume_data (dict): A dictionary containing 'volume' and 'location'.
        """
        try:
            volume, thickness, tsunami_potential_ratio, no2d = self.calculate_volume_features(volume_data["area"], 
                                                                                        volume_data["mean_slope"],
                                                                                        volume_data["mean_elevation"])
            self.cursor.execute("""
                INSERT INTO volumes (
                    released, condprob, area, mean_elevation, mean_easting, mean_northing, mean_slope,
                    seed_triangles, p_fos_seed, volume, thickness, tsunami_potential_ratio, no2d, slopeunit
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                json.dumps(volume_data["released"]),  # Serialize list to JSON
                volume_data["condprob"],
                volume_data["area"],
                volume_data["mean_elevation"],
                volume_data["mean_easting"],
                volume_data["mean_northing"],
                volume_data["mean_slope"],
                json.dumps(volume_data["seed_triangles"]),
                volume_data["p_fos_seed"],
                volume,
                thickness,
                tsunami_potential_ratio,
                no2d,
                volume_data["slopeunit"]
            ))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error inserting volume data: {e}")

    def insert_seed_triangles(self, triangles, slopeunits, p_fos):
        rows = [
            (int(tri), int(su), float(ps))
            for tri, su, ps in zip(triangles, slopeunits, p_fos)
        ]
        self.cursor.executemany(
            "INSERT INTO seed_triangles (triangle_id, slopeunit, p_fos) VALUES (?, ?, ?)",
            rows
        )
        self.conn.commit()
    
    def calculate_volume_features(self, area, mean_slope, mean_elevation):
        """
        Assign thickness:
        Statistical relation for assigning thickness of the volume (Zengafinnen-Morris et al. 2022).
        V = 0.0298*A^{1.36}
        
        Estimate tsunami potential:
        Equation for maximum sea level depression.
        
        eta_0,2D = 0.2139d[1 - 0.7458sin(theta) + 0.1704sin(theta)^2](Lsin(theta)/H)^(5/4)
        
        where d[m] is the thickness of the landslide, theta is the slope angle,
        L[m] is the length of the landslide, and H[m] is the water depth.
        Watts et al. (2003):
        
        Estimate tsunami potential ratio:
        
        V_{min}=0.0957*slope^{-1.609}*H^{1.3807} 
        
        where V_{min} is the minimum volume that can create a significant tsunami for a given water 
        depth H (in m) and slope angle (in degrees) and V_{min} in Mm^{3}. Based on equations of 
        Watts et al. (2003). 
        
        Note: This ratio goes to infinity as H goes to zero. Therefore, we truncate H at -1 m 
        (a small negative value) to avoid division by zero errors.
        """
        volume = 0.0298*area**1.36
        thickness = volume/area
        
        if mean_elevation >= -1.0:
            self.logger.warning(f"Mean elevation is non-negative: {mean_elevation}. \
                Setting elevation to -1.")
            mean_elevation = -1.  # Set to a small negative value to avoid division by zero.
        
        L = np.sqrt(area) # Asume A = L*L
        mean_slope_rad = mean_slope*np.pi/180
        
        eta = 0.2139*thickness*(1-0.7458*np.sin(mean_slope_rad)+0.1704*(np.sin(mean_slope_rad))**2)\
            *(L*np.sin(mean_slope_rad)/(0-mean_elevation))**(5/4)
        tsunami_potential_ratio = volume/(0.0957*(mean_slope**-1.609)*((0-mean_elevation)**1.3807)*1e6)
        
        return(volume, thickness, tsunami_potential_ratio, eta)

    def assign_probabilities_to_seed_triangles(self, displacement_threshold, displacement_dir,\
        table_filename="exceedance_displacement.npz", column_name="p_shake"):
        """
        Loads shakemap and computes the release probability of each seed triangle is computed based
        on P(displacement > displacement_threshold) for each individual seed triangle.
        
        Parameters:
            displacement_threshold (float): Threshold for calculation of probability. 
                Must lie within the range of calculated probabilities.
            displacement_dir: directory of the computed displacements.
            table_filename (str): filename of the lookuptable.
            column_name (str): Column name of the assigned variable.
            
        TODO: Probabilities of initial values is just an approximation.
        """
        # Add column if not present
        if column_name not in SEED_TRIANGLES_SCHEMA.keys():
            try:
                self.cursor.execute(f"ALTER TABLE seed_triangles ADD COLUMN {column_name} REAL")
                self.logger.info(f"Added '{column_name}' column to the seed_triangle table.")
                self.conn.commit()
            except Exception as e:
                self.logger.warning(f"Could not add column '{column_name}': {e}")
        
        # Load lookuptable
        lookup_table_path = os.path.join(displacement_dir, table_filename)
        self.logger.info(f"Load exceedance probabilities: {lookup_table_path}.")
        displacement_exceedance = np.load(lookup_table_path)
        thresholds, exceedance_probs = displacement_exceedance["thresholds"], displacement_exceedance["probs"]

        interpolator = interp1d(x=thresholds, y=exceedance_probs, fill_value=(1., 0.), bounds_error=True)
        probabilities = interpolator(displacement_threshold)

        # Fetch all triangle_ids from seed_triangles
        self.cursor.execute("SELECT id, triangle_id FROM seed_triangles")
        rows = self.cursor.fetchall()

        updates = []
        for row in rows:
            seed_triangle_id = row["id"]
            triangle_index = row["triangle_id"]
            prob = float(probabilities[triangle_index])
            updates.append((prob, seed_triangle_id))

        
        # Update p_shake for each seed triangle
        self.cursor.executemany(
            "UPDATE seed_triangles SET p_shake = ? WHERE id = ?",
            updates
        )
        self.conn.commit()
        self.logger.info("Assigned probabilities to all seed triangles.")
    
    def compute_release_probabilities(self, column_name="p_shake", batch_size=1000):
        
        """
        Compute probabilities of the initial state of each volume based on the release probability
        of each seed triangle in the initial state.
        
        If the volume is assigned to a slopeunit the sample space is all initial states within the 
        given slopeunit. If the volume is not assigned to a slopeunit (i.e. slopeunit=-1) the 
        volume is considered an independent trial (We assume no interaction between volumes).
        
        Processes the database in batches for efficiency.
        
        Parameters:
            column_name (str): Column name of the assigned variable.
        """
        # Add column if not present
        if column_name not in VOLUMES_SCHEMA.keys():
            try:
                self.cursor.execute(f"ALTER TABLE volumes ADD COLUMN {column_name} REAL")
                self.logger.info(f"Added '{column_name}' column to the table.")
                self.conn.commit()
            except Exception as e:
                self.logger.warning(f"Could not add column '{column_name}': {e}")

        # Load all triangle probabilities from seed_triangles table into a dict
        self.cursor.execute("SELECT triangle_id, p_shake FROM seed_triangles")
        triangle_probs = {row["triangle_id"]: row[column_name] for row in self.cursor.fetchall()}
        seed_triangles_in_su = self.load_slopeunits_to_seedtriangles()
        # Assign release probabilities to volumes.
        offset = 0
        while True:
            query = f"SELECT id, seed_triangles, slopeunit FROM volumes LIMIT {batch_size} OFFSET {offset}"
            df = pd.read_sql_query(query, self.conn)
            if df.empty:
                break

            # Vectorized probability calculation
            df['prob'] = df.apply(
                lambda row: self.get_release_probability(
                    json.loads(row['seed_triangles']) if isinstance(row['seed_triangles'], str) else row['seed_triangles'],
                    seed_triangles_in_su[row['slopeunit']],
                    row['slopeunit'],
                    triangle_probs
                ),
                axis=1
            )
            updates = list(zip(df['prob'], df['id']))

            if updates:
                self.cursor.executemany(
                    f"UPDATE volumes SET {column_name} = ? WHERE id = ?", updates
                )
                self.conn.commit()

            offset += batch_size

        self.logger.info(f"Finished assigning '{column_name}' to all volumes.")
    
    def load_slopeunits_to_seedtriangles(self):
        """
        Build a dictionary mapping slopeunit -> list of seed triangle indices,
        using the seed_triangles table in the database.
        Returns:
            dict: {slopeunit: [triangle_id, ...], ...}
        """
        self.cursor.execute("SELECT triangle_id, slopeunit FROM seed_triangles")
        slopeunits_to_seedtriangles = {}
        for row in self.cursor.fetchall():
            su = row["slopeunit"]
            tri = row["triangle_id"]
            if su not in slopeunits_to_seedtriangles:
                slopeunits_to_seedtriangles[su] = []
            slopeunits_to_seedtriangles[su].append(tri)
        return slopeunits_to_seedtriangles

    def get_release_probability(self, seed_triangles, seed_triangles_in_su, slope_unit, probabilities):
        """
        Calculates the probability of the initial state defined by the seed triangles within the 
        given slopeunit. If slopeunits are not defined (slope_unit = -1) the event is considered 
        as independent (non exclusive).

        Args:
            seed_triangles (list[int]): Indices of seed triangles (released in the initial state).
            slope_unit (int): Slope unit identifier.
            seed_triangles_in_su (list[int]): All seed triangles (none released and released)
                in the slopeunit.
            probabilities (dict or np.ndarray): Mapping from triangle index to P(displacement > delta).
                                                If ndarray, index is triangle index.

        Returns:
            float: Probability of the initial state.
        
        """
        if slope_unit == -1:
            return float(np.prod([probabilities[tri] for tri in seed_triangles]))
        else:
            # Fetch all release triangles in the slopeunit
            # Assume you have a method or mapping: self.slopeunit_to_triangles[slope_unit]
            # which gives a list/array of triangle indices in the slopeunit.

            # Prepare indicator vector: alpha[i] = 1 if triangle[i] is a seed triangle, else 0
            seed_set = set(seed_triangles)
            alpha = np.array([1 if tri in seed_set else 0 for tri in seed_triangles_in_su])

            # Get probabilities for all triangles in the slopeunit
            p = np.array([probabilities[tri] for tri in seed_triangles_in_su])
            
            # Compute probability: prod(p_i^alpha_i * (1-p_i)^(1-alpha_i))
            return float(np.prod(p**alpha * (1 - p)**(1 - alpha)))

    def write_volumes_to_csv(self, max_rasters=100):
        #self.df.drop(columns=["released"]).to_csv(os.path.join(self.output_dir, "volumes.csv"),
        #                                          float_format="%.6e")
        csv_filename = os.path.join(self.output_dir, "volumes.csv")
        
        # Create a generator to fetch rows from the database
        row_generator = self.fetch_volumes(max_rows=max_rasters, only_clusters=True, ordered=True)
        
        # Get the first row to determine the header (fieldnames)
        first_row = next(row_generator)
        fieldnames = list(first_row.keys()) + ['LONLO', 'LONHI', 'LATLO', 'LATHI']
        
        upstream_dict = np.load(self.upstream_dict_path, allow_pickle=True).item()
        
       
        # Open the CSV file for writing
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write the header (column names)
            writer.writeheader()
            
            # Write the first row
            LONLO, LONHI, LATLO, LATHI = self.landslide_grid_extent(first_row, upstream_dict, self.lon_tri, self.lat_tri, self.tri_mask)
            first_row.update({'LONLO': LONLO, 'LONHI': LONHI, 'LATLO': LATLO, 'LATHI': LATHI})
            #first_row.pop("released", None)
            # Alternatively the list can be convert to a string
            first_row["released"] = json.dumps(first_row["released"])
            writer.writerow(first_row)
            for i,row in enumerate(row_generator):
                
                # The calculations are not very fast, so avoid doing this for all volumes.
                if i < max_rasters: 
                    LONLO, LONHI, LATLO, LATHI = self.landslide_grid_extent(row, upstream_dict, self.lon_tri, self.lat_tri, self.tri_mask)
                    row.update({'LONLO': LONLO, 'LONHI': LONHI, 'LATLO': LATLO, 'LATHI': LATHI})
                # Including a list in a csv can be a bit messy, so a simple solutions is to just remove it
                #row.pop("released", None)
                # Alternatively the list can be convert to a string
                row["released"] = json.dumps(row["released"])
                writer.writerow(row)
                
        return csv_filename

    def fetch_volumes(self, max_rows=None, only_clusters=False, ordered=True):
        """ Return python generator with volumes from database.
        """ 
        query = "SELECT * FROM volumes"
        params = []

        if only_clusters:
            query += " WHERE is_representative = 1"

        if ordered:
            query += " ORDER BY no2d DESC"

        if max_rows is not None:
            query += " LIMIT ?"
            params.append(max_rows)
        
        self.cursor.execute(query, params)
        for row in self.cursor:
            row_dict = dict(row)
            row_dict["released"] = json.loads(row_dict["released"])
            yield row_dict
    
    def write_volumes_to_rasters(self, tsunami_potential_ratio_threshold=1., max_rasters=100, raster_driver="GTiff", crop=True):
        """Write rasters of volumes, optionally cropped to extent of each volume (non-zero region)."""
        rasterdir = os.path.join(self.output_dir, "rasters")
        create_dir(rasterdir, logger=self.logger, clear=True)

        self.logger.info('reading triangles')
        with rasterio.open(self.tri_mask_path) as src:
            tri_mask = src.read(1)
            tri_profile = src.profile

        #tri_profile.update(dtype=rasterio.float32, count=1, driver=raster_driver)

        for i, volume in enumerate(self.fetch_volumes(max_rows=max_rasters, only_clusters=True, ordered=True)):
            self.write_raster(volume, tri_mask, tri_profile, rasterdir=rasterdir, raster_driver=raster_driver, crop=True)
            if i >= max_rasters or tsunami_potential_ratio_threshold > volume["tsunami_potential_ratio"]:
                break

    def write_raster(self, volume, tri_mask, tri_profile, rasterdir, raster_driver="AAIGrid", crop=True):
        volume_mask = np.isin(tri_mask, json.loads(str(volume["released"])))
        volume_raster = convolve2d(volume_mask.astype(float) * volume["thickness"], np.ones((3, 3)) / 9., mode="same")        
        profile = tri_profile.copy()
        
        if crop:
            nonzero = np.nonzero(volume_raster)
            if nonzero[0].size != 0 or nonzero[1].size != 0:
                row_min, row_max = nonzero[0].min(), nonzero[0].max()
                col_min, col_max = nonzero[1].min(), nonzero[1].max()

                pad = 10
                row_min = max(row_min - pad, 0)
                row_max = min(row_max + pad, volume_raster.shape[0])
                col_min = max(col_min - pad, 0)
                col_max = min(col_max + pad, volume_raster.shape[1])

                volume_raster = volume_raster[row_min:row_max, col_min:col_max]
                cropped_transform = tri_profile["transform"] * rasterio.Affine.translation(col_min, row_min)

                profile.update({
                    "height": volume_raster.shape[0],
                    "width": volume_raster.shape[1],
                    "transform": cropped_transform
                })

        lst = ast.literal_eval(volume["seed_triangles"])  # Converts to [1, 2, 3, 4]
        st_str = '-'.join(map(str, lst))
        volume_path = os.path.join(
            rasterdir,
            f'volume_id-{volume["id"]}_seed-{st_str}-{"crop"}{SUFFIX_MAP[raster_driver]}'
        ) 
        # Set nodata value
        profile.update({
        "driver": raster_driver,
        "dtype": rasterio.float32,
        "count": 1,
        "nodata": -9999  # Set your NoData value here
        })

        #os.makedirs(os.path.dirname(volume_path), exist_ok=True)
        with rasterio.open(volume_path, 'w', **profile) as dst:
            dst.write(volume_raster.astype(rasterio.float32), 1)
        self.logger.info(f"Wrote volume raster: {volume_path}")

    def get_cluster_probability(self, cluster_id, probability_column="p_shake"):
        """
        Compute probability of the given clusters based on probability of volumes.
        
        parameters:
            clusters (list): List of cluster IDs.
            probability_column (str): Column name for the probability to use.
        """
        cluster_probabilities = {}
        query = f""" SELECT * FROM volumes WHERE cluster = ?  """, (cluster_id,)
        self.cursor.execute(query)
        df_cluster = pd.read_sql_query(query, self.conn)
        # TODO: Caclulate probability that 1 or more volumes in the cluster is released.
        
    def landslide_grid_extent(self, volume, upstream_dict, lon_tri, lat_tri, tri_mask):
        """ 
        Calculate bounding box for landslide simulations. 
        Calculations are based on polygons made from upstream triangles. All polygons covering the 
        the seed triangle is used to define the extent of the simulation grid.
        """
        
        # Find all polygons in the upstream_dict that covers the seed triangle
        matching_keys = []
        for seed_triangle in ast.literal_eval(volume['seed_triangles']):
            for k, v in upstream_dict.items():
                if seed_triangle in v:
                    matching_keys.append(k)
        
        # Merge all polygons into one
        merged = np.concatenate([upstream_dict[k] for k in matching_keys])
        # Find all unique triangles from the polygons
        unique_merged = np.unique(merged)
        # Get a mask og triangles
        mask = np.isin(tri_mask, unique_merged)
        # Get the coordinates of the triangles
        temp_lon = lon_tri[mask]
        temp_lat = lat_tri[mask]

        # Compute corners
        lat_min, lat_max = temp_lat.min(), temp_lat.max()
        lon_min, lon_max = temp_lon.min(), temp_lon.max()

        dx = lon_max - lon_min
        dy = lat_max - lat_min

        # Add some extra space outside the polygons 
        # This could potentially be scaled based on the volume
        # Where 0.1 is x*volume/reference volume
        default_diff = 0.025
        xextra = [default_diff, 0.05, 0.07, 0.09]
        yextra = [default_diff, 0.05, 0.07, 0.09]

        # This seems to work well for all test cases
        xe = xextra[1]
        ye = yextra[1]
        # This hardcoded and suboptimal, but works for this excat case, but should
        # be updated in the future
        if lon_max > 15.6:
            #xcoords = [lon_min - xe, lon_min - xe, lon_max, lon_max]
            LONLO = lon_min - xe
            LONHI = lon_max + default_diff
        else:
            #xcoords = [lon_min, lon_min, lon_max + xe, lon_max + xe]
            LONLO = lon_min - default_diff
            LONHI = lon_max + xe
        #ycoords = [lat_min - ye, lat_max, lat_max, lat_min - ye]
        LATHI = lat_max + default_diff
        LATLO = lat_min - ye
            
        return LONLO, LONHI, LATLO, LATHI
            
    def plot_distribution(self, seed_prob="p_fos_seed"):
        """Create figures to display the volume distribution.
        """
        query = f"SELECT area, {seed_prob}, condprob, thickness, tsunami_potential_ratio FROM volumes;"
        df = pd.read_sql_query(query, self.conn)
        probability = df[seed_prob]*df.condprob
        
        # Create a figure and axis objects
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns of plots
        fig.suptitle(f"Release Characteristics - seed_prob: {seed_prob}", fontsize=16)

        # Top-left: Histogram of area
        axes[0, 0].hist(df.area, bins=30, color="skyblue", edgecolor="black")
        axes[0, 0].set_yscale('log', nonpositive='clip')
        axes[0, 0].grid(which='major', axis='y')
        axes[0, 0].set_title("Area of Release")
        axes[0, 0].set_xlabel("Area")
        axes[0, 0].set_ylabel("Frequency")
        
        # Top-right: Weighted histogram of area
        axes[0, 1].hist(df.area, bins=30, weights=probability, color="salmon", edgecolor="black", density=True)
        axes[0, 1].set_yscale('log', nonpositive='clip')
        axes[0, 1].grid(which='major', axis='y')
        axes[0, 1].set_title("Weighted Area of Release")
        axes[0, 1].set_xlabel("Area")
        axes[0, 1].set_ylabel("Weighted Frequency")

        # Middle-left: Histogram of thickness
        axes[1, 0].hist(df.thickness, bins=30, color="skyblue", edgecolor="black")
        axes[1, 0].set_yscale('log', nonpositive='clip')
        axes[1, 0].grid(which='major', axis='y')
        axes[1, 0].set_title("Thickness of Release")
        axes[1, 0].set_xlabel("Thickness")
        axes[1, 0].set_ylabel("Frequency")

        # Middle-right: Weighted histogram of thickness
        axes[1, 1].hist(df.thickness, bins=30, weights=probability, color="salmon", edgecolor="black", density=True)
        axes[1, 1].set_yscale('log', nonpositive='clip')
        axes[1, 1].grid(which='major', axis='y')
        axes[1, 1].set_title("Weighted Thickness of Release")
        axes[1, 1].set_xlabel("Thickness")
        axes[1, 1].set_ylabel("Weighted Frequency")
        
        # Bottom-left: Weighted histogram of area
        axes[2, 0].hist(df.tsunami_potential_ratio, bins=30, color="skyblue", edgecolor="black")
        axes[2, 0].set_yscale('log', nonpositive='clip')
        axes[2, 0].grid(which='major', axis='y')
        axes[2, 0].set_title("Tsunami Potential Ratio of Release")
        axes[2, 0].set_xlabel("Tsunami Potential Ratio")
        axes[2, 0].set_ylabel("Frequency")

        # Bottom-right: Weighted histogram of thickness
        axes[2, 1].hist(df.tsunami_potential_ratio, bins=30, weights=probability, color="salmon", edgecolor="black", density=True)
        axes[2, 1].set_yscale('log', nonpositive='clip')
        axes[2, 1].grid(which='major', axis='y')
        axes[2, 1].set_title("Weighted Tsunami Potential Ratio of Release")
        axes[2, 1].set_xlabel("Tsunami Potential Ratio")
        axes[2, 1].set_ylabel("Weighted Frequency")

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
        plt.savefig(os.path.join(self.output_dir, f"release_characteristics-{seed_prob}.png"))


    def plot_release_density_plots(self, seed_prob="p_fos_seed"):
        """ Create figure to display the location of the sampled release volumes.
        """
        
        self.logger.info(" Create spatial release density figure.")
        n_triangles = int(self.tri_mask.max()) + 1
        counts = np.zeros(n_triangles)
        probs = np.zeros(n_triangles)

        self.cursor.execute(f"""
            SELECT released, {seed_prob}, condprob FROM volumes
        """)
        for row in self.cursor:
            released = np.array(json.loads(row["released"]), dtype=int)
            counts[released] += 1.
            probs[released] += row[seed_prob]*row["condprob"]

        counts_raster = np.zeros(self.tri_mask.shape)
        probs_raster = np.ones(self.tri_mask.shape)

        for tri_index in range(n_triangles):
            counts_raster += np.where(self.tri_mask == tri_index, counts[tri_index], 0)
            probs_raster *= np.where(self.tri_mask == tri_index, 1. - probs[tri_index], 1.)
            if tri_index % 500 == 0:
                self.logger.info(f"Aggregated triangles: {tri_index}") 
        probs_raster = 1. - probs_raster
        # Create subplots
        
         # Convert bounds to extent
        extent = [self.bounds.left, self.bounds.right, self.bounds.bottom, self.bounds.top]

        # Create a map with the raster CRS
        if self.crs.is_geographic:
            projection = ccrs.PlateCarree()
        else:
            projection = ccrs.epsg(self.crs.to_epsg())
        
        # Calculate tick values dynamically based on raster bounds
        lon_ticks = np.linspace(self.bounds.left, self.bounds.right, num=5)  # 5 ticks along longitude
        lat_ticks = np.linspace(self.bounds.bottom, self.bounds.top, num=5)  # 5 ticks along latitude
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': projection})  # 1 row, 2 columns of plots
        fig.suptitle(f"Spatial distribution of release volumes - seed_prob: {seed_prob}", fontsize=16)

        # Plot the first raster
        counts_im = axes[0].imshow(
            counts_raster,
            origin='upper',
            extent=extent,
            transform=projection,
            cmap="tab20b",
            norm=LogNorm(vmin=1, vmax=np.nanmax(counts_raster), clip=True),
        )

        # Plot the second raster
        prob_im = axes[1].imshow(
            probs_raster,
            origin='upper',
            extent=extent,
            transform=projection,
            cmap="tab20b",
            norm=LogNorm(vmin=1e-6, vmax=1., clip=True),
        )
        
        # Add gridlines
        for ax in axes:
            gridlines = ax.gridlines(
                draw_labels=True,  # Show labels on gridlines
                xlocs=lon_ticks,  # X ticks (longitudes)
                ylocs=lat_ticks,  # Y ticks (latitudes)
                linewidth=0.5,
                color='gray',
                linestyle='--'
            )

            # Customize tick labels
            gridlines.xlabel_style = {'size': 10, 'color': 'black'}
            gridlines.ylabel_style = {'size': 10, 'color': 'black'}

            gridlines.right_labels = False

        # Create colorbars
        fig.colorbar(counts_im, ax=axes[0], label="Counts")
        fig.colorbar(prob_im, ax=axes[1], label="Probability")

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
        plt.savefig(os.path.join(self.output_dir, f"release_distribution-{seed_prob}.png")) 
        plt.close()
        
    def test_no_duplicate_triangles_in_released(self):
        """
        Test that no volume in the database contains duplicate triangles in the 'released' field.
        
        Usage example:
        
        db_handler = VolumeDatabaseHandler(...)
        assert test_no_duplicate_triangles_in_released(db_handler)
        """
        for volume in self.fetch_volumes():
            released = volume["released"]
            if len(released) != len(set(released)):
                self.logger.warning(f"Duplicate triangles found in volume with id {volume.get('id', 'unknown')}: {released}")
                return False
        self.logger.info("All volumes have unique triangles in 'released'.")
        return True
    
    def insert_volumes_batch(self, volume_list):
        """
        Insert multiple volume records into the database.
        Args:
            volume_list (List[dict]): List of dictionaries each containing 'volume' and 'location' data.
        """
        try:
            rows = []

            for volume_data in volume_list:
                try:
                    volume, thickness, tsunami_potential_ratio, no2d = self.calculate_volume_features(
                        volume_data["area"],
                        volume_data["mean_slope"],
                        volume_data["mean_elevation"]
                    )

                    row = (
                        json.dumps(volume_data["released"]),
                        volume_data["condprob"],
                        volume_data["area"],
                        volume_data["mean_elevation"],
                        volume_data["mean_easting"],
                        volume_data["mean_northing"],
                        volume_data["mean_slope"],
                        json.dumps(volume_data["seed_triangles"]),
                        volume_data["p_fos_seed"],
                        volume,
                        thickness,
                        tsunami_potential_ratio,
                        no2d,
                        volume_data["slopeunit"]
                    )

                    rows.append(row)
                except KeyError as e:
                    self.logger.error(f"Missing field in volume_data: {e} | data: {volume_data}")
                except Exception as e:
                    self.logger.error(f"Error preparing volume_data: {e} | data: {volume_data}")

            if rows:
                self.cursor.executemany("""
                    INSERT INTO volumes (
                        released, condprob, area, mean_elevation, mean_easting, mean_northing, mean_slope,
                        seed_triangles, p_fos_seed, volume, thickness, tsunami_potential_ratio, no2d, slopeunit
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                self.conn.commit()

        except Exception as e:
            self.logger.error(f"Error during batch insert: {e}")

    def fetch_all_seed_triangles(self):
        """
        Fetch all unique seed triangles from the database.
        Returns:
            set: A set of all unique seed triangle indices.
        """
        seed_triangles_set = set()
        query = "SELECT seed_triangles FROM volumes"
        self.cursor.execute(query)
        for row in self.cursor:
            # Parse the JSON list of seed triangles
            triangles = json.loads(row["seed_triangles"])
            seed_triangles_set.update(triangles)
        return seed_triangles_set

if __name__ == "__main__":
    main()
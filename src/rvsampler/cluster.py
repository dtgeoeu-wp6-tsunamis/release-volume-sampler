
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

import sqlite3
from rvsampler.set_logg import setup_logger, close_logger


def main():
    # Usage example. Called from preparational.py.:
    rundir = "/home/ebr/projects/release-volume-sampler/generated/messina_003"
    
    config = {
        "rundir": rundir,
        "n_clusters": 500,
        "random_state": 0,
        "batch_size": 1024,
        "feature_columns": ['area', 'no2d', 'mean_elevation',
                            'mean_northing', 'mean_easting'],
        "columns_to_scale": ['area', 'no2d', 'mean_elevation',
                             'mean_northing', 'mean_easting'],
        "weights": {
            'area': 1.0,
            'no2d': 1.0,
            'mean_elevation': 1.0,
            'mean_northing': 1.0,
            'mean_easting': 1.0
        },
    }
    # Initialize ClusterAnalysis object
    with ClusterAnalysis(**config) as cluster_analysis:
        # Fit the model
        cluster_analysis.fit()
        # Write to database
        cluster_analysis.write_to_database()
        # Find representatives
        cluster_analysis.find_representatives()

class ClusterAnalysis:
    """
    Class to perform clustering analysis on release volumes. Information 
    about the clusters is stored in the database.
    
    Attributes:
        rundir (str): Directory where the analysis is run.
        
        output_dir (str): Directory where output files are stored.
        
        logger (Logger): Logger for logging messages.
        
        db_path (str): Path to the SQLite database.
        
        conn (sqlite3.Connection): SQLite connection object.
        
        n_clusters (int): Number of clusters to form.
        
        random_state (int): Random state for reproducibility.
        
        batch_size (int): Size of batches for processing data.
        
        feature_columns (list): List of feature columns to use for clustering.
        
        columns_to_scale (list): List of columns to scale before clustering.
        
        mbk (MiniBatchKMeans): MiniBatchKMeans model for clustering.
        
        scaler (StandardScaler): StandardScaler for scaling features.
    """
    
    def __init__(self, rundir, n_clusters, random_state, batch_size, 
                 columns_to_scale, feature_columns, weights):
        self.rundir = rundir
        self.output_dir = os.path.join(rundir, 'volumes') # Writes to the volumes directory.
        
        self.db_path = os.path.join(rundir, 'volumes', 'volumes.db')
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.batch_size = batch_size
        self.feature_columns = feature_columns
        self.columns_to_scale = columns_to_scale
        self.weights = weights

        # These are created in __enter__
        self.logger = None
        self.conn = None

        # Pointers to MiniBatchKMeans and StandardScaler created during fitting
        self.mbk = None
        self.scaler = None
        

    def __enter__(self):
        self.logger = setup_logger("cluster", self.output_dir)
        self.logger.info(f"Initialized ClusterAnalysis with config: {self.get_params_dict()}")
        self.conn = sqlite3.connect(self.db_path)
        self.logger.info(f"Database connection established.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.")
        if self.logger:
            close_logger(self.logger)

    def fit(self):
        """ 
        This function reads data from the database in chunks, scales the relevant columns,
        and fits the MiniBatchKMeans model incrementally.
        """
        self.logger.info("Fitting MiniBatchKMeans model...")
        query = "SELECT * FROM volumes ORDER BY RANDOM() LIMIT 1000000;"

        # Fit scaler on a sample (first chunk)
        self.fit_scaler()
        
        # Initialize MiniBatchKMeans with the number of clusters and random state
        self.mbk = MiniBatchKMeans(n_clusters=self.n_clusters, 
                                   random_state=self.random_state, 
                                   batch_size=self.batch_size)

        # Fit the MiniBatchKMeans model incrementally using the data from the database
        for df_chunk in pd.read_sql_query(query, self.conn, chunksize=self.batch_size):
            X = self.get_scaled_features(df_chunk)
            self.mbk.partial_fit(X)
    
    def fit_scaler(self):
        """
        Fit the StandardScaler on the DataFrame chunk.
        This is used to scale the features before clustering.
        """
        self.logger.info("Fitting StandardScaler...")
        query = "SELECT * FROM volumes ORDER BY RANDOM() LIMIT 1000000;"
        random_chunk_df = pd.read_sql_query(query, self.conn, chunksize=self.batch_size).__next__()
        X = random_chunk_df[self.feature_columns]
        self.scaler = StandardScaler().fit(X[self.columns_to_scale])
        return self.scaler
    
    def get_scaled_features(self, df_chunk):
        """
        Scale the features in the DataFrame chunk using the fitted scaler and weights.
        """
        X = df_chunk[self.feature_columns].copy()
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet.")
        X[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        return X[self.columns_to_scale] * np.array([self.weights[col] for col in self.columns_to_scale])
    
    
    def write_to_database(self):
        """
        Assign labels to the entire dataset and compute distances to cluster centers.
        This function updates the database with cluster labels and distances.
        """
        self.logger.info("Writing cluster labels and distances to database...")
        query = "SELECT * FROM volumes;"
        cursor = self.conn.cursor()

        for df_chunk in pd.read_sql_query(query, self.conn, chunksize=self.batch_size):
            # Scale the relevant columns
            X = self.get_scaled_features(df_chunk)

            clusters = self.mbk.predict(X)
            centers = self.mbk.cluster_centers_

            # Compute distances to assigned cluster center
            distances = np.linalg.norm(X - centers[clusters], axis=1)

            # Write the labels and distances to database
            for rowid, cluster, distance in zip(df_chunk['id'], clusters, distances):
                cursor.execute("UPDATE volumes SET cluster = ?, cluster_center_dist = ?, is_representative = ? WHERE id = ?",
                            (int(cluster), float(distance), False, int(rowid)))
            self.conn.commit()

    def find_representatives(self):
        """
        Find representative volumes for each cluster by minimizing distance to the 
        cluster center and write result to database (column: is_representative).
        """
        self.logger.info("Finding representatives for each cluster...")
        representatives = {}
        query = "SELECT id, cluster, cluster_center_dist FROM volumes"
        for df_chunk in pd.read_sql_query(query, self.conn, chunksize=self.batch_size):
            for _, row in df_chunk.iterrows():
                cluster = row['cluster']
                distance = row['cluster_center_dist']
                id = row['id']
                if cluster not in representatives:
                    representatives[cluster] = (distance, id)
                else:
                    if distance < representatives[cluster][0]:
                        representatives[cluster] = (distance, id)
        
        # write representatives to database
        cursor = self.conn.cursor()
        for cluster, (distance, id) in representatives.items():
            cursor.execute("UPDATE volumes SET is_representative = ? WHERE id = ?", (True, int(id)))
        self.conn.commit()
        
    def get_params_dict(self):
        """
        Return all relevant parameters as a dictionary.
        """
        return {
            "rundir": self.rundir,
            "output_dir": self.output_dir,
            "db_path": self.db_path,
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "batch_size": self.batch_size,
            "feature_columns": self.feature_columns,
            "columns_to_scale": self.columns_to_scale,
            "weights": self.weights,
        }

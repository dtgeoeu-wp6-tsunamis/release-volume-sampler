import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

from rvsampler.database_handler import VolumeDatabaseHandler
from rvsampler.utils import create_dir
from rvsampler.set_logg import setup_logger

class ProbabilityAggregator:
    """ Class to aggregate probabilities for clustered release volumes. """
    def __init__(self, rundir):
        self.rundir = rundir
        self.output_dir = os.path.join(rundir, 'aggregation')
        create_dir(self.output_dir, logger=None, clear=True)
        self.logger = setup_logger("probability_aggregator", self.output_dir)

    def get_cluster_release_probability(self, cluster_id, p_shake_cols, db_handler, condprob_column="condprob"):
        """
        Compute the probability that at least one volume in the given cluster is released,
        for all p_shake_* columns at once.
        Returns a numpy array of probabilities (one per p_shake column).
        """
        # Fetch seed_triangles and condprob for this cluster
        query = f"SELECT seed_triangles, {condprob_column} FROM volumes WHERE cluster = ?"
        db_handler.cursor.execute(query, (cluster_id,))
        rows = db_handler.cursor.fetchall()
        if not rows:
            self.logger.warning(f"No volumes found for cluster {cluster_id}.")
            return np.zeros(len(p_shake_cols))

        df = pd.DataFrame(rows, columns=["seed_triangles", condprob_column])
        df["seed_triangles_tuple"] = df["seed_triangles"].apply(
            lambda val: tuple(ast.literal_eval(val)) if isinstance(val, str) else tuple(val)
        )
        condprob_sum = df.groupby("seed_triangles_tuple")[condprob_column].sum().reset_index()

        # Collect all triangle_ids needed
        triangle_ids = set()
        for tup in condprob_sum["seed_triangles_tuple"]:
            triangle_ids.update(tup)
        if not triangle_ids:
            return np.zeros(len(p_shake_cols))

        # Fetch all p_shake columns for these triangles in one query
        placeholders = ",".join("?" for _ in triangle_ids)
        col_str = ", ".join([f'"{col}"' for col in p_shake_cols])
        query = f"SELECT triangle_id, {col_str} FROM seed_triangles WHERE triangle_id IN ({placeholders})"
        db_handler.cursor.execute(query, tuple(triangle_ids))
        shake_rows = db_handler.cursor.fetchall()
        # Build a DataFrame: index=triangle_id, columns=p_shake_cols
        shake_df = pd.DataFrame(shake_rows, columns=["triangle_id"] + p_shake_cols).set_index("triangle_id")

        # For each seed_triangles_tuple, compute the product of p_shake values for all columns
        # Result: shape (n_tuples, n_cols)
        tuple_arr = np.zeros((len(condprob_sum), len(p_shake_cols)))
        for i, tup in enumerate(condprob_sum["seed_triangles_tuple"]):
            vals = shake_df.loc[list(tup)].values  # shape (len(tup), n_cols)
            tuple_arr[i, :] = np.prod(vals, axis=0)

        # Multiply by condprob_sum and compute 1 - prod(1 - p_row) for each column
        p_row = condprob_sum[condprob_column].values[:, None] * tuple_arr  # shape (n_tuples, n_cols)
        # For each column: 1 - prod(1 - p_row[:, col])
        probs = 1 - np.prod(1 - p_row, axis=0)
        return probs

    def compute_cluster_release_probabilities(self, p_shake_cols="p_shake_", condprob_column="condprob"):
        """
        Compute cluster release probabilities for all clusters and all p_shake_* columns (vectorized).
        Returns a DataFrame with clusters as index and p_shake_* columns as columns.
        """
        with VolumeDatabaseHandler(self.rundir) as db_handler:
            res = db_handler.cursor.execute("PRAGMA table_info(seed_triangles)")
            p_shake_cols = [col[1] for col in res.fetchall() if col[1].startswith('p_shake_')]

            # Get all clusters
            clusters = pd.read_sql_query("SELECT DISTINCT cluster FROM volumes", db_handler.conn)['cluster'].sort_values().tolist()
            self.logger.info(f"Computing cluster release probabilities for {len(clusters)} clusters and {len(p_shake_cols)} p_shake columns.")

            # Prepare result DataFrame
            result = pd.DataFrame(index=clusters, columns=p_shake_cols, dtype=float)
            for cluster in clusters:
                self.logger.info(f"Computing probabilities for cluster {cluster}")
                probs = self.get_cluster_release_probability(cluster, p_shake_cols, db_handler, condprob_column=condprob_column)
                result.loc[cluster, :] = probs

        # Sort columns numerically from p_shake_0 to p_shake_120
        def p_shake_key(col):
            try:
                return int(col.split('_')[-1])
            except Exception:
                return float('inf')

        sorted_cols = sorted([col for col in result.columns if col.startswith('p_shake')], key=p_shake_key)
        result = result[sorted_cols]

        # write result to file.
        np.savez(os.path.join(self.output_dir, "cluster_release_probabilities.npz"), 
                 probabilities=result.values, 
                 clusters=result.index,
                 scenarios=result.columns)
        self.logger.info(f"Cluster release probabilities saved to {self.output_dir}.")
    
    def plot_cluster_probability_heatmap(self, save_fig=False):
        # load cluster_release_probabilities
        result = np.load(os.path.join(self.output_dir, "cluster_release_probabilities.npz"), allow_pickle=True)

        # Create a heatmap of the probabilities
        fig, ax = plt.subplots(figsize=(10, 6))
        log_result = np.log10(result['probabilities'] + 1e-20)
        im = ax.imshow(log_result, aspect='auto', cmap='coolwarm', vmin=-3, vmax=0)
        fig.colorbar(im, label='log10(Probability)')

        # Set ticks every 10 clusters (rows)
        yticks = np.arange(0, len(result['clusters']), 10)
        ax.set_yticks(ticks=yticks)
        ax.set_yticklabels([result['clusters'][i] for i in yticks])

        # Set ticks every 10 shakemaps (columns)
        xticks = np.arange(0, len(result['scenarios']), 10)
        ax.set_xticks(ticks=xticks)
        ax.set_xticklabels([result['scenarios'][i] for i in xticks], rotation=90)

        ax.set_title("Cluster probabilities (log10 scale)")
        fig.tight_layout()
        if save_fig:
            fig.savefig(os.path.join(os.path.join(self.rundir, "aggregation"), "cluster_release_probabilities.png"))
            
        return fig
    
    def completed(self):
        self.logger.info("Cluster release probabilities computation completed.")   
        with open(os.path.join(self.output_dir, "completed"), "w") as f:
            pass
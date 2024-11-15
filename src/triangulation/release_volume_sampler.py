import meshio
import numpy as np
from pyproj import Transformer
import rasterio
import matplotlib.pyplot as plt
import os
import logging

from itertools import chain, combinations, product
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("release volume sampler")


class TriangularMeshAnalysis:
    def __init__(self, mesh_path, cumprob_logfos_path, utm_epsg_code, fos_threshold, output_dir):
        self.utm_epsg_code = utm_epsg_code
        self.cumprob_logfos_path = cumprob_logfos_path
        self.fos_threshold = fos_threshold
        self.output_dir = output_dir
        
        logger.info(f"Load mesh: {mesh_path}")
        self.mesh = meshio.read(mesh_path)
        self.elevation = self.mesh.point_data["Elevation"]
        self.neighbours = self.mesh.cell_data["neighbours"][0]
        self.is_interior = self.mesh.cell_data["is_interior"][0] == 1
        self.triangles = self.mesh.cells_dict["triangle"]
        self.n_triangles = self.triangles.shape[0]
        
        logger.info("Calculate vertice locations (easting, northing) using projection EPSG:{self.utm_epsg_code}.")
        self.transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg_code}", always_xy=True)
        self.easting, self.northing = self.transformer.transform(
            self.mesh.points[:, 0], self.mesh.points[:, 1]
        )
        
        #self.points = np.vstack([self.easting, self.northing, self.elevation]).T
        logger.info(" Calculating triangle normals, sides and areas.") # Has to be executed in correct order.
        self.normals, self.sides, self.areas = self._compute_triangle_normals_and_sides()
        logger.info(" Compute boundary normals and gradients.") 
        self.side_normals = self._compute_side_normals()
        self.grads = self.calculate_boundary_gradients()
        
        
        # Load lookuptable
        logger.info(f"Load cumulative probabilities: {self.cumprob_logfos_path}.")
        cumprob_logfos_npz = np.load(self.cumprob_logfos_path)
        self.cumprob_thresholds, self.cumprob_logfos = cumprob_logfos_npz["thresholds"], cumprob_logfos_npz["cummulative_probs"]
        
        # Load triangulation mask.
        #logger.info(f"Load triangulation mask: {self.tri_mask_path}")
        #with rasterio.open(self.tri_mask_path) as src:
        #    self.tri_mask = src.read(1)
        

    def _compute_triangle_normals_and_sides(self):
        """Compute triangle normals (in 3D), side lengths and area.
        """
        points = np.vstack([self.easting, self.northing, self.elevation]).T
        
        p1 = points[self.triangles][:, 0, :]
        p2 = points[self.triangles][:, 1, :]
        p3 = points[self.triangles][:, 2, :]
        
        cross_products = np.linalg.cross(p2 - p1, p3 - p1)
        areas = 0.5 * np.linalg.norm(cross_products, axis=1) 
        normals = cross_products/cross_products[:, 2].reshape(-1, 1) # scale so that n3 = 1. (Easier to calculate gradient)
        
        # Compute side lengths (i-th side is opposite of i-th vertice).
        sides = np.vstack([np.linalg.norm(s, axis=1) for s in [p3 - p2, p1 - p3, p2 - p1]]).T
        
        return normals, sides, areas

    
    def _compute_side_normals(self):
        """Returns outward pointing side normals (2D) of counterclockwise oriented triangles.
        """
        points = np.vstack([self.easting, self.northing]).T

        p1 = points[self.triangles][:,0,:]
        p2 = points[self.triangles][:,1,:]
        p3 = points[self.triangles][:,2,:]

        # counterclockwise side vector ri opposite of i'th vertice
        r3, r1, r2 = p2 - p1, p3 - p2, p1 - p3

        # Normalize
        r1 = np.divide(r1, np.sqrt(np.sum(r1**2, axis=1)).reshape((-1,1)))
        r2 = np.divide(r2, np.sqrt(np.sum(r2**2, axis=1)).reshape((-1,1)))
        r3 = np.divide(r3, np.sqrt(np.sum(r3**2, axis=1)).reshape((-1,1)))
        
        R = np.array([[0, -1],[1, 0]])

        # Side normals (si pointing towards i'th neighbour)
        s1 = np.matmul(R, r1.T).T
        s2 = np.matmul(R, r2.T).T
        s3 = np.matmul(R, r3.T).T
        
        return np.stack([s1, s2, s3])


    def calculate_boundary_gradients(self):
        # Directional gradients across boundary
        return np.sum(self.normals[:, :-1] * self.side_normals, axis=2).T


    def find_seed_triangles(self):
        # Identify triangles below release threshold
        seed_triangles = np.arange(self.n_triangles)[self.triangle_fos < self.fos_threshold]
        return seed_triangles


    def probability_of_release(self, triangles, released_volume):
        # Determine if a triangle will be released based on its FOS and its neighbors
        probs = []
        for triangle in triangles:
            neighbors_in_release = [t in released_volume for t in self.neighbours[triangle]]
            delta = self.sides[triangle][neighbors_in_release].sum() / self.sides[triangle].sum()
            probability_of_release = self.get_cummulative_logfos(triangle, np.log(self.fos_threshold) - np.log(1-delta))
            probs.append(probability_of_release)
            #probs.append(self.triangle_fos[triangle] < self.fos_threshold/(1 - delta))
        return np.array(probs)


    def get_cummulative_logfos(self, triangle, threshold):
        return np.interp(threshold, xp = self.cumprob_thresholds, fp = self.cumprob_logfos[triangle,:], left=0., right=1.)


    def get_upstream_triangles(self, released_triangle):
        # Get upstream triangles relative to a released triangle
        upstream_triangles = self.neighbours[released_triangle, self.grads[released_triangle, :] > 0]
        released_is_downstream = self.grads[upstream_triangles][self.neighbours[upstream_triangles] == released_triangle] < 0
        released_is_interior = self.is_interior[upstream_triangles]
        return upstream_triangles[released_is_downstream & released_is_interior].astype(int)

   
    def run(self):
        #seed_triangles = self.find_seed_triangles()
        #logger.info(f"Found {len(seed_triangles)} seed triangles.")
        #seed_triangle = seed_triangles[3]
        
        # Initiate recursion for the guven seed.
        seed_triangle = 4851
        release = Release(triangulation=self, released=[seed_triangle], released_at_step = [0], probability = 1., step = 1)

        outdict = {"cumprob" : 0.}
        release.write_release(outdict)
        cumprob =  outdict["cumprob"]
        logger.info(f"Seed: {seed_triangle}, Cummulative probability: {cumprob}")

    def write_volume_to_file(self, tri_mask_path, volume_path, triangle_indices):
        # Write binary mask of release volume to file
        logger.info(f"Write volume to file: {volume_path}")
        with rasterio.open(tri_mask_path) as src:
            tri_mask = src.read(1)
            profile = src.profile
        
        volume_mask = np.isin(tri_mask, triangle_indices).astype(np.uint8)
        profile.update(dtype=rasterio.uint8, count=1)
        
        with rasterio.open(volume_path, 'w', **profile) as dst:
            dst.write(volume_mask, 1)


    def plot_fos_distribution(self):
        # Plot FOS distribution histogram
        plt.hist(self.triangle_fos, range=(1, 20), bins=40)
        plt.xlabel("Factor of Safety")
        plt.ylabel("Frequency")
        plt.title("Triangle Factor of Safety Distribution")
        plt.show()

        
    def plot_grad_distribution(self, grads):
        # Plot directional gradient histogram
        plt.hist(grads.flatten(), range=(-0.5, 0.5), bins=30)
        plt.xlabel("Directional Gradient")
        plt.ylabel("Frequency")
        plt.title("Directional Gradient Distribution")
        plt.show()
    
    @staticmethod
    def _read_tif(fname):
        "Read .tif data and profile using rasterio."
        #logger.info(f"Read file: {fname}")
        with rasterio.open(fname) as src:
            #data = np.ma.masked_equal(src.read(1), src.nodata)
            data = src.read(1)
            msk = np.where(src.read_masks(1) == src.nodata, False, True)
            profile = src.profile.copy()
        return data, msk, profile


def main():
    # Usage example
    mesh_path = "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/triangulation.vtk"
    fos_path = "/home/ebr/projects/release-volume-sampler/generated/messina_001/fos/quantiles/fos_quantiles_2.tif" # 0.5 quantile
    cumprob_logfos_path = "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/cummulative_fos.npz"
    tri_mask_path = "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/triangulation_raster.tif"
    utm_epsg_code = 32633 # Messina strait
    fos_threshold = 1.4
    output_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/volumes"
    
    # Execute analysis.
    analysis = TriangularMeshAnalysis(mesh_path, cumprob_logfos_path, utm_epsg_code, fos_threshold, output_dir)
    analysis.run()


class Release():
    def __init__(self, triangulation, released, released_at_step, probability, step):
        self.triangulation = triangulation
        self.released = released.copy()                           # List of previously released triangles
        self.released_at_step = released_at_step.copy()           # List containing the step at which each triangle was released.
        self.probability = probability   # Probability of allready released volume.
        self.step = step
        self.children = []
        
        #logger.info(f"New Release: released: {released}, probability: {probability}, step: {step}, released_at_step: {released_at_step}")
        # Find upstream triangles of the ones released at previous step and calculate probability of release.
        for triangle, released_at_step in zip(self.released, self.released_at_step):
            if released_at_step == self.step-1:
                upstream_triangles = self.triangulation.get_upstream_triangles(triangle)
                indep_prob = self.triangulation.probability_of_release(upstream_triangles, self.released) # Independent release probabilities
                total_prob = 0
                for sub in self._subsets(upstream_triangles): 
                    new_release = upstream_triangles[sub]
                    probability_of_new_release = np.concat([(1-indep_prob)[~sub], indep_prob[sub]]).prod()
                    total_prob += probability_of_new_release
                    released = self.released.copy()
                    released.extend((new_release.tolist()))
                    released_at_step = self.released_at_step.copy()
                    released_at_step.extend([self.step for i in new_release])
                    #logger.info(released)
                    self.children.append(
                        Release(
                            triangulation = self.triangulation, 
                            released = released,
                            released_at_step = released_at_step,
                            probability = self.probability*probability_of_new_release,
                            step = self.step + 1)
                        )
                print(total_prob)

    def write_release(self, outdict):
        """Pass recursively throug set of releases.
        """
        if len(self.children) == 0:
            outdict["cumprob"] += self.probability
            logger.info(f"End node (New volume). Released triangles: {self.released}, Probability: {self.probability}")
        else:
            for children in self.children:
                children.write_release(outdict)

    
    @staticmethod
    def _subsets(upstream):
        """
        Generates all combinations of subsets for the given list.
        Each subset is represented as a list of the same length as upstream,
        where 1 indicates inclusion and 0 indicates exclusion.
        """
        n = len(upstream)
        # Create all binary masks of length n
        for mask in product([False, True], repeat=n):
            yield np.array(mask).astype(bool)


if __name__ == "__main__":
    main()
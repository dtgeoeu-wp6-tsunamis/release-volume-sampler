import meshio
import numpy as np
from pyproj import Transformer
import rasterio
import matplotlib.pyplot as plt
import os
import logging
import json

from itertools import product
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("release volume sampler")


def main():
    # Usage example
    config = {
        "mesh_path": "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/triangulation.vtk",
        "cumprob_logfos_path": "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/cummulative_fos.npz",
        "output_dir": "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/volumes",
        "utm_epsg_code": 32633, # Messina strait
        "fos_threshold": 1.1,
        "recursive_probability_threshold": 0.01,
        "seed_triangle_probability_threshold": 0.1,
    }
    # Execute analysis.
    analysis = RecursiveReleaseAnalysis(**config)


class RecursiveReleaseAnalysis:
    """
    This class implements a probabalistic propagation algorithm for the selection of potential release volumes. The algorithm proceeds 
    along the following steps:
        
        1. Seed triangles are selected based on the condition: P(fos < fos_threshold) > seed_triangle_probability_threshold.
        2. Recursion is initiated for each seed triangle in a stepwise proceedure. For each triangle released in previous step, 
            the probability that upstream triangles will be released are calculated (indipendently):
            - Find the reduction factor (delta) based on the ratio of their perimeter in contact with the released volume by the ratio of the 
            total perimeter (P) of the upstream triangle and the perimeter of the upstream triangle in contact with the released volume (Pc). 
            I.e., delta = Pc/P.
            - The probability that an upstream triangle is released is P(fos < fos_threshold*(1-delta)).
            - Total probabilities are calculated by assuming the release probabilities as independent (This is a considerable simplification).
        3. Recursion is terminated if the probability of a given volume is below the recursive_probability_threshold or no triangles where released in the previous step.
   
    Attributes:
        utm_epsg_code (int): The EPSG code for the UTM projection used in calculating the areas and sides of triangles.
        
        cumprob_logfos_path (str): The file path to the lookup table for cumulative probabilities of log(FOS).
        
        fos_threshold (float): The threshold factor of safety (FOS) value used to assign probabilities to released upstream triangles.
        
        output_dir (str): The directory where output results will be stored.
        
        recursive_probability_threshold (float): The probability threshold below which recursive propagation is truncated.
        
        seed_triangle_probability_threshold (float): The threshold probability used to filter seed triangles. 
            A triangle is considered a seed if P(FOS < fos_threshold) exceeds this value.
    
    """
    def __init__(self, mesh_path, cumprob_logfos_path, utm_epsg_code, fos_threshold, output_dir, recursive_probability_threshold, seed_triangle_probability_threshold):
        self.utm_epsg_code = utm_epsg_code             # Projection used for calculation of areas and sides of triangles.
        self.cumprob_logfos_path = cumprob_logfos_path # Path to lookuptable for cumprob of logfos.
        self.fos_threshold = fos_threshold             # To assign probability of released upstream triangles.
        self.output_dir = output_dir
        Release.recursive_probability_threshold = recursive_probability_threshold # Recursive propagation truncated if probabiliy is below.
        self.seed_triangle_probability_threshold = seed_triangle_probability_threshold # Filtration of seed triangles: P(fos < fos_threshold) > threshold
        
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
        
        logger.info(" Calculating triangle normals, sides and areas.") # Has to be executed in correct order.
        self.normals, self.sides, self.areas = self._compute_triangle_normals_and_sides()
        logger.info(" Compute boundary normals and gradients.") 
        self.side_normals = self._compute_side_normals()
        self.grads = self.calculate_boundary_gradients()
        
        # Load lookuptable
        logger.info(f"Load cumulative probabilities: {self.cumprob_logfos_path}.")
        cumprob_logfos_npz = np.load(self.cumprob_logfos_path)
        self.cumprob_thresholds, self.cumprob_logfos = cumprob_logfos_npz["thresholds"], cumprob_logfos_npz["cummulative_probs"]
        
        # Run analysis
        self.run()


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


    def probability_of_release(self, triangles, released_volume):
        # Determine if a triangle will be released based on its FOS and its neighbors
        probs = []
        for triangle in triangles:
            neighbors_in_release = [t in released_volume for t in self.neighbours[triangle]]
            delta = self.sides[triangle][neighbors_in_release].sum() / self.sides[triangle].sum()
            probability_of_release = self.get_cummulative_logfos(triangle, np.log10(self.fos_threshold) - np.log10(1-delta))
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
        #seed_triangles, seed_probabilities = self.find_seed_triangles()
        # Identify triangles below release threshold
        all_triangles = np.arange(self.n_triangles-1)
        seed_probability = np.array([self.get_cummulative_logfos(triangle, np.log10(self.fos_threshold)) for triangle in all_triangles])
        seed_triangles = all_triangles[np.logical_and(seed_probability > self.seed_triangle_probability_threshold, seed_probability != 9999.0)]
        logger.info(f"Found {len(seed_triangles)} seed triangles.")
        #seed_triangle = seed_triangles[3]
        
        recursive_propagations = []
        for seed_triangle in seed_triangles:
            
            volumes = []
            # Initiate recursion for the given seed.
            release = Release(triangulation=self, released=[int(seed_triangle)], released_at_step = [0], probability = 1., step = 1)
            
            # traverse the released volumes.
            release.write_release(volumes)
            for volume in volumes:
                volume["area"] = self.areas[volume["released"]].sum()
                # volume["normals"] = self.normals[volume["released"]].tolist()
            
            recursive_propagations.append({
                "seed_triangle": int(seed_triangle),
                "seed_triangle_probability": seed_probability[seed_triangle],
                "volumes": volumes
            })
        
        # Write to file.
        with open(os.path.join(self.output_dir, 'recursive_propagations.json'), 'w') as f:
            json.dump(recursive_propagations, f, indent=4)


class Release():
    """
    Class representing a state of the release process. Recursion is initiated upon the creation of an instance.
    
    Class Attributes:
        recursive_probability_threshold (float): Probability threshold above which progression is discontinued.
    
    Instance Attributes:

        triangulation (RecursiveReleaseAnalysis): The current Recursive analysis.
        released (list[int]): Previously released triangles.
        released_at_step (list[int]) Step in the recursion at which the coresponding triangle was released.
        probability (float): Probability of the current state.
        step (int): Current step of the release process.
        children (list[Release]): Progressing release states.
    
    """
    recursive_probability_threshold = None # Stop recursion if probability of release is smaller.
    
    def __init__(self, triangulation, released, released_at_step, probability, step):
        self.triangulation = triangulation
        self.released = released                           # List of previously released triangles
        self.released_at_step = released_at_step           # List containing the step at which each triangle was released.
        self.probability = probability                            # Probability of released volume.
        self.step = step
        self.children = []
        
        if self.probability > self.recursive_probability_threshold:
            # Find upstream triangles of the ones released at previous step and calculate probability of release.
            upstream_triangles = set() # Define as set to ensure every triangle occurs only once..
            for triangle, released_at_step in zip(self.released, self.released_at_step):
                if released_at_step == self.step-1:
                    # Triangle was released at previous step. Find upstream...
                    upstream_triangles = upstream_triangles.union(set(self.triangulation.get_upstream_triangles(triangle)))
            
            if len(upstream_triangles) > 0:
                upstream_triangles = np.array(list(upstream_triangles))
                indep_prob = self.triangulation.probability_of_release(upstream_triangles, self.released) # Independent release probabilities
                for sub in self._subsets(upstream_triangles): 
                    new_release = upstream_triangles[sub]
                    probability_of_new_release = np.concat([(1-indep_prob)[~sub], indep_prob[sub]]).prod()
                    
                    # Initiate new release.
                    released = self.released.copy()
                    released.extend((new_release.tolist()))
                    released_at_step = self.released_at_step.copy()
                    released_at_step.extend([self.step for i in new_release])
                    self.children.append(
                        Release(
                            triangulation = self.triangulation, 
                            released = released,
                            released_at_step = released_at_step,
                            probability = self.probability*probability_of_new_release,
                            step = self.step + 1)
                        )


    def write_release(self, volumes):
        """Pass recursively throug set of releases and append terminated processes.
        """
        if len(self.children) == 0:
            new_volume = {
                "released": self.released, 
                "probability": self.probability, 
                "steps": self.released_at_step,
            }
            volumes.append(new_volume)
            logger.info(f"Terminated release. Released: {self.released}, Probability: {self.probability}, Steps: {self.released_at_step}")
        else:
            for children in self.children:
                children.write_release(volumes)

    
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
import meshio
import numpy as np
from pyproj import Transformer
import os
from itertools import product

from rvsampler.utils import create_dir
from rvsampler.set_logg import setup_logger
from rvsampler.database_handler import VolumeDatabaseHandler
from rvsampler.triangulate import Triangulation

import math
from math import comb  
import itertools
from collections import defaultdict
import random
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor


def main():
    """ To ensure that module imports work, 
    ../src/rvsampler$ python release_volume_sampler.py
    """
    # Usage example
    config = {
        "rundir": "/home/ebr/projects/release-volume-sampler/generated/messina_001",
        "mesh_path": "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/triangulation.vtk",
        "cumprob_logfos_path": "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/cumulative_fos.npz",
        "utm_epsg_code": 32633, # Messina strait
    }
    
    run_config = {
        "fos_threshold": 1.5,
        "recursive_probability_threshold": 0.01,
        "seed_triangle_probability_threshold": 1e-3,
        "max_n_seed_triangles": 100,
    }
    # Execute analysis.
    analysis = RecursiveReleaseAnalysis(**config)
    analysis.run(**run_config)


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
        
        mesh_path (str): The file path to the mesh (triangulation.vtk) used for the analysis.
        
        recursive_probability_threshold (float): The probability threshold below which recursive propagation is truncated.
        
        seed_triangle_probability_threshold (float): The threshold probability used to filter seed triangles. 
            A triangle is considered a seed if P(FOS < fos_threshold) exceeds this value.
    
    """

    def __init__(self, rundir, mesh_path, cumprob_logfos_path, utm_epsg_code):
        self.rundir = rundir
        self.output_dir = os.path.join(rundir, "volumes")
        create_dir(self.output_dir)
        self.logger = setup_logger("volume_sampler", self.output_dir)
        
        self.utm_epsg_code = utm_epsg_code             # Projection used for calculation of areas and sides of triangles.
        self.cumprob_logfos_path = cumprob_logfos_path # Path to lookuptable for cumprob of logfos.
        self.triangulation = Triangulation(self.rundir)
        
        self.logger.info(" Calculating triangle normals, sides and areas.") # Has to be executed in correct order.
        self.normals, self.sides, self.areas, self.slopes = self.triangulation.get_normals_sides_areas_slopes()
       
        # A0 definert som middel trekant størrelse
        # Litt usikker på om det er en god definisjon at denne endrer seg med n_triangles. Det vil si at hvis man har mange flere 
        # å mindre trinagler så endres ikke sannsyneligheten.
        #self.A0 = np.sum(self.areas)/self.n_triangles
        # Use 200000 as this is approximately the result of the above calculation for the default run of the script.
        self.A0 = 200000
        self.logger.info(" Compute boundary normals and gradients.") 
        self.side_normals = self.triangulation.get_side_normals()
        self.grads = self.calculate_boundary_gradients()
        
        # Load lookuptable
        self.logger.info(f"Load cumulative probabilities: {self.cumprob_logfos_path}.")
        cumprob_logfos_npz = np.load(self.cumprob_logfos_path)
        self.cumprob_thresholds, self.cumprob_logfos = cumprob_logfos_npz["thresholds"], cumprob_logfos_npz["probs"]

    def calculate_boundary_gradients(self):
        # Directional gradients across boundary
        return np.sum(self.normals[:, :-1] * self.side_normals, axis=2).T

    def probability_of_release(self, triangles, released_volume, fos_threshold):
        # Determine if a triangle will be released based on its FOS and its neighbors
        probs = []
        for triangle in triangles:
            neighbors_in_release = [t in released_volume for t in self.triangulation.neighbours[triangle]]
            delta = self.triangulation.sides[triangle][neighbors_in_release].sum() / self.triangulation.sides[triangle].sum()
            probability_of_release = np.exp(np.log(self.get_cumulative_logfos(triangle, np.log10(fos_threshold) 
                                        - np.log10(1-delta)))*(self.triangulation.areas[triangle]/self.A0))
            probs.append(probability_of_release)
        return np.array(probs)

    def get_cumulative_logfos(self, triangle, threshold):
        return np.interp(threshold, xp = self.cumprob_thresholds, fp = self.cumprob_logfos[triangle,:], left=0., right=1.)
    
    def get_initial_states(self, seed_triangles, max_n_slopeunits, use_slopeunits, max_n_neighbours):
        # Use slopeunits file for checking seed triangles that are dependent
        if use_slopeunits:
            # Read slopeunit file
            
            
            # Step 1: Group seed_triangles by their slopeunit
            grouped = defaultdict(list)
            for tri, su in zip(seed_triangles, self.triangulation.slopeunits[seed_triangles]):
                grouped[su].append(tri)
              
            """   This code can be used to asses how many seed triangles that can happen simultaneously, to many will lead to
            an insande amount of combinations
            MAX_R = 4
            def count_limited_combinations(n, max_r):
                return sum(comb(n, r) for r in range(1, min(n, max_r)+1))

            total_combinations = sum(count_limited_combinations(size, MAX_R) for size in group_sizes.values())
            print(f"Total number of combinations (r ≤ {MAX_R}): {total_combinations}")
            """
            
            # Limit number of slopeunit groups
            all_groups = list(grouped.values())
            random_subset = random.sample(all_groups, min(max_n_slopeunits, len(all_groups)))
            
            # Step 2: For each group, generate combinations
            MAX_R = max_n_neighbours  # max combination size (adjust as needed)

            all_combinations = []
            for vals in random_subset:
                if len(vals) == 0:
                    continue
                for r in range(1, min(MAX_R+1, len(vals)+1)):
                    all_combinations.extend(itertools.combinations(vals, r))
                    
            initial_states = [list(map(int, combo)) for combo in all_combinations]

        else:
            # Convert seed_triangles to a list
            seed_triangles = np.array(list(seed_triangles))
            
            # Also calculate seed triangle pairs
            st_pairs = set()
            
            # Coordinates
            points = np.vstack([self.triangulation.easting, self.triangulation.northing]).T
            
            for ist, ss in enumerate(seed_triangles):
                distances = np.linalg.norm(points[self.triangulation.triangles[ss]].mean(axis=0) - points[self.triangulation.triangles[seed_triangles]].mean(axis=1), axis=1)
                
                within_1km_indices = np.where(distances <= 1000)[0]
                
                for i in seed_triangles[within_1km_indices]:
                    other = i
                    if ss != other:  # remove self-pairs
                        pair = tuple(sorted((ss, other)))  # sort to handle (a,b) == (b,a)
                        st_pairs.add(pair)
                        
            initial_states = [[x] for x in seed_triangles]
            # add pairs
            initial_states.extend([list(t) for t in st_pairs])
            initial_states = [[int(x) for x in sublist] for sublist in initial_states]
            
        return initial_states
    
    
    def run(self, seed_triangle_probability_threshold, fos_threshold, max_n_seed_triangles, recursive_probability_threshold, use_slopeunits=True, max_n_slopeunits=5, max_n_neighbours=2, max_workers=4):
        
        # Asssign class variables to Release. 
        Release.fos_threshold = fos_threshold             # To assign probability of released upstream triangles.
        Release.recursive_probability_threshold = recursive_probability_threshold # Recursive propagation truncated if probabiliy is below.
        Release.logger = self.logger
        
        self.logger.info(f"Run volume sampler: \
            seed_triangle_probability: {seed_triangle_probability_threshold}, \
            fos_threshold: {fos_threshold} \
            recursive_probability_threshold: {recursive_probability_threshold}")
        
        # Filtration of seed triangles: P(fos < fos_threshold) > threshold
        all_triangles = np.arange(self.triangulation.n_triangles)
        seed_probability = np.array([self.get_cumulative_logfos(triangle, np.log10(Release.fos_threshold)) for triangle in all_triangles])
        seed_triangles = all_triangles[np.logical_and(seed_probability > seed_triangle_probability_threshold, seed_probability != 9999.0)]

        
        
        self.logger.info(f"Found {len(seed_triangles)} seed triangles.")
        # Limit the number of seed triangles       
        if len(seed_triangles) > max_n_seed_triangles:
            self.logger.warning(
                f"Number of seed triangles ({len(seed_triangles)}) exceeds max_n_seed_triangles ({max_n_seed_triangles}). "
                "Limiting to max_n_seed_triangles."
            )
            seed_triangles = random.sample(list(seed_triangles), k=max_n_seed_triangles)  
        
        # Get the initial states                
        initial_states = self.get_initial_states(seed_triangles, max_n_slopeunits, use_slopeunits, max_n_neighbours)  

        # Run volume generation in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_seed_pair, seed_pair, self.triangulation, self, seed_probability)
                for seed_pair in initial_states
            ]

            for future in concurrent.futures.as_completed(futures):
                all_results.append(future.result())
                
        with VolumeDatabaseHandler(self.rundir) as volumes_db:
            for volumes, seed_pair, prob in all_results:
                self.append_features_to_volumes(volumes, seed_pair, prob)
                for volume in volumes:
                    volumes_db.insert_volume(volume_data=volume)

        self.logger.info(f"Finished triangle initiation. Total: {len(seed_triangles)} initial combinations.")


            
        #with VolumeDatabaseHandler(self.rundir) as volumes_db:
        #    for seed_pairs in initial_states:        
        #        volumes = []
        #        # Initiate recursion for the given seed.                       
        #        release = Release(triangulation=self.triangulation, 
        #                          sampler=self, 
        #                          released=seed_pairs, 
        #                          released_at_step = [0], 
        #                          probability = 1., 
        #                          step = 1)
        #        
        #        # traverse the released volumes.
        #        release.write_release(volumes)
        #        
        #        # Append features
        #        self.append_features_to_volumes(volumes, seed_pairs, float(math.prod(seed_probability[seed_pairs])))
        #        
        #        # Add volumes to database
        #        for volume in volumes:
        #            volumes_db.insert_volume(volume_data=volume)
        #            
        #    self.logger.info(f"Finished triangle initiation. Total: {len(seed_triangles)} initial combinations.")
        
    
    def append_features_to_volumes(self, volumes, seed_triangles, p_fos_seed):
        """
        Append features to the volumes list.
        """
        for volume in volumes:
            volume["seed_triangles"] = seed_triangles
            volume["p_fos_seed"] = p_fos_seed
            volume["area"] = self.triangulation.areas[volume["released"]].sum()
            volume["mean_elevation"] = float(self.triangulation.elevation[np.unique(self.triangulation.triangles[volume["released"]]).flatten()].mean())
            volume["mean_slope"] = self.triangulation.slopes[volume["released"]].mean()
            volume["mean_easting"] = self.triangulation.easting[np.unique(self.triangulation.triangles[volume["released"]]).flatten()].mean()
            volume["mean_northing"] = self.triangulation.northing[np.unique(self.triangulation.triangles[volume["released"]]).flatten()].mean()
            volume["slopeunit"] = self.triangulation.slopeunits[seed_triangles[0]]
        return volumes

def process_seed_pair(seed_pair, triangulation, sampler, seed_probability):
    volumes = []

    release = Release(
        triangulation=triangulation,
        sampler=sampler,
        released=seed_pair,
        released_at_step=[0],
        probability=1.0,
        step=1
    )
    
    release.write_release(volumes)
    
    # Attach seed probability
    prob = float(math.prod(seed_probability[seed_pair]))
    
    return volumes, seed_pair, prob

class Release():
    """
    Class representing a state of the release process. Recursion is initiated upon the creation of an instance.
    
    Class Attributes:
        recursive_probability_threshold (float): Probability threshold above which progression is discontinued.
    
    Instance Attributes:

        triangulation (Triangulation): The current triangulation.
        sampler (RecursiveReleaseAnalysis): The current Recursive analysis.
        released (list[int]): Previously released triangles.
        released_at_step (list[int]) Step in the recursion at which the coresponding triangle was released.
        probability (float): Probability of the current state.
        step (int): Current step of the release process.
        children (list[Release]): Progressing release states.
    
    """
    recursive_probability_threshold = None # Stop recursion if probability of release is smaller.
    fos_threshold = None
    logger = None
    
    def __init__(self, triangulation, sampler, released, released_at_step, probability, step):
        self.triangulation = triangulation
        self.sampler = sampler
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
                indep_prob = self.sampler.probability_of_release(upstream_triangles, self.released, self.fos_threshold) # Independent release probabilities
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
                            sampler= self.sampler, 
                            released = released,
                            released_at_step = released_at_step,
                            probability = self.probability*probability_of_new_release,
                            step = self.step + 1)
                        )

    def write_release(self, volumes):
        """Pass recursively throug set of releases and append terminated processes with nonzero probabilities.
        """
        if len(self.children) == 0 and self.probability > 0.:
            new_volume = {
                "released": self.released, 
                "condprob": self.probability, 
                "steps": self.released_at_step,
            }
            volumes.append(new_volume)
            #self.logger.info(f"Terminated release. Released: {self.released}, Probability: {self.probability}, Steps: {self.released_at_step}")
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

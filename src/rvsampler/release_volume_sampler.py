import meshio
import numpy as np
from pyproj import Transformer
import os
from itertools import product

from rvsampler.utils import create_dir
from rvsampler.set_logg import setup_logger
from rvsampler.database_handler import VolumeDatabaseHandler


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
        
        
        self.logger.info(f"Load mesh: {mesh_path}")
        self.mesh = meshio.read(mesh_path)
        self.elevation = self.mesh.point_data["Elevation"]
        self.neighbours = self.mesh.cell_data["neighbours"][0]
        self.is_interior = self.mesh.cell_data["is_interior"][0] == 1
        self.triangles = self.mesh.cells_dict["triangle"]
        self.n_triangles = self.triangles.shape[0]
        self.logger.info(f"n_triangles: {self.n_triangles}")
        
        self.logger.info("Calculate vertice locations (easting, northing) using projection EPSG:{self.utm_epsg_code}.")
        self.transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.utm_epsg_code}", always_xy=True)
        self.easting, self.northing = self.transformer.transform(
            self.mesh.points[:, 0], self.mesh.points[:, 1]
        )
        
        self.logger.info(" Calculating triangle normals, sides and areas.") # Has to be executed in correct order.
        self.normals, self.sides, self.areas, self.slopes = self._compute_triangle_properties()
        # A0 definert som middel trekant størrelse
        # Litt usikker på om det er en god definisjon at denne endrer seg med n_triangles. Det vil si at hvis man har mange flere 
        # å mindre trinagler så endres ikke sannsyneligheten.
        #self.A0 = np.sum(self.areas)/self.n_triangles
        # Use 200000 as this is approximately the result of the above calculation for the default run of the script.
        self.A0 = 200000
        self.logger.info(" Compute boundary normals and gradients.") 
        self.side_normals = self._compute_side_normals()
        self.grads = self.calculate_boundary_gradients()
        
        # Load lookuptable
        self.logger.info(f"Load cumulative probabilities: {self.cumprob_logfos_path}.")
        cumprob_logfos_npz = np.load(self.cumprob_logfos_path)
        self.cumprob_thresholds, self.cumprob_logfos = cumprob_logfos_npz["thresholds"], cumprob_logfos_npz["probs"]

    def _compute_triangle_properties(self):
        """Compute triangle geometric properties.
        - Normals scaled so that z-axis has lenght 1.
        - slopes in degrees.
        """
        points = np.vstack([self.easting, self.northing, self.elevation]).T
        
        p1 = points[self.triangles][:, 0, :]
        p2 = points[self.triangles][:, 1, :]
        p3 = points[self.triangles][:, 2, :]
        
        cross_products = np.linalg.cross(p2 - p1, p3 - p1)
        areas = 0.5 * np.linalg.norm(cross_products, axis=1) 
        normals = cross_products/cross_products[:, 2].reshape(-1, 1) # scale so that n3 = 1. (Easier to calculate gradient)
        
        normal_lenghts = np.sqrt((normals**2).sum(axis=-1))
        slopes = np.rad2deg(np.arccos(1/normal_lenghts))
        
        # Compute side lengths (i-th side is opposite of i-th vertice).
        sides = np.vstack([np.linalg.norm(s, axis=1) for s in [p3 - p2, p1 - p3, p2 - p1]]).T
        
        return normals, sides, areas, slopes

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

    def probability_of_release(self, triangles, released_volume, fos_threshold):
        # Determine if a triangle will be released based on its FOS and its neighbors
        probs = []
        for triangle in triangles:
            neighbors_in_release = [t in released_volume for t in self.neighbours[triangle]]
            delta = self.sides[triangle][neighbors_in_release].sum() / self.sides[triangle].sum()
            probability_of_release = np.exp(np.log(self.get_cumulative_logfos(triangle, np.log10(fos_threshold) 
                                        - np.log10(1-delta)))*(self.areas[triangle]/self.A0))
            probs.append(probability_of_release)
        return np.array(probs)

    def get_cumulative_logfos(self, triangle, threshold):
        return np.interp(threshold, xp = self.cumprob_thresholds, fp = self.cumprob_logfos[triangle,:], left=0., right=1.)

    def get_upstream_triangles(self, released_triangle):
        # Get upstream triangles relative to a released triangle
        upstream_triangles = self.neighbours[released_triangle, self.grads[released_triangle, :] > 0]
        released_is_downstream = self.grads[upstream_triangles][self.neighbours[upstream_triangles] == released_triangle] < 0
        upstream_is_interior = self.is_interior[upstream_triangles]
        return upstream_triangles[released_is_downstream & upstream_is_interior].astype(int)
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius in kilometers

        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c  # Distance in km
   
    def run(self, seed_triangle_probability_threshold, fos_threshold, recursive_probability_threshold):
        
        # Asssign class variables to Release. 
        Release.fos_threshold = fos_threshold             # To assign probability of released upstream triangles.
        Release.recursive_probability_threshold = recursive_probability_threshold # Recursive propagation truncated if probabiliy is below.
        Release.logger = self.logger
        
        self.logger.info(f"Run volume sampler: \
            seed_triangle_probability: {seed_triangle_probability_threshold}, \
            fos_threshold: {fos_threshold} \
            recursive_probability_threshold: {recursive_probability_threshold}")
        
        # Filtration of seed triangles: P(fos < fos_threshold) > threshold
        all_triangles = np.arange(self.n_triangles)
        seed_probability = np.array([self.get_cumulative_logfos(triangle, np.log10(Release.fos_threshold)) for triangle in all_triangles])
        seed_triangles = all_triangles[np.logical_and(seed_probability > seed_triangle_probability_threshold, seed_probability != 9999.0)]
        #seed_triangles = all_triangles[np.logical_and(seed_probability > 3e-1, seed_probability != 9999.0)]
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #print(len(seed_triangles))
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #trast
        
        self.logger.info(f"Found {len(seed_triangles)} seed triangles.")
        
        # Also calculate seed triangle pairs
        st_pairs = set()
        
        # Coordinates
        points = np.vstack([self.easting, self.northing]).T

        #p1 = points[self.triangles][:,0,:]
        #p2 = points[self.triangles][:,1,:]
        #p3 = points[self.triangles][:,2,:]

        for ist, ss in enumerate(seed_triangles):
            distances = np.linalg.norm(points[self.triangles[ss]].mean(axis=0) - points[self.triangles[seed_triangles]].mean(axis=1), axis=1)
            
            within_1km_indices = np.where(distances <= 1000)[0]
            
            for i in seed_triangles[within_1km_indices]:
                other = i
                if ss != other:  # remove self-pairs
                    pair = tuple(sorted((ss, other)))  # sort to handle (a,b) == (b,a)
                    st_pairs.add(pair)
        
        #st_pairs = set(list(st_pairs)[:50])
        
       
        with VolumeDatabaseHandler(self.rundir) as volumes_db:
            for seed_triangle in seed_triangles:
                
                volumes = []
                # Initiate recursion for the given seed.
                release = Release(triangulation=self, released=[int(seed_triangle)], released_at_step = [0], probability = 1., step = 1)
                
                # traverse the released volumes.
                release.write_release(volumes)
                
                # Append features
                for volume in volumes:
                    volume["area"] = self.areas[volume["released"]].sum()
                    volume["mean_elevation"] = float(self.elevation[self.triangles[volume["released"]].flatten()].mean()) # Elevation is point data.
                    volume["mean_slope"] = self.slopes[volume["released"]].mean()
                    volume["seed_triangle"] = int(seed_triangle)
                    volume["seed_triangle2"] = -1
                    volume["p_fos_seed"] = seed_probability[seed_triangle]
                
                # Add volumes to database
                for volume in volumes:
                    volumes_db.insert_volume(volume_data=volume)
                  
            for pair in st_pairs:
                volumes2 = []
                # Initiate recursion for the given seed.
                pair2 = [int(x) for x in pair] # Convert to list
                release = Release(triangulation=self, released=pair2, released_at_step = [0], probability = 1., step = 1)
                
                # traverse the released volumes.
                release.write_release(volumes2)
                
                # Append features
                for volume in volumes2:
                    volume["area"] = self.areas[volume["released"]].sum()
                    volume["mean_elevation"] = float(self.elevation[self.triangles[volume["released"]].flatten()].mean()) # Elevation is point data.
                    volume["mean_slope"] = self.slopes[volume["released"]].mean()
                    volume["seed_triangle"] = int(pair2[0])#", ".join(str(x) for x in pair2)
                    volume["seed_triangle2"] = int(pair2[1])#", ".join(str(x) for x in pair2)
                    volume["p_fos_seed"] = seed_probability[pair2[0]]*seed_probability[pair2[1]]
                
                # Add volumes to database
                for volume in volumes2:
                    volumes_db.insert_volume(volume_data=volume)


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
    fos_threshold = None
    logger = None
    
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
                indep_prob = self.triangulation.probability_of_release(upstream_triangles, self.released, self.fos_threshold) # Independent release probabilities
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
        """Pass recursively throug set of releases and append terminated processes with nonzero probabilities.
        """
        if len(self.children) == 0 and self.probability > 0.:
            new_volume = {
                "released": self.released, 
                "condprob": self.probability, 
                "steps": self.released_at_step,
            }
            volumes.append(new_volume)
            self.logger.info(f"Terminated release. Released: {self.released}, Probability: {self.probability}, Steps: {self.released_at_step}")
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

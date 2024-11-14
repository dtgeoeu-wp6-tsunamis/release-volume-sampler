import meshio
import numpy as np
from pyproj import Transformer
import rasterio
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger("release volume sampler")


class TriangularMeshAnalysis:
    def __init__(self, mesh_path, fos_path, tri_mask_path, utm_epsg_code, fos_threshold, output_dir):
        self.utm_epsg_code = utm_epsg_code
        self.fos_path = fos_path
        self.tri_mask_path = tri_mask_path
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
        
        logger.info(f"Load triangulation mask: {self.tri_mask_path}")
        with rasterio.open(self.tri_mask_path) as src:
            self.tri_mask = src.read(1)
        
        logger.info(f"Load factor of safety from file (FOS): {self.fos_path}")
        with rasterio.open(self.fos_path) as src:
            logfos = src.read(1)
            self.triangle_fos = np.array([np.nanmin(10**logfos[self.tri_mask == tri_index], initial=9999) for tri_index in range(self.n_triangles)])
        

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


    def _will_be_released(self, triangle, release_volume):
        # Determine if a triangle will be released based on its FOS and its neighbors
        neighbors_in_release = [t in release_volume for t in self.neighbours[triangle]]
        delta = self.sides[triangle][neighbors_in_release].sum() / self.sides[triangle].sum()
        return self.triangle_fos[triangle] * (1 - delta) < self.fos_threshold


    def _get_upstream_triangles(self, released_triangle):
        # Get upstream triangles relative to a released triangle
        upstream_triangles = self.neighbours[released_triangle, self.grads[released_triangle, :] > 0]
        released_is_downstream = self.grads[upstream_triangles][self.neighbours[upstream_triangles] == released_triangle] < 0
        released_is_interior = self.is_interior[upstream_triangles]
        return(upstream_triangles[released_is_downstream & released_is_interior])


    def get_release_volume(self, init_triangle):
        # Recursive release volume calculation
        release_volume = []
        released_triangles = [init_triangle]
        
        while released_triangles:
            released_triangle = released_triangles.pop()
            release_volume.append(int(released_triangle))
            upstream_triangles = self._get_upstream_triangles(released_triangle)
            upstream_is_released = [self._will_be_released(triangle, release_volume) for triangle in upstream_triangles]
            released_triangles.extend(upstream_triangles[upstream_is_released])
        return release_volume

    
    def run(self):
        seed_triangles = self.find_seed_triangles()
        logger.info(f"Found {len(seed_triangles)} seed triangles.")
        
        release_volumes = []
        for seed_triangle in seed_triangles:
            released_triangles = self.get_release_volume(seed_triangle)
            release_volume = {
                "seed_triangle": int(seed_triangle),
                "released_triangles": released_triangles,
                "area": float(np.sum(self.areas[released_triangles]))
            }
            release_volumes.append(release_volume)
            logger.info(release_volume)
            #output_path = os.path.join(self.output_dir, f"volume_{seed_triangle}.tif")
            #self.write_volume_to_file(self.tri_mask_path, output_path, release_volume)


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





def main():
    # Usage example
    mesh_path = "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/triangulation.vtk"
    fos_path = "/home/ebr/projects/release-volume-sampler/generated/messina_001/fos/quantiles/fos_quantiles_2.tif" # 0.5 quantile
    tri_mask_path = "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/triangulation_raster.tif"
    utm_epsg_code = 32633 # Messina strait
    fos_threshold = 1.4
    output_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation/volumes"
    
    # Execute analysis.
    analysis = TriangularMeshAnalysis(mesh_path, fos_path, tri_mask_path, utm_epsg_code, fos_threshold, output_dir)
    analysis.run()


if __name__ == "__main__":
    main()
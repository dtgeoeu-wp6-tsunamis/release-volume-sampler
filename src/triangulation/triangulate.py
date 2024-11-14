import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import rasterio
from scipy.signal import convolve2d
from scipy.ndimage import label
import numpy as np
from rasterio.transform import rowcol
from pyproj import Transformer
import matplotlib.cm as cm
import os
import meshio

class Triangulation:
    
    def __init__(self):
        """ Class to triangulate topography subject to optimization of triangle shape, 
            approximation of the topography and equally sized triangles.
            
            TODO: Several parameters related to the optimization is hardcoded.
        """
        # Load real raster data
        self.output_dir = "/home/ebr/projects/release-volume-sampler/generated/messina_001/triangulation"
        self.bathyfile = "/home/ebr/projects/release-volume-sampler/input/bathy/messina_001/bathy_truncated.tif"
        self.src = rasterio.open(self.bathyfile)
        self.bathy = self.src.read(1)
        self.bathy_msk = np.where(self.src.read_masks(1) == self.src.nodata, False, True)
        self.profile = self.src.profile.copy()
        self.UTM_epsg_code = 32633 #Messina strait

        # Partition bathy into connected components and select.
        self.labeled, num_features = label(self.bathy_msk)
        self.label_component = 2

        # Get the shape of the data (number of rows and columns)
        self.rows, self.cols = self.bathy.shape
       
        # Initial Triangulation 
        self.triang_points, self.triang_point_is_boundary = self.create_points((100, 130))
        self.tri = Delaunay(self.triang_points)
        self.vertices = tf.Variable(self.triang_points, dtype=tf.float32)  # TensorFlow variable for optimization
        
       
        #self.triangles = self.tri.simplices  # Indices of vertices for each triangle
        #self.triangle_is_interior = np.array([not all(self.triang_point_is_boundary[t]) for t in self.triangles])
        #self.neighbors = self.tri.neighbors
        
        # Points for evaluation of topography.
        all_eval_points, eval_point_is_boundary = self.create_points(dims = (400, 300))
        self.eval_points  = tf.cast(all_eval_points[eval_point_is_boundary == 0], tf.float32) # remove boundary points
        self.true_elevations = tf.cast(self.target_elevation_function(self.eval_points), tf.float32)


    def triangle_is_interior(self):
        return(np.array([not all(self.triang_point_is_boundary[t]) for t in self.tri.simplices]))


    def target_elevation_function(self, points):
        eastings, northings = points[:, 0], points[:, 1]
        
        # Create a transformer to convert UTM to geographic coordinates (lon, lat)
        transformer = Transformer.from_crs(f"EPSG:{self.UTM_epsg_code}", self.src.crs, always_xy=True)  # Replace 32633 with your UTM zone
        lons, lats = transformer.transform(eastings, northings)
        rows, cols = rowcol(self.src.transform, lons, lats)
        
        # Clamp row and col to ensure they are within the raster bounds
        rows = np.clip(rows, 0, self.bathy.shape[0] - 1)
        cols = np.clip(cols, 0, self.bathy.shape[1] - 1)
        
        return tf.cast(np.nan_to_num(self.bathy[rows, cols]), tf.float32)  # Synthetic elevation function
    
    
    def get_3d_vertices(self, tri_indices): 
        vertice_elevations = tf.reshape(self.target_elevation_function(self.vertices),[-1,1])
        vertices_3d = tf.concat([self.vertices, vertice_elevations], axis=1)
        
        # Gather all vertices for the triangles containing the points
        p1_p2_p3 = tf.gather(vertices_3d, tri_indices)  # Get the triangle's vertices for the valid points
        p1, p2, p3 = p1_p2_p3[:,0,:], p1_p2_p3[:,1,:], p1_p2_p3[:,2,:]
        return(p1, p2, p3)

 
    def evaluate_triangulated_elevation(self, points):
        """
        Evaluates the elevation of points based on the plane defined by the containing triangle's vertices.
        Vectorized to avoid loops for efficiency.
        """
        
        # Step 1: Find which triangle contains each point
        # Assumption: points are 2D, so we need to add them to the 3D triangulation.
        points = tf.convert_to_tensor(points, dtype=tf.float32)

        # Find the triangle containing each point
        simplexes = self.tri.find_simplex(points)  # Index of triangle containing each point
        simplex_is_valid = simplexes != -1  # Filter out points outside the triangulation
        simplexes = simplexes[simplex_is_valid]
        valid_points = points[simplex_is_valid]
 
        # Gather all vertices for the triangles containing the points
        tri_indices = self.tri.simplices[simplexes]  # Get the triangle indices for valid points
        p1, p2, p3 = self.get_3d_vertices(tri_indices)
        
        # Compute the normal vector of the plane for each triangle using the cross product
        normal = tf.linalg.cross(p2 - p1, p3 - p1)
        a, b, c = tf.unstack(normal, axis=1)  # Extract coefficients a, b, c of the plane

        # Compute the d coefficient of the plane equation (ax + by + cz + d = 0)
        d = -(a * p1[:, 0] + b * p1[:, 1] + c * p1[:, 2])

        # Compute the elevation (z) for each point (x, y)
        elevation_valid = -(a * valid_points[:, 0] + b * valid_points[:, 1] + d) / c

        # Use tf.scatter_nd to create the full elevation tensor
        elevation = tf.scatter_nd(
            indices=tf.where(simplex_is_valid),
            updates=elevation_valid,
            shape=[tf.shape(points)[0]]
        )
        return elevation
    
    
    def compute_shape_loss(self):
        """
        Computes the shape loss based on the aspect ratio for all triangles,
        vectorized for efficiency.
        """
        # Mask triangles with boundary points only
        boundary_mask = tf.reduce_all(tf.gather(self.triang_point_is_boundary, self.tri.simplices) == 1, axis=1)
        non_boundary_triangles = tf.boolean_mask(self.tri.simplices, ~boundary_mask)

        p1, p2, p3 = self.get_3d_vertices(non_boundary_triangles)
        # Gather vertices for each triangle (shape: [num_triangles, 3, 2])
        #p1, p2, p3 = tf.gather(self.vertices, non_boundary_triangles[:, 0]), \
        #    tf.gather(self.vertices, non_boundary_triangles[:, 1]), \
        #    tf.gather(self.vertices, non_boundary_triangles[:, 2])

        # Compute edge lengths (shape: [num_triangles])
        edge1 = tf.norm(p2 - p1, axis=1)
        edge2 = tf.norm(p3 - p2, axis=1)
        edge3 = tf.norm(p1 - p3, axis=1)

        # Compute perimeter and semi-perimeter
        perimeter = edge1 + edge2 + edge3
        semi_perimeter = perimeter / 2.0

        # Compute area using Heron's formula
        area = tf.sqrt(semi_perimeter * (semi_perimeter - edge1) * (semi_perimeter - edge2) * (semi_perimeter - edge3))

        # Calculate aspect ratio for each triangle
        aspect_ratio = (12 * tf.sqrt(3.0) * area) / (perimeter ** 2)

        # Step 7: Compute shape loss
        shape_loss = tf.reduce_mean(1.0 - aspect_ratio)  # Sum of penalties for low aspect ratio
        # Step 8: Add a penalty for large triangle areas (squared area penalty)
        area_loss = tf.reduce_mean(area ** 2)

        return shape_loss, area_loss
    
    
    def create_batches(self, batch_size):
        """Create batches from the evaluation points."""
        epoch=0
        while True:
            epoch += 1
            print(f"Epoch: {epoch}")
            indices = np.random.permutation(len(self.eval_points))  # Shuffle the points
            for i in range(0, len(self.eval_points), batch_size):
                batch_indices = indices[i:i+batch_size]
                yield batch_indices
    
    
    def fit(self):
        # 4. Optimization Loop
        optimizer = tf.optimizers.Adam(learning_rate=1., beta_1=0.7)
        interior_mask =  tf.cast(1- self.triang_point_is_boundary, dtype=tf.float32)
        num_iterations = 2000
        batch_size = 3000  # Set your batch size

        shape_weight = 5e1 # Recall that this is weighted against average depth residuals.
        area_weight = 5e-11
        elevation_weight = 1e-2


        # Create batches from the evaluation points
        batch_generator = self.create_batches(batch_size)

        for i in range(num_iterations):
            batch_indices = next(batch_generator)  # Get the next batch of points
            self.tri = Delaunay(self.vertices.numpy())
            with tf.GradientTape() as tape:
                # Shape term: Penalize triangles with bad aspect ratios 
                shape_loss, area_loss = self.compute_shape_loss()
                
                # Evaluate predicted and true elevations at sample points within triangles
                predicted_elevations = self.evaluate_triangulated_elevation(tf.gather(self.eval_points, batch_indices))
                elevation_loss = tf.reduce_mean((predicted_elevations - tf.gather(self.true_elevations, batch_indices)) ** 2)  # L2 difference from true elevation
                
                # Weighted loss
                weighted_loss = shape_weight*shape_loss + area_weight*area_loss + elevation_weight*elevation_loss # Add loss for points outside of domain?
            gradients = tape.gradient(weighted_loss, [self.vertices])
            
            # Apply mask to gradients: zero out gradients for boundary points
            gradients_fixed = gradients[0] * tf.reshape(interior_mask, (-1, 1)) 
            optimizer.apply_gradients([(gradients_fixed, self.vertices)])
            
            if i % 50 == 0:
                print_str = f"""
---------------------------------------------------------
Iteration: {i}
Loss: {weighted_loss.numpy():.10e}
Elevation loss {elevation_weight*elevation_loss.numpy():.10e}
Shape loss: {shape_weight*shape_loss.numpy():.10e}
Area loss: {area_weight*area_loss.numpy():.10e}
"""
                print(print_str)
    
    
    def create_points(self, dims=None):
        # Construct pixel indices
        nr_of_rows, nr_of_cols = dims if dims is not None else (self.rows, self.cols)
        row_indices, col_indices = np.meshgrid(
            np.linspace(0, self.rows-1, num=nr_of_rows, dtype=int), 
            np.linspace(0, self.cols-1, num=nr_of_cols, dtype=int), 
            indexing="ij"
        )
        # Convert pixel indices to UTM coordinates
        lons, lats = rasterio.transform.xy(self.src.transform, row_indices, col_indices)
        eastings, northings = self.lonlat_to_meters(np.array(lons), np.array(lats))
        #eastings, northings = np.array(eastings), np.array(northings)
        mask = self.labeled[row_indices, col_indices] == self.label_component

        # Add extra boundry vertices to ensure that the entire region is contained in triangulation.
        mask_buff = convolve2d(mask, np.ones((3, 3)), mode='same')> 0.

        is_raster_boundary = np.full(mask.shape, False)
        is_raster_boundary[0,:] = True
        is_raster_boundary[-1,:] = True
        is_raster_boundary[:,0] = True
        is_raster_boundary[:,-1] = True

        mask_boundary = is_raster_boundary & mask | mask_buff & ~mask
        mask_interior = mask & ~mask_boundary

        interior_points = np.vstack([eastings[mask_interior], northings[mask_interior]]).T
        boundary_points = np.vstack([eastings[mask_boundary], northings[mask_boundary]]).T
        points = np.vstack([interior_points, boundary_points])
        point_is_boundary = np.hstack([np.zeros(interior_points.shape[0]), np.ones(boundary_points.shape[0])])

        return points, point_is_boundary


    def lonlat_to_meters(self, lon, lat):
        """
        Converts longitude and latitude to meters using a specified EPSG projection.
        
        Args:
            lon (array-like): Array of longitudes.
            lat (array-like): Array of latitudes.
            epsg_code (int): EPSG code for the target projection in meters.
                            Common values: 3857 (Web Mercator), or any UTM zone code (e.g., 32633 for UTM zone 33N).
        Returns:
            x, y: Coordinates in meters in the target projection specified by UTM_epsg_code.
        """
        # Initialize transformer with WGS84 (EPSG:4326) to target projection in meters
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{self.UTM_epsg_code}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return x, y


    def plot_triangulation(self, output_file="triangulation_plot.png"):
        """
        Plot the optimized triangulation with color-coded elevations and save to a file.

        Parameters:
        - vertices: Tensor or array of vertex coordinates.
        - eval_points: Array of evaluation points where elevations are known.
        - true_elevations: Tensor of true elevation values corresponding to eval_points.
        - filtered_triangles: Array of triangles defining the triangulation.
        - output_file: File path to save the plot.
        """
        # Convert tensors to numpy arrays if necessary
        vertices = self.vertices.numpy() if hasattr(self.vertices, 'numpy') else self.vertices
        eval_points = self.eval_points.numpy() if hasattr(self.eval_points, 'numpy') else self.eval_points
        true_elevations = self.true_elevations.numpy() if hasattr(self.true_elevations, 'numpy') else self.true_elevations

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Set colormap for the scatter plot of evaluation points
        cmap = cm.gist_ncar
        norm = plt.Normalize(vmin=np.min(true_elevations), vmax=np.max(true_elevations))
        
        # Scatter plot for evaluation points with color representing elevation
        sc = ax.scatter(eval_points[:, 0], 
                        eval_points[:, 1], 
                        c=true_elevations, 
                        cmap=cmap, 
                        norm=norm, s=10, edgecolor='k', linewidth=0.1)
        
        plt.colorbar(sc, ax=ax, label="Elevation (m)")
        
        # Triangulation plot
        ax.triplot(vertices[:, 0], vertices[:, 1], self.tri.simplices[self.triangle_is_interior()], color='gray', linewidth=0.5)
        ax.scatter(vertices[:, 0], vertices[:, 1], color="black", s=2, label="Vertices")

        # Title and labels
        ax.set_title("Optimized Triangulation", fontsize=16)
        ax.set_xlabel("X (meters)", fontsize=14)
        ax.set_ylabel("Y (meters)", fontsize=14)

        # Grid and legend
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        # Save to file
        plt.savefig(os.path.join(self.output_dir, output_file), dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close figure after saving to avoid display if running in a notebook
        print(f"Plot saved to {output_file}")


    def write_to_file(self):
        """
        Write triangulation to files.
        """
        # Write to mesh.
        cells = [
                    ("triangle", self.tri.simplices),
                ]
            
        
        # Convert meters to lonlat
        transformer = Transformer.from_crs(f"EPSG:{self.UTM_epsg_code}" ,"EPSG:4326" , always_xy=True)
        eastings, northings = self.vertices[:,0].numpy(), self.vertices[:,1].numpy()
        lons, lats = transformer.transform(eastings, northings)
        elevations = self.target_elevation_function(self.vertices).numpy()
        
        mesh = meshio.Mesh(
                np.vstack([lons, lats, elevations]).T,
            cells,
            # Optionally provide extra data on points, cells, etc.
            point_data={"Elevation": elevations,
                        "is_boundary": self.triang_point_is_boundary},
            cell_data={"is_interior": [self.triangle_is_interior().astype(int)],
                       "neighbours": [self.tri.neighbors]}
        )
        mesh.write(
            os.path.join(self.output_dir, "triangulation.vtk"),  # str, os.PathLike, or buffer/open file
            # file_format="vtk",  # optional if first argument is a path; inferred from extension
        )

        # Write to raster:
        row_indices, col_indices = np.meshgrid(np.arange(self.src.height), np.arange(self.src.width), indexing='ij')

        lons, lats = rasterio.transform.xy(self.src.transform, row_indices, col_indices)
        lons, lats = np.array(lons), np.array(lats)
        
        eastings, northings = self.lonlat_to_meters(lons, lats)

        # Find the triangle containing each point
        simplexes = self.tri.find_simplex(np.vstack([eastings.flatten(), northings.flatten()]).T)  # Index of triangle containing each point
        
        with rasterio.open(os.path.join(self.output_dir, 'triangulation_raster.tif'),'w', **self.src.profile) as dst:
            dst.write(simplexes.reshape((self.src.height, self.src.width)), 1)

def main():
    triang = Triangulation()
    triang.fit()
    triang.plot_triangulation()
    triang.write_to_file()

if __name__ == "__main__":
    main()
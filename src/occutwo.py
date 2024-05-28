import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import time
class OccuGrid:
    def __init__(self, resolution, num_samples, device):
        self.grid = torch.zeros((resolution+1, resolution+1, resolution+1))
        self.resolution = resolution
        self.num_samples = num_samples
        self.device = device

    def filter_outliers_alt(self, point_cloud, k=3, outlier_threshold=1):
            # Fit KNN model
        #if tensor is on gpu, convert to numpy
        if point_cloud.is_cuda:
            point_cloud = point_cloud.cpu().numpy()
            
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(point_cloud)
        
        # Compute distances and indices of k nearest neighbors
        distances, indices = nbrs.kneighbors(point_cloud)
        
        # Compute median distance to k nearest neighbors for each point
        median_distances = np.median(distances[:, 1:], axis=1)
        
        # Compute outlier threshold
        outlier_threshold *= np.median(median_distances)
        
        # Identify outliers
        outliers_mask = median_distances > outlier_threshold
        
        # Remove outliers
        filtered_point_cloud = point_cloud[~outliers_mask]
        

        return filtered_point_cloud, outlier_threshold
    
    def filter_outliers(self, point_cloud, k=3, outlier_threshold=1):
         # Convert point cloud to PyTorch tensor if it's not already
        if not isinstance(point_cloud, torch.Tensor):
            point_cloud = torch.tensor(point_cloud, device=self.device)
        
        # Use torch.cdist to compute pairwise distances
        distances = torch.cdist(point_cloud, point_cloud)
        
        # Sort distances for each point and take the first k+1 since it includes the point itself
        distances, _ = torch.sort(distances, dim=1)
        k_distances = distances[:, 1:k+1]  # Exclude the distance to itself (0)
        
        # Compute median distance to k nearest neighbors for each point
        median_distances = torch.median(k_distances, dim=1)[0]
        
        # Compute outlier threshold for each point
        thresholds = outlier_threshold * median_distances
        
        # Identify outliers
        outliers_mask = median_distances > thresholds
        
        # Remove outliers
        filtered_point_cloud = point_cloud[~outliers_mask]
        
        return filtered_point_cloud, thresholds
    
    def update(self, xyz, filter = False, alt = False):
        if filter and alt:
            start = time.time()
            xyz, _ = self.filter_outliers_alt(xyz)
            #print(f"Time to filter: {time.time() - start}")
            #convert to tensor
            xyz = torch.tensor(xyz, device=self.device)
        elif filter:
            start = time.time()
            xyz, _ = self.filter_outliers(xyz)
            #print(f"Time to filter: {time.time() - start}")
        
        
        start = time.time()
        normalized_pcd = self.normalize(xyz)
        #print(f"Time to normalize: {time.time() - start}")

        start = time.time()
        # Find the voxel indices for each point
        indices = (normalized_pcd * self.resolution).floor().long()
        
        indices = torch.clamp(indices, 0, self.resolution)
        #print(f"Time to find indices: {time.time() - start}")

        start = time.time()
        #uses an in place update method (indices, values, accumulate)
        self.grid.index_put_((indices[:, 0], indices[:, 1], indices[:, 2]), 
                            torch.tensor(1.0, device = self.device), accumulate=True)
        #print(f"Time to update grid: {time.time() - start}")

        start = time.time()
        self.grid /= self.grid.sum()
        #print(f"Time to normalize grid: {time.time() - start}")

    def sample(self, randomize = False):
        flat_grid = self.grid.view(-1)
        
        dist = torch.distributions.Categorical(flat_grid)
        samples = dist.sample((self.num_samples,))
        #a linear index i can be converted back to the 3D coordinates (x,y,z) by
        #i = z * (width * height) + y * width + x
        #z = i // (width * height) but W = H = resolution
        z = samples // (self.resolution + 1) ** 2
        y = (samples % (self.resolution + 1) ** 2) // (self.resolution + 1)
        x = samples % (self.resolution + 1)
        #float conversion is need for adding random offsets
        sampled_points = torch.stack((x, y, z), dim=1).float().to(self.device)

        if randomize:
            random_offsets = torch.rand(self.num_samples, 3, device=self.device)
            sampled_points += random_offsets
        
        return sampled_points / self.resolution

    def sample_uniformly(self, randomize = False):
        # Step 1: Calculate the total number of voxels
        total_voxels = (self.resolution + 1) ** 3

        # Step 2: Generate uniform random indices from the total number of voxels
        sampled_indices = torch.randint(0, total_voxels, (self.num_samples,), device=self.device)

        # Step 3: Convert linear indices to 3D coordinates
        # The logic here assumes a row-major order
        z = sampled_indices // ((self.resolution + 1) ** 2)
        y = (sampled_indices % ((self.resolution + 1) ** 2)) // (self.resolution + 1)
        x = sampled_indices % (self.resolution + 1)

        # Stack the coordinates to get a [num_samples, 3] tensor
        sampled_points = torch.stack((x, y, z), dim=1).float()

        # Step 4: Optionally normalize the coordinates
        
        if randomize:
            random_offsets = torch.rand(self.num_samples, 3, device=self.device)
            sampled_points += random_offsets
            
        sampled_points /= self.resolution
        return sampled_points

    def normalize(self, point_cloud):
        # Find min and max values for each axis
        mins = torch.min(point_cloud, dim=0)[0]
        maxs = torch.max(point_cloud, dim=0)[0]
        
        # Calculate the longest dimension
        ranges = maxs - mins
        max_range = torch.max(ranges)
        
        # Calculate the scaling factor
        scale_factor = 1.0 / max_range
        
        # Normalize each point by scaling uniformly
        normalized_point_cloud = (point_cloud - mins) * scale_factor
        
        self.scale_factor = scale_factor.to(self.device)
        self.mins = mins.to(self.device)

        return normalized_point_cloud
    
    def reverse_normalize(self, normalized_point_cloud):
        #print(self.scale_factor.is_cuda)
        #print(self.mins.is_cuda)
        return (normalized_point_cloud / self.scale_factor) + self.mins

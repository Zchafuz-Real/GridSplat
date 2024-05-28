import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import time
class OccuGrid:
    def __init__(self, 
                 resolution, 
                 num_samples, 
                 device, 
                 factor = None,
                 bd = 2,
                 min_temp = 1.5,
                 max_temp = 3.0,
                 ema_decay = 0.9):
        self.grid = torch.ones((resolution, resolution, resolution)).to(device)
        self.resolution = resolution
        self.num_samples = num_samples
        self.factor = factor
        self.scale = (1/2) ** (1/2) * (1/resolution)
        self.device = device
        self.timers = {"sample": [], "normalize": [], "reverse": []}
        self.bd = bd
        self.fixed_min = self.bd * torch.tensor([-0.5, -0.5, -0.5], device=self.device)
        self.fixed_max = self.bd * torch.tensor([0.5, 0.5, 0.5], device=self.device)
        self._weights = torch.ones((resolution, resolution, resolution)).to(device)
        self.ema_decay = ema_decay
        self.center = torch.tensor([0.5, 0.5, 0.5], device=self.device)
        self.temperature = 3.0
        self.min_temperature = min_temp 
        self.max_temperature = max_temp
    def update(self, xyz):    
        normalized_pcd = self.normalize(xyz)
    
        indices = (normalized_pcd * self.resolution).floor().long()
        indices = torch.clamp(indices, 0, self.resolution - 1)
        
        self.grid.index_put_((indices[:, 0], indices[:, 1], indices[:, 2]), 
                            torch.tensor(1.0, device = self.device), accumulate=True)

    def update_with_importance(self, xyz, weights):
        # Ensure opacity is a column vector
        if len(weights.shape) == 1:
            weights = weights.unsqueeze(1)
        
        # Normalize point cloud
        normalized_pcd = self.normalize(xyz)
        
        # Calculate grid indices
        indices = (normalized_pcd * self.resolution).floor().long()
        indices = torch.clamp(indices, 0, self.resolution - 1)
        
        self.grid = self.ema_decay * self.grid
        # Flatten the grid indices to a 1D array
        flattened_indices = indices[:, 0] * (self.resolution ** 2) + indices[:, 1] * self.resolution + indices[:, 2]
        
        # Create a flat grid to accumulate the opacity values
        flat_grid = self.grid.view(-1)
        flat_grid.scatter_add_(0, flattened_indices, weights.squeeze())
        
    def sample(self, randomize = False):
        start = time.time()
        total_voxels = (self.resolution) ** 3
        sampled_indices = torch.randint(0, total_voxels, (self.num_samples,))
        x = sampled_indices // (self.resolution ** 2)
        y = (sampled_indices % (self.resolution ** 2)) // self.resolution
        z = sampled_indices % self.resolution

        sampled_points = torch.stack((x, y, z), dim=1).float().to(self.device)
        sampled_points /= self.resolution 
        sampled_points += (1 / (2 * self.resolution))
        
        if randomize:
            scale = (1/2) * (1/self.resolution)
            random_offsets = 2 * torch.rand_like(sampled_points, device=self.device) - 1
            sampled_points += (scale * random_offsets) 
        
        self.timers["sample"].append(time.time() - start)
        return sampled_points


    def importance_sample(self, randomize = False):
        #flat_grid = torch.ones((self.resolution, self.resolution, self.resolution)).to(self.device)
        flat_grid = self.grid.flatten() 
        
        
        #grid_coords = torch.stack(torch.meshgrid(torch.arange(self.resolution, device=self.device),
        #                                         torch.arange(self.resolution, device=self.device),
        #                                         torch.arange(self.resolution, device=self.device)), dim=-1).float()
        #grid_coords /= self.resolution
        #grid_coords += (1 / (2 * self.resolution))
        #distances_to_center = torch.norm(grid_coords - self.center, dim=-1).flatten()
        
        #importance_weights = flat_grid / flat_grid.sum()
        #importance_weights *= torch.exp(- 10.0 * distances_to_center)
        #importance_weights /= importance_weights.sum()

        importance_weights = flat_grid / flat_grid.sum()
        importance_weights = torch.pow(importance_weights, 1.0 / self.temperature)
        importance_weights /= importance_weights.sum()
        #print(importance_weights)
        assert not torch.isnan(importance_weights).any(), "importance_weights contains nan values before creating Categorical distribution"
        dist = torch.distributions.Categorical(importance_weights)
        sampled_indices = dist.sample((self.num_samples,))
        x = sampled_indices // (self.resolution ** 2)
        y = (sampled_indices % (self.resolution ** 2)) // self.resolution
        z = sampled_indices % self.resolution
        
        sampled_points = torch.stack((x, y, z), dim=1).float().to(self.device)
        #put the points in the middle of the voxel
        sampled_points /= self.resolution 
        sampled_points += (1 / (2 * self.resolution))
        if randomize:
            scale = (1/2) * (1/self.resolution)
            random_offsets = 2 * torch.rand_like(sampled_points) - 1
            sampled_points += (scale * random_offsets) 
            #sampled_points = torch.clamp(sampled_points, 0, 1)
        return sampled_points


    def adjust_temperature(self, current_iter, max_iterations):
        self.temperature = self.max_temperature + (self.min_temperature - self.max_temperature) * (current_iter / max_iterations)
        
    def update_nbustesrs(self, xyz, weights):
         # xyz points should be in range [0, 1]
        xyz = self.normalize(xyz)
        assert xyz.min() >= 0, f"xyz min {xyz.min()}"
        assert xyz.max() <= 1, f"xyz max {xyz.max()}"

        # verify the shapes are correct
        assert len(xyz.shape) == 2
        assert xyz.shape[0] == weights.shape[0]
        assert xyz.shape[1] == 3
        assert weights.shape[1] == 1

        self._weights = self.ema_decay * self._weights
        indices = (xyz * self.resolution).floor().long()
        indices = torch.clamp(indices, 0, self.resolution - 1)
        self._weights[indices[:, 0], indices[:, 1], indices[:, 2]] += weights.squeeze(-1)
    
    def sample_nbusters(self, randomize = False):
        probs = self._weights.view(-1) / self._weights.view(-1).sum()
        dist = torch.distributions.categorical.Categorical(probs)
        sample = dist.sample((self.num_samples,))

        h = torch.div(sample, self.resolution**2, rounding_mode="floor")
        d = sample % self.resolution
        w = torch.div(sample, self.resolution, rounding_mode="floor") % self.resolution

        idx = torch.stack([h, w, d], dim=1).float()
        if randomize:
            return (idx + torch.rand_like(idx).to(self.device)) / self.resolution
        else:
            return idx / self.resolution
    
    def normalize(self, point_cloud):
        # Define fixed boundaries for the grid
        start = time.time()
        
        # Normalize the input data to the fixed range
        normalized_point_cloud = (point_cloud - self.fixed_min) / (self.fixed_max - self.fixed_min)
        self.timers["normalize"].append(time.time() - start)
        """ print("normalized")
        print(normalized_point_cloud.min())
        print(normalized_point_cloud.max())
        print(normalized_point_cloud.mean()) """
        return normalized_point_cloud
    
    def reverse_normalize(self, normalized_point_cloud):
        start = time.time()
        # Reverse the normalization to map the normalized point cloud back to original data space
        original_point_cloud = normalized_point_cloud * (self.fixed_max - self.fixed_min) + self.fixed_min
        self.timers["reverse"].append(time.time() - start)
        """ print("reversed")
        print(original_point_cloud.min())
        print(original_point_cloud.max())
        print(original_point_cloud.mean()) """
        return original_point_cloud

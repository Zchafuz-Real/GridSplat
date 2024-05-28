import math
import torch

from PIL import Image
from pathlib import Path
import numpy as np
from gsplat.sh import num_sh_bases
class Gaussians:
    def __init__(self,
                 config: dict,
                 ):
        self.config = config
        self.device = torch.device(config['device'])
        #self.gt_image = self.image_path_to_tensor(config['image_path']).to(self.device)
        self.gt_image = torch.zeros(800, 800, 3)
        self.num_points = config['num_points']

        BLOCK_X, BLOCK_Y = 16, 16
        fov_x = math.pi / 2.0
        self.H, self.W = self.gt_image.shape[0], self.gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.tile_bounds = (
            (self.W + BLOCK_X - 1) // BLOCK_X,
            (self.H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
        self.block = torch.tensor([BLOCK_X, BLOCK_Y, 1], device=self.device)

        

        self._init_gaussians()

    def image_path_to_tensor(self, image_path: Path):
        import torchvision.transforms as transforms
        print(image_path)
        img = Image.open(image_path)
        transform = transforms.ToTensor()
        img_tensor = transform(img).permute(1, 2, 0)[..., :3]
        return img_tensor
    
    def _init_gaussians(self):
        """Random gaussians"""
        #change this to 2 in original
        bd = self.config['bd']

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        #self.means = (torch.rand(self.num_points, 3, device=self.device))
        
        #distances, _  = self.intialize_scales(self.means.data, 10)
        #distances = torch.from_numpy(distances)
        #avg_dist = distances.mean(dim=-1, keepdim=True)
        #self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        self.rgbs = torch.rand(self.num_points, 3, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points, 1), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.rand(3, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False
        
    def intialize_scales(self, x, k):
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
    


        
class GaussiansSH:
    def __init__(self,
                 config: dict,
                 ):
        
        self.device = torch.device(config['device'])
        #self.gt_image = self.image_path_to_tensor(config['image_path']).to(self.device)
        self.gt_image = torch.zeros(800, 800, 3)
        self.num_points = config['num_points']

        BLOCK_X, BLOCK_Y = 16, 16
        fov_x = math.pi / 2.0
        self.H, self.W = self.gt_image.shape[0], self.gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.tile_bounds = (
            (self.W + BLOCK_X - 1) // BLOCK_X,
            (self.H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
        self.block = torch.tensor([BLOCK_X, BLOCK_Y, 1], device=self.device)

        

        self._init_gaussians()

    def image_path_to_tensor(self, image_path: Path):
        import torchvision.transforms as transforms
        print(image_path)
        img = Image.open(image_path)
        transform = transforms.ToTensor()
        img_tensor = transform(img).permute(1, 2, 0)[..., :3]
        return img_tensor
    
    def _init_gaussians(self):
        """Random gaussians"""
        #change this to 2 in original
        bd = 2

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        
        #distances, _  = self.intialize_scales(self.means.data, 10)
        #distances = torch.from_numpy(distances)
        #avg_dist = distances.mean(dim=-1, keepdim=True)
        #self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        dim_sh = num_sh_bases(3)
        self.features_dc = torch.rand(self.num_points, 3,device=self.device)
        self.features_rest = torch.zeros((self.num_points, dim_sh - 1, 3), device=self.device)
        
        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points, 1), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.rand(3, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.features_dc.requires_grad = True
        self.features_rest.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False
        
    def intialize_scales(self, x, k):
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
    
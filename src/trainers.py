import torch
import numpy as np
import json
import math
import copy
import optuna
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from pytorch_msssim import SSIM
from src.occutwo import OccuGrid
from src.gaussians import Gaussians
from src.sampling import SamplingStrategy
from src.data_collector import DataLogger
from src.data_loader import DataLoader
from src.viewer_test import ViserViewer
from src.color_utils import get_bg_color, BackgroundColor
from pathlib import Path
from gsplat.sh import spherical_harmonics
import torch.nn as nn


def load_example_camera(width):
    path = Path("extended_gsplat/data/blender/hotdog/transforms_train.json")
    meta = None
    with open(path, "r") as f:
        meta = json.load(f)
    #pick a random camera from meta
    cam_x_angle = float(meta["camera_angle_x"])
    frame = meta["frames"][56]
    camera = np.array(frame["transform_matrix"]).astype(np.float32)
    focal_length = 0.5 * width / math.tan(0.5 * cam_x_angle)
    c2w = torch.from_numpy(camera)
    image = Path(f"extended_gsplat/data/blender/hotdog/{frame['file_path']}.png")

    return c2w, torch.Tensor([focal_length]), image
    
import time
from typing import Optional

class TrainerClassic:

    def __init__(self, 
                 model: torch.nn.Module, 
                 gaussians: Gaussians, 
                 config: dict, 
                 sampling_strategy: Optional[SamplingStrategy] = None,
                 datalogger: Optional[DataLogger] = None,
                 dataloader: Optional[DataLoader] = None):
        self.times = [[],[],[],[]]
        self.config = config
        self.device = config['device']
        self.model = model.to(config['device'])
        self.gt_image = gaussians.gt_image
        self.image_type = config['image_type']
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

        self.iterations = config['iterations']
        self.mean_lr = config['mean_lr']
        self.mlp_lr = config['mlp_lr']
        self.gaussians = gaussians
        self.lrs = {
            "quat": config["lrs"]["quat"],
            "rgb":config["lrs"]["rgb"],
            "opacity":config["lrs"]["opacity"],
            "scale":config["lrs"]["scale"],
            "layers":config["lrs"]["layers"]
        }
        self.background = get_bg_color(config['background'])
        self.view_background = get_bg_color(config['background'])
        if config['background'] == 'random':
            self.view_background = get_bg_color('white')
        
        
        self.sampling_strategy = sampling_strategy
        #self.occu_grid = OccuGrid(config['resolution'], 
        #                          config['num_samples'], 
        #                          config['device'])
        
        self.allow_sampling = config['allow_sampling']
        self.sample_every = config['sample_every']
        self.randomize = config['randomize']

        self.allow_scheduler = config['allow_scheduler']
        self.gamma = config['gamma']
        self.step_size = config['step_size']

        self.optimizer_means = torch.optim.Adam([self.gaussians.means], 
                                          lr=self.mean_lr)
        self.allow_means_opptimization_iter = config['allow_means_opptimization_iter']

        #self.mlp_optimizer = torch.optim.Adam(self.model.parameters(),
        #                                        lr=self.mlp_lr)
        self.mlp_optimizer = torch.optim.Adam([
            {'params': model.quaternion_head.parameters(), 'lr': self.lrs["quat"] },
            {'params': model.rgb_head.parameters(), 'lr': self.lrs["rgb"]},
            {'params': model.opacity_head.parameters(), 'lr': self.lrs["opacity"]},
            {'params': model.scales_head.parameters(), 'lr': self.lrs["scale"]},
            {'params': model.layers.parameters(), 'lr': self.lrs["layers"]}]
            )
        
        if self.allow_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.mlp_optimizer, 
                                                             step_size=self.step_size, 
                                                             gamma=self.gamma)
        if datalogger:
            self.xys = None
            self.allow_logging = True
            self.datalogger = datalogger
            self.save_model = config["save_model"]
        else:
            self.allow_logging = False
            self.save_model = config["save_model"]
        if config['strategy'] == 'pool':
            self.sampling_strategy.initiate_pool(self.gaussians.means)
        
        self.dataloader = dataloader
        self.camera_idx = None
        
        self.MS_loss = torch.nn.MSELoss()
         
        self.loss_function = {
            'ssim_L1': self.ssim_L1_loss,
            'L1': self.L1_loss,
            'ssim': self.ssim_loss, 
            'mse_loss': self.mse_loss,
            'mse_ssim': self.mse_ssim_loss
        }
        assert config["loss_function"] in self.loss_function, "!!!This loss is not implemented!!!"
        print(config["loss_function"])
        
        self.viewer_enabled = config["viewer_enabled"]
        if self.viewer_enabled:
            self.viewer = ViserViewer(
                port = 9800,
                occugrid = self.sampling_strategy.occu_grid,
                cameras = copy.deepcopy(self.dataloader.cameras),
                images = copy.deepcopy(self.dataloader.cached_images),
                bg_color = copy.deepcopy(self.view_background),
                grid = {"resolution": self.config["resolution"],
                        "num_samples": self.config["num_samples"]}
                #is_training_mode = True
            )
        
    def disable_means_optimization(self):
        self.gaussians.means.requires_grad = False

    def update_features(self):
        self.gaussians.quats, self.gaussians.rgbs, self.gaussians.opacities , self.gaussians.scales = self.model(self.gaussians.means)
        
    def create_image(self):
        #self.gaussians.means = self.sampling_strategy.occu_grid.reverse_normalize(self.gaussians.means)
        with torch.no_grad():
            background = self.background.to(self.device)
            R = self.c2w[:3, :3]
            T = self.c2w[:3, 3:4]
            R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        
            R = R @ R_edit
            R_inv = R.T
            T_inv = -R_inv @ T
            viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
            viewmat[:3, :3] = R_inv
            viewmat[:3, 3:4] = T_inv
            viewmat.requires_grad = False

            W, H = int(self.gaussians.W), int(self.gaussians.H)
            
        xys, depths, radii, conics, compensation, num_tiles_hit, cov3d = project_gaussians(
                self.gaussians.means,
                self.gaussians.scales,
                1,
                self.gaussians.quats,
                viewmat,
                self.focal_length,
                self.focal_length,
                self.gaussians.W / 2,
                self.gaussians.H / 2,
                H,
                W,
                16,
            )
        if self.allow_logging:
            self.xys = xys
        
        xys.retain_grad()
        
        out_img = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                self.gaussians.rgbs,
                self.gaussians.opacities,
                H,
                W,
                16,
                background,
            )
        self.background = background
            
        return out_img

    def calculate_bg_penalty(self, image, bg_color):
        diff = 1 - torch.abs(bg_color - image)
        penalty = diff.mean()
        return penalty

    def ssim_L1_loss(self, pred_img):
        lamba = self.config["lamba"]
        Ll1 = torch.abs(self.gt_image - pred_img).mean()
        simloss = 1 - self.ssim(self.gt_image.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        return (1 - lamba) * Ll1 + lamba * simloss
    
    def L1_loss(self, pred_img):
        Ll1 = torch.abs(self.gt_image - pred_img).mean()
        return Ll1
    
    def ssim_loss(self, pred_img):
        simloss = 1 - self.ssim(self.gt_image.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        return simloss
    
    def mse_loss(self, pred_img):
        return  self.MS_loss(self.gt_image, pred_img)
    
    def mse_ssim_loss(self, pred_img):
        lamba = self.config["lamba"]
        MS_loss = self.MS_loss(self.gt_image, pred_img)
        simloss = simloss = 1 - self.ssim(self.gt_image.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        return (1 - lamba) * MS_loss + lamba * simloss
    
    def calculate_loss(self, pred_img):
        if self.allow_logging:
            with torch.no_grad():
                bg_penalty = self.calculate_bg_penalty(pred_img, self.background)
                self.datalogger.log_bg_penalty(bg_penalty.detach().cpu().numpy())
        
        return self.loss_function[self.config['loss_function']](pred_img)
    
    def reinitialize_weights(self, m):
        if isinstance(m, nn.Linear):
            print("reinit linear")
            m.reset_parameters()
            
    def train_step(self, iter):
        """ if self.allow_sampling and iter > self.allow_means_opptimization_iter:
                print("first iter")
                self.gaussians.means  = self.sampling_strategy.update_and_sample(iter,
                                                                                self.gaussians.means.detach()) """
        with torch.no_grad():
            self.gaussians.means  = self.sampling_strategy.update_and_sample(iter,
                                                                    self.gaussians.means,
                                                                    self.gaussians.opacities)
        start = time.time()
        self.update_features()
        self.times[0].append(time.time() - start)
          
        start = time.time()
        pred_img = self.create_image()
        self.times[1].append(time.time() - start)
        
        start = time.time()
        loss = self.calculate_loss(pred_img)
        self.times[2].append(time.time() - start)
        
        loss.backward()

        self.mlp_optimizer.step()
        self.optimizer_means.step()

        if self.allow_scheduler:
            self.scheduler.step()
        
        return {"loss": loss.item(), "pred_img": pred_img.detach()}
           
    def train(self, trial = None):
        
        for iter in range(self.iterations):
            self.model.train()
             
            image, camera_to_world, intrinsics, self.camera_idx  = self.dataloader.next_train()
             
            if self.image_type == "uint8":
                self.gt_image = image.float() / 255.0
            elif self.image_type == "float32":
                self.gt_image = image
                
            self.c2w = camera_to_world
            self.focal_length, self.focal_length, self.gaussians.W, self.gaussians.H  = intrinsics.fx, intrinsics.fy, image.shape[1], image.shape[0]
            
            if iter == self.allow_means_opptimization_iter:
                self.disable_means_optimization()

             
            outputs = self.train_step(iter)
             

            if trial is not None and iter == 1000:
                trial.report(outputs["loss"], iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if self.allow_logging:
                self.datalogger.log_mlp_grads(self.model)
            
            if iter % 100 == 0:
                print('Iter: ', iter, 'Loss: ', outputs["loss"])

            self.mlp_optimizer.zero_grad()
            self.optimizer_means.zero_grad() 
            
             
            if self.allow_logging:
                start = time.time()
                self.datalogger.log_iter_data(iter, 
                                         outputs["loss"],
                                         outputs["pred_img"],
                                         self.camera_idx,
                                         self.c2w,
                                         self.gaussians.means,
                                         self.xys,
                                         self.gaussians.rgbs,
                                         self.gaussians.opacities
                                         )
                self.times[3].append(time.time() - start)
        
             
            if self.viewer_enabled:
                self.viewer.update(
                    outputs["loss"],
                    outputs["pred_img"],
                    self.camera_idx,
                    self.gaussians.means,
                    self.gaussians.rgbs,
                    self.gaussians.opacities,
                    self.gaussians.scales,
                    self.model
                )
             

        if self.allow_logging:
            self.datalogger.log_post_iter_data(self.gaussians.means, 
                                               self.gaussians.rgbs,
                                               self.gaussians.opacities)
        
        update_time = np.mean(self.times[0] )
        image_time = np.mean(self.times[1] )
        loss_time = np.mean(self.times[2] )
        
        if self.allow_logging:
            log_time = np.mean(self.times[3] )
            print(f'update: {update_time}, image: {image_time}, loss: {loss_time}, log: {log_time}')
        else:
            print(f'update: {update_time}, image: {image_time}, loss: {loss_time}')
        if self.viewer_enabled:
            self.viewer.idle()
        
        if self.save_model:
            grid_params = {
                "resolution": self.sampling_strategy.occu_grid.resolution,
                "num_samples": self.sampling_strategy.occu_grid.num_samples,
                "grid": self.sampling_strategy.occu_grid.grid,
                "ema_decay": self.sampling_strategy.occu_grid.ema_decay,
                "temp": self.sampling_strategy.occu_grid.temperature,
                "min_temp": self.sampling_strategy.occu_grid.min_temperature,
                "max_temp": self.sampling_strategy.occu_grid.max_temperature,
            }
            self.model.save_model_state(grid_params)
        
        return outputs["loss"]
    
    
class TrainerFactory:
    @staticmethod
    def create_trainer(
                        model: torch.nn.Module, 
                        gaussians: Gaussians, 
                        config: dict, 
                        sampling_strategy: Optional[SamplingStrategy] = None,
                        datalogger: Optional[DataLogger] = None, 
                        data_loader: Optional[DataLoader] = None):

        if not config['allow_logging']:
            datalogger = None

        if config['type'] == 'test':
            return TrainerClassic(model, 
                                   gaussians, 
                                   config, 
                                   sampling_strategy, 
                                   datalogger,
                                   data_loader)
        
        else: 
            return ValueError(f"This trainer ({config['type']}) is not implemented yet! ")
        
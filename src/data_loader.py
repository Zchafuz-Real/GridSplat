import random
import numpy as np
import torch
from src.dataparser_outputs import DataparserOutputs
from src.color_utils import get_bg_color, BackgroundColor
from dataclasses import dataclass
from PIL import Image


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    

class DataLoader:
    def __init__(self, config, dataparser_outputs: DataparserOutputs, device):
        self.image_filenames = dataparser_outputs.image_filenames
        print("Data loader init")
        self.config = config
        self.background_color = get_bg_color(config["background_color"])
        self.NS_conversion = config["convert_NS_way"]
        self.image_type = config["image_type"]
        if self.NS_conversion:
            self.convert_pil_image = self.convert_NS_way
        else:
            self.convert_pil_image = self.convert_other_way

        self.cameras = dataparser_outputs.cameras
        
        self.unseen_cameras = [i for i in range(len(self.cameras.camera_to_worlds))]
        self.device = device
        self.cached_images =  []
        self.cache_images()
        print(f"Number of Images: {len(self.cached_images)}")
       
    def cache_images(self):
        
        for image_path in self.image_filenames:
            pil_image = Image.open(image_path)
            image = self.convert_pil_image(pil_image)
            self.cached_images.append(image)
    
    def convert_NS_way(self, pil_image):
        if self.image_type == "uint8":
            image = np.array(pil_image, dtype = "uint8")
            image = torch.from_numpy(image)
            image = image[:, :, :3] * (image[:, :, -1:] / 255.0) + 255.0 * self.background_color * (
                1.0 - image[:, :, -1:] / 255.0) 
            image = torch.clamp(image, min = 0, max = 255).to(torch.uint8).to(self.device)
        elif self.image_type == "float32":
            image = np.array(pil_image, dtype = "uint8")
            image = torch.from_numpy(image.astype("float32") / 255.0)
            
            image = image[:, :, :3] * image[:, :, -1:] + self.background_color * (
                1.0 - image[:, :, -1:]) 
            image = image.to(self.device)
            
            
        return image
    

    
    def convert_other_way(self, pil_image):
        image = pil_image.convert("RGB")
        image = np.array(pil_image, dtype = "uint8")
        image = torch.from_numpy(image)[:, :, :3]
        image = torch.clamp(image, min = 0, max = 255).to(torch.uint8).to(self.device)
        return image
        
    def next_train(self):
        image_idx = self.unseen_cameras.pop(random.randint(0, len(self.unseen_cameras) - 1))
        image_path = self.image_filenames[image_idx]
        
        image = self.cached_images[image_idx]
        camera_to_world = self.cameras.camera_to_worlds[image_idx].to(self.device)
        camera_idx = self.cameras.camera_idxs[image_idx]
        fx = self.cameras.fx.to(self.device)
        fy = self.cameras.fy.to(self.device)
        cx = self.cameras.cx.to(self.device)
        cy = self.cameras.cy.to(self.device)
        
        if len(self.unseen_cameras) == 0:
            self.unseen_cameras = [i for i in range(len(self.cameras.camera_to_worlds))]
        
        return image, camera_to_world, Intrinsics(fx, fy, cx, cy), camera_idx
        

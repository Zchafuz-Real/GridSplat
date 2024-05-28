
import json
import torch
from src.models import ModelFactory
from src.gaussians import Gaussians
from src.data_collector import DataLoggerFactory
from src.sampling import StrategyFactory
from src.trainers import TrainerFactory
from src.data_loader import DataLoader
from src.dataparser_outputs import DataparserOutputs, Cameras
from datetime import datetime
from pathlib import Path

def load_from_json(filename):
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)
      
class Experiment:
    def __init__(self, experiment_config, idx):

        if experiment_config['name']:
            self.experiment_name = f"{experiment_config['name']}_{str(idx)}" 
        else:    
            self.experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M_%S") + "_" + str(idx)

        if experiment_config["training"]["image_type"] != experiment_config["data_loader"]["image_type"]:
            print("Warning: Image type in training and data loader are different")
            print("Setting image type in training to image type in data loader")
        experiment_config["training"]["image_type"] = experiment_config["data_loader"]["image_type"]
        self.experiment_config = experiment_config
        self.data_file = Path(self.experiment_config["data"])
        self.scale_factor = 1.0
        
        self.gaussians_config = experiment_config['gaussians']
        self.model_config = experiment_config['model']
        self.training_config = experiment_config['training']
        self.data_loader_config = experiment_config['data_loader']
        self.device = self.training_config['device']

        self.data = self.load_data()

    def setup_environment(self):
        print("Setup environment")
        print(f"Sampling Strategy: {self.training_config["strategy"]}")
        self.model = ModelFactory.create_model(self.model_config, self.experiment_name)
        self.sampling_strategy = StrategyFactory.create_strategy(self.training_config)
        self.data_loader = DataLoader(self.data_loader_config, self.data, self.device)
        self.gaussians = Gaussians(self.gaussians_config) 
        self.data_logger = DataLoggerFactory.create_data_logger(
            self.experiment_config,
            self.gaussians.H,
            self.gaussians.W,
            self.experiment_name
        )
 
        self.trainer = TrainerFactory.create_trainer(self.model,
                                                     self.gaussians,
                                                     self.training_config,
                                                     self.sampling_strategy,
                                                     self.data_logger,
                                                     self.data_loader)

    def load_data(self):
        import numpy as np
        import imageio
        from pathlib import Path

        meta = load_from_json(self.data_file / f"transforms_train.json")
        image_filenames = []
        camera_idxs = []
        poses = []
        for idx, frame in enumerate(meta["frames"]):
            fname = self.data_file / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            camera_idxs.append(idx)
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform
        camera_to_world[..., 3] *= self.scale_factor
        #filtered_filenames = [file for file in image_filenames if file.name in [f"r_{i}.png" for i in [5]]]
        filtered_filenames = image_filenames
        #find the indices of the filtered filenames
        indices = [image_filenames.index(file) for file in filtered_filenames]
        #filter the camera_to_worlds
        cameras = Cameras(
            camera_idxs = torch.Tensor(camera_idxs).to(self.device),
            camera_to_worlds=camera_to_world[indices],
            fx=torch.Tensor([focal_length]).to(self.device),
            fy=torch.Tensor([focal_length]).to(self.device),
            cx=torch.Tensor([cx]).to(self.device),
            cy=torch.Tensor([cy]).to(self.device),
        )
        dataparser_outputs = DataparserOutputs(
            image_filenames=filtered_filenames,
            cameras=cameras,
        )

        return dataparser_outputs
    
    def run(self, trial = None):
        self.setup_environment()
        loss = self.trainer.train(trial)
        if self.training_config['allow_logging']:
            self.data_logger.create_visualizations()
        return loss
import torch
from src.experiment_3d import Experiment
from src.color_utils import get_bg_color
from src.viewer_test import ViserViewer
from src.models import DynamicMLP
import json
import pickle
#load a json file

import argparse
from pathlib import Path
def main():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--model_dir', type=str, help='Directory containing the model')
    parser.add_argument('--port', type=int, help='Port number to run the viewer on')
    parser.add_argument('--data_dir', type=str, help='Directory containing the data')
    args = parser.parse_args()
    
    config = json.load(open(Path(args.model_dir) / "config.json"))    
    model_config = config["model"]
    data_loader_config = config["data_loader"]
    
    model = DynamicMLP(model_config, "testing")
    #join the model_dir path with "model_state.pth"
    model_state_file = Path(args.model_dir) / "model_state.pth"
    grid_params_file = Path(args.model_dir) / "grid_params.pkl"
    
    model.load_state_dict(torch.load(model_state_file))
    model.to("cuda:0")
    data_file = config["data"] if args.data_dir is None else args.data_dir
    
    exp = Experiment(data_loader_config, data_file= data_file)
    cameras = exp.data_loader.cameras
    images = exp.data_loader.cached_images
    bg_color = get_bg_color("white")

    #load the pickle grid_params.pkl file
    grid_params = pickle.load(open(grid_params_file, "rb"))

    viewer = ViserViewer(args.port, 
                         cameras = cameras, images = images,
                     bg_color = bg_color, model = model,
                     grid_parameters = [
                         grid_params["grid"],
                         grid_params["resolution"],
                         grid_params["num_samples"],
                         grid_params["ema_decay"],
                         grid_params["temp"],
                         grid_params["min_temp"],
                         grid_params["max_temp"]
                     ],
                     is_training_mode = False)
    viewer.run_eval()
if __name__ == "__main__":
    main()
# Grid Splat

This project is a Master Thesis project for DTU exploring a field predicting Gaussian Splats features. 

# Installation
These are the steps that work on my machine so far 
1. Setup a conda environment
```bash
conda create -n gridsplat
conda activate gridsplat
```
2. Install dependencies
- clone the repository
- either install dependencies manually or use the `setup.py` file
- installing torch with cuda support is necessary either way (for me it works with the latest build and cuda 18)

```bash
cd path/to/GridSplat
pip install -e setup.py
```
# Using the model

1. Download the blender dataset
```bash
mkdir data
python download_data.py
```
2. Start training the model
```bash
python start3d.py --data-dir path/to/data/blender/datset (eg. hotdog)
```
- you can modify the test config dictionary inside the start3d.py

# Using custom configs
- you can make a configs directory and add your own configs
- initialy copy the test config dictionary inside start3d.py and make a file "config0.json" for example
- you can add as many configs you like the names of the files do not matter but set the "name" key inside the config
- configs with the same `"name"` key will output directories with `name_key_<idx>`

```bash
python start3d.py --config-dir path/to/configs
```

# Output diectories
- if `"allow_logging"` is set to `true` inside the config the program outputs directories with:

config_name_idx
|-- losses  
  |--camera_losses.png  
  |--loss.png  
  |--gradient logs  
  ...  
|-- points  
  |-- 3D_point_distribution.png  
  |-- point_distribution.gif  
|-- training  
  |-- final_image.png  
  |-- training.gif  
|-- config.json  
|-- results.json  

- if `"save_model"` is set to `true` the program will output the trained model state and grid_parameters used
- grid_params can be extracted with:
```python
import pickle
from pathlib import Path
grid_params_file = Path("model_dir_name/grid_params.pkl")
grid_params = pickle.load(open(grid_params_file, "rb")
grid, res, n_samples, ema_decay, temp, min_temp, max_temp = grid_params
```
|-- grid_params.pkl
|-- model_state.pth

# Using the viewer 

- if `"enable_viewer"` is set to true in the config the viewer will lunch at training on port 9800
- if using dtu hpc you can tunnel the port using:
```cmd
ssh -i ./.ssh/gbar -L 9800:<host_name_on_hpc>:9800 <dtu_id>@login2.hpc.dtu.dk
```
- change the `update interval` inside the viewer to see the training quicker
- or click `stop update` to let the training proceed without being slowed down


- or if you have a trained model you can use:
```bash
python eval_model.py --model_dir path/to_model_dir --port 9800 --data_dir path/to/data_used
```
- the `eval_model` viewer is very simple for now
- click `sample` to sample some points and move the camera around to get the images


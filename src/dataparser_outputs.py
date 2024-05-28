from dataclasses import dataclass
from typing import List
from pathlib import Path
from torch import Tensor

@dataclass 
class Cameras:
    camera_idxs: Tensor
    camera_to_worlds: Tensor
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class DataparserOutputs:
    """Dataparser outputs for the which will be used by the DataManager
    for creating RayBundle and RayGT objects."""

    image_filenames: List[Path]
    """Filenames for the images."""
    cameras: Cameras

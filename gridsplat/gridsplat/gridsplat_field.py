# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions


class GridsplatField(Field):
    """Compound Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        appearance_embedding_dim: int = 32,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.base_res = base_res
        self.average_init_density = average_init_density
        self.step = 0

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2 - 1, implementation=implementation
        )

        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )


        self.rgb_head = MLP(
            in_dim=self.geo_feat_dim + 1,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        self.quat_head = MLP(
            in_dim=self.geo_feat_dim + 1,
            num_layers=3,
            layer_width=64,
            out_dim=4,
            activation=nn.ReLU(),
            out_activation=nn.Tanh(),
            implementation=implementation,
        )

        self.scale_head = MLP(
            in_dim=self.geo_feat_dim + 1,
            num_layers=3,
            layer_width=64,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        self.opacities_head = MLP(
            in_dim=self.geo_feat_dim + 1,
            num_layers=3,
            layer_width=64,
            out_dim=1,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def get_outputs(
        self, points: Tensor
    ) -> Dict[FieldHeadNames, Tensor]:
        
        h = self.mlp_base(points)

        rgb = self.rgb_head(h).view(-1, 3)
        quats = self.quat_head(h).view(-1, 4)
        scales = self.scale_head(h).view(-1, 3)
        opacities = self.opacities_head(h).view(-1, 1)
        outputs = {
            FieldHeadNames.RGB: rgb, 
            "opacities": opacities, 
            "scales": scales,
            "quats": quats,
        }
        

        return outputs

import torch
import torch.nn as nn
from torch.nn import functional as F


class LinearBatchNormReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearBatchNormReLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        return self.relu(x)
    
class DynamicMLP(nn.Module):

    def __init__(self, config, experiment_name):
        
        super(DynamicMLP, self).__init__()
        self.experiment_name = experiment_name
        
        print("______Dynamic MLP initialization______")
        #self.L = config['L']
        self.config = config
        self.input_dim = config['input_dim']
        self.include_input = config['include_input']
        self.use_positional_encoding = config['use_positional_encoding']
        print("num_layers", config['num_layers'])
        self.num_layers = config['num_layers']
        self.shared_hidden_dim = config['shared_hidden_dim']
        self.hidden_dim = config['hidden_dim']
        print("skip_connections", config['skip_connections'])
        self.skip_connections = config['skip_connections'] 
        print("weight_init", config["weight_init"])     
        self.weight_initialization = config["weight_init"]

        # Adjust input dimension based on positional encoding and include_input
        self.encodings = {
            "old": self.positional_encoding,
            "positional": self.positional_encoding_ns,
            "fourier": self.fourier_encoding,
        }
        if config['use_positional_encoding']:
            self.encoding = self.encodings[config['input_encoding']] 
            print(f"Using {config['input_encoding']} encoding")
            if config['input_encoding'] == "fourier":
                self.basis = torch.normal(mean=0, 
                                          std = self.config["scale"],
                                          size=(self.input_dim, self.config["n_freq"]))
        self.initialize_input_dim()
        print("input_dim", self.input_dim)
        
        
        self.build_nn_modules() #code from NS
        self.activation = nn.ReLU()
        print("layers", self.layers)
        
        # Specialized heads
        self.quaternion_head = nn.Linear(self.hidden_dim, 4)
        self.rgb_head = nn.Linear(self.hidden_dim, 3)
        self.opacity_head = nn.Linear(self.hidden_dim, 1) 
        self.scales_head = nn.Linear(self.hidden_dim, 3)

        if self.weight_initialization: 
            self.initialize_weights()
        
        self.save_config()
    
    def save_config(self):
        #save the input dimensions, config and the layers 
        # Save the input dimensions, config and the layers
        import os
        import json
        config_data = {
            'input_dim': self.input_dim,
            'num_layers': self.num_layers,
            'shared_hidden_dim': self.shared_hidden_dim,
            'hidden_dim': self.hidden_dim,
            'skip_connections': self.skip_connections,
            'weight_initialization': self.weight_initialization,
            'use_positional_encoding': self.use_positional_encoding,
            'include_input': self.include_input,
            'n_freqs': self.config["n_freq"],
            'layers': [str(layer) for layer in self.layers]
        }
        out_dir = os.path.join(os.getcwd(), "data_log", self.experiment_name)
        os.makedirs(out_dir, exist_ok = True)
        with open(os.path.join(out_dir, 'mlp_config.json'), 'w') as f:
            json.dump(config_data, f)
    
    def save_model_state(self, grid_params):
        import os
        import pickle
        out_dir = os.path.join(os.getcwd(), "data_log", self.experiment_name)
        os.makedirs(out_dir, exist_ok = True)
        torch.save(self.state_dict(), os.path.join(out_dir, 'model_state.pth'))
        with open(os.path.join(out_dir, 'grid_params.pkl'), 'wb') as f:
            pickle.dump(grid_params, f)
        
    def build_nn_modules(self):
        layers = []
        if self.num_layers == 1:
            layers.append(LinearBatchNormReLU(self.input_dim, self.hidden_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self.skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(LinearBatchNormReLU(self.input_dim, self.shared_hidden_dim))
                elif i in self.skip_connections:
                    layers.append(LinearBatchNormReLU(self.shared_hidden_dim + self.input_dim, self.shared_hidden_dim))
                else:
                    layers.append(LinearBatchNormReLU(self.shared_hidden_dim, self.shared_hidden_dim))
            layers.append(LinearBatchNormReLU(self.shared_hidden_dim, self.hidden_dim))
            
        self.layers = nn.ModuleList(layers)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def initialize_input_dim(self):
        if self.use_positional_encoding:
            if self.config["input_encoding"] == "fourier":
                encoded_dim = self.basis.shape[1] * 2
            elif self.config["input_encoding"] == "positional":
                encoded_dim = self.input_dim * self.config['n_freq'] * 2
            self.input_dim = encoded_dim + self.input_dim if self.include_input else encoded_dim
        else:
            self.input_dim = self.input_dim

    def positional_encoding(self, x):
        frequencies = 2 ** torch.linspace(0, self.L-1, self.L)
        frequencies = frequencies.view(1, 1, -1).to(x.device)  # Shape [1, 1, L]
        x = x.unsqueeze(-1)  # Shape [N, C, 1]
        encoded = torch.cat((torch.sin(frequencies * x), torch.cos(frequencies * x)), dim=2)
        return encoded.view(x.shape[0], -1)

    def positional_encoding_ns(self, x):
        scaled_in_tensor = 2 * torch.pi * x
        freqs = 2 ** torch.linspace(self.config["min_freq"], self.config["max_freq"], self.config["n_freq"]).to(x.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)
        encoded_inputs = torch.sin(torch.cat((scaled_inputs, scaled_inputs + torch.pi / 2.0), dim=-1))
        return encoded_inputs

    def fourier_encoding(self, x):
        self.basis = self.basis.to(x.device)
        scaled_in_tensor = 2 * torch.pi * x
        scaled_inputs = scaled_in_tensor @ self.basis
        freqs = 2 ** torch.linspace(0.0, 0.0, 1).to(x.device)
        scaled_inputs = scaled_inputs[..., None] * freqs
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)
        encoded_inputs = encoded_inputs = torch.sin(torch.cat((scaled_inputs, scaled_inputs + torch.pi / 2.0), dim=-1))
        return encoded_inputs
        
    def initialize_input(self, x):
        #normalize the inputs to be between 0 and 1 where 0 is the minimum value and 1 is the maximum value
        x = (x - x.min()) / (x.max() - x.min())
        
        if not self.use_positional_encoding:
            return x
        if self.include_input:
            #y = (x - x.min()) / (x.max() - x.min())
            return torch.cat((x, self.encoding(x)), dim=1)
        else:
            #x = (x - x.min()) / (x.max() - x.min())
            return self.encoding(x)

    def heads(self, x):
        quaternions = self.quaternion_head(x)
        quaternions = F.normalize(quaternions, p=2, dim=1)

        rgbs = torch.sigmoid(self.rgb_head(x))
        opacities = torch.sigmoid(self.opacity_head(x))
        scales = torch.sigmoid(self.scales_head(x))
        
        return quaternions, rgbs, opacities, scales

    def forward(self, x):
        in_tensor = self.initialize_input(x)
        x = in_tensor
        
        #if self.test_run:
        #   x = self.layers(x)
        #else:
        for i, layer in enumerate(self.layers):
            #print(i, layer)
            if i in self.skip_connections:
                #print(i, "skip")
                #print(x.shape)
                x = torch.cat((in_tensor, x), dim= - 1) #think about -1 or 1 for concat
                #print(x.shape)
            #print("forward layer")
            #print(x.shape)
            x = layer(x)
            #print(x.shape)
        
        return self.heads(x)
    
class SpecializedMLPSH(nn.Module):

    def __init__(self, config):
        
        super(SpecializedMLPSH, self).__init__()
        self.L = config['L']
        self.input_dim = config['input_dim']
        self.include_input = config['include_input']
        self.use_positional_encoding = config['use_positional_encoding']
        self.shared_hidden_dim = config['shared_hidden_dim']
        self.hidden_dim = config['hidden_dim']

        # Adjust input dimension based on positional encoding and include_input
        self.initialize_input_dim()

        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.shared_hidden_dim),
            nn.BatchNorm1d(self.shared_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.shared_hidden_dim, self.shared_hidden_dim), #add this
            nn.BatchNorm1d(self.shared_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.shared_hidden_dim, self.shared_hidden_dim),
            nn.BatchNorm1d(self.shared_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.shared_hidden_dim, self.shared_hidden_dim), #add this
            nn.BatchNorm1d(self.shared_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.shared_hidden_dim, self.shared_hidden_dim), #add this
            nn.BatchNorm1d(self.shared_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.shared_hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        # Specialized heads
        self.quaternion_head = nn.Linear(self.hidden_dim, 4)
        self.features_dc_head = nn.Linear(self.hidden_dim, 3)
        self.features_rest_head = nn.Linear(self.hidden_dim, 15 * 3)
        #make the features_rest_head ouput a tensor that has the shape (15, 3)
        
        
        self.opacity_head = nn.Linear(self.hidden_dim, 1)
        self.scales_head = nn.Linear(self.hidden_dim, 3)

    
    def initialize_input_dim(self):
        if self.use_positional_encoding:
            if self.encoding == "fourier":
                encoded_dim = self.basis.shape[1] * self.config['n_freq'] * 2
            elif self.encoding == "positional":
                encoded_dim = self.input_dim * self.config['n_freq'] * 2
            self.input_dim = encoded_dim + self.input_dim if self.include_input else encoded_dim
        else:
            self.input_dim = self.input_dim

    def positional_encoding(self, x):
        frequencies = 2 ** torch.linspace(0, self.L-1, self.L)
        frequencies = frequencies.view(1, 1, -1).to(x.device)  # Shape [1, 1, L]
        x = x.unsqueeze(-1)  # Shape [N, C, 1]
        encoded = torch.cat((torch.sin(frequencies * x), torch.cos(frequencies * x)), dim=2)
        return encoded.view(x.shape[0], -1)

    def initialize_input(self, x):
        #x = (x - x.min()) / (x.max() - x.min())
        if not self.use_positional_encoding:
            return x
        if self.include_input:
            return torch.cat((x, self.positional_encoding(x)), dim=1)
        else:
            return self.positional_encoding(x)

    def heads(self, x):
        quaternions = self.quaternion_head(x)
        quaternions = F.normalize(quaternions, p=2, dim=1)

        features_rest = self.features_rest_head(x)
        features_rest = features_rest.view(features_rest.size(0), 15, 3)
        
        features_dc = self.features_dc_head(x)
        opacities = torch.sigmoid(self.opacity_head(x))
        scales = torch.sigmoid(self.scales_head(x))
        
        return quaternions, features_dc, features_rest, opacities, scales

    def forward(self, x):
        x = self.initialize_input(x)

        # Pass through shared layers
        x = self.shared_layers(x)

        # Pass through specialized heads
        return self.heads(x)
       


class ModelFactory:
    @staticmethod
    def create_model(config, experiment_name = None):
        if config['model_type'] == 'dynamic':
            return DynamicMLP(config, experiment_name)
        else:
            raise ValueError(f"Unknown model type: {config['model_type']}")
        



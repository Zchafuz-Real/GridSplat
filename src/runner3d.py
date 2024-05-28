import os
import json
from src.experiment_3d import Experiment

class Runner:
    def __init__(self, input_arg, *args):

        self.experiments = []

        if isinstance(input_arg, str): 
            self.initialize_with_directory(input_arg, *args)
        elif isinstance(input_arg, dict):
            self.initialize_with_dict(input_arg)
        else:
            raise ValueError('Invalid input type, must be str or dict')
    
    def modify_field(self, field, values):

        config_type, field_name = field

        if len(self.configs) != len(values):
            raise ValueError('Number of values must match the number of experiments')

        combined_data = zip(self.configs, values)
        for config, value in combined_data:
            config[config_type][field_name] = value


    def load_configs(self, directory, configs):
        for idx, config in enumerate(configs):
            with open(f'{directory}/{config}', 'r') as f:
                config = json.load(f)
                self.configs.append(config)
            f.close()

    def initialize_with_directory(self, directory, *args):
        configs = os.listdir(directory)
        print(configs)
        self.configs = []

        # If there are arguments, load only the specified configs
        if args:
            self.load_configs(directory, args)
            return
        
        configs = os.listdir(directory)
        self.load_configs(directory, configs)


    def initialize_with_dict(self, config):
        self.configs = [config]
        #self.experiments = [Experiment(config, 0)]

    def initialize_experiments(self):

        names = {}

        for config in self.configs:
            
            index = names.get(config["name"], 0)
            names[config["name"]] = index + 1

            self.experiments.append(Experiment(config, names[config["name"]]))

    def run_experiments(self, trial = None):
        self.initialize_experiments()

        for experiment in self.experiments:
            loss = experiment.run(trial)
        return loss

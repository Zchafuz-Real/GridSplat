from src.occutwo import OccuGrid
from src.occuthree import OccuGrid as OC3

import torch
from torch import Tensor

class SamplingStrategy:
    def __init__(self, config): 
        self.occu_grid = OccuGrid(config['resolution'], 
                                  config['num_samples'], 
                                  config['device'])
        self.filter = config['filter']
        self.randomize = config['randomize']
        self.alt = config['alt']
        self.sample_every = config['sample_every']

    def update_and_sample(self, iter, xyz) -> Tensor:
        pass
        


class RandomSampling(SamplingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.occu_grid = OC3(config["resolution"],
                             config["num_samples"],
                             config["device"],
                             config["factor"],
                             bd = config["bd"])
        
    def update_and_sample(self, iter, xyz, weights = None):
        if iter % self.sample_every == 0:
            #self.occu_grid.update(xyz, self.filter, self.alt)
            sampled_points = self.occu_grid.sample(self.randomize)
            return self.occu_grid.reverse_normalize(sampled_points)
            #return sampled_points
        return xyz

class ImportanceSampling(SamplingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.max_iterations = config["iterations"]
        self.occu_grid = OC3(config["resolution"],
                             config["num_samples"],
                             config["device"],
                             config["factor"],
                             bd = config["bd"],
                             min_temp = config["min_temp"],
                             max_temp = config["max_temp"],
                             ema_decay = config["ema_decay"])

    def update_and_sample(self, iter, xyz, weights):
        if iter % self.sample_every == 0:
            self.occu_grid.adjust_temperature(iter, self.max_iterations)
            #print(self.occu_grid.temperature)
            self.occu_grid.update_with_importance(xyz, weights)
            #self.occu_grid.update_nbustesrs(xyz, weights)
            sampled_points = self.occu_grid.importance_sample(self.randomize)
            #sampled_points = self.occu_grid.sample_nbusters(self.randomize)
            return self.occu_grid.reverse_normalize(sampled_points)
        sampled_points = self.occu_grid.importance_sample(self.randomize)
        return self.occu_grid.reverse_normalize(sampled_points)
        #return xyz

class StrategyFactory:
    @staticmethod
    def create_strategy(config):
        strategy = config['strategy']
        if strategy == 'random':
            return RandomSampling(config)
        if strategy == 'random_fix':
            return RandomSampling(config)
        if strategy == 'importance':
            return ImportanceSampling(config)
        else:
            raise ValueError(f"Unknown strategy ({config['strategy']})")
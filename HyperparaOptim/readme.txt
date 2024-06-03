import numpy as np
import random
import matplotlib.pyplot as plt
import ray
from ray import tune
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
from ray.tune import CLIReporter
from ray.tune.analysis import ExperimentAnalysis

TuneResultFolder = 'C:/ray_results3/Trainable_2024-05-26_09-15-07' 
analysis = ExperimentAnalysis(TuneResultFolder)
best_config = analysis.get_best_config(metric="meanReward", mode="max")
print("Best Hyperparameters: ", best_config)
Best Hyperparameters:  {'batch_size': 64, 'filtersize': 32, 'filtersize2': 64, 'filtersize3': 64, 'learningrate': 0.005, 'replay_after': 100, 'update_every': 100}
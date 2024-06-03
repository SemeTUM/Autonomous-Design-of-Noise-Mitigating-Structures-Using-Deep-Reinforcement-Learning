# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:07:22 2024

@author: admin
"""

import sys
import os
import mph
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import scipy.io as sio
import torch.optim as optim
from ComEnv import DDPGEnv
import multiprocessing as mp
import mpi4py


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions, seed, hyperparameter):
        super(DQNCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.hyperparameter= hyperparameter
        self.seed=seed
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], self.hyperparameter["filtersize"], kernel_size=2, stride=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(self.hyperparameter['filtersize'], self.hyperparameter['filtersize2'], kernel_size=2, stride=2),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(self.hyperparameter['filtersize2'], self.hyperparameter['filtersize3'], kernel_size=2, stride=1),
            ##nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), self.hyperparameter['filtersize3']),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(self.hyperparameter['filtersize3'], self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features((torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
#Replay buffer 

class ReplayMemory:
    def __init__(self, buffer_size, batch_size, seed, device):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = seed
        self.device= device
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
    
    
# dqnAgent with Fixed Target network
class DDQNAgent():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    def __init__(self, input_shape, action_size, hyperparameter, seed, buffer_size, batch_size, gamma, tau):

        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(0)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_after = replay_after
        self.tau = tau
        self.hyperparameter= hyperparameter
        # Q-Network
        self.policy_net = DQNCnn(input_shape, action_size, self.seed, self.hyperparameter).to(self.device)
        self.target_net = DQNCnn(input_shape, action_size, self.seed, self.hyperparameter).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.hyperparameter['learningrate'])
        
        # Replay memory
        self.memory = ReplayMemory(self.buffer_size, self.batch_size, self.seed, self.device)
        
        self.t_step = 0
        
    def get_dict_to_save(self):
        return {'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict()}
    
    def load(self, dict):
        self.policy_net.load_state_dict(dict['policy_net'])
        self.target_net.load_state_dict(dict['target_net'])
        self.optimizer.load_state_dict(dict['optimizer'])
        
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.hyperparameter['update_every']

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.hyperparameter['replay_after']:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.0):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones=experiences
        
        Q_policy_next= self.policy_net(next_states).detach()
        Q_targets_next=self.target_net(next_states).detach()
        
        _, policy_actions=Q_policy_next.max(1, keepdim=True)
        Q_targets_next= Q_targets_next.gather(1, policy_actions)
        
        Q_targets= rewards + (self.gamma*Q_targets_next*(1 - dones))
        
        Q_expected= self.policy_net(states).gather(1, actions.unsqueeze(1))
    
        loss = F.mse_loss(Q_expected, Q_targets[0].unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()       
        self.soft_update(self.policy_net, self.target_net, self.tau)
        
    def checkpoint(self, filename):
         torch.save(self.policy_net.state_dict(), filename)

    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)




INPUT_SHAPE = (1, 30, 30)
ACTION_SIZE = 900#900#original 101
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 1000   # replay buffer size1000
BATCH_SIZE = 64       # Update batch size
LR = 0.001            # learning rate 
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 100     # how often to update the network
replay_after = 100  # After which thershold replay to be started 100
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 100        # Rate by which epsilon to be decayed
FC1_SIZE= 32
FC2_SIZE= 64

start_epoch=0
scores= []
scores_window=deque(maxlen=100)


#agent = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, reply_after, hyperparameter, DQNCnn)

##epsilon
epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)


result_directory = "Absorption_Opt2D__DDQNCNN"
os.makedirs(result_directory, exist_ok=True)



#agent = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)

##epsilon
#epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)
#env = DDPGEnv()
##train
class modeltoTrain:
    def __init__(self, hyperparameter):
        super().__init__()
        self.hyperparameter= hyperparameter
        self.model= DQNCnn(INPUT_SHAPE, ACTION_SIZE, SEED, self.hyperparameter)
        self.model= DQNCnn(INPUT_SHAPE, ACTION_SIZE, SEED, self.hyperparameter)
        self.agent= DDQNAgent(input_shape=INPUT_SHAPE, action_size=ACTION_SIZE, hyperparameter=self.hyperparameter, seed=42, buffer_size=2000,
                              batch_size=self.hyperparameter['batch_size'], gamma=GAMMA, tau=TAU)
        self.scores=[]
        self.STATES=[]
        self.ACTIONS=[]
        self.absorption=[]
        self.absorpSum=[]
        self.scores_window= deque(maxlen=1000)
    
    def trainOneEpisode(self):
        meanReward= self.train(n_episodes= 1000)
        return meanReward  
    
    def save(self, path):
        dict = {'scores': self.scores, 'scores_window': self.scores_window}
        agent_dict = self.agent.get_dict_to_save()
        dict.update(agent_dict)
        torch.save(dict, path)
        
    def load(self, path):
        dict = torch.load(path)
        self.agent.load(dict)
        self.scores = dict['scores']
        self.scores_window = dict['scores_window']
            
    def train(self, n_episodes):#100
        eps = EPS_START
        
        for i_episode in range(start_epoch + 1, n_episodes+1):
            state = env.reset()
            state= np.reshape(state, INPUT_SHAPE)
            score = 0
            #eps = max(eps * EPS_DECAY, EPS_MIN)
            eps = epsilon_by_epsiode(i_episode)
            
            action = self.agent.act(state, eps)
            next_state, reward, done, absorpSum, absorp = env.step(action)
            next_state= np.reshape(next_state, INPUT_SHAPE)
            score += reward
            self.agent.step(state, action, reward, next_state, done)
            state = next_state
    
            self.scores_window.append(score)       # save most recent score
            self.scores.append(score)              # save most recent score
            
            self.ACTIONS.append(action)
            self.STATES.append(state)
            self.absorption.append(absorpSum)
            self.absorpSum.append(absorp)
                   
            #if i_episode % 100==0:
                #torch.save(agent.policy_net.state_dict(), "ddqn{}.pth".format(i_episode))
             #   model_path = os.path.join(result_directory, "TL_HALFRULE_trained_ddqn{}.pth".format(i_episode))
             #   self.agent.checkpoint(model_path)
             #   print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(self.scores_window)))
            #if absorpSum > 19:
                #torch.save(agent.policy_net.state_dict(), "ddqn{}.pth".format(i_episode))
            #    model_path = os.path.join(result_directory, "TL_trained_ddqn{}.pth".format(i_episode))
            #    self.agent.checkpoint(model_path)
        
        return np.mean(self.scores_window)

###run the program
env=DDPGEnv()
#scores = train(2000)
'''Train the model'''
import ray
from ray import tune
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp
from ray.tune import CLIReporter

ModelFileName= "checkpoint.pth"
TuneResultFolder= 'C:/ray_results3'
ray_directory = "ray_temp_dir3"

os.makedirs(TuneResultFolder, exist_ok=True)
os.makedirs(ray_directory, exist_ok=True)
MaxTrainIter= 4

reporter= CLIReporter(max_progress_rows=10)
reporter.add_metric_column("mean_reward")

class Trainable(tune.Trainable):
    def setup(self, hyperparameter):
        self.modeltoTrain= modeltoTrain(hyperparameter)
        
    def step(self):
        meanReward= self.modeltoTrain.trainOneEpisode()
        return {'meanReward': meanReward}
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpointPath=os.path.join(tmp_checkpoint_dir, ModelFileName)
        self.modeltoTrain.save(checkpointPath)
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpointPath= os.path.join(tmp_checkpoint_dir, ModelFileName)
        self.modeltoTrain.load(checkpointPath)
 
def main():
    
    ray.init()
    #ray.init(_temp_dir="C:/Users/admin/DEEP REINFORCEMENT LEARNING PAPER/DRL 3D/ray_temp_dir")

    space= {"batch_size": hp.choice("batch_size", [32, 64, 128]),
            "learningrate": hp.choice("learningrate", [0.01, 0.005, 0.001]),
            "update_every": hp.choice("update_every", [100, 200, 400]),
            "filtersize": hp.choice("filtersize", [32,64,128]),
            "filtersize2": hp.choice("filtersize2", [64,128,256]),
            "filtersize3": hp.choice("filtersize3", [64,128,256]),
            "replay_after": hp.choice("replay_after", [100,500,600]) }
    
    #resources_per_trial={'cpu':3, 'gpu':1},
    hypSearchOpt= HyperOptSearch(space, metric="meanReward", mode="max")
    analysis= tune.run(
        Trainable,
        resources_per_trial={'cpu':7, 'gpu':1},
        stop={'training_iteration': MaxTrainIter},
        num_samples=10,
        scheduler= MedianStoppingRule(metric="meanReward", mode="max"),
        search_alg=hypSearchOpt,
        local_dir= TuneResultFolder,
        progress_reporter=reporter,
        checkpoint_freq=100,
        verbose=1)

    
if __name__=="__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 09:49:35 2023

@author: admin
"""
import time
import math
import sys
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
from ComEnv import DDPGEnv
from collections import namedtuple, deque
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.optim as optim

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
#Replay buffer 

class ReplayMemory:
    def __init__(self, buffer_size, batch_size, seed, device):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(0)
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
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model):

        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(0)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.replay_after = replay_after
        self.DQN = model
        self.tau = tau
        
        # Q-Network
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayMemory(self.buffer_size, self.batch_size, self.seed, self.device)
        
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
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


# HyperParameters
INPUT_SHAPE = (1, 30, 30)
ACTION_SIZE = 450#900
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 1000   # replay buffer size1000
BATCH_SIZE = 64       # Update batch size
LR = 0.001            # learning rate 
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 100     # how often to update the network
UPDATE_TARGET = 100  # After which thershold replay to be started 100
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 100        # Rate by which epsilon to be decayed

start_epoch=0
scores= []
scores_window=deque(maxlen=100)

result_directory = "RefProb_RuleEnvDDQNCNN1"
os.makedirs(result_directory, exist_ok=True)


agent = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)

##epsilon
epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

def train(n_episodes=2000):#100
    ACTIONS= []
    STATES= []
    ABSP= []
    ABSPSUM= []
    
    for i_episode in range(start_epoch + 1, n_episodes+1):
        state = env.reset()
        state= np.reshape(state, INPUT_SHAPE)
        score = 0
        eps = epsilon_by_epsiode(i_episode)

        action = agent.act(state, eps)
        next_state, reward, done, absorpSum, absp = env.step(action)
        next_state= np.reshape(next_state, INPUT_SHAPE)
        score += reward
        agent.step(state, action, reward, next_state, done)
        state = next_state

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        ACTIONS.append(action)
        STATES.append(state)
        ABSPSUM.append(absorpSum)
        ABSP.append(absp)
      
        np.save(os.path.join(result_directory, "DDQNCNNactions.npy"), ACTIONS)
        np.save(os.path.join(result_directory, "DDQNCNNscores.npy"), scores)
        np.save(os.path.join(result_directory, "DDQNCNNstates.npy"), STATES)
        np.save(os.path.join(result_directory, "DDQNCNNabsorption.npy"), ABSP)
        np.save(os.path.join(result_directory, "DDQNCNNabsum.npy"), ABSPSUM)
        
        if i_episode % 100==0:
            #torch.save(agent.policy_net.state_dict(), "ddqn{}.pth".format(i_episode))
            model_path = os.path.join(result_directory, "Refruletrained_ddqncnn{}.pth".format(i_episode))
            agent.checkpoint(model_path)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score))
    
    return scores

###run the program
#seed = 0
#torch.manual_seed(seed)
#np.random.seed(seed)
#random.seed(seed)

env=DDPGEnv()
scores = train(2000)
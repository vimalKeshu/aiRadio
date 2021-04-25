import gym
import math
import random
import numpy as np  
from  collections import namedtuple 
from itertools import count 
from PIL import Image 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision.transforms as T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        layers = [
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train() # set training mode

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.model(x)

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32)) # to tensor
        pdparam = self.forward(x) # forward pass
        pd = Categorical(logits=pdparam) # probability distribution
        action = pd.sample() # pi(a/s) in action via pd 
        log_prob = pd.log_prob(action) #log_prob of pi(a/s)
        self.log_probs.append(log_prob) #store for training
        return action.item()
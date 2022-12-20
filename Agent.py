import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from collections import namedtuple, deque
from itertools import count
import random

class DQN(nn.Module):
    def __init__(self,num_actions): # num_actions = dim of obs
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(num_actions,64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.critic = nn.Linear(128,num_actions) # 2 action values
        
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
     
    def forward(self,x): # (batch) obs -> (batch) 2 action values 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('before batched,',x)
        x = self.d_D_batch(x)
        # print('fwd x',x)
        x = x.to(device)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.critic(x)
        return x
        
        # int to float
    def d_D(self,aaa):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        aaa = aaa.float()
        aaa = torch.unsqueeze(aaa, dim=0)
        return aaa.to(device)
    
    def d_D_batch(self,aaa):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        aaa = aaa.float()
        return aaa.to(device)

    def select_1st(self,tensor):
        toReturn = []
        for i in range(len(tensor)):
            toReturn.append(tensor[i][0])
        toReturn = torch.stack(toReturn,dim=0)
        return toReturn
    
'''non-learning agent'''
class Agent_1(nn.Module):
    def __init__(self):
        super().__init__()
        '''todo'''
        pass
        
    def forward(self):
        '''todo'''
        pass
    
    def get_action(self,obs): # obs: list
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        X = []
        for i in obs:
            X.append(i.cpu().numpy().copy())
        x = X
        length = len(x)
        COOP = 1
        for i in range(length):
            if x[i][1] == 0:
                continue
            else:
                COOP = 0
        if COOP == 0:
            y = 1 # defect
        else:
            y = 0
            # y = np.random.choice([0,1],1, p=[0.85, 0.15])[0] # cooperate
        y = torch.tensor([y]).to(device)
        return y

    
    def d_D(self,aaa):
        bbb = []
        for i in range(len(aaa)):
            bbb.append(aaa[i].to(torch.float))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return torch.tensor([bbb]).to(device)


'''変わるnon-learning agent'''
class Agent_2(nn.Module):
    def __init__(self):
        super().__init__()
        '''todo'''
        pass
        
    def forward(self):
        '''todo'''
        pass
    

    
    
    
    
    
    
    


    
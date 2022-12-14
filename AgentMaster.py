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

''' source
https://github.com/johannah/bootstrap_dqn/blob/master/dqn_model.py

'''

class HeadNet(nn.Module):
    
    def __init__(self,num_actions):
        super(HeadNet, self).__init__()
        self.critic = nn.Linear(128,num_actions) # 2 actions

    def forward(self,x):
        x = self.critic(x)
        return x 



class BootDQN(nn.Module):
    
    def __init__(self,num_actions,num_heads):
        super(BootDQN, self).__init__()
        ''' body net '''
        self.fc1 = nn.Linear(num_actions,64)
        self.fc2 = nn.Linear(64,128)
        ''' head net '''
        self.net_list = nn.ModuleList([HeadNet(num_actions) for k in range(num_heads)])
        ''' NaiveDQN '''
        self.num_heads = num_heads


    def forward(self,x,k):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = self.d_D_batch(x)
        x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)  
        
        if k is not None and k < self.num_heads: 
            '''BootDQN'''
            x = self.net_list[k](x)
        else:
            '''NaiveDQN'''
            x = self.net_list[0](x)


        
        return x



    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    ''' tensor: int to float '''
    def d_D_batch(self,aaa):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        aaa = aaa.float()
        return aaa.to(device)



class DQN(nn.Module):
    
    def __init__(self,num_actions): # num_actions = dim of obs
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(num_actions,64)
        # self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,128)
        # self.bn2 = nn.BatchNorm1d(128)
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
    
    # noise? 厳しくなる　tft agent 一旦valleyに落ちちゃったら回復できるかどうか
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


'''todo'''
class Agent_2(nn.Module):
    def __init__(self):
        super().__init__()
        '''todo'''
        pass
        
    def forward(self):
        '''todo'''
        pass
    

    
    
    
    
    
    
    


    
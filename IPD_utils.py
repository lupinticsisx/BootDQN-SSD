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

'''Util functions: ptz format <-> tensor''' 
def batchify_obs(obs, device): # obs is a dict {agent0:action, agent1:action}
    """Converts PZ style observations to batch of torch arrays."""
    # convert dict to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    # obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device): # x.type = dict
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env): # returns a dict. 
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy().copy()
    # print('list unbatch ',list(x))
    # x = np.array(list(x))
    # print('unbatchified: ',x)
    # x = [int(x[0][0]),int(x[0][1])]
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    # print('unbatchified: ',x)
    return x
    

''' returns to_pd object, not buffer_csv'''
def save_buffer(data,output_dir): # input: return_buffer
    to_pd = pd.DataFrame(data, columns=['episodic_return'])
    to_pd.to_csv(output_dir,index=False)
    return to_pd
    ''' return none''' 

def plot_buffer(data,color="rocket_r",rolling=50,y='episodic_return'): # input: csv
    data = pd.read_csv(data)
    # sns.set_theme(style="whitegrid")
    data = data[y].rolling(rolling).mean()
    sns.lineplot(data=data, palette=sns.color_palette(color), linewidth=1.0)
    return

def plot(data,rolling=50,y='episodic_return'): # input: csv
    data = pd.read_csv(data)
    # sns.set_theme(style="whitegrid")
    data = data[y].rolling(rolling).mean()
    sns.set(rc={'figure.figsize':(13.7,7.27)})
    sns.lineplot(data=data, palette=sns.color_palette("magma", as_cmap=True), linewidth=1.0)
    return

def save_loss(data,output_dir):
    D = []
    for i in data:
        D.append(i.cpu().detach().numpy())
    data = D
    to_pd = pd.DataFrame(data, columns=['episodic_loss'])
    to_pd.to_csv(output_dir,index=False)
    return to_pd

def pop(L):
    L.pop(0)
    return L


def save_action(data,output_dir):
    D = []
    for i in data:
        D.append(i.cpu().detach().numpy())
    data = D
    to_pd = pd.DataFrame(data, columns=['player 0','player 1'])
    to_pd.to_csv(output_dir,index=False)
    return to_pd
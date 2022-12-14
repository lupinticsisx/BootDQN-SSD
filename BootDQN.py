
''' 
BootStrapped DQN implemented by minato

source and inspired by
https://github.com/johannah/bootstrap_dqn/blob/master/dqn_model.py

'''

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
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1


from ipd_v6 import ipd
from IPD_utils import batchify_obs, batchify, unbatchify, save_buffer, plot_buffer, save_loss, pop
from AgentMaster import HeadNet, BootDQN, DQN, Agent_1, Agent_2 

import argparse

parser = argparse.ArgumentParser(description='BootDQN IPD')
# args.batch_size <-- 128
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)') 
# args.num_episodes <-- 10
parser.add_argument('--num-episodes', type=int, default=10, help='number of epochs to train (default: 10)') 
# args.seed <-- 1
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)') 
# args.max_cycles <-- 96
parser.add_argument('--max-cycles', type=int, default=96, help='number of cycles per episode (default: 96)') 
# args.obs_lim <-- 3
parser.add_argument('--obs-lim', type=int, default=3, help='observation chain length (default: 3)') 
# args.memory_size <-- 98304
parser.add_argument('--memory-size', type=int, default=98304, help='max buffer size (default: 98304)') 
# args.learning_rate <-- 0.001
parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate. Default Optimizer: Adam (default: 0.001)') 
# args.target_update <-- 10
parser.add_argument('--target-update', type=int, default=10, help='freeze target Q-net for every ... steps (default: 10)')
# args.num_heads <-- 4
parser.add_argument('--num-heads', type=int, default=4, help='number of Q heads for BootDQN (default: 4)')
# args.bernoulli <-- 0.5
parser.add_argument('--bernoulli', type=float, default=0.5, help='probablity of sharing (default: 0.5)')




args = parser.parse_args(args=[])
# torch.manual_seed(args.seed)

''' replay buffer for DQN''' # state = obs, next_state = next_obs
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward', 'mask'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

'''ENV SETUP'''


''' LEARNER SETUP '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent_1 = Agent_1().to(device)


'''buffers to return'''


'''epsilon greedy'''



class TRAINER(object): # trainer for DQN agent
    
    def __init__(self,args,env):
        ''' args '''
        self.seed = args.seed
        self.max_cycles = args.max_cycles
        self.batch_size = args.batch_size
        self.num_episodes = args.num_episodes
        self.obs_lim = args.obs_lim
        self.memory_size = args.memory_size
        self.learning_rate = args.learning_rate
        self.target_update = args.target_update
        self.env = env
        print('agents: ',self.env.possible_agents)
        self.num_agents = len(self.env.possible_agents)
        self.num_actions = self.env.action_space(self.env.possible_agents[0]).n
        self.observation_size = self.env.observation_space(self.env.possible_agents[0]).shape

        ''''''
        self.random_state = np.random.RandomState(self.seed)
        self.p = args.bernoulli

        '''epsilon greedy''' 
        self.GAMMA = 0.999 # args.gamma
        # self.EPS_START = 0.9 # args.eps_start
        # self.EPS_END = 0.05 # args.eps_end
        # self.EPS_DECAY = 200 # args.eps_decay

        ''' buffer dtype '''
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'mask'))
        '''GPU'''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ''' net '''
        self.num_heads = args.num_heads
        self.policy_net = BootDQN(num_actions=2, num_heads=self.num_heads).to(self.device)
        self.target_net = BootDQN(num_actions=2, num_heads=self.num_heads).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        '''optimizer'''
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        '''buffers to return'''
        self.loss_buffer = [] 
        self.return_buffer = []
        self.action_buffer = []
        
    def optimize_model(self,memory):
        if len(memory) < self.batch_size: # 128
            return
        transitions = memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.unsqueeze(s,dim=0) for s in batch.next_state
                                                    if s is not None])

        help_batch_state = []
        for i in batch.state:
            help_batch_state.append(torch.unsqueeze(i,dim=0))

        
        '''BootDQN

        '''


        state_batch = torch.cat(help_batch_state,0)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        mask_batch = torch.cat(batch.mask)

        # take out the k-th element of mask_batch and dot corresp. head 
        for k in range(self.num_heads):
            mask_batch_k = []
            for i in range(batch_size):
                mask_batch_k.append(mask_batch[i][k])
            mask_batch_k = torch.tensor(mask_batch_k).to(device)
            state_action_values = self.policy_net(state_batch,k).gather(1, action_batch)
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states,k).max(1)[0].detach()
            criterion = nn.SmoothL1Loss()
            loss = criterion(torch.dot(state_action_values,mask_batch_k), torch.dot(expected_state_action_values,mask_batch_k).unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # # Compute V(s_{t+1}) for all next states.
        # # Expected values of actions for non_final_next_states are computed based
        # # on the "older" target_net; selecting their best reward with max(1)[0].
        # # This is merged based on the mask, such that we'll have either the expected
        # # state value or 0 in case the state was final.
        # next_state_values = torch.zeros(self.batch_size, device=self.device)
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # # Compute the expected Q values
        # expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # # Compute Huber loss
        # criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # # Optimize the model
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.loss_buffer.append(loss)
        # self.optimizer.step()
        
    def select_action(self,state,steps_done,k): # state = input(obs)
        with torch.no_grad():
            return self.policy_net(state,k).max(-1)[1].view(1, 1)

    def sample_mask(self,p,num_heads):
        # prob = torch.empty(num_heads).uniform_(0, 1)
        return torch.bernoulli(torch.tensor[p]).to(self.device)
    

    def train(self):
        '''variables'''
        
        SEED = self.seed
        MAX_CYCLES = self.max_cycles
        OBS_LIM = self.obs_lim
        device = self.device
        MEMORY_SIZE = self.memory_size
        memory = ReplayMemory(MEMORY_SIZE)
        LEARNING_RATE = self.learning_rate
        BATCH_SIZE = self.batch_size
        TARGET_UPDATE = self.target_update
        num_episodes = self.num_episodes
        env = self.env
        OBS_LIM = self.obs_lim
        max_cycles = MAX_CYCLES # turns within an episode
        end_step = MAX_CYCLES
        seed = SEED
        
        for i_episode in range(num_episodes):
            print('>>> episode :',i_episode)
            ''' BootDQN '''
            heads = list(range(self.num_heads))
            self.random_state.shuffle(heads)
            active_head = heads[0]
            #
            steps_done = 0
            episode_durations = []
            total_episodic_return = 0
            env.reset(seed=seed) # returns None
            next_obs = env.first_obs() # type: dict; {agent: 0 for agent in self.agents}
            OBS = [] # batch_obs
            # will gradually increase the depth challenge of exploration
            if i_episode%100 == 0 and i_episode>3:
                OBS_LIM += 1
            OBS_lim = OBS_LIM # exploration depth
            
            for step in range(0, max_cycles):
                obs = batchify_obs(next_obs, device)
                if len(OBS) < OBS_lim:
                    OBS.append(obs)
                else:
                    pop(OBS)
                    OBS.append(obs)
                action = self.select_action(obs,steps_done,active_head)
                steps_done += 1
                action_1 = agent_1.get_action(OBS)
                actions = torch.cat((action[0],action_1))
                self.action_buffer.append(actions)
                unbatchified = unbatchify(actions, env)
                env.step(unbatchified)
                ''' rwd:  {'player_0': 5, 'player_1': 0}
                    next obs: {'player_0': 1, 'player_1': 0} '''
                rewards = env.rewards
                obs = env.observations
                next_obs = env.observations
                # to store into the buffer
                tensor_obs = batchify_obs(obs,device) # tensor([0., 0.], device='cuda:0')
                rb_rewards = batchify(rewards, device) # tensor([5., 0.], device='cuda:0')
                # make rewards to the right shape
                m_rewards = torch.tensor([rb_rewards[0]]).to(device)
                mask = self.sample_mask(self.p, self.num_heads)
                memory.push(actions, action, tensor_obs, m_rewards, mask) # memory.push(state, action, next_state, reward, mask)
                # to record the return during training
                total_episodic_return += rb_rewards[0].cpu().numpy()
                # update target every TARGET_UPDATE steps
                if step % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            # plotting
            episode_durations.append(max_cycles + 1)
            # plot_durations()
            # to record the return during training
            print('Data Buffering Complete')
            self.return_buffer.append(np.mean(total_episodic_return))
            # optimization
            self.optimize_model(memory)
            print('Training Complete on this episode')
    
        
        print('ALL Trainings Complete')
        print('>>> actions saved')
        print('>>> training loss saved')
        return self.return_buffer, self.loss_buffer, self.action_buffer
        
        
        
        


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        


    
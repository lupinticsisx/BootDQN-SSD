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


from ipd_v7 import ipd
import IPD_utils
from IPD_utils import batchify_obs, batchify, unbatchify, save_buffer, plot_buffer, save_loss, pop
import Agent
from Agent import DQN, Agent_1


import argparse

parser = argparse.ArgumentParser(description='DQN IPD')
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

args = parser.parse_args(args=[])
# torch.manual_seed(args.seed)

''' replay buffer for DQN''' # state = obs, next_state = next_obs
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
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
# SEED = args.seed
# MAX_CYCLES = args.max_cycles
# OBS_LIM = args.obs_lim
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = ipd('ansi').env
# num_agents = len(env.possible_agents)
# num_actions = env.action_space(env.possible_agents[0]).n
# observation_size = env.observation_space(env.possible_agents[0]).shape

''' LEARNER SETUP '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent_1 = Agent_1().to(device)
# policy_net = DQN(num_actions=2).to(device)
# target_net = DQN(num_actions=2).to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()
# print('agents: ',env.possible_agents)
# MEMORY_SIZE = args.memory_size
# memory = ReplayMemory(MEMORY_SIZE)
# LEARNING_RATE = args.learning_rate
# optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
# BATCH_SIZE = args.batch_size
# TARGET_UPDATE = args.target_update
# num_episodes = args.num_episodes
# max_cycles = MAX_CYCLES # turns within an episode
# end_step = MAX_CYCLES
# seed = SEED
# steps_done = 0
# episode_durations = []

'''buffers to return'''
# return_buffer = [] # for storing mean episodic return during training
# loss_buffer = [] # for recording the loss
# action_buffer = [] # for recording agent's actions

'''epsilon greedy'''
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200


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

        '''epsilon greedy''' 
        self.GAMMA = 0.999 # args.gamma
        self.EPS_START = 0.9 # args.eps_start
        self.EPS_END = 0.05 # args.eps_end
        self.EPS_DECAY = 200 # args.eps_decay
        ''' ... '''
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ''' net '''
        self.policy_net = DQN(num_actions=2).to(self.device)
        self.target_net = DQN(num_actions=2).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        self.loss_buffer = [] # for recording the loss
        self.return_buffer = []
        self.return_buffer_x = []
        self.action_buffer = []
        
    def optimize_model(self,memory):
        if len(memory) < self.batch_size:
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
        state_batch = torch.cat(help_batch_state,0)
        # print('state batch',state_batch)
        action_batch = torch.cat(batch.action)
        # print('action batch',action_batch)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # print('loss',loss)
        self.loss_buffer.append(loss)
        # print('ok')
        # for param in policy_net.parameters():
        #     print('param.grad',param.grad)
        #     print('param.grad.data',param.grad.data)
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def select_action(self,state,steps_done): # state = input(obs)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(-1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)
    
    

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
        '''buffers to return''' 
        # return_buffer = [] # for storing mean episodic return during training
        # loss_buffer = [] # for recording the loss
        # action_buffer = [] # for recording agent's actions
        OBS_LIM = self.obs_lim
        max_cycles = MAX_CYCLES # turns within an episode
        end_step = MAX_CYCLES
        seed = SEED
        
        for i_episode in range(num_episodes):
            print('>>> episode :',i_episode)
            steps_done = 0
            episode_durations = []
            total_episodic_return = 0
            total_episodic_return_x = 0
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
                action = self.select_action(obs,steps_done)
                steps_done += 1
                # print('action 0',action)
                # action_1 = agent_1.get_action(obs)
                action_1 = agent_1.get_action(OBS)
                # print('action 1',action_1)
                actions = torch.cat((action[0],action_1))
                # print('actions',actions)
                self.action_buffer.append(actions)
                unbatchified = unbatchify(actions, env)
                env.step(unbatchified)
                ''' rwd:  {'player_0': 5, 'player_1': 0}
                    next obs: {'player_0': 1, 'player_1': 0} '''
                rewards = env.rewards
                obs = env.observations
                # print('obs-2',env.observations)
                next_obs = env.observations
                # to store into the buffer
                tensor_obs = batchify_obs(obs,device) # tensor([0., 0.], device='cuda:0')
                rb_rewards = batchify(rewards, device) # tensor([5., 0.], device='cuda:0')
                # make rewards to the right shape
                m_rewards = torch.tensor([rb_rewards[0]]).to(device)
                memory.push(actions, action, tensor_obs, m_rewards) # memory.push(state, action, next_state, reward)
                # to record the return during training
                total_episodic_return += rb_rewards[0].cpu().numpy()
                total_episodic_return_x += rb_rewards[1].cpu().numpy()
                # update target every TARGET_UPDATE steps
                if step % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            # plotting
            episode_durations.append(max_cycles + 1)
            # plot_durations()
            # to record the return during training
            print('Data Buffering Complete')
            print('Training Complete')
            self.return_buffer.append(np.mean(total_episodic_return))
            self.return_buffer_x.append(np.mean(total_episodic_return_x))
            # optimization
            self.optimize_model(memory)
    
        
        print('ALL Training Complete')
        print('actions saved')
        print('training loss saved')
        return self.return_buffer, self.return_buffer_x, self.loss_buffer, self.action_buffer
        
        
        
        
# def train_agent(args = args):
#     torch.manual_seed(args.seed)
#     Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))
#     '''ENV SETUP'''
#     SEED = args.seed
#     MAX_CYCLES = args.max_cycles
#     OBS_LIM = args.obs_lim
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ''' LEARNER SETUP '''
#     MEMORY_SIZE = args.memory_size
#     memory = ReplayMemory(MEMORY_SIZE)
#     LEARNING_RATE = args.learning_rate
#     optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
#     BATCH_SIZE = args.batch_size
#     TARGET_UPDATE = args.target_update
#     num_episodes = args.num_episodes
#     max_cycles = MAX_CYCLES # turns within an episode
#     end_step = MAX_CYCLES
#     seed = SEED
    
#     # not included in init
#     steps_done = 0 
#     episode_durations = [] 
#     '''buffers to return''' 
#     return_buffer = [] # for storing mean episodic return during training
#     loss_buffer = [] # for recording the loss
#     action_buffer = [] # for recording agent's actions
    
#     '''train the agent upon every episode completion'''
#     OBS_LIM = args.obs_lim
    
#     for i_episode in range(num_episodes):
#         print('>>> episode :',i_episode)
#         total_episodic_return = 0
#         env.reset(seed=seed) # returns None
#         next_obs = env.first_obs() # type: dict; {agent: 0 for agent in self.agents}
#         OBS = [] # batch_obs
#         # will gradually increase the depth challenge of exploration
#         if i_episode%100 == 0 and i_episode>3:
#             OBS_LIM += 1
#         OBS_lim = OBS_LIM # depth_exploration
        
#         for step in range(0, max_cycles):
#             # print('next_obs',next_obs)
#             obs = batchify_obs(next_obs, device)
#             if len(OBS) < OBS_lim:
#                 OBS.append(obs)
#             else:
#                 pop(OBS)
#                 OBS.append(obs)
#             # print('obs',obs)
#             # print('OBS',OBS)
#             action = select_action(obs)
#             # print('action 0',action)
#             # action_1 = agent_1.get_action(obs)
#             action_1 = agent_1.get_action(OBS)
#             # print('action 1',action_1)
#             actions = torch.cat((action[0],action_1))
#             # print('actions',actions)
#             action_buffer.append(actions)
#             unbatchified = unbatchify(actions, env)
#             env.step(unbatchified)
#             # rwd:  {'player_0': 5, 'player_1': 0}
#             # next obs: {'player_0': 1, 'player_1': 0}
#             rewards = env.rewards
#             obs = env.observations
#             # print('obs-2',env.observations)
#             next_obs = env.observations
#             # to store into the buffer
#             tensor_obs = batchify_obs(obs,device) # tensor([0., 0.], device='cuda:0')
#             rb_rewards = batchify(rewards, device) # tensor([5., 0.], device='cuda:0')
#             # make rewards to the right shape
#             m_rewards = torch.tensor([rb_rewards[0]]).to(device)
#             memory.push(actions, action, tensor_obs, m_rewards) # memory.push(state, action, next_state, reward)
#             # to record the return during training
#             total_episodic_return += rb_rewards[0].cpu().numpy()
#             # update target every TARGET_UPDATE steps
#             if step % TARGET_UPDATE == 0:
#                 target_net.load_state_dict(policy_net.state_dict())
#         # plotting
#         episode_durations.append(max_cycles + 1)
#         # plot_durations()
#         # to record the return during training
#         print('Data Buffering Complete')
#         print('Training Complete')
#         return_buffer.append(np.mean(total_episodic_return))
#         # optimization
#         optimize_model(memory)
    
#     '''train ends'''
#     print('ALL Training Complete')
#     print('actions saved')
#     print('training loss saved')
    
    
#     return return_buffer, loss_buffer, action_buffer




# def select_action(state): # state = input(obs)
#     global steps_done
#     sample = random.random()
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#         math.exp(-1. * steps_done / EPS_DECAY)
#     steps_done += 1
#     if sample > eps_threshold:
#         with torch.no_grad():
#             # t.max(1) will return largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             return policy_net(state).max(-1)[1].view(1, 1)
#     else:
#         return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)


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
        
# def optimize_model(memory):
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))
#     # print(batch)
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                           batch.next_state)), device=device, dtype=torch.bool)
#     non_final_next_states = torch.cat([torch.unsqueeze(s,dim=0) for s in batch.next_state
#                                                 if s is not None])

#     help_batch_state = []
#     for i in batch.state:
#         help_batch_state.append(torch.unsqueeze(i,dim=0))
#     state_batch = torch.cat(help_batch_state,0)
#     # print('state batch',state_batch)
#     action_batch = torch.cat(batch.action)
#     # print('action batch',action_batch)
#     reward_batch = torch.cat(batch.reward)

#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     state_action_values = policy_net(state_batch).gather(1, action_batch)
    
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1)[0].
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch

#     # Compute Huber loss
#     criterion = nn.SmoothL1Loss()
#     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     # print('loss',loss)
#     loss_buffer.append(loss)
#     # print('ok')
#     # for param in policy_net.parameters():
#     #     print('param.grad',param.grad)
#     #     print('param.grad.data',param.grad.data)
#     #     param.grad.data.clamp_(-1, 1)
#     optimizer.step()

    
    

    
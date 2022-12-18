'''ipd_v7: 
first_obs: {0,0} -> {random, random}
episodic randomized (just a little)
when Bob C, Alice will C only if Bob has been C for 3+(or more).
When bob D, Alice will D for sure.
changed sigma 0.5->0.3
'''

import functools
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

COOPERATE = 0
DEFECT = 1
MOVES = ['COOPERATE','DEFECT']
NUM_ITERS = 100



def ipd(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """
    
    metadata = {"render_modes": ["ansi"], "name": "ipd"}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(2)] # 2 players
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        ) # no change

        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        self._action_spaces = {agent: Discrete(2) for agent in self.possible_agents} # 2 actions, C or D
        # self._action_spaces = {agent: Y for (agent,Y) in zip(self.possible_agents,[0,1])}
        self._observation_spaces = {
            agent: Discrete(2) for agent in self.possible_agents
        } # 2 possible actions - C or D
        self.render_mode = render_mode
        self._none = None
        self.REWARD = {
            (COOPERATE, COOPERATE): (3,3),
            (COOPERATE, DEFECT): (0,5),
            (DEFECT, COOPERATE): (5,0),
            (DEFECT, DEFECT): (1,1),
        }

        
    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(2) # consistant

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2) # consistant

    def render(self): # no change
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.WARN(
                "You are calling render method without specifying any render mode."
            )
            return

        return 

    def observe(self, agent): # no change
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self): # no change
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, return_info=False, options=None): # 
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        NONE = None
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents} # None
        self.observations = {agent: None for agent in self.agents} # None
        self.num_moves = 0
        # np.random.seed(seed)
        c_c = np.random.normal(3, 0.3, 1)[0]
        c_d_c = np.random.normal(0, 0.3, 1)[0]
        c_d_d = np.random.normal(5, 0.3, 1)[0]
        d_c_d = np.random.normal(5, 0.3, 1)[0]
        d_c_c = np.random.normal(0, 0.3, 1)[0]
        d_d = np.random.normal(1, 0.3, 1)[0]
        REWARD_MAP = {
            (COOPERATE, COOPERATE): (c_c,c_c),
            (COOPERATE, DEFECT): (c_d_c,c_d_d),
            (DEFECT, COOPERATE): (d_c_d,d_c_c),
            (DEFECT, DEFECT): (d_d,d_d),
        }
        # print(REWARD_MAP)
        
        '''change this to make randomized reward'''
        self.REWARD = REWARD_MAP
        print(self.REWARD)
        
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        
    def first_obs(self):
        # x = np.random.randint(2,size=1)[0]
        # y = np.random.randint(2,size=1)[0]
        # self.to_return = {self.agents[0]:x,self.agents[1]:y}
        self.to_return = {agent: 0 for agent in self.agents}
        return self.to_return
    


    def step(self, action):  # for batch action execution. action.type = dict
        """
        step(action) takes in an action (actions) for both agentsand needs to update
        - rewards
        - terminations
        - truncations
        - infos
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
     
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        # self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agents[0]],self.state[self.agents[1]] = action[self.agents[0]],action[self.agents[1]]
        # print('self.state: ',self.state)
        
        # rewards for all agents are placed in the .
        # rewards: dictionary
        
        self.rewards[self.agents[0]], self.rewards[self.agents[1]] = self.REWARD[
            (self.state[self.agents[0]], self.state[self.agents[1]])
        ]
        # print(self.rewards)

        self.num_moves += 1
        # The truncations dictionary must be updated for all players.
        self.truncations = {
            agent: self.num_moves >= NUM_ITERS for agent in self.agents
        }

        # observe the current state
        for i in self.agents:
            self.observations[i] = self.state[
                self.agents[1 - self.agent_name_mapping[i]]
            ]

                
        # Adds .rewards to ._cumulative_rewards
        # self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
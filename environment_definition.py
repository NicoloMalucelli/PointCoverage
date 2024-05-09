import functools

import gymnasium
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

import numpy as np
import random

ACTIONS_ENCODING = ((0,-1),(0,1),(-1,0),(1,0),(0,0))

class EnvironmentSettings:
    def __init__(self, width, height, n_agents, n_targets):
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.n_targets = n_targets


def env(environmentSettings: EnvironmentSettings, render_mode=None,):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(environmentSettings, render_mode=internal_render_mode)
    """
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    """
    return env


def raw_env(environmentSettings: EnvironmentSettings,render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(environmentSettings, render_mode=render_mode)
    #env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "collaboration_to_cover_targets_v1"}

    def __init__(self, environmentSettings: EnvironmentSettings, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        These attributes should not be changed after initialization.
        """
        self.environmentSettings = environmentSettings
        self.possible_agents = ["agent-" + str(i) for i in range(environmentSettings.n_agents)]
        self.agent_name_mapping = {self.possible_agents[i]: i for i in range(environmentSettings.n_agents)}
        self.render_mode = render_mode

    # Each agent knows the position of each agent and the position of each target
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        multi_shape = tuple([(self.environmentSettings.width, self.environmentSettings.height)] * (self.environmentSettings.n_agents + self.environmentSettings.n_targets))
        return MultiDiscrete(np.array(multi_shape))

    # Each agent can move UP, DOWN, LEFT, RIGHT or stay still
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(ACTIONS_ENCODING))

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        str = '_' * (self.environmentSettings.width+2) + '\n'
        
        for i in range(self.environmentSettings.height):
            str = str + "|"
            for j in range(self.environmentSettings.width):
                if (j,i) in self.agents_positions:
                    str = str + 'o'
                elif (j,i) in self.targets_positions:
                    str = str + 'x'
                else:
                    str = str + ' '
            str = str + '|\n'
        str = str + '‾' * (self.environmentSettings.width+2)
        print(str)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.targets_positions = [self.__get_random_position() for i in range(self.environmentSettings.n_targets)]
        self.agents_positions = [self.__get_random_position() for i in range(self.environmentSettings.n_agents)]

        observations = {agent: self.__get_observation() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos # each agent has the same view about the environment (redundant info returned)

    def __get_observation(self):
        return self.agents_positions + self.targets_positions
    
    def __get_random_position(self):
        return (random.randint(0, self.environmentSettings.width-1), 
            random.randint(0, self.environmentSettings.height-1)) 
    
    def __get_covered_targets(self):
        return set(self.targets_positions) - self.__get_uncovered_targets()
    
    def __get_uncovered_targets(self):
        return set(self.targets_positions) - set(self.agents_positions)

    def __compute_global_reward(self):
        # the number of covered targets multiplied by the number of agents
        return len(self.__get_covered_targets()) * len(self.agents)

    def __compute_local_reward(self, agent):
        #if len(self.__get_uncovered_targets()) == 0:
        #    return 0 # is it any non-null value the same? even zero?
        # the local reward of an agent is the euclidean distance from the closest (not covered) target
        agent_id = self.agent_name_mapping[agent]
        agent_position = self.agents_positions[agent_id]
        if agent_position in self.targets_positions:
            return 0
        return -min([np.sqrt(pow(agent_position[0]-target[0], 2)+pow(agent_position[1]-target[1], 2)) for target in self.__get_uncovered_targets()])

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """

        if  len(actions) != len(self.agents):
            gymnasium.logger.warn(f"the number of actions ({len(actions)}) does not match the number of agents ({len(self.agents)})")
            return {}, {}, {}, {}, {}

        # update the position of each agent based on the actions received as argument
        for agent in self.agents:
            self.agents_positions[self.agent_name_mapping[agent]] = tuple(map(sum, zip(
                ACTIONS_ENCODING[actions[agent]], 
                self.agents_positions[self.agent_name_mapping[agent]]
            ))) 
            

        global_reward = self.__compute_global_reward()        
        rewards = {
            agent: global_reward + self.__compute_local_reward(agent)
            for agent in self.agents
        }

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        #self.state = self.__get_observation() #TODO I don't know if self.state is mandatory or not!
        observations = {agent: self.state  for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

        """
        if self.render_mode == "human":
            self.render()
        """

"""
To interact with your custom parallel environment, use the following code:

import parallel_rps

env = parallel_rps.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
"""
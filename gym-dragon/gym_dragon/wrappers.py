"""
This module defines various wrappers for modifying environments and observations.
"""
import gym
import numpy as np
import scipy

from collections import defaultdict
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID
from typing import Hashable

from .core import *



### Environment Wrappers

class EnvWrapper(gym.Wrapper, MultiAgentEnv):
    """
    Wraps a MultiAgentEnv to allow a modular transformation of
    the `step()` and `reset()` methods.
    """

    def __init__(self, env: MultiAgentEnv):
        """
        Parameters
        ----------
        env : DragonBaseEnv
            Base environment
        """
        gym.Wrapper.__init__(self, env)
        MultiAgentEnv.__init__(self)

        # RLlib compatibility
        self.get_agent_ids = env.get_agent_ids
        self.observation_space_contains = env.observation_space_contains
        self.observation_space_sample = env.observation_space_sample
        self.action_space_contains = env.action_space_contains
        self.action_space_sample = env.action_space_sample

    def reset(self, **kwargs) -> dict:
        """
        Reset the environment.
        """
        return super().reset(**kwargs)

    def step(self, actions: dict) -> tuple[dict, float, bool, dict]:
        """
        Take a step in the environment.

        Parameters
        ----------
        actions : dict
            Dictionary mapping from agent ID to agent action
        """
        return super().step(actions)

class GlobalObservationsWrapper(EnvWrapper):
    """
    Wrapper to feed global observations to each agent from step().
    """
    @property
    def observation_space(self):
        return gym.spaces.Dict({
            agent_id: self.env.observation_space
            for agent_id in self.env.agents
        })

    def step(self, actions: dict) -> tuple[dict, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics.
        Observations for each agent are global observations.

        Parameters
        ----------
        actions : dict
            Dictionary mapping from agent ID to agent action
        """
        obs, reward, done, info = self.env.step(actions)
        obs = {agent_id: obs for agent_id in self.env.agents}
        return obs, reward, done, info

class GymWrapper(EnvWrapper):
    """
    Wrapper to remove dict in observations, rewards, done
    """
    def reset(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment.
        """
        obs = super().reset(**kwargs)
        obs_list = []
        available_actions_list = []
        # iterate over agent_id in ray multiagentdict
        for agent_id, agent_obs in obs.items():
            agent_obs_list = []
            agent_available_actions_list = []
            # iterate over observations type in agent_obs
            for k, v in agent_obs.items():
                agent_obs_list.append(v.astype(np.float32).reshape(1, -1))
                if k == 'action_mask':
                    agent_available_actions_list.append(v.astype(np.float32).reshape(1, -1))
            # obs: [shape: (1, obs_dims)]
            # available_actions: [shape: (1, action_dims)]
            obs_list.append(np.concatenate(agent_obs_list, axis=-1))
            available_actions_list.append(np.concatenate(agent_available_actions_list, axis=-1))
        # obs: [shape: (num_agents, obs_dims)]
        # available_actions: [shape: (num_agents, action_dims)]
        obs = np.concatenate(obs_list, axis=0)
        available_actions = np.concatenate(available_actions_list, axis=0)

        return obs, available_actions

    def step(self, actions: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Take a step in the environment.

        Parameters
        ----------
        actions : dict
            Dictionary mapping from agent ID to agent action
        """
        # Take step in the base environment
        obs, rewards, dones, info = super().step(actions)

        obs_list = []
        rewards_list = []
        dones_list = []
        available_actions_list = []
        # iterate over agent_id in ray multiagentdict
        for agent_id, agent_obs in obs.items():
            agent_obs_list = []
            agent_available_actions_list = []
            # iterate over observations type in agent_obs
            for k, v in agent_obs.items():
                agent_obs_list.append(v.astype(np.float32).reshape(1, -1))
                if k == 'action_mask':
                    agent_available_actions_list.append(v.astype(np.float32).reshape(1, -1))
            # obs: [shape: (1, obs_dims)]
            obs_list.append(np.concatenate(agent_obs_list, axis=-1))
            # agent rewards
            rewards_list.append(rewards[agent_id])
            # agent dones
            dones_list.append(dones[agent_id])
            # available_actions: [shape: (1, action_dims)]
            available_actions_list.append(np.concatenate(agent_available_actions_list, axis=-1))
        # obs: [shape: (num_agents, obs_dims)]
        obs = np.concatenate(obs_list, axis=0)
        # rewards [shape: (num_agents, 1)]
        rewards = np.expand_dims(np.array(rewards_list), axis=-1)
        # dones [shape: (num_agents)]
        dones = np.array(dones_list)
        # available_actions: [shape: (num_agents, action_dims)]
        available_actions = np.concatenate(available_actions_list, axis=0)

        return obs, rewards, dones, info, available_actions

class RewardShapingWrapper(EnvWrapper):
    """
    Environment wrapper with additional rewards.

    Subclasses should implement `additional_reward()`.

    Examples
    --------
    Reward shaping wrappers can be additively composed.

    For instance, to have `reward = r_env + 0.01 * r_A + 0.1 * r_B`, do:

    >>> env = DragonEnv()
    >>> env = RewardShapingWrapperA(env, weight=0.01)
    >>> env = RewardShapingWrapperB(env, weight=0.1)
    """

    def __init__(self, env: MultiAgentEnv, weight=1):
        """
        Parameters
        ----------
        env : MultiAgentEnv
            Base environment
        weight : float, default=1
            Multiplicative weighting factor for reward shaping terms
        """
        super().__init__(env)
        self.weight = weight

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict]:
        """
        Run one timestep of the environment's dynamics.

        Parameters
        ----------
        actions : dict
            Dictionary mapping from agent ID to agent action
        """
        args = self.additional_reward_args()

        # Take step in the base environment
        obs, env_reward, done, info = super().step(actions)

        # Combine rewards
        reward = self.additional_reward(actions, *args)
        reward = {
            agent_id: self.weight * reward[agent_id] + env_reward[agent_id]
            for agent_id in env_reward
        }

        return obs, reward, done, info

    def additional_reward_args(self) -> tuple:
        """
        Calculate info **before** taking a step in the base environment.

        Returns
        -------
        args : tuple
            Extra arguments to be passed to `additional_reward()`
            after base environment step
        """
        return ()

    def additional_reward(self, actions: dict, *args) -> dict:
        """
        Calculate additional rewards **after** taking a step in the base environment.

        Parameters
        ----------
        actions : dict
            Dictionary mapping from agent ID to agent action
        args : Any
            Info returned by `additional_reward_args()`

        Returns
        -------
        reward : dict
            Dictionary mapping agent ID to rewards
        """
        return defaultdict(float)


class TimePenalty(RewardShapingWrapper):
    """
    Time penalty to motivate complete task asap.
    """

    def additional_reward(self, actions):
        Action = self.action_enum
        reward = defaultdict(float) # indexed by agent ID

        for agent_id, action in actions.items():
            agent, action = self.agents[agent_id], Action(action)
            reward[agent.id] -= 1

        return reward

class ExploreReward(RewardShapingWrapper):
    """
    Reward agents for visiting new nodes.
    """

    def reset(self, **kwargs):
        self.visited_nodes = defaultdict(set) # indexed by agent ID
        return super().reset(**kwargs)

    def additional_reward(self, actions):
        Action = self.action_enum
        reward = defaultdict(float) # indexed by agent ID

        for agent_id, action in actions.items():
            agent, action = self.agents[agent_id], Action(action)
            if agent.node not in self.visited_nodes[agent.id]:
                self.visited_nodes[agent.id].add(agent.node)
                reward[agent.id] += 1

        return reward


class InspectReward(RewardShapingWrapper):
    """
    Reward agents for inspecting new bombs.
    """

    def reset(self, **kwargs):
        self.inspected_bombs = defaultdict(set) # indexed by agent ID
        return super().reset(**kwargs)

    def additional_reward(self, actions):
        Action = self.action_enum
        reward = defaultdict(float) # indexed by agent ID

        for agent_id, action in actions.items():
            agent, action = self.agents[agent_id], Action(action)
            if action == Action.inspect_bomb:
                if agent.bomb not in self.inspected_bombs[agent.id]:
                    self.inspected_bombs[agent.id].add(agent.bomb)
                    reward[agent.id] += 1

        return reward


class DefusalReward(RewardShapingWrapper):
    """
    Reward agents for each step in bomb defusal sequence.
    """

    def additional_reward_args(self):
        return ({agent.id: agent.bomb for agent in self.agents.values()},)

    def additional_reward(self, actions, bombs):
        Action = self.action_enum
        reward = defaultdict(float) # indexed by agent ID

        for agent_id, action in actions.items():
            agent, action = self.agents[agent_id], Action(action)
            if action.tool() in Tool.bomb_tools():
                if bombs[agent.id] is not None:
                    if bombs[agent.id].state == Bomb.BombState.exploded:
                        reward[agent.id] -= 1
                    else:
                        reward[agent.id] += 1
        return reward


class BeaconReward(RewardShapingWrapper):
    """
    Reward agents for properly using beacons.
    """

    def additional_reward_args(self):
        return ({agent.id: agent.node for agent in self.agents.values()},)

    def additional_reward(self, actions, prev_nodes):
        Action = self.action_enum
        reward = defaultdict(float) # indexed by agent ID

        for agent_id, action in actions.items():
            agent, action = self.agents[agent_id], Action(action)

            if action.node():
                # Reward moving to node with rally beacon
                if Tool.rally_beacon in agent.node.beacons:
                    reward[agent.id] += 1

                # Penalize moving away from node with rally beacon
                if Tool.rally_beacon in prev_nodes[agent.id].beacons:
                    reward[agent.id] -= 1

            # Reward placing rally beacon if all agents not at same node
            if action == Action.use_tool(Tool.rally_beacon):
                all_agent_nodes = {_agent.node for _agent in self.agents.values()}
                reward[agent.id] += (1 if len(all_agent_nodes) > 1 else -1)

            # Reward removing rally beacon if all agents at same node
            if action == Action.remove_beacon(Tool.rally_beacon):
                all_agent_nodes = {_agent.node for _agent in self.agents.values()}
                reward[agent.id] += (1 if len(all_agent_nodes) == 1 else -1)

            # Reward placing bomb beacon at node with bombs
            if action == Action.use_tool(Tool.bomb_beacon):
                reward[agent.id] += (1 if agent.node.bombs else -1)

            # Reward removing bomb beacon at node without bombs
            if action == Action.remove_beacon(Tool.bomb_beacon):
                reward[agent.id] += (1 if not agent.node.bombs else -1)

        return reward


class ProximityReward(RewardShapingWrapper):
    """
    Reward agents for being close to each other.
    """

    def additional_reward(self, actions):
        locations = np.array([agent.node.centroid for agent in self.agents.values()])
        mean_pairwise_dist = scipy.spatial.distance.pdist(locations).mean()
        return {agent_id: -mean_pairwise_dist for agent_id in self.agents}



### Observation Wrappers

class ObservationWrapper(Observation):
    """
    Wrapper used to modify the observations returned by the environment.
    To do this, override the `space()` and `to_dict()` methods.

    Attributes
    ----------
    obs : Observation
        The base observation

    Examples
    --------
    Use a single observation wrapper:

    >>> env = DragonEnv(obs_wrapper=Memory)

    Compose multiple observation wrappers:

    >>> def obs_wrapper(obs):
    ...     obs = ObsWrapperA(obs, **kwargs_A)
    ...     obs = ObsWrapperB(obs, **kwargs_B)
    ...     obs = ObsWrapperC(obs, **kwargs_C)
    ...     return obs

    >>> env = DragonEnv(obs_wrapper=obs_wrapper)
    """

    def __init__(self, obs: Observation):
        """
        Parameters
        ----------
        obs : Observation
            Base `Observation` instance
        """
        self.obs = obs

        # `Observation` methods
        self.tick = self.obs.tick
        self.reset = self.obs.reset
        self.update_from_node = self.obs.update_from_node
        self.update_from_agent_node = self.obs.update_from_agent_node
        self.update_from_inspection = self.obs.update_from_inspection
        self.update_from_sensor = self.obs.update_from_sensor

    def __repr__(self) -> str:
        """
        String representation of this object.
        """
        return f'{self.__class__.__name__}<{self.obs}>'

    @property
    def space(self) -> gym.spaces.Dict:
        """
        Return the observation space.
        """
        return self.obs.space

    @property
    def agent(self) -> Agent:
        """
        The agent this observation corresponds to.
        """
        return self.obs.agent

    @property
    def graph(self) -> Graph:
        """
        The graph being observed.
        """
        return self.obs.graph

    @property
    def agent_id_to_index(self) -> dict[AgentID, int]:
        """
        Agent ID to integer index (i.e. for 'agents' field of the observation).
        """
        return self.obs.agent_id_to_index

    @property
    def node_id_to_index(self) -> dict[Hashable, int]:
        """
        Node ID to integer index (i.e. for 'graph' field of the observation).
        """
        return self.obs.node_id_to_index

    def to_dict(self, *args, **kwargs) -> dict[str, np.ndarray]:
        """
        Return observation as a dictionary.
        """
        return self.obs.to_dict(*args, **kwargs)


class DummyObs(ObservationWrapper):
    """
    Observation wrapper that returns a dummy observation.
    """

    def __init__(self, obs: Observation):
        super().__init__(obs)

    @property
    def space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict()

    def to_dict(self, *args, **kwargs) -> dict[str, np.ndarray]:
        return {}

class MiniObs(ObservationWrapper):
    """
    Only returns the observation vector of the current node that the agent is in
    Reduce the observation space from dict to array
    """

    @property
    def space(self) -> gym.spaces.Box:
        self_indicator_arr = 1
        self_tool_arr = 3
        current_node_arr = self.obs._graph_arr.shape[1]
        obs_shape = self_indicator_arr+self_tool_arr+current_node_arr
        return gym.spaces.Box(
                low=-1, high=float('inf'), shape=(obs_shape,), dtype=float)


    def to_dict(self, *args, **kwargs):

        # if self.obs.agent.node:
        #     self.update_from_agent_node()
        self_indicator = self.node_id_to_index[self.obs.agent.node.id]
        #print(self_indicator)
        self_tool = self.obs.agent.available_tools(return_mask=True)[0:3]
        #print(self_tool)
        node_obs = self.obs._graph_arr[self_indicator]
        #print(node_obs)
        dim = self.obs._graph_arr.shape[1] + 1 + 3
        x, index = np.empty(dim, dtype=float), 1
        x[0] = self_indicator
        for vec in [self_tool,node_obs]:
            x[index:index + len(vec)] = vec
            index += len(vec)

        return x



class Memory(ObservationWrapper):
    """
    Observation wrapper that carries over data from previous observations.
    This `Memory` object essentially acts as an "belief state".

    Observation space includes an additional field:

    * 'memory_weights'
        An array of shape (num_nodes,) indicating how recently each node was observed
    """



    def __init__(self, obs: Observation, decay=None):
        """
        Parameters
        ----------
        obs : Observation
            Base `Observation` instance
        decay : float, optional
            Float within [0, 1] that controls how fast memory should decay per second
        """
        super().__init__(obs)
        self._decay = 1 if decay is None else decay
        self._memory_weights = np.zeros(len(self.node_id_to_index))

    @property
    def space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            **self.obs.space,
            'memory_weights': gym.spaces.Box(
                low=0, high=1, shape=self._memory_weights.shape, dtype=float),
        })

    def reset(self):
        super().reset()
        self._memory_weights = np.zeros_like(self._memory_weights)

    def tick(self, dt=1, **kwargs):
        super().tick(dt=dt, reset=False)
        self._memory_weights *= (self._decay ** dt)

    def update_from_node(self, node: Node, **kwargs):
        super().update_from_node(node, **kwargs)
        node_index = self.node_id_to_index[node.id]
        self._memory_weights[node_index] = 1

    def to_dict(self, *args, **kwargs) -> dict[str, np.ndarray]:
        obs_dict = super().to_dict(*args, **kwargs)
        obs_dict['memory_weights'] = self._memory_weights
        return obs_dict


class EdgeIndex(ObservationWrapper):
    """
    Observation wrapper that includes an edge index of graph connectivity.
    Useful for graph neural networks in frameworks like `torch_geometric`.

    Observation space includes an additional field:

    * 'edge_index'
        An array of shape (2, num_edges) indicating graph connectivity
    """

    @property
    def space(self) -> gym.spaces.Dict:
        edge_index = self.graph.to_vec()[1]
        return gym.spaces.Dict({
            **self.obs.space,
            'edge_index': gym.spaces.Box(
                low=0, high=len(self.graph.nodes)-1, shape=edge_index.shape, dtype=float),
        })

    def to_dict(self, *args, **kwargs) -> dict[str, np.ndarray]:
        obs_dict = super().to_dict(*args, **kwargs)
        obs_dict['edge_index'] = self.graph.to_vec()[1]
        return obs_dict


class ShowAllAgentLocations(ObservationWrapper):
    """
    Observation wrapper that always includes locations of all agents.
    """
    def to_dict(self, *args, **kwargs) -> dict[str, np.ndarray]:
        for agent in self.graph.agents:
            self.update_from_node(
                agent.node, include_agents=True, include_beacons=False, include_bombs=False)

        return super().to_dict(*args, **kwargs)


class ShowAllAgentNodes(ObservationWrapper):
    """
    Observation wrapper that shows all agent nodes, effectively merging observations.
    """
    def to_dict(self, *args, **kwargs) -> dict[str, np.ndarray]:
        for agent in self.graph.agents:
            self.update_from_node(
                agent.node, include_agents=True, include_beacons=True, include_bombs=True)

        return super().to_dict(*args, **kwargs)


class FullyObservable(ObservationWrapper):
    """
    Fully observable wrapper.
    """
    def __init__(self, obs: Observation):
        super().__init__(Memory(obs))

    def to_dict(self, *args, **kwargs) -> dict[str, np.ndarray]:
        for agent in self.graph.agents:
            self.update_from_node(
                agent.node,
                bomb=agent.bomb,
                include_agents=True,
                include_beacons=True,
                include_bombs=True,
                include_bomb_inspection=True
            )

        return super().to_dict(*args, **kwargs)

class QMIX_mini(ObservationWrapper):
    """
    Fully observable wrapper.
    """

    @property
    def space(self) -> gym.spaces.Dict:
        self_indicator_arr = 1
        self_tool_arr = 3
        current_node_arr = self.obs._graph_arr.shape[1]
        obs_shape = self_indicator_arr+self_tool_arr+current_node_arr
        obs_space =  gym.spaces.Box(
                low=-1, high=float('inf'), shape=(obs_shape,), dtype=float)

        state_space = gym.spaces.Box(
                low=-1, high=float('inf'), shape=self.obs._graph_arr.shape, dtype=float)

        return gym.spaces.Dict({
            'obs':obs_space,
            'state': state_space,
        })



    def to_dict(self, *args, **kwargs):

        if self.obs.agent.node:
            self.update_from_agent_node()
        self_indicator = self.node_id_to_index[self.obs.agent.node.id]
        #print(self_indicator)
        self_tool = self.obs.agent.available_tools(return_mask=True)[0:3]
        #print(self_tool)
        node_obs = self.obs._graph_arr[self_indicator]
        #print(node_obs)
        dim = self.obs._graph_arr.shape[1] + 1 + 3
        x, index = np.empty(dim, dtype=float), 1
        x[0] = self_indicator
        for vec in [self_tool,node_obs]:
            x[index:index + len(vec)] = vec
            index += len(vec)


        for agent in self.graph.agents:
            self.update_from_node(
                agent.node,
                bomb=agent.bomb,
                include_agents=True,
                include_beacons=False,
                include_bombs=True,
                include_bomb_inspection=True
            )


        return {'obs':x,'state':self.obs._graph_arr}

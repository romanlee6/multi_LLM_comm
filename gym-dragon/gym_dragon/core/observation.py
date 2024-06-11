"""
Module for representing agent observations in the environment.

**Observation Space**

A standard observation for each agent contains the following items:

* 'agents'
    An array of shape (num_agents, num_agent_features)
* 'graph'
    An array of shape (num_nodes, num_node_features)
* 'action_mask'
    An array of shape (num_actions,)

**Agent Representation**

Each "agent vector" in the agent array contains the following variables:

* boolean indicator (i.e. "this is me")
* whether agent is frozen
* whether agent is protected
* number of remaining tool uses for each tool

Variables that are unknown (e.g. for other agents) are marked as `UNKNOWN`.

**Node Representation**

Each "node vector" in the node array contains the following variables:

* whether each agent is present or absent
* whether each help beacon type is present or absent
* number of bombs in node
* bomb index within node of "currently loaded" bomb for the observing agent
* vector for the "currently loaded" bomb if one is present, otherwise all zeros
* number of ignited cells within node

Variables that are unknown (e.g. bomb sequence that has not yet been inspected)
are marked as `UNKNOWN`.

**Bomb Representation**

Each "bomb vector" contains the following:

* boolean indicator that bomb is present
* whether bomb is active
* bomb ID
* remaining countdown for the bomb
* bomb ID of parent bomb in chain dependency (if any)
* the remaining bomb sequence (with previous sequence steps masked to 0)

Variables that are unknown (e.g. bomb sequence that has not yet been inspected)
are marked as `UNKNOWN`.
"""
import gym.spaces
import numpy as np

from ray.rllib.utils.typing import AgentID
from typing import Hashable, Iterable

from .agent import Agent
from .graph import Node, Graph
from .world import WorldObj

from ..utils import UNKNOWN



### Classes

class Observation(WorldObj):
    """
    Class for representing and updating observations from the environment.

    Observations contain the following items:

    * 'agents'
        An array of shape (num_agents, num_agent_features) <see `Agent.to_vec()`>
    * 'graph'
        An array of shape (num_nodes, num_node_features) <see `Node.to_vec()`>
    * 'action_mask'
        An array of shape (num_actions,)

    Attributes
    ----------
    space : gym.spaces.Dict
        The observation space
    agent : Agent
        The agent this observation corresponds to
    graph : Graph
        The graph being observed
    agent_id_to_index : dict[AgentID, int]
        Agent ID to integer index (i.e. for 'agents' field of the observation)
    node_id_to_index : dict[Hashable, int]
        Node ID to integer index (i.e. for 'graph' field of the observation)
    """

    def __init__(
        self, agent: Agent, graph: Graph, agent_ids: Iterable[AgentID], num_actions: int):
        """
        Parameters
        ----------
        agent : Agent
            The agent this observation corresponds to
        graph : Graph
            The graph being observed
        agent_ids : Iterable[AgentID]
            Agent IDs
        num_actions : int
            The number of actions in the action space (for action mask)
        """
        super().__init__()
        self._agent = agent
        self._graph = graph

        # Indexing
        self._agent_id_to_index = {agent_id: i for i, agent_id in enumerate(sorted(agent_ids))}
        self._node_id_to_index = {node_id: i for i, node_id in enumerate(sorted(graph.nodes))}

        # Observation data
        self._agents_arr = UNKNOWN * np.ones((len(agent_ids), Agent.dim))
        self._graph_arr = graph.to_vec()[0]

        # Observation space
        self._space = gym.spaces.Dict({
            'agents': gym.spaces.Box(
                low=UNKNOWN, high=1, shape=self._agents_arr.shape, dtype=float),
            'graph': gym.spaces.Box(
                low=UNKNOWN, high=float('inf'), shape=self._graph_arr.shape, dtype=float),
            'action_mask': gym.spaces.Box(0, 1, shape=(num_actions,), dtype=bool),
        })

    def __repr__(self) -> str:
        """
        String representation of this object.
        """
        return f'{self.__class__.__name__}({self._agent.id})'

    @property
    def space(self) -> gym.spaces.Dict:
        """
        The observation space.
        """
        return self._space

    @property
    def agent(self) -> Agent:
        """
        The agent this observation corresponds to.
        """
        return self._agent

    @property
    def graph(self) -> Graph:
        """
        The graph being observed.
        """
        return self._graph

    @property
    def agent_id_to_index(self) -> dict[AgentID, int]:
        """
        Agent ID to integer index (i.e. for 'agents' field of the observation).
        """
        return self._agent_id_to_index

    @property
    def node_id_to_index(self) -> dict[Hashable, int]:
        """
        Node ID to integer index (i.e. for 'graph' field of the observation).
        """
        return self._node_id_to_index

    def reset(self, *args, **kwargs):
        """
        Clear all data from this observation.
        """
        super().reset(*args, **kwargs)
        self._agents_arr.fill(UNKNOWN)
        self._graph_arr = self._graph.to_vec()[0]

    def tick(self, dt: float = 1, reset: bool = True, **kwargs):
        """
        Advance time by a given interval.

        Parameters
        ----------
        dt : float, default=1
            Seconds per tick
        reset : bool, default=True
            Whether or not to reset this observation
        """
        super().tick(dt=dt, **kwargs)
        if reset:
            self.reset()

    def update_from_node(self, node: Node, **kwargs):
        """
        Update observation with information from the given node.

        Parameters
        ----------
        node : Node
            The observed node
        **kwargs : dict
            Keyword arguments passed to `Node.to_vec()`
        """
        node_index = self.node_id_to_index[node.id]
        node_vector = node.to_vec(**kwargs)
        mask = (node_vector != UNKNOWN)
        self._graph_arr[node_index][mask] = node_vector[mask]

    def update_from_agent_node(self):
        """
        Update observation with information from the agent's current node.
        """
        # Update observation for any agents observed at current node
        for agent in self._agent.node.agents:
            # Update agent vector
            this_is_me = (agent.id == self._agent.id)
            agent_index = self.agent_id_to_index[agent.id]
            agent_vector = agent.to_vec(indicator=this_is_me)
            mask = (agent_vector != UNKNOWN)
            self._agents_arr[agent_index][mask] = agent_vector[mask]

            # Update graph vector to show agents are no longer present at other nodes
            self._graph_arr[:, agent_index] = 0

        # Update observation at observer agent's current node
        self.update_from_node(self._agent.node, bomb=self._agent.bomb)

    def update_from_inspection(self):
        """
        Update observation from an inspected bomb.
        """
        self.update_from_node(
            self._agent.node, bomb=self._agent.bomb, include_bomb_inspection=True)

    def update_from_sensor(self, node: Node):
        """
        Update observation from the sensor tool.

        Parameters
        ----------
        node : Node
            The node indicated by the sensor
        """
        if node.bombs:
            self.update_from_node(
                node,
                bomb=node.bombs[0],
                include_agents=False,
                include_beacons=False,
                include_hazards=False,
            )

    def to_dict(self, action_mask=None) -> dict[str, np.ndarray]:
        """
        Return observation as a dictionary.

        Parameters
        ----------
        action_mask : Optional[np.ndarray]
            Boolean mask over available actions
        """
        if self._agent.node:
            self.update_from_agent_node()

        if action_mask is None:
            action_mask = np.ones(self.space['action_mask'].shape)

        return {
            'agents': self._agents_arr,
            'graph': self._graph_arr,
            'action_mask': action_mask,
        }

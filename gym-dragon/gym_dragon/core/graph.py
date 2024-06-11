"""
Module for representing a spatial graph over the environment.
"""
import numpy as np

from collections import defaultdict
from ray.rllib.utils.typing import AgentID
from typing import Hashable, Iterable, Optional

from .agent import Agent
from .world import Bomb, BombBeacon, HelpBeacon, Region, Tool, WorldObj

from ..utils import cycle_encoding, np_replace, UNKNOWN



class Node(WorldObj):
    """
    Representation of a node in the environment graph.

    Attributes
    ----------
    id : Hashable
        Unique identifier for this node
    centroid : tuple[int, int]
        The (x, z) centroid for this node
    area : Optional[int]
        2D cell area of this node
    region : Region
        The region this node is located in
    agents : list[Agent]
        List of agents currently present at this node
    bombs : list[Bomb]
        List of bombs currently present at this node
    bomb_beacons : list[BombBeacon]
        List of bomb beacons currently present at this node
    help_beacons : list[HelpBeacon]
        List of help beacons currently present at this node
    fire_locations : set[tuple[int, int]]
        Set of fire cell locations within this node
    """

    def __init__(
        self,
        node_id: Hashable,
        centroid: tuple[int, int],
        agent_ids: Iterable[AgentID],
        graph=None):
        """
        Parameters
        ----------
        node_id : Hashable
            Unique identifier for this node
        centroid : tuple[int, int]
            The (x, z) centroid for this node
        agent_ids : Iterable[AgentID]
            Unique IDs for all agents
        graph : Graph, optional
            Parent graph for this node
        """
        super().__init__()
        self._agent_to_index = {agent_id: i for i, agent_id in enumerate(sorted(agent_ids))}
        self._dim = Node.dim(num_agents=len(self._agent_to_index))
        self._id = node_id
        self._centroid = centroid
        self._region = Region.from_location(centroid)
        self._area = None
        self._graph = graph
        self.reset()

    def __repr__(self) -> str:
        """
        String representation of this object.
        """
        return f'Node({self.id.__repr__()})'

    @property
    def id(self) -> Hashable:
        """
        Unique identifier for this node.
        """
        return self._id

    @property
    def centroid(self) -> tuple[int, int]:
        """
        The (x, z) centroid for this node.
        """
        return self._centroid

    @property
    def area(self) -> Optional[int]:
        """
        2D cell area of this node.
        """
        return self._area

    @area.setter
    def area(self, value) -> int:
        """
        Set the 2D cell area of this node.
        """
        if self._area is None:
            self._area = value

    @property
    def region(self) -> Region:
        """
        The region this node is located in.
        """
        return self._region

    @property
    def agents(self) -> list['Agent']:
        """
        List of agents currently present at this node.
        """
        return self._child_dict[Agent]

    @property
    def bombs(self) -> list[Bomb]:
        """
        List of bombs currently present at this node.
        """
        return self._child_dict[Bomb]

    @property
    def bomb_beacons(self) -> list[BombBeacon]:
        """
        List of bomb beacons present at this node.
        """
        return self._child_dict[BombBeacon]

    @property
    def help_beacons(self) -> list[HelpBeacon]:
        """
        List of help beacons present at this node.
        """
        return self._child_dict[HelpBeacon]

    @property
    def fire_locations(self) -> set[tuple[int, int]]:
        """
        Set of fire cell locations within this node.
        """
        return self._fire_locations

    def reset(self, *args, **kwargs):
        """
        Reset this node.
        """
        self.clear_children()
        super().reset(*args, **kwargs)
        self._child_dict = {
            Agent: [],
            Bomb: [],
            BombBeacon: [],
            HelpBeacon: [],
        }

        self._agent_is_present = np.zeros(len(self._agent_to_index), dtype=bool)
        self._num_help_beacons = np.zeros(len(HelpBeacon.MessageType), dtype=bool)
        self._fire_locations = set()

    def tick(self, **kwargs):
        """
        Advance time by a given interval.
        """
        super().tick(**kwargs)

        bomb_beacons = list(self.bomb_beacons)
        for beacon in bomb_beacons:
            if beacon.bomb not in self.bombs:
                self.remove_child(beacon)

    def unfreeze_agents(self):
        """
        Unfreeze all agents at this node.
        """
        for agent in self.agents:
            agent.unfreeze()

    def ignite(self, loc: tuple[int, int]):
        """
        Ignite the given cell location.

        Parameters
        ----------
        loc : tuple[int, int]
            Cell location to ignite
        """
        self._fire_locations.add(loc)
        for bomb in self.bombs:
            if bomb.location == loc:
                bomb.state = Bomb.BombState.exploded

    def extinguish(self, loc: tuple[int, int]):
        """
        Extinguish the given cell location.

        Parameters
        ----------
        loc : tuple[int, int]
            Cell location to extinguish
        """
        self._fire_locations.discard(loc)

    def to_vec(
        self, bomb=None,
        include_agents=True, include_beacons=True,
        include_bombs=True, include_bomb_inspection=False,
        include_hazards=True) -> np.ndarray:
        """
        Return a fixed-size vector representation of an observation over this
        node in the graph.

        Node features include:

        * whether each agent is present or absent
        * whether each help beacon type is present or absent
        * number of bombs in node
        * bomb index for given bomb, within this node
        * vector for given bomb (see `Bomb.to_vec()`)
        * number of ignited cells within node

        Attributes
        ----------
        bomb : Bomb, optional
            The current bomb being observed at this node
        include_agents : bool, default=True
            Whether to include information about which agents are present at this node
        include_beacons : bool, default=True
            Whether to include information about which beacons are active at this node
        include_bombs : bool, default=True
            Whether to include information about bombs at this node
        include_bomb_inspection : bool, default=False
            Whether to include inspected information for observed bomb
        include_hazards : bool, default=True
            Whether to include information about hazards at this node (i.e. fire)
        """
        if not bomb and self.bomb_beacons and include_beacons:
            # Select the bomb corresponding to the most recent bomb beacon
            bomb = self.bomb_beacons[-1].bomb
            include_bombs = True
            include_bomb_inspection = True

        if bomb:
            bomb_index = cycle_encoding(self.bombs.index(bomb), len(self.bombs))
            bomb_vec = bomb.to_vec(include_inspection=include_bomb_inspection)
        else:
            bomb_index = np.zeros(2)
            bomb_vec = np.zeros(Bomb.dim)

        vector_components = [
            (self._agent_is_present, include_agents),
            (self._num_help_beacons, include_beacons),
            ([len(self.bombs)], include_bombs),
            (bomb_index, include_bombs),
            (bomb_vec, include_bombs),
            ([len(self.fire_locations)], include_hazards)
        ]

        x, index = np.empty(self._dim, dtype=float), 0
        for vec, flag in vector_components:
            x[index:index+len(vec)] = vec if flag else UNKNOWN
            index += len(vec)

        return x

    @staticmethod
    def dim(num_agents: int):
        """
        Dimension of `Node.to_vec()`.

        Parameters
        ----------
        num_agents : int
            Total number of agents in the environment
        """
        num_other_features = 3 # number of bombs, bomb index, number of fire cells
        return num_agents + len(HelpBeacon.MessageType) + num_other_features + Bomb.dim

    def add_child(self, obj: WorldObj):
        """
        Add a child object.

        Parameters
        ----------
        obj : WorldObj
            Child object
        """
        super().add_child(obj)
        if type(obj) in self._child_dict:
            self._child_dict[type(obj)].append(obj)
            if isinstance(obj, Agent):
                self._agent_is_present[self._agent_to_index[obj.id]] = True
            elif isinstance(obj, HelpBeacon):
                self._num_help_beacons[obj.message_type] += 1

    def remove_child(self, obj: WorldObj):
        """
        Remove a child object.

        Parameters
        ----------
        obj : WorldObj
            Child object
        """
        super().remove_child(obj)
        if type(obj) in self._child_dict and obj in self._child_dict[type(obj)]:
            self._child_dict[type(obj)].remove(obj)
            if isinstance(obj, Agent):
                self._agent_is_present[self._agent_to_index[obj.id]] = False
            elif isinstance(obj, HelpBeacon):
                self._num_help_beacons[obj.message_type] -= 1


class Graph(WorldObj):
    """
    The environment graph.

    Each node has a (x, z) location, which falls within at most one `Region`.

    Two nodes are connected by an edge if it is possible for an agent to
    traverse between them.

    Attributes
    ----------
    nodes : dict[Hashable, Node]
        Dictionary mapping node ID to `Node` instance
    edges : set[tuple[Hashable, Hashable]]
        Set of edges as `(in_node_id, out_node_id)` tuples
    agents : set[Agent]
        Set of agents in the graph
    regions : set[Region]
        Set of regions in the graph
    """

    def __init__(
        self,
        centroids: dict[Hashable, tuple[int, int]],
        edges: Iterable[tuple[Hashable, Hashable]],
        agents: Iterable[Agent]):
        """
        Parameters
        ----------
        centroids : dict[Hashable, tuple[int, int]]
            Dictionary mapping node ID to (x, z) node centroid
        edges : Iterable[tuple[Hashable, Hashable]]
            Set of edges as `(in_node_id, out_node_id)` tuples
        agents : Iterable[Agent]
            Agents within the graph
        """
        super().__init__()
        self._agents = set(agents)

        # Create nodes
        self._nodes = {}
        for node_id, centroid in centroids.items():
            node = Node(node_id, centroid, (agent.id for agent in agents), graph=self)
            self.add_child(node)
            self._nodes[node_id] = node

        # Create adjacency list
        self._edges = set(edges)
        self._adj_list = defaultdict(list)
        for u, v in self._edges:
            self._adj_list[u].append(v)

        # Vector representation
        nodes_list = sorted(self.nodes.values(), key=(lambda node: node.id))
        self._node_to_index = {node.id: i for i, node in enumerate(nodes_list)}
        self._nodes_arr = None
        self._edge_index = np_replace(np.array(list(self.edges)).T, self._node_to_index)

    def tick(self, **kwargs):
        """
        Advance time by a given interval.
        """
        super().tick(**kwargs)

        # Update vector representation
        if self._nodes_arr is None:
            self.to_vec()
        else:
            for node in {agent.node for agent in self._agents}:
                node_index = self._node_to_index[node.id]
                self._nodes_arr[node_index] = node.to_vec(
                    bomb=None,
                    include_agents=True,
                    include_bombs=False,
                    include_beacons=True,
                    include_hazards=bool(node.help_beacons),
                )

    @property
    def nodes(self) -> dict[Hashable, Node]:
        """
        Dictionary mapping node ID to `Node` instance.
        """
        return self._nodes

    @property
    def edges(self) -> set[tuple[Hashable, Hashable]]:
        """
        Set of edges as `(in_node_id, out_node_id)` tuples.
        """
        return self._edges

    @property
    def agents(self) -> set[Agent]:
        """
        Set of agents in the graph.
        """
        return self._agents

    @property
    def regions(self) -> set[Region]:
        """
        Set of regions in the graph.
        """
        return {node.region for node in self.nodes.values()}

    def neighbors(self, node_id: Hashable) -> list[Hashable]:
        """
        Returns a list of node IDs adjacent to the given node.

        Parameters
        ----------
        node_id : Hashable
            Source node ID
        """
        return self._adj_list[node_id]

    def to_vec(self) -> np.ndarray:
        """
        Return a vector representation of this graph.

        Returns
        -------
        nodes_arr : np.ndarray
            Array of shape (num_nodes, num_node_features) indicating node features
        edge_index : np.ndarray
            Array of shape (2, num_edges) indicating graph connectivity
        """
        if self._nodes_arr is None:
            self._nodes_arr = np.stack([
                node.to_vec(
                    bomb=None,
                    include_agents=True,
                    include_bombs=False,
                    include_beacons=True,
                    include_hazards=bool(node.help_beacons),
                )
                for node in sorted(
                    self.nodes.values(), key=lambda node: self._node_to_index[node.id])
            ])

        return self._nodes_arr, self._edge_index

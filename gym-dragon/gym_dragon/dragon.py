"""
Base implementation of the Dragon environment.
"""
import csv
import gym.spaces
import numpy as np
import pickle

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID
from typing import Callable, Iterable, Optional

from .core import *
from .render import Renderer
from .utils import BoundedGrid, get_item, random_allocation

from .wrappers import ShowAllAgentLocations
from .core.action import MiniActionEnum

### Constants

RESOURCES_DIR = Path(__file__).parent / '../resources'



### Environment

class DragonBaseEnv(MultiAgentEnv, WorldObj):
    """
    Base Dragon environment.

    **Representation**

    Rather than modeling the map on the grid level, we segment the traversable part of
    the map into areas, and create a graph where each node represents one of these areas.
    Observations and actions are abstracted on the graph-level (e.g. observing bombs at
    current node, or traveling to adjacent node).

    **Task**

    The basic Dragon environment has 3 players - red, green, and blue - who must cooperate
    to defuse bombs throughout the map.

    To complete the bomb defusal task, players are required to search for, identify,
    and dispose of two types of bombs: Standard and Volatile (see **gym_dragon.core.world.Bomb**).

    The essential process of bomb defusal requires applying Tools in the correct
    sequence for a given bomb. There are 3 types of sequence steps: red, green, and blue.
    They may be addressed either by application of a player's specialty tool
    (without consuming resources) or by one of their purchased tools (note: purchased tools
    have a limited number of usages, but cover some of the other player’s capabilities;
    see **gym_dragon.core.world.Tool**).

    If a Bomb explodes, nearby players become "injured" and frozen in place for a fixed time.
    Non-frozen players can stabilize and "unfreeze" injured players.

    **Action Space**

    The environment uses a discrete action space, with actions consisting of the following:

    * Action.unfreeze_agents
        Unfreeze agents at the current node
    * Action.inspect_bomb
        Inspect a bomb at the current node
    * Action.find_next_bomb
        Update the agent's observation with the next bomb in the current node
    * Action.place_bomb_beacon
        Place a bomb beacon at the current node
    * Action.place_help_beacon
        Place a help beacon at the current node
    * Action.remove_bomb_beacon
        Remove a bomb beacon from the current node
    * Action.remove_help_beacon
        Remove a help beacon from the current node
    * Action.go_to(node_id)
        Move the agent to the specified node
    * Action.use_tool(tool)
        Apply the specified tool to the current node

    See **gym_dragon.core.action** module for more detail.

    **Observation Space**

    A standard observation for each agent contains the following items:

    * 'agents'
        An array of shape (num_agents, num_agent_features)
    * 'graph'
        An array of shape (num_nodes, num_node_features)
    * 'action_mask'
        An array of shape (num_actions,)

    See **gym_dragon.core.observation** module for more detail.

    **Rewards**

    The team receives the following rewards:

    * +10n for each n-stage bomb successfully defused
    * -0.1 for each percentage of damage taken by agents
    * -10 for each time an agent is frozen

    Attributes
    ----------
    observation_space : gym.spaces.Dict
        The observation space
    action_space : gym.spaces.Discrete
        The action space
    agents : dict[AgentID, Agent]
        Dictionary mapping agent IDs to `Agent` instances
    graph : Graph
        The environment graph
    observations : dict[AgentID, Observation]
        Dictionary mapping Agent IDs to `Observation` instances
    time : float
        Current mission time, in seconds
    timestep : int
        Number of environment steps taken so far in the current epsiode
    score : int
        Total team score
    """
    default_budget_weights = {
        region: {'perturbations': 0.2, 'communications': 0.2, 'bomb_additonal': 0.2}
        for region in Region
    }

    def __init__(
        self,
        mission_length: float = 10*60,
        recon_phase_length: float = 2*60,
        seconds_per_timestep: float = 2.0,
        valid_regions: set[Region] = set(Region),
        obs_wrapper: Optional[Callable[[Observation], Observation]] = None,
        budget_weights: dict[Region, dict[str, float]] = default_budget_weights,
        color_tools_only: bool = False,
        include_fuse_bombs: bool = True,
        include_fire_bombs: bool = True,
        include_chained_bombs: bool = True):
        """
        Parameters
        ----------
        mission_length : float, default=10*60
            Total mission length, in seconds
        recon_phase_length : float, default=2*60
            Length of reconnaissance phase, in seconds
        seconds_per_timestep : float, default=2.0
            Number of seconds per timestep in the environment
        valid_regions : set[Region], optional
            Set of regions to include in the environment
        obs_wrapper : Callable(Observation) -> Observation, optional
            Callable observation wrapper
        budget_weights : dict[Region, dict[str, float]]
            Weights of factors in budget calculation (if budget is not explictly given)
        color_tools_only : bool, default=False
            Whether to restrict agents to color tools only
        include_fuse_bombs : bool, default=True
            Whether to include fuse bombs in the environment
        include_fire_bombs : bool, default=True
            Whether to include fire bombs in the environment
        include_chained_bombs : bool, default=True
            Whether to include chained bombs in the environment
        """
        WorldObj.__init__(self)

        # Initialize agents
        self._agents = {
            'alpha': Agent('alpha', color=Color.red),
            'bravo': Agent('bravo', color=Color.green),
            'charlie': Agent('charlie', color=Color.blue),
        }

        # Load map data
        with open(RESOURCES_DIR / 'dragon_v2.pkl', 'rb') as file:
            data = pickle.load(file)
            self._init_graph(
                data['centroids'], data['edges'], valid_regions=valid_regions)
            self._init_grid(
                data['block_grid'], data['segmentation_grid'], valid_regions=valid_regions)

        # Environment variables
        self._mission_length = mission_length
        self._recon_phase_length = recon_phase_length
        self._seconds_per_timestep = seconds_per_timestep
        self._valid_regions = valid_regions

        # Mission variables
        self._time, self._score = 0, 0

        # Config variables
        self._budget_weights = budget_weights
        self._color_tools_only = color_tools_only
        self._include_fuse_bombs = include_fuse_bombs
        self._include_fire_bombs = include_fire_bombs
        self._include_chained_bombs = include_chained_bombs

        # Create action enum
        Action = ActionEnum(self.agents.values(), self.graph.nodes.values())
        self.action_enum = Action
        self.action_array = np.array(Action)

        # Set action space
        self._action_space = gym.spaces.Discrete(len(Action))

        # Create observations for each agent
        self._observations = {}
        obs_wrapper = obs_wrapper if obs_wrapper else (lambda obs: obs)
        for agent in self.agents.values():
            obs = Observation(agent, self.graph, self.agents, len(Action))
            obs = obs_wrapper(obs)
            self.add_child(obs)
            self._observations[agent.id] = obs

        # Renderer
        self._renderer = None

        MultiAgentEnv.__init__(self)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """
        The observation space.
        """
        return get_item(self.observations.values()).space

    @property
    def share_observation_space(self) -> gym.spaces.Dict:
        """
        The shared observation space
        """
        return gym.spaces.Dict({agent_id: self.observation_space for agent_id in self._agents})

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """
        The action space.
        """
        return self._action_space

    @property
    def agents(self) -> dict[AgentID, Agent]:
        """
        Dictionary mapping agent IDs to `Agent` instances.
        """
        return self._agents

    @property
    def graph(self) -> Graph:
        """
        The environment graph.
        """
        return self._graph

    @property
    def observations(self) -> dict[AgentID, Observation]:
        """
        Dictionary mapping Agent IDs to `Observation` instances.
        """
        return self._observations

    @property
    def time(self) -> float:
        """
        Current mission time, in seconds.
        """
        return self._time

    @property
    def timestep(self) -> int:
        """
        Number of environment steps taken so far in the current epsiode.
        """
        return self._time // self._seconds_per_timestep

    @property
    def score(self) -> int:
        """
        Total team score.
        """
        return self._score

    @property
    def bomb_inspected(self) -> int:
        """
        number of bombs inspected
        """
        return len(self._inspected_bombs)

    @property
    def bomb_defused(self) -> int:
        """
        number of bombs defused
        """
        return len(self._terminal_bombs)

    def reset(
        self,
        env: Optional['DragonBaseEnv'] = None,
        seed: Optional[int] = None,
        csv_path: Optional[str] = None,
        num_bombs_per_region: int = 15,
        budget_per_region: Optional[int] = None,
        start_location: Optional[tuple[int, int]] = (24, 149),
        start_regions: set[Region] = set(Region),
        tool_allocation: Optional[dict[AgentID, dict[Tool, int]]] = None,
        **kwargs) -> MultiAgentDict:
        """
        Reset the environment to an initial state
        and return initial observations for each agent.

        Parameters
        ----------
        seed : int, optional
            Seed for randomness
        env : DragonBaseEnv, optional
            Reference environment to reset state to (overrides other arguments)
        csv_path : str, optional
            Path to CSV file containing bomb distribution
        num_bombs_per_region : int, default=15
            Number of bombs to be placed randomly in each region
        budget_per_region : int, optional
            Team budget per region, to be spent randomly on purchasing tools
        start_location : tuple[int, int], optional, default=(24, 149)
            Initial location for all agents
        start_regions : set[Region], optional
            Set of valid regions for initial mission locations
        tool_allocation : dict[AgentID, dict[Tool, int]], optional
            Tool allocation for each agent, to be used instead of random allocation
        """
        WorldObj.reset(self)
        self._random = np.random.default_rng(seed)

        # Reset to default initial state
        if env:
            self._reset_from_env(env)
        else:
            self._time, self._score = 0, 0
            self._in_recon_phase = True
            self._bombs = []
            self._inspected_bombs = defaultdict(set) # sets of inspected bombs, indexed by agent ID
            self._terminal_bombs = [] # store terminal (defused / exploded) bombs
            self._fires = []
            self._block_grid[...] = self._base_grid[...] # reset grid
            for agent in self.agents.values(): agent.reset(random=self._random) # reset agents

            # Place bombs
            self._place_bombs(
                csv_path=csv_path,
                num_bombs_per_region=num_bombs_per_region)

            # Calculate team budget
            if tool_allocation:
                self._tool_allocation = tool_allocation
            else:
                if budget_per_region:
                    budget = budget_per_region * len(self.graph.regions)
                else:
                    budget = self._calculate_budget()
                self._tool_allocation = self.random_tool_allocation(budget, self._random)

            # Set start node
            start_node = None
            if start_location:
                if self._node_grid.is_in_bounds(start_location):
                    start_node = self._node_grid[start_location]
            if not start_node:
                start_regions = start_regions.intersection(self.graph.regions)
                if not start_regions:
                    start_regions = self.graph.regions
                start_node_candidates = [
                    node for node in self.graph.nodes.values() if node.region in start_regions]
                start_node = self._random.choice(start_node_candidates)
            for agent in self.agents.values():
                agent.go_to(start_node)

            # Reset renderer
            if self._renderer:
                self._renderer.reset(self._block_grid)

        self.tick(dt=0)
        return self._get_obs()

    def step(
        self, actions: MultiAgentDict
    ) -> tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """
        Run one timestep of the environment's dynamics.

        Parameters
        ----------
        actions : MultiAgentDict
            Dictionary mapping from agent ID to agent action

        Returns
        -------
        obs : MultiAgentDict
            Dictionary mapping agent ID to observations
        reward : MultiAgentDict
            Dictionary mapping agent ID to rewards
        done : MultiAgentDict
            Dictionary mapping agent ID to done values
        info : MultiAgentDict
            Dictionary mapping agent ID to info values
        """
        reward = {agent_id: 0 for agent_id in self.agents.keys()}
        info = {agent_id: {} for agent_id in self.agents.keys()}
        prev_agent_health = {agent.id: agent.health for agent in self.agents.values()}

        # Apply actions
        Action = self.action_enum
        for agent_id in self._random.permutation(list(actions.keys())):
            agent, action = self.agents[agent_id], Action(actions[agent_id])

            if action == Action.unfreeze_agents:
                # Unfreeze all agents at the current node
                agent.node.unfreeze_agents()

            if action == Action.inspect_bomb:
                # Inspect the currently observed bomb
                if agent.bomb:
                    agent.bomb.inspect()
                    self._inspected_bombs[agent.id].add(agent.bomb)
                    self.observations[agent.id].update_from_inspection()

            elif action == Action.find_next_bomb:
                # Cycle to the next bomb at the current node
                agent.find_next_bomb()

            elif action == Action.place_bomb_beacon:
                # Place bomb beacon at the current node
                if agent.bomb:
                    agent.node.add_child(BombBeacon(agent.bomb))
                    for _agent_id in self._inspected_bombs:
                        self._inspected_bombs[_agent_id].add(agent.bomb)

            elif action == Action.place_help_beacon:
                # Place help beacon at the current node
                agent.node.add_child(agent.create_help_beacon())

            elif action == Action.remove_bomb_beacon:
                # Remove oldest bomb beacon from the current node
                if agent.node.bomb_beacons:
                    agent.node.remove_child(agent.node.bomb_beacons[0])

            elif action == Action.remove_help_beacon:
                # Remove oldest help beacon from the current node
                if agent.node.help_beacons:
                    agent.node.remove_child(agent.node.help_beacons[0])

            elif action.node() is not None:
                # Go to the given node
                agent.go_to(action.node())

            elif action.tool() is not None:
                # Apply the specified tool
                tool = agent.get_tool_from_inventory(action.tool()) # check inventory
                if tool == Tool.ppe:
                    agent.is_protected = True
                elif tool == Tool.sensor:
                    self.observations[agent.id].update_from_sensor(self._random_node(agent))
                elif tool == Tool.fire_extinguisher:
                    for fire in self._fires:
                        fire.extinguish(*list(agent.node.fire_locations)[:tool.area])
                elif tool in Tool.bomb_tools():
                    if agent.bomb:
                        agent.bomb.apply_tool(tool)
                        if agent.bomb.state == Bomb.BombState.defused and agent.bomb.color == tool.color:
                            for _agent_id in reward:
                                reward[_agent_id] += agent.bomb.value / len(self.agents)

        # Advance time
        self.tick()

        # Penalties for taking damage and becoming frozen
        for agent in self.agents.values():
            damage = max(0, prev_agent_health[agent.id] - agent.health)
            self._score -= 10 * damage
            self._score -= 10 * bool(damage) * bool(agent.is_frozen)
            reward[agent.id] -= 10 * damage
            reward[agent.id] -= 10 * bool(damage) * bool(agent.is_frozen)

        for agent_id in info:
            info[agent_id]['score'] = self.score

        return self._get_obs(), reward, self._get_done(), info

    def tick(self, dt: Optional[float] = None):
        """
        Advance time by a given interval.

        Parameters
        ----------
        dt : float, optional
            Seconds per tick
        """
        dt = self._seconds_per_timestep if dt is None else dt
        self._time += dt

        # Check for end of recon phase
        if self._in_recon_phase and self._time >= self._recon_phase_length:
            self._in_recon_phase = False
            self._assign_tools_from_allocation()

        # Remove terminal bombs (exploded or defused)
        while self._terminal_bombs:
            bomb = self._terminal_bombs.pop()
            node = self._node_grid[bomb.location]
            if bomb in node.bombs:
                node.remove_child(bomb)

        super().tick(dt=dt)

    def random_tool_allocation(
        self, budget: int, random: np.random.Generator) -> dict[AgentID, dict[Tool, int]]:
        """
        Generate a random allocation of tools to each agent given a fixed team budget.

        Parameters
        ----------
        budget : int
            Total budget for generating random tool allocation
        random : np.random.Generator
            Random number generator
        """
        allocation = {}
        agent_budgets = self._get_agent_budgets(budget, random=random)

        # Randomly allocate tools to agents
        for agent_id, agent_budget in agent_budgets.items():
            costs = [tool.cost for tool in Tool]
            exclude = [random.choice(list(Tool.color_tools()))] # random color tool
            if self._color_tools_only:
                exclude += set(Tool) - Tool.color_tools() # all non-color tools
            agent_allocation = random_allocation(
                agent_budget, costs, exclude, random=random)
            allocation[agent_id] = {Tool(t): n for t, n in enumerate(agent_allocation)}

        return allocation

    def render(self, *args, **kwargs):
        """
        Render the environment.

        For kwargs, see `Renderer.__init__()` and `Renderer.render()` in **gym_dragon.render**.
        """
        if not self._renderer:
            self._renderer = Renderer(self, **kwargs)
        self._renderer.render(self._block_grid, **kwargs)

    def observation_space_sample(self, agent_ids=None) -> MultiAgentDict:
        """
        Randomly sample an observation for each agent.

        Parameters
        ----------
        agent_ids : Iterable[AgentID], optional
            Agent IDs to sample observations for
        """
        agent_ids = agent_ids if agent_ids else self.get_agent_ids()
        return {agent_id: obs.space.sample() for agent_id, obs in self.observations.items()}

    def action_space_sample(self, agent_ids=None) -> MultiAgentDict:
        """
        Randomly sample an action for each agent.

        Parameters
        ----------
        agent_ids : Iterable[AgentID], optional
            Agent IDs to sample actions for
        """
        agent_ids = agent_ids if agent_ids else self.get_agent_ids()
        Action = self.action_enum
        actions = {}
        for agent_id in agent_ids:
            mask = self._get_action_mask(agent_id)
            actions[agent_id] = Action(self._random.choice(self.action_array[mask]))

        return actions

    def get_agent_ids(self) -> set:
        """
        Returns a set of agent IDs in the environment.

        This method is needed for rllib compatibility.

        :meta private:
        """
        return set(self.agents.keys())

    def observation_space_contains(self, obs: MultiAgentDict) -> bool:
        """
        Checks if the observation space contains the given key.

        This method is needed for rllib compatibility.

        :meta private:

        Parameters
        ----------
        obs : MultiAgentDict[AgentID, Any]
            Dictionary mapping agent IDs to agent observations
        """
        return all(self.observation_space.contains(obs[agent_id]) for agent_id in obs)

    def action_space_contains(self, actions: MultiAgentDict) -> bool:
        """
        Checks if the action space contains the given key.

        This method is needed for rllib compatibility.

        :meta private:

        Parameters
        ----------
        actions : MultiAgentDict[AgentID, Any]
            Dictionary mapping agent IDs to agent actions
        """
        return all(self.action_space.contains(actions[agent_id]) for agent_id in actions)

    def _get_obs(self) -> MultiAgentDict:
        """
        Get the current observation for each agent.
        """
        return {
            agent_id: obs.to_dict(action_mask=self._get_action_mask(agent_id))
            for agent_id, obs in self.observations.items()
        }

    def _get_done(self) -> MultiAgentDict:
        """
        Get the current done value for each agent.
        """
        done = (
            all(bomb.state.is_terminal() for bomb in self._bombs)
            or self._time >= self._mission_length
        )
        return {key: done for key in ['__all__', *self.agents.keys()]}

    def _get_action_mask(self, agent_id: AgentID) -> np.ndarray[bool]:
        """
        Return a boolean mask over available actions for the given agent.

        Parameters
        ----------
        agent_id : AgentID
            ID of agent to find available actions for

        Returns
        -------
        mask : np.ndarray[bool] of shape (num_actions,)
            Mask over available actions, where `mask[a]` is True
            if and only if `a` is an available action
        """
        Action = self.action_enum
        agent = self.agents[agent_id]
        mask = np.zeros(len(Action), dtype=bool)

        # Placing beacons
        mask[Action.place_bomb_beacon] = bool(agent.node.bombs)
        mask[Action.place_help_beacon] = True

        # Removing beacons
        mask[Action.remove_bomb_beacon] = bool(agent.node.bomb_beacons)
        mask[Action.remove_help_beacon] = bool(agent.node.help_beacons)

        # Handle frozen agent
        if agent.is_frozen:
            return mask

        # Unfreezing other agents
        mask[Action.unfreeze_agents] = (len(agent.node.agents) > 1)

        # Bomb actions
        if agent.node and agent.node.bombs:
            mask[Action.inspect_bomb] = True
            mask[Action.find_next_bomb] = (len(agent.node.bombs) > 1)

        # Going to other nodes
        available_node_ids = (
            self.graph.neighbors(agent.node.id) if agent.node else self.graph.nodes.keys())
        mask[Action.go_to(available_node_ids)] = True

        # Using tools
        mask[Action.use_tool(agent.available_tools(return_mask=True))] = True
        mask[Action.use_fire_extinguisher_tool] &= bool(agent.node.fire_locations)

        return mask

    def _calculate_budget(self) -> int:
        """
        Calculate team budget based on total bomb sequence length.
        Other factors (e.g. perturbations) are weighted relative to bomb defusal budget.
        """
        budget = 0
        for region in Region:
            total_num_bomb_stages = sum([
                bomb.num_stages for bomb in self._bombs if bomb.region == region])
            budget_per_stage = (1 + sum(self._budget_weights[region].values()))
            budget += total_num_bomb_stages * budget_per_stage

        return int(np.ceil(budget))

    def _get_agent_budgets(
        self, team_budget: int, random: np.random.Generator) -> dict[AgentID, int]:
        """
        Randomly allocate team budget to each agent.

        Parameters
        ----------
        team_budget : int
            Total team budget
        random : np.random.Generator
            Random number generator
        """
        # Randomly decide portion of team budget to be used (unless color tools only)
        if not self._color_tools_only:
            team_budget = int(team_budget * random.beta(5, 1))

        # Randomly allocate team budget across agents
        n = len(self.agents)
        agent_budgets = random.uniform(size=n)
        agent_budgets *= team_budget / agent_budgets.sum()
        agent_budgets = agent_budgets.astype(int)
        agent_budgets[random.permutation(n)[:team_budget - agent_budgets.sum()]] += 1

        return dict(zip(self.agents.keys(), agent_budgets))

    def _create_bomb_dependency_chains(
        self,
        region: Region,
        num_chains: int = 1,
        max_chain_length: int = 3,
    ):
        """
        Create bomb dependency chains within the given region.

        Parameters
        ----------
        region : Region
            Region to create bomb dependency chains in
        num_chains : int, default=1
            Number of bomb dependency chains to create
        max_chain_length : int, default=3
            Maximum length of each bomb dependency chain
        """
        region_bombs = [bomb for bomb in self._bombs if bomb.region == region]
        chain_lengths = sorted(
            self._random.integers(2, max_chain_length + 1, size=num_chains))

        self._random.shuffle(region_bombs)
        for chain_length in chain_lengths:
            if len(region_bombs) < chain_length:
                break
            bomb_chain = region_bombs[:chain_length]
            for bomb in bomb_chain:
                region_bombs.remove(bomb)
            for i, bomb in enumerate(bomb_chain[:-1]):
                bomb.dependency = bomb_chain[i + 1]

    def _random_node(self, agent: Agent) -> Optional[Node]:
        """
        Return a random node with an uninspected bomb, within the same region.

        Parameters
        ----------
        agent : Agent
            Agent to find a random node for
        """
        candidate_nodes = [
            node for node in self.graph.nodes.values()
            if node.region == agent.node.region
            and not self._inspected_bombs[agent.id].intersection(node.bombs)
        ]
        return self._random.choice(candidate_nodes) if candidate_nodes else None

    def _on_bomb_state_change(self, bomb: Bomb):
        """
        Callback for when a bomb changes state.

        Parameters
        ----------
        bomb : Bomb
            The bomb that has changed its state
        """
        node = self._node_grid[bomb.location]

        if bomb.state == Bomb.BombState.exploded:
            self._terminal_bombs.append(bomb)
            if self._block_grid[bomb.location] != 'fire':
                self._block_grid[bomb.location] = 'ground'

            # Freeze nearby agents if the bomb exploded
            for agent in node.agents:
                agent.explode()

            def on_fire_change():
                if self._renderer:
                    self._renderer.update_map(self._block_grid)

            # Start fire if the bomb was a fire bomb
            if bomb.is_fire_bomb:
                fire = Fire(
                    bomb.location,
                    self._block_grid,
                    self._node_grid,
                    on_change=on_fire_change,
                )
                node.add_child(fire)
                self._fires.append(fire)

        elif bomb.state == Bomb.BombState.defused:
            self._terminal_bombs.append(bomb)
            self._block_grid[bomb.location] = 'ground'
            self._score += bomb.value

        elif bomb.state == Bomb.BombState.active:
            self._block_grid[bomb.location] = 'bomb_active'

        elif bomb.state == Bomb.BombState.inactive:
            self._block_grid[bomb.location] = 'bomb_inactive'

        # Update renderer
        if self._renderer:
            self._renderer.update_map(self._block_grid)

    def _assign_tools_from_allocation(self):
        """
        Assign agents tools from allocation.
        """
        for agent_id, agent_allocation in self._tool_allocation.items():
            for tool, quantity in agent_allocation.items():
                self.agents[agent_id].assign_tool(tool, quantity)

    def _get_csv_bomb_list(self, csv_path: str, csv_subtract_fuse: int = 0) -> list[dict]:
        """
        Get bombs from CSV file as a list of dictionaries

        Parameters
        ----------
        csv_path : str
            Path to CSV file containing bomb distribution
        csv_subtract_fuse : int, default=0
            Recon phase length to subtract from fuse if reading bomb distribution from CSV
        """
        # Load bomb information from CSV file
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            csv_bomb_list = []
            for row in reader:
                if row['FeatureType'] == 'Bomb':
                    loc = row['LocationXYZ'].split()
                    dependency_id = None if row['ChainedID'] == 'NONE' else int(row['ChainedID'])
                    bomb = {
                        'bomb_id': int(row['BombID']),
                        'location': (int(loc[0]), int(loc[2])),
                        'sequence': [Color.from_char(c) for c in row['Sequence']],
                        'fuse': (int(row['FuseStartTime']) - csv_subtract_fuse) * 60 \
                                if self._include_fuse_bombs else None,
                        'dependency': dependency_id if self._include_chained_bombs else None,
                        'is_fire_bomb': row['BlockType'] == 'block_bomb_fire' if self._include_fire_bombs else False
                    }
                    csv_bomb_list.append(bomb)

            # Sort bombs based on 1) bomb_id, 2) dependency_id, putting bombs with None dependency at start of list
            csv_bomb_list = sorted(
                csv_bomb_list, key=lambda x: (x['dependency'] is not None, x['bomb_id'], x['dependency']))

        return csv_bomb_list

    def _place_bombs(self,
        csv_path: str = None,
        num_bombs_per_region: int = 15):
        """
        Place bombs on the map.

        Parameters
        ----------
        csv_path : str
            Path to CSV file containing bomb distribution
        num_bombs_per_region : int, default=15
            Number of bombs to place in each region (if CSV file is not provided)
        """
        bombs = {}
        # Load bomb information from CSV file
        if csv_path:
            # Create Bomb objects from sorted list
            for bomb_dict in self._get_csv_bomb_list(csv_path):
                bomb = Bomb(
                    bomb_id=bomb_dict['bomb_id'],
                    location=bomb_dict['location'],
                    sequence=bomb_dict['sequence'],
                    fuse=bomb_dict['fuse'],
                    dependency=bombs.get(bomb_dict['dependency'], None),
                    is_fire_bomb=bomb_dict['is_fire_bomb'],
                    on_state_change=self._on_bomb_state_change
                )
                bombs[bomb.id] = bomb
                self._bombs.append(bomb)

        # Randomly create bombs
        else:
            # Randomly select bomb locations from each region
            bomb_locations = []
            for region in self.graph.regions:
                n, k = len(self._empty_locations[region]), num_bombs_per_region
                bomb_locations += list(
                    self._empty_locations[region][self._random.permutation(n)[:k]])

            # Create random bomb for each location
            for bomb_id, location in enumerate(bomb_locations):
                fuse = 60 * self._random.integers(
                    self._recon_phase_length / 60 + 1, self._mission_length / 60)
                bomb = Bomb.random(
                    bomb_id, tuple(location),
                    fuse=(fuse if self._include_fuse_bombs else None),
                    on_state_change=self._on_bomb_state_change,
                    random=self._random,
                    is_fire_bomb=(None if self._include_fire_bombs else False),
                )
                bombs[bomb.id] = bomb
                self._bombs.append(bomb)

            # Randomly create bomb dependency chains
            if self._include_chained_bombs:
                for region in Region:
                    self._create_bomb_dependency_chains(
                        region, num_chains=self._random.integers(1, 4))

        # Place bombs on the map
        for bomb in bombs.values():
            self._node_grid[bomb.location].add_child(bomb) # update graph
            self._block_grid[bomb.location] = 'bomb_inactive' # update grid

    def _reset_from_env(self, env: 'DragonBaseEnv'):
        """
        Reset environment from another environment.

        Parameters
        ----------
        env : DragonBaseEnv
            Reference environment to reset state to
        """
        env = env.unwrapped

        # Reset initial state to current state of reference environment
        self._time, self._score = env.time, env.score
        self._in_recon_phase = env._in_recon_phase
        self._bombs = deepcopy(env._bombs)
        self._inspected_bombs = deepcopy(env._inspected_bombs)
        self._terminal_bombs = deepcopy(env._terminal_bombs)
        self._fires = deepcopy(env._fires)
        self._block_grid[...] = env._block_grid[...]

        # Reset agents
        self._agents.clear()
        self._agents = {agent_id: Agent(agent_id) for agent_id in env.agents}
        for agent in self._agents.values():
            # Set current node
            agent.go_to(env.graph.nodes[env.agents[agent.id].node.id])

            # Set tool allocation
            agent._tool_remaining_uses = env.agents[agent.id]._tool_remaining_uses
            agent._tool_available = env.agents[agent.id]._tool_available

            # Set health
            agent._health = env.agents[agent.id]._health

        # Add bombs
        for bomb in self._bombs:
            self._node_grid[bomb.location].add_child(bomb) # update graph
            self._block_grid[bomb.location] = 'bomb_inactive' # update grid

        # Reset renderer
        if self._renderer:
            self._renderer.reset(self._block_grid)

    def _init_graph(self, centroids: dict, edges: Iterable, valid_regions=set(Region)):
        """
        Initialize the underlying graph for the environment.

        Parameters
        ----------
        centroids : dict[Hashable, tuple[int, int]]
            Dictionary mapping node ID to (x, z) node centroid
        edges : Iterable[tuple[Hashable, Hashable]]
            Set of edges, where (i, j) indicates an edge between nodes i & j
        valid_regions : set[Region]
            Set of regions to include in the graph
        """
        # Filter by region
        centroids = {
            k: v for k, v in centroids.items() if Region.from_location(v) in valid_regions}
        edges = {(i, j) for i, j in edges if i in centroids and j in centroids}

        # Create graph
        self._graph = Graph(centroids, edges, self.agents.values())
        self.add_child(self._graph)

    def _init_grid(
        self, base_grid: BoundedGrid, node_id_grid: BoundedGrid, valid_regions=set(Region)):
        """
        Initialize the underlying grid for the environment.

        Parameters
        ----------
        base_grid : BoundedGrid
            Grid indicating the block type at each (x, z) location
        node_id_grid : BoundedGrid
            Grid indicating the node ID for each (x, z) location
        valid_regions : set[Region]
            Set of regions to include in the grid
        """
        base_grid.replace(None, 'obstacle')
        base_grid.replace('stone_slab', 'water') # for visualization purposes

        # Get bounds
        z_max, z_min = base_grid.bounds[1]
        for region in valid_regions:
            if region == Region.forest:
                z_min, z_max = min(0, z_min), max(50, z_max)
            elif region == Region.village:
                z_min, z_max = min(50, z_min), max(100, z_max)
            elif region == Region.desert:
                z_min, z_max = min(100, z_min), max(151, z_max)

        # Create grid from bounds
        self._base_grid = base_grid.view[0:52, z_min:z_max+1]
        self._block_grid = BoundedGrid(*self._base_grid.bounds)

        # Create node grid with Node objects
        self._node_grid = BoundedGrid(*self._base_grid.bounds)
        for loc in self._node_grid.locations():
            if node_id_grid[loc] in self.graph.nodes:
                self._node_grid[loc] = self.graph.nodes[node_id_grid[loc]]

        # Set node areas
        for node in self.graph.nodes.values():
            node.area = node_id_grid.count(node.id)


        # Get empty locations in each region (for potential bomb placement)
        all_empty_locations = [
            loc for loc in self._base_grid.locations()
            if self._base_grid[loc] == 'ground' and self._node_grid[loc] is not None
        ]
        self._empty_locations = {
            region: np.array([
                loc for loc in all_empty_locations
                if Region.from_location(loc) == region
            ])
            for region in Region
        }


### Environment

class MiniDragonBaseEnv(DragonBaseEnv):
    """
    Base Dragon environment.

    **Representation**

    Rather than modeling the map on the grid level, we segment the traversable part of
    the map into areas, and create a graph where each node represents one of these areas.
    Observations and actions are abstracted on the graph-level (e.g. observing bombs at
    current node, or traveling to adjacent node).

    **Task**

    The basic Dragon environment has 3 players - red, green, and blue - who must cooperate
    to defuse bombs throughout the map.

    To complete the bomb defusal task, players are required to search for, identify,
    and dispose of two types of bombs: Standard and Volatile (see **gym_dragon.core.world.Bomb**).

    The essential process of bomb defusal requires applying Tools in the correct
    sequence for a given bomb. There are 3 types of sequence steps: red, green, and blue.
    They may be addressed either by application of a player's specialty tool
    (without consuming resources) or by one of their purchased tools (note: purchased tools
    have a limited number of usages, but cover some of the other player’s capabilities;
    see **gym_dragon.core.world.Tool**).

    If a Bomb explodes, nearby players become "injured" and frozen in place for a fixed time.
    Non-frozen players can stabilize and "unfreeze" injured players.

    **Action Space**

    The environment uses a discrete action space, with actions consisting of the following:

    * Action.unfreeze_agents
        Unfreeze agents at the current node
    * Action.inspect_bomb
        Inspect a bomb at the current node
    * Action.find_next_bomb
        Update the agent's observation with the next bomb in the current node
    * Action.place_bomb_beacon
        Place a bomb beacon at the current node
    * Action.place_help_beacon
        Place a help beacon at the current node
    * Action.remove_bomb_beacon
        Remove a bomb beacon from the current node
    * Action.remove_help_beacon
        Remove a help beacon from the current node
    * Action.go_to(node_id)
        Move the agent to the specified node
    * Action.use_tool(tool)
        Apply the specified tool to the current node

    See **gym_dragon.core.action** module for more detail.

    **Observation Space**

    A standard observation for each agent contains the following items:

    * 'agents'
        An array of shape (num_agents, num_agent_features)
    * 'graph'
        An array of shape (num_nodes, num_node_features)
    * 'action_mask'
        An array of shape (num_actions,)

    See **gym_dragon.core.observation** module for more detail.

    **Rewards**

    The team receives the following rewards:

    * +10n for each n-stage bomb successfully defused
    * -0.1 for each percentage of damage taken by agents
    * -10 for each time an agent is frozen

    Attributes
    ----------
    observation_space : gym.spaces.Dict
        The observation space
    action_space : gym.spaces.Discrete
        The action space
    agents : dict[AgentID, Agent]
        Dictionary mapping agent IDs to `Agent` instances
    graph : Graph
        The environment graph
    observations : dict[AgentID, Observation]
        Dictionary mapping Agent IDs to `Observation` instances
    time : float
        Current mission time, in seconds
    timestep : int
        Number of environment steps taken so far in the current epsiode
    score : int
        Total team score
    """
    default_budget_weights = {
        region: {'perturbations': 0.2, 'communications': 0.2, 'bomb_additonal': 0.2}
        for region in Region
    }

    def __init__(
            self,
            mission_length: float = 900,
            recon_phase_length: float = 0,
            seconds_per_timestep: float = 2.0,
            valid_regions: set[Region] = set(Region),
            obs_wrapper: Optional[Callable[[Observation], Observation]] = None,
            budget_weights: dict[Region, dict[str, float]] = default_budget_weights,
            color_tools_only: bool = True,
            include_fuse_bombs: bool = False,
            include_fire_bombs: bool = False,
            include_chained_bombs: bool = False):
        """
        Parameters
        ----------
        mission_length : float, default=10*60
            Total mission length, in seconds
        recon_phase_length : float, default=2*60
            Length of reconnaissance phase, in seconds
        seconds_per_timestep : float, default=2.0
            Number of seconds per timestep in the environment
        valid_regions : set[Region], optional
            Set of regions to include in the environment
        obs_wrapper : Callable(Observation) -> Observation, optional
            Callable observation wrapper
        budget_weights : dict[Region, dict[str, float]]
            Weights of factors in budget calculation (if budget is not explictly given)
        color_tools_only : bool, default=False
            Whether to restrict agents to color tools only
        include_fuse_bombs : bool, default=True
            Whether to include fuse bombs in the environment
        include_fire_bombs : bool, default=True
            Whether to include fire bombs in the environment
        include_chained_bombs : bool, default=True
            Whether to include chained bombs in the environment
        """
        WorldObj.__init__(self)

        # Initialize agents
        self._agents = {
            'alpha': Agent('alpha', color=Color.red),
            'bravo': Agent('bravo', color=Color.green),
            'charlie': Agent('charlie', color=Color.blue),
        }
        self._agent_ids = set(agent_id for agent_id in self._agents)
        self.valid_nodes = [(28, 56), (15, 55), (40, 56), (33, 64), (22, 63)]

        # Load map data
        with open(RESOURCES_DIR / 'dragon_v2.pkl', 'rb') as file:
            data = pickle.load(file)
            self._init_graph(
                data['centroids'], data['edges'], valid_nodes=self.valid_nodes)
            self._init_grid(
                data['block_grid'], data['segmentation_grid'], valid_regions=valid_regions)

        # Environment variables
        self._mission_length = mission_length
        self._recon_phase_length = recon_phase_length
        self._seconds_per_timestep = seconds_per_timestep
        self._valid_regions = valid_regions

        # Mission variables
        self._time, self._score = 0, 0

        # Config variables
        self._budget_weights = budget_weights
        self._color_tools_only = color_tools_only
        self._include_fuse_bombs = include_fuse_bombs
        self._include_fire_bombs = include_fire_bombs
        self._include_chained_bombs = include_chained_bombs

        # Create action enum
        Action = MiniActionEnum(self.agents.values(), self.graph.nodes.values())
        self.action_enum = Action
        self.action_array = np.array(Action)

        # Set action space
        self._action_space = gym.spaces.Discrete(len(Action))

        # Create observations for each agent
        self._observations = {}
        obs_wrapper = obs_wrapper if obs_wrapper else (lambda obs: obs)
        for agent in self.agents.values():
            obs = Observation(agent, self.graph, self.agents, len(Action))
            obs = obs_wrapper(obs)
            self.add_child(obs)
            self._observations[agent.id] = obs

        # Renderer
        self._renderer = None
        # self.seed = None
        MultiAgentEnv.__init__(self)

    def seed(self,seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """
        The observation space.
        """
        return get_item(self.observations.values()).space

    @property
    def share_observation_space(self) -> gym.spaces.Dict:
        """
        The shared observation space
        """
        return gym.spaces.Dict({agent_id: self.observation_space for agent_id in self._agents})


    @property
    def action_space(self) -> gym.spaces.Discrete:
        """
        The action space.
        """
        return self._action_space

    @property
    def agents(self) -> dict[AgentID, Agent]:
        """
        Dictionary mapping agent IDs to `Agent` instances.
        """
        return self._agents

    @property
    def graph(self) -> Graph:
        """
        The environment graph.
        """
        return self._graph

    @property
    def observations(self) -> dict[AgentID, Observation]:
        """
        Dictionary mapping Agent IDs to `Observation` instances.
        """
        return self._observations

    @property
    def time(self) -> float:
        """
        Current mission time, in seconds.
        """
        return self._time

    @property
    def timestep(self) -> int:
        """
        Number of environment steps taken so far in the current epsiode.
        """
        return self._time // self._seconds_per_timestep

    @property
    def score(self) -> int:
        """
        Total team score.
        """
        return self._score

    DOUBLE_ALLOCATION  = {'alpha': {Tool.red: 99, Tool.green: 99}, 'bravo': {Tool.blue: 99, Tool.green: 99},
                                     'charlie': {Tool.red: 99, Tool.blue: 99}}

    def get_avail_actions(self):
        mask = []
        for agent_id in self.agents:
            mask.append(self._get_action_mask(agent_id))

        return mask


    def reset(
            self,
            env: Optional['DragonBaseEnv'] = None,
            seed: Optional[int] = None,
            csv_path: Optional[str] = RESOURCES_DIR / 'mturk_bomb.csv',
            # csv_path: Optional[str] ='D:\py\gym_DRAGON\gym-dragon-mturk\mturk_bomb.csv',
            num_bombs_per_region: int = 5,
            budget_per_region: Optional[int] = None,
            start_location: Optional[tuple[int, int]] = (28, 56),
            start_regions: set[Region] = set(Region),
            tool_allocation: Optional[dict[AgentID, dict[Tool, int]]] = DOUBLE_ALLOCATION,
            **kwargs) -> MultiAgentDict:
        """
        Reset the environment to an initial state
        and return initial observations for each agent.

        Parameters
        ----------
        seed : int, optional
            Seed for randomness
        env : DragonBaseEnv, optional
            Reference environment to reset state to (overrides other arguments)
        csv_path : str, optional
            Path to CSV file containing bomb distribution
        num_bombs_per_region : int, default=15
            Number of bombs to be placed randomly in each region
        budget_per_region : int, optional
            Team budget per region, to be spent randomly on purchasing tools
        start_location : tuple[int, int], optional, default=(24, 149)
            Initial location for all agents
        start_regions : set[Region], optional
            Set of valid regions for initial mission locations
        tool_allocation : dict[AgentID, dict[Tool, int]], optional
            Tool allocation for each agent, to be used instead of random allocation
        """
        WorldObj.reset(self)


        if hasattr(self,"rng"):
            self._random = self.rng
        elif seed is not None:
            self._random = np.random.default_rng(seed)
        else:
            seed = 0
            self._random = np.random.default_rng(seed)

        # Reset to default initial state
        if env:
            self._reset_from_env(env)
        else:
            self._time, self._score = 0, 0
            self._in_recon_phase = True
            self._bombs = []
            self._inspected_bombs = defaultdict(set)  # sets of inspected bombs, indexed by agent ID
            self._terminal_bombs = []  # store terminal (defused / exploded) bombs
            self._fires = []
            self._block_grid[...] = self._base_grid[...]  # reset grid
            for agent in self.agents.values(): agent.reset(random=self._random)  # reset agents

            # Place bombs
            if seed == 0:
                self._place_bombs(
                    csv_path=csv_path,
                    num_bombs_per_region=num_bombs_per_region)
            else:
                self._place_bombs(
                    csv_path=None,
                    num_bombs_per_region=num_bombs_per_region)
            # Calculate team budget
            if tool_allocation:
                self._tool_allocation = tool_allocation
            else:
                if budget_per_region:
                    budget = budget_per_region * len(self.graph.regions)
                else:
                    budget = self._calculate_budget()
                self._tool_allocation = self.random_tool_allocation(budget, self._random)

            # Set start node
            start_node = None
            if seed == 0 and start_location:
                if self._node_grid.is_in_bounds(start_location):
                    start_node = self._node_grid[start_location]
            if not start_node:
                start_regions = start_regions.intersection(self.graph.regions)
                if not start_regions:
                    start_regions = self.graph.regions
                start_node_candidates = [
                    node for node in self.graph.nodes.values() if node.region in start_regions]
                start_node = self._random.choice(start_node_candidates)
            for agent in self.agents.values():
                agent.go_to(start_node)
                self.observations[agent.id].update_from_agent_node()

            # Reset renderer
            if self._renderer:
                self._renderer.reset(self._block_grid)
        obs = self._get_obs()
        self.tick(dt=0)
        return obs

    def step(
            self, actions: MultiAgentDict
    ) -> tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """
        Run one timestep of the environment's dynamics.

        Parameters
        ----------
        actions : MultiAgentDict
            Dictionary mapping from agent ID to agent action

        Returns
        -------
        obs : MultiAgentDict
            Dictionary mapping agent ID to observations
        reward : MultiAgentDict
            Dictionary mapping agent ID to rewards
        done : MultiAgentDict
            Dictionary mapping agent ID to done values
        info : MultiAgentDict
            Dictionary mapping agent ID to info values
        """
        reward = {agent_id: 0 for agent_id in self.agents.keys()}
        info = {agent_id: {} for agent_id in self.agents.keys()}
        prev_agent_health = {agent.id: agent.health for agent in self.agents.values()}

        # Apply actions
        Action = self.action_enum
        for agent_id in self._random.permutation(list(actions.keys())):
            agent, action = self.agents[agent_id], Action(actions[agent_id])

            # if action == Action.unfreeze_agents:
            #     # Unfreeze all agents at the current node
            #     agent.node.unfreeze_agents()

            if action == Action.inspect_bomb:
                # Inspect the currently observed bomb
                if agent.bomb:
                    agent.bomb.inspect()
                    self._inspected_bombs[agent.id].add(agent.bomb)
                    self.observations[agent.id].update_from_inspection()
                else:
                    self.observations[agent.id].update_from_agent_node()
            # elif action == Action.find_next_bomb:
            #     # Cycle to the next bomb at the current node
            #     agent.find_next_bomb()
            #
            # elif action == Action.place_bomb_beacon:
            #     # Place bomb beacon at the current node
            #     if agent.bomb:
            #         agent.node.add_child(BombBeacon(agent.bomb))
            #         for _agent_id in self._inspected_bombs:
            #             self._inspected_bombs[_agent_id].add(agent.bomb)
            #
            # elif action == Action.place_help_beacon:
            #     # Place help beacon at the current node
            #     agent.node.add_child(agent.create_help_beacon())
            #
            # elif action == Action.remove_bomb_beacon:
            #     # Remove oldest bomb beacon from the current node
            #     if agent.node.bomb_beacons:
            #         agent.node.remove_child(agent.node.bomb_beacons[0])
            #
            # elif action == Action.remove_help_beacon:
            #     # Remove oldest help beacon from the current node
            #     if agent.node.help_beacons:
            #         agent.node.remove_child(agent.node.help_beacons[0])

            elif action.node() is not None:
                # Go to the given node
                agent.go_to(action.node())
                self.observations[agent.id].update_from_agent_node()
            elif action.tool() is not None:
                # Apply the specified tool
                tool = agent.get_tool_from_inventory(action.tool())  # check inventory
                if tool == Tool.ppe:
                    agent.is_protected = True
                elif tool == Tool.sensor:
                    self.observations[agent.id].update_from_sensor(self._random_node(agent))
                elif tool == Tool.fire_extinguisher:
                    for fire in self._fires:
                        fire.extinguish(*list(agent.node.fire_locations)[:tool.area])
                elif tool in Tool.bomb_tools():
                    if agent.bomb:
                        ## interagted defusal reward wrapper here
                        if agent.bomb.color == tool.color:
                            for _agent_id in reward:
                                reward[_agent_id] += 1

                            agent.bomb.apply_tool(tool)
                            if agent.bomb.state == Bomb.BombState.defused:
                                for _agent_id in reward:
                                    reward[_agent_id] += agent.bomb.value
                self.observations[agent.id].update_from_agent_node()
            else:
                self.observations[agent.id].update_from_agent_node()

        obs = self._get_obs()

        # Advance time
        self.tick()

        # Penalties for taking damage and becoming frozen
        for agent in self.agents.values():
            damage = max(0, prev_agent_health[agent.id] - agent.health)
            self._score -= 10 * damage
            self._score -= 10 * bool(damage) * bool(agent.is_frozen)
            reward[agent.id] -= 10 * damage
            reward[agent.id] -= 10 * bool(damage) * bool(agent.is_frozen)

        for agent_id in info:
            info[agent_id]['score'] = self.score

        return obs, reward, self._get_done(), info

    def tick(self, dt: Optional[float] = None):
        """
        Advance time by a given interval.

        Parameters
        ----------
        dt : float, optional
            Seconds per tick
        """
        # dt = self._seconds_per_timestep if dt is None else dt
        # self._time += dt
        #
        # # Check for end of recon phase
        # if self._in_recon_phase and self._time >= self._recon_phase_length:
        #     self._in_recon_phase = False
        #     self._assign_tools_from_allocation()
        #
        # # Remove terminal bombs (exploded or defused)
        # while self._terminal_bombs:
        #     bomb = self._terminal_bombs.pop()
        #     node = self._node_grid[bomb.location]
        #     if bomb in node.bombs:
        #         node.remove_child(bomb)

        super().tick(dt=dt)

    def random_tool_allocation(
            self, budget: int, random: np.random.Generator) -> dict[AgentID, dict[Tool, int]]:
        """
        Generate a random allocation of tools to each agent given a fixed team budget.

        Parameters
        ----------
        budget : int
            Total budget for generating random tool allocation
        random : np.random.Generator
            Random number generator
        """
        allocation = {}
        agent_budgets = self._get_agent_budgets(budget, random=random)

        # Randomly allocate tools to agents
        for agent_id, agent_budget in agent_budgets.items():
            costs = [tool.cost for tool in Tool]
            exclude = [random.choice(list(Tool.color_tools()))]  # random color tool
            if self._color_tools_only:
                exclude += set(Tool) - Tool.color_tools()  # all non-color tools
            agent_allocation = random_allocation(
                agent_budget, costs, exclude, random=random)
            allocation[agent_id] = {Tool(t): n for t, n in enumerate(agent_allocation)}

        return allocation

    def render(self, *args, **kwargs):
        """
        Render the environment.

        For kwargs, see `Renderer.__init__()` and `Renderer.render()` in **gym_dragon.render**.
        """
        if not self._renderer:
            self._renderer = Renderer(self, **kwargs)
        self._renderer.render(self._block_grid, **kwargs)

    def observation_space_sample(self, agent_ids=None) -> MultiAgentDict:
        """
        Randomly sample an observation for each agent.

        Parameters
        ----------
        agent_ids : Iterable[AgentID], optional
            Agent IDs to sample observations for
        """
        agent_ids = agent_ids if agent_ids else self.get_agent_ids()
        return {agent_id: obs.space.sample() for agent_id, obs in self.observations.items()}

    def action_space_sample(self, agent_ids=None) -> MultiAgentDict:
        """
        Randomly sample an action for each agent.

        Parameters
        ----------
        agent_ids : Iterable[AgentID], optional
            Agent IDs to sample actions for
        """
        agent_ids = agent_ids if agent_ids else self.get_agent_ids()
        Action = self.action_enum
        actions = {}
        for agent_id in agent_ids:
            mask = self._get_action_mask(agent_id)
            actions[agent_id] = Action(self._random.choice(self.action_array[mask]))

        return actions

    def get_agent_ids(self) -> set:
        """
        Returns a set of agent IDs in the environment.

        This method is needed for rllib compatibility.

        :meta private:
        """
        return set(self.agents.keys())

    def observation_space_contains(self, obs: MultiAgentDict) -> bool:
        """
        Checks if the observation space contains the given key.

        This method is needed for rllib compatibility.

        :meta private:

        Parameters
        ----------
        obs : MultiAgentDict[AgentID, Any]
            Dictionary mapping agent IDs to agent observations
        """
        return all(self.observation_space.contains(obs[agent_id]) for agent_id in obs)

    def action_space_contains(self, actions: MultiAgentDict) -> bool:
        """
        Checks if the action space contains the given key.

        This method is needed for rllib compatibility.

        :meta private:

        Parameters
        ----------
        actions : MultiAgentDict[AgentID, Any]
            Dictionary mapping agent IDs to agent actions
        """
        return all(self.action_space.contains(actions[agent_id]) for agent_id in actions)

    def _get_obs(self) -> MultiAgentDict:
        """
        Get the current observation for each agent.
        """
        return {
            agent_id: obs.to_dict(action_mask=self._get_action_mask(agent_id))
            for agent_id, obs in self.observations.items()
        }

    def _get_done(self) -> MultiAgentDict:
        """
        Get the current done value for each agent.
        """

        done = (
                all(bomb.state.is_terminal() for bomb in self._bombs)
                or self._time >= self._mission_length
        )
        return {key: done for key in ['__all__', *self.agents.keys()]}

    def _get_action_mask(self, agent_id: AgentID) -> np.ndarray[bool]:
        """
        Return a boolean mask over available actions for the given agent.

        Parameters
        ----------
        agent_id : AgentID
            ID of agent to find available actions for

        Returns
        -------
        mask : np.ndarray[bool] of shape (num_actions,)
            Mask over available actions, where `mask[a]` is True
            if and only if `a` is an available action
        """
        Action = self.action_enum
        agent = self.agents[agent_id]
        mask = np.zeros(len(Action), dtype=bool)

        # Placing beacons
        # mask[Action.place_bomb_beacon] = bool(agent.node.bombs)
        # mask[Action.place_help_beacon] = True

        # Removing beacons
        # mask[Action.remove_bomb_beacon] = bool(agent.node.bomb_beacons)
        # mask[Action.remove_help_beacon] = bool(agent.node.help_beacons)

        # Handle frozen agent
        # if agent.is_frozen:
        #     return mask

        # Unfreezing other agents
        # mask[Action.unfreeze_agents] = (len(agent.node.agents) > 1)

        # Bomb actions
        if agent.node and agent.node.bombs:
            mask[Action.inspect_bomb] = True
            # mask[Action.find_next_bomb] = (len(agent.node.bombs) > 1)

        # Going to other nodes
        available_node_ids = (
            self.graph.neighbors(agent.node.id) if agent.node else self.graph.nodes.keys())
        mask[Action.go_to(available_node_ids)] = True

        # Using tools
        mask[Action.use_tool(agent.available_tools(return_mask=True)[0:3])] = True
        # mask[Action.use_fire_extinguisher_tool] &= bool(agent.node.fire_locations)

        return mask

    def _calculate_budget(self) -> int:
        """
        Calculate team budget based on total bomb sequence length.
        Other factors (e.g. perturbations) are weighted relative to bomb defusal budget.
        """
        budget = 0
        for region in Region:
            total_num_bomb_stages = sum([
                bomb.num_stages for bomb in self._bombs if bomb.region == region])
            budget_per_stage = (1 + sum(self._budget_weights[region].values()))
            budget += total_num_bomb_stages * budget_per_stage

        return int(np.ceil(budget))

    def _get_agent_budgets(
            self, team_budget: int, random: np.random.Generator) -> dict[AgentID, int]:
        """
        Randomly allocate team budget to each agent.

        Parameters
        ----------
        team_budget : int
            Total team budget
        random : np.random.Generator
            Random number generator
        """
        # Randomly decide portion of team budget to be used (unless color tools only)
        if not self._color_tools_only:
            team_budget = int(team_budget * random.beta(5, 1))

        # Randomly allocate team budget across agents
        n = len(self.agents)
        agent_budgets = random.uniform(size=n)
        agent_budgets *= team_budget / agent_budgets.sum()
        agent_budgets = agent_budgets.astype(int)
        agent_budgets[random.permutation(n)[:team_budget - agent_budgets.sum()]] += 1

        return dict(zip(self.agents.keys(), agent_budgets))

    def _create_bomb_dependency_chains(
            self,
            region: Region,
            num_chains: int = 1,
            max_chain_length: int = 3,
    ):
        """
        Create bomb dependency chains within the given region.

        Parameters
        ----------
        region : Region
            Region to create bomb dependency chains in
        num_chains : int, default=1
            Number of bomb dependency chains to create
        max_chain_length : int, default=3
            Maximum length of each bomb dependency chain
        """
        region_bombs = [bomb for bomb in self._bombs if bomb.region == region]
        chain_lengths = sorted(
            self._random.integers(2, max_chain_length + 1, size=num_chains))

        self._random.shuffle(region_bombs)
        for chain_length in chain_lengths:
            if len(region_bombs) < chain_length:
                break
            bomb_chain = region_bombs[:chain_length]
            for bomb in bomb_chain:
                region_bombs.remove(bomb)
            for i, bomb in enumerate(bomb_chain[:-1]):
                bomb.dependency = bomb_chain[i + 1]

    def _random_node(self, agent: Agent) -> Optional[Node]:
        """
        Return a random node with an uninspected bomb, within the same region.

        Parameters
        ----------
        agent : Agent
            Agent to find a random node for
        """
        candidate_nodes = [
            node for node in self.graph.nodes.values()
            if node.region == agent.node.region
               and not self._inspected_bombs[agent.id].intersection(node.bombs)
        ]
        return self._random.choice(candidate_nodes) if candidate_nodes else None

    def _on_bomb_state_change(self, bomb: Bomb):
        """
        Callback for when a bomb changes state.

        Parameters
        ----------
        bomb : Bomb
            The bomb that has changed its state
        """
        node = self._node_grid[bomb.location]

        if bomb.state == Bomb.BombState.exploded:
            self._terminal_bombs.append(bomb)
            if self._block_grid[bomb.location] != 'fire':
                self._block_grid[bomb.location] = 'ground'

            # Freeze nearby agents if the bomb exploded
            for agent in node.agents:
                agent.explode()

            def on_fire_change():
                if self._renderer:
                    self._renderer.update_map(self._block_grid)

            # Start fire if the bomb was a fire bomb
            if bomb.is_fire_bomb:
                fire = Fire(
                    bomb.location,
                    self._block_grid,
                    self._node_grid,
                    on_change=on_fire_change,
                )
                node.add_child(fire)
                self._fires.append(fire)

        elif bomb.state == Bomb.BombState.defused:
            self._terminal_bombs.append(bomb)
            self._block_grid[bomb.location] = 'ground'
            self._score += bomb.value

        elif bomb.state == Bomb.BombState.active:
            self._block_grid[bomb.location] = 'bomb_active'

        elif bomb.state == Bomb.BombState.inactive:
            self._block_grid[bomb.location] = 'bomb_inactive'

        # Update renderer
        if self._renderer:
            self._renderer.update_map(self._block_grid)

    def _assign_tools_from_allocation(self):
        """
        Assign agents tools from allocation.
        """
        for agent_id, agent_allocation in self._tool_allocation.items():
            for tool, quantity in agent_allocation.items():
                self.agents[agent_id].assign_tool(tool, quantity)

    def _get_csv_bomb_list(self, csv_path: str, csv_subtract_fuse: int = 0) -> list[dict]:
        """
        Get bombs from CSV file as a list of dictionaries

        Parameters
        ----------
        csv_path : str
            Path to CSV file containing bomb distribution
        csv_subtract_fuse : int, default=0
            Recon phase length to subtract from fuse if reading bomb distribution from CSV
        """
        # Load bomb information from CSV file
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            csv_bomb_list = []
            for row in reader:
                if row['FeatureType'] == 'Bomb':
                    loc = row['LocationXYZ'].split()
                    dependency_id = None if row['ChainedID'] == 'NONE' else int(row['ChainedID'])
                    bomb = {
                        'bomb_id': int(row['BombID']),
                        'location': (int(loc[0]), int(loc[2])),
                        'sequence': [Color.from_char(c) for c in row['Sequence']],
                        'fuse': (int(row['FuseStartTime']) - csv_subtract_fuse) * 60 \
                            if self._include_fuse_bombs else None,
                        'dependency': dependency_id if self._include_chained_bombs else None,
                        'is_fire_bomb': row['BlockType'] == 'block_bomb_fire' if self._include_fire_bombs else False
                    }
                    csv_bomb_list.append(bomb)

            # Sort bombs based on 1) bomb_id, 2) dependency_id, putting bombs with None dependency at start of list
            csv_bomb_list = sorted(
                csv_bomb_list, key=lambda x: (x['dependency'] is not None, x['bomb_id'], x['dependency']))

        return csv_bomb_list

    def _place_bombs(self,
                     csv_path: str = None,
                     num_bombs_per_region: int = 15):
        """
        Place bombs on the map.

        Parameters
        ----------
        csv_path : str
            Path to CSV file containing bomb distribution
        num_bombs_per_region : int, default=15
            Number of bombs to place in each region (if CSV file is not provided)
        """
        bombs = {}
        # Load bomb information from CSV file
        if csv_path:
            # Create Bomb objects from sorted list
            for bomb_dict in self._get_csv_bomb_list(csv_path):
                bomb = Bomb(
                    bomb_id=bomb_dict['bomb_id'],
                    location=bomb_dict['location'],
                    sequence=bomb_dict['sequence'],
                    fuse=bomb_dict['fuse'],
                    dependency=bombs.get(bomb_dict['dependency'], None),
                    is_fire_bomb=bomb_dict['is_fire_bomb'],
                    on_state_change=self._on_bomb_state_change
                )
                bombs[bomb.id] = bomb
                self._bombs.append(bomb)

        # Randomly create bombs
        else:
            # Randomly select bomb locations from each region
            color_list = [Color.red,Color.green,Color.blue]
            bomb_locations = self._random.permutation([(28,56),(14,55),(40,56),(33,64),(22,63)])
            for bomb_id, location in enumerate(bomb_locations):
                if bomb_id <= 1:
                    sequence = [color_list[x] for x in self._random.choice(3,1)]
                elif bomb_id <=3:
                    sequence = [color_list[x] for x in self._random.choice(3, 2,replace = False)]
                else:
                    sequence = [color_list[x] for x in self._random.choice(3, 3,replace = False)]
                bomb = Bomb(
                    bomb_id=bomb_id+1,
                    location=tuple(location),
                    sequence=sequence,
                    fuse=None,
                    dependency=None,
                    is_fire_bomb=False,
                    on_state_change=self._on_bomb_state_change
                )
                bombs[bomb.id] = bomb
                self._bombs.append(bomb)

            # Randomly create bomb dependency chains
            if self._include_chained_bombs:
                for region in Region:
                    self._create_bomb_dependency_chains(
                        region, num_chains=self._random.integers(1, 4))

        # Place bombs on the map
        for bomb in bombs.values():
            self._node_grid[bomb.location].add_child(bomb)  # update graph
            self._block_grid[bomb.location] = 'bomb_inactive'  # update grid

    def _reset_from_env(self, env: 'DragonBaseEnv'):
        """
        Reset environment from another environment.

        Parameters
        ----------
        env : DragonBaseEnv
            Reference environment to reset state to
        """
        env = env.unwrapped

        # Reset initial state to current state of reference environment
        self._time, self._score = env.time, env.score
        self._in_recon_phase = env._in_recon_phase
        self._bombs = deepcopy(env._bombs)
        self._inspected_bombs = deepcopy(env._inspected_bombs)
        self._terminal_bombs = deepcopy(env._terminal_bombs)
        self._fires = deepcopy(env._fires)
        self._block_grid[...] = env._block_grid[...]

        # Reset agents
        self._agents.clear()
        self._agents = {agent_id: Agent(agent_id) for agent_id in env.agents}
        for agent in self._agents.values():
            # Set current node
            agent.go_to(env.graph.nodes[env.agents[agent.id].node.id])

            # Set tool allocation
            agent._tool_remaining_uses = env.agents[agent.id]._tool_remaining_uses
            agent._tool_available = env.agents[agent.id]._tool_available

            # Set health
            agent._health = env.agents[agent.id]._health

        # Add bombs
        for bomb in self._bombs:
            self._node_grid[bomb.location].add_child(bomb)  # update graph
            self._block_grid[bomb.location] = 'bomb_inactive'  # update grid

        # Reset renderer
        if self._renderer:
            self._renderer.reset(self._block_grid)

    def _init_graph(self, centroids: dict, edges: Iterable, valid_nodes=[(28,56),(15,55),(40,56),(33,64),(22,63)]):
        """
        Initialize the underlying graph for the environment.

        Parameters
        ----------
        centroids : dict[Hashable, tuple[int, int]]
            Dictionary mapping node ID to (x, z) node centroid
        edges : Iterable[tuple[Hashable, Hashable]]
            Set of edges, where (i, j) indicates an edge between nodes i & j
        valid_regions : set[Region]
            Set of regions to include in the graph
        """
        # Filter by region
        centroids = {
            k: v for k, v in centroids.items() if v in valid_nodes}
        edges = {(i, j) for i, j in edges if i in centroids and j in centroids}

        # Create graph
        self._graph = Graph(centroids, edges, self.agents.values())
        self.add_child(self._graph)

    def _init_grid(
            self, base_grid: BoundedGrid, node_id_grid: BoundedGrid, valid_regions=set(Region)):
        """
        Initialize the underlying grid for the environment.

        Parameters
        ----------
        base_grid : BoundedGrid
            Grid indicating the block type at each (x, z) location
        node_id_grid : BoundedGrid
            Grid indicating the node ID for each (x, z) location
        valid_regions : set[Region]
            Set of regions to include in the grid
        """
        base_grid.replace(None, 'obstacle')
        base_grid.replace('stone_slab', 'water')  # for visualization purposes

        # Get bounds
        z_max, z_min = base_grid.bounds[1]
        for region in valid_regions:
            if region == Region.forest:
                z_min, z_max = min(0, z_min), max(50, z_max)
            elif region == Region.village:
                z_min, z_max = min(50, z_min), max(100, z_max)
            elif region == Region.desert:
                z_min, z_max = min(100, z_min), max(151, z_max)

        # Create grid from bounds
        self._base_grid = base_grid.view[0:52, z_min:z_max + 1]
        self._block_grid = BoundedGrid(*self._base_grid.bounds)

        # Create node grid with Node objects
        self._node_grid = BoundedGrid(*self._base_grid.bounds)
        for loc in self._node_grid.locations():
            if node_id_grid[loc] in self.graph.nodes:
                self._node_grid[loc] = self.graph.nodes[node_id_grid[loc]]

        # Set node areas
        for node in self.graph.nodes.values():
            node.area = node_id_grid.count(node.id)

        # Get empty locations in each region (for potential bomb placement)
        all_empty_locations = [
            loc for loc in self._base_grid.locations()
            if self._base_grid[loc] == 'ground' and self._node_grid[loc] is not None
        ]
        self._empty_locations = {
            region: np.array([
                loc for loc in all_empty_locations
                if Region.from_location(loc) == region
            ])
            for region in Region
        }

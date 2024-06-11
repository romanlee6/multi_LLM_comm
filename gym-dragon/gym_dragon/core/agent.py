"""
Module for representing agents acting in the environment.
"""
import numpy as np

from ray.rllib.utils.typing import AgentID
from typing import Optional, Union, TYPE_CHECKING

from .world import Bomb, Color, Tool, WorldObj, HelpBeacon
from ..utils import UNKNOWN

if TYPE_CHECKING:
    from .graph import Node




class Agent(WorldObj):
    """
    Class representing an individual agent.

    Attributes
    ----------
    id : AgentID
        Unique identifier for the agent
    color : Color
        Color of the agent
    node : Node or None
        Node for the agent's current location
    bomb : Bomb or None
        Current bomb that the agent can observe or act upon
    health : float
        Current health of the agent (value from 0 to 1)
    is_frozen : bool
        Whether the agent is currently frozen
    is_protected : bool
        Whether the agent is wearing PPE (personal protective equipment)
    tool_remaining_uses : np.ndarray[Tool]
        Number of remaining uses for each tool
    """
    dim = 3 + len(Tool) # dimension of Agent.to_vec()
    bomb_damage = 0.5
    fire_damage = 0.05

    def __init__(self, agent_id: AgentID, color=None):
        """
        Parameters
        ----------
        agent_id : AgentID
            Unique identifier for the agent
        color : Color, optional
            Color of the agent
        """
        super().__init__()
        self._id = agent_id
        self._color = color
        self.reset()

    def __repr__(self) -> str:
        """
        String representation of this object.
        """
        return f'Agent({self.id.__repr__()})'

    @property
    def id(self) -> AgentID:
        """
        Unique identifier for the agent.
        """
        return self._id

    @property
    def color(self) -> Color:
        """
        Color of the agent.
        """
        return self._color

    @property
    def node(self) -> Optional['Node']:
        """
        Node for the agent's current location.
        """
        return self._current_node

    @property
    def bomb(self) -> Optional[Bomb]:
        """
        Current bomb that the agent can observe or act upon.
        """
        if self.node:
            if self.node.bombs and self._bomb_index is not None:
                return self.node.bombs[self._bomb_index % len(self.node.bombs)]

    @property
    def health(self) -> float:
        """
        Current health of the agent (value from 0 to 1).
        """
        return self._health

    @property
    def is_frozen(self) -> bool:
        """
        Return whether or not the agent is currently frozen.
        """
        return self._health <= 0

    @property
    def is_protected(self) -> bool:
        """
        Whether the agent is wearing PPE (personal protective equipment).
        """
        return self._is_protected

    @is_protected.setter
    def is_protected(self, value):
        """
        Update whether agent is wearing PPE (personal protective equipment).
        """
        self._is_protected = bool(value)

    @property
    def tool_remaining_uses(self) -> np.ndarray[Tool]:
        """
        Number of remaining uses for each tool.
        """
        return self._tool_remaining_uses

    def reset(self, random: Optional[np.random.Generator] = None):
        """
        Reset the agent.

        Parameters
        ----------
        random : np.random.Generator, optional
            Random number generator
        """
        super().reset()

        self._random = random if random else np.random.default_rng()
        self._health = 1
        self._bomb_index = None
        self._current_node = None
        self._is_protected = False

        # Reset tools
        self._tool_remaining_uses = np.zeros(len(Tool))
        self._tool_available = np.zeros(len(Tool), dtype=bool)

    def tick(self, *args, **kwargs):
        """
        Advance time by a given interval.
        """
        super().tick(*args, **kwargs)
        if self.node.fire_locations:
            p = len(self.node.fire_locations) / self.node.area
            if self._random.random() <= p:
                self.burn()

    def go_to(self, node: 'Node'):
        """
        Move the agent to the specified node.

        Parameters
        ----------
        node : Node
            Node to move to
        """
        # Update old node
        if self._current_node:
            self._current_node.remove_child(self)

        # Update new node
        self._current_node = node
        self._current_node.add_child(self)

        # Update bomb index
        self._bomb_index = 0 if node.bombs else None

    def find_next_bomb(self):
        """
        Cycle to the next bomb at the current node.
        """
        if self.node.bombs:
            if self._bomb_index is None:
                self._bomb_index = 0
            else:
                self._bomb_index += 1
                self._bomb_index %= len(self.node.bombs)

    def create_help_beacon(self) -> HelpBeacon:
        """
        Create help beacon of relevant message type.
        """
        if self.is_frozen:
            return HelpBeacon(HelpBeacon.MessageType.frozen)
        if self.node.fire_locations:
            return HelpBeacon(HelpBeacon.MessageType.fire)
        if self.bomb:
            for color in self.bomb.sequence:
                if not self._tool_available[Tool.from_color(color)]:
                    return HelpBeacon.from_color(color)

        return HelpBeacon(HelpBeacon.MessageType.all_clear)

    def explode(self):
        """
        Agent takes damage from an explosion.
        """
        if self._is_protected:
            self._is_protected = False
        else:
            self._health = max(0, self._health - Agent.bomb_damage)

    def burn(self):
        """
        Agent takes damage from fire.
        """
        self._health = max(0, self._health - Agent.fire_damage)

    def unfreeze(self):
        """
        Unfreeze the agent.
        """
        if self._health == 0:
            self._health = 1

    def available_tools(
        self, return_mask=False) -> Union[np.ndarray[Tool], np.ndarray[bool]]:
        """
        Return an array of available tools.

        Parameters
        ----------
        return_mask : bool, default=False
            Whether or not to return a boolean mask over all Tool values.
        """
        if return_mask:
            return self._tool_available
        else:
            return np.where(self._tool_available)[0]

    def get_tool_from_inventory(self, tool: Tool) -> Optional[Tool]:
        """
        Get tool from inventory for usage. Decrements the number of remaining uses by 1.

        Parameters
        ----------
        tool : Tool
            Tool to be used by agent if available
        """
        if self._tool_remaining_uses[tool] > 0:
            self._tool_remaining_uses[tool] -= 1 # decrement
            self._tool_available = (self._tool_remaining_uses > 0)
            return tool

    def assign_tool(self, tool: Tool, quantity: int = 1):
        """
        Assign tool to agent.

        Parameters
        ----------
        tool : Tool
            Tool to be assigned to agent
        quantity : int, default=1
            Quanity of tool to be assigned to agent
        """
        self._tool_remaining_uses[tool] += quantity * tool.uses
        self._tool_available = (self._tool_remaining_uses > 0)

    def to_vec(self, indicator=False, fully_observable=False) -> np.ndarray:
        """
        Return a vector representation of this agent.

        Parameters
        ----------
        indicator : bool, default=False
            Boolean variable indicator variable for this agent (e.g. "this is me")
        fully_observable : bool, default=False
            Whether or not to include all agent information (e.g. tools)
        """
        x = np.empty(Agent.dim)
        x[0] = indicator
        x[1] = self.is_frozen
        x[2] = self.is_protected if indicator or fully_observable else UNKNOWN
        x[3:] = np.tanh(self._tool_remaining_uses)
        return x

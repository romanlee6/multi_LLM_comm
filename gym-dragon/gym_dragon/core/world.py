"""
Module for representing basic attributes and objects in the environment.

Contains enumerations for colors, tools, and regions,
as well as classes for representing world objects such as bombs, beacons, and fire.
"""
import enum
import numpy as np

from typing import Any, Callable, Iterable, Optional
from ..utils import one_hot, positional_encoding, UNKNOWN, BoundedGrid



class Color(enum.IntEnum):
    """
    Enumeration of colors, as used for agents, bombs, and tools.
    """
    red = 0
    green = enum.auto()
    blue = enum.auto()

    @staticmethod
    def from_char(char: str) -> 'Color':
        """
        Return the Color corresponding to the given character.
        """
        char_to_color = {'r': Color.red, 'g': Color.green, 'b': Color.blue}
        return char_to_color[char.lower()]


class Region(str, enum.Enum):
    """
    Enumeration of map regions.
    """
    forest = 'forest'
    village = 'village'
    desert = 'desert'

    @staticmethod
    def from_location(location: tuple[int, int]) -> 'Region':
        """
        Return the region for the specified (x, z) location.
        """
        x, z = location
        if 0 <= z < 50:
            return Region.forest
        elif 50 <= z < 100:
            return Region.village
        elif 100 <= z: 
            return Region.desert


class Tool(enum.IntEnum):
    """
    Enumeration of available agent tools.

    Attributes
    ----------
    color : Color or None
        Color of this tool
    uses : int
        Default number of uses of this tool
    cost : int
        Cost of this tool
    area : int
        Area of effect of this tool
    """
    red = 0 #: Allows any agent to execute a red sequence step
    green = enum.auto() #: Allows any agent to execute a green sequence step
    blue = enum.auto() #: Allows any agent to execute a blue sequence step
    sensor = enum.auto() #: Points in the direction of a random bomb in the same region
    ppe = enum.auto() #: Personal protective equipment, protects from a bomb explosion
    disposer = enum.auto() #: Completely disposes of a bomb
    fire_extinguisher = enum.auto() #: Puts out all fire in surrounding 3x3 cell area

    @staticmethod
    def bomb_tools() -> set['Tool']:
        """
        Return the set of tools that can be used on a bomb.
        """
        return {Tool.red, Tool.green, Tool.blue, Tool.disposer}

    @staticmethod
    def color_tools() -> set['Tool']:
        """
        Return the set of color tools.
        """
        return {tool for tool in Tool if tool.color is not None}

    @staticmethod
    def from_color(color: Color) -> 'Tool':
        """
        Return the tool corresponding to the given Color.
        """
        color_to_tool = {
            Color.red: Tool.red, Color.green: Tool.green, Color.blue: Tool.blue}
        return color_to_tool[color]

    @property
    def color(self) -> Optional[Color]:
        """
        Return the color of this tool.
        """
        tool_to_color = {
            Tool.red: Color.red, Tool.green: Color.green, Tool.blue: Color.blue}
        return tool_to_color.get(self, None)

    @property
    def uses(self) -> int:
        """
        Return default number of uses of tool.
        """
        tool_to_uses = {
            Tool.red: 1,
            Tool.green: 1,
            Tool.blue: 1,
            Tool.sensor: 1,
            Tool.ppe: 1,
            Tool.disposer: 1,
            Tool.fire_extinguisher: 1,
        }
        return tool_to_uses[self]

    @property
    def cost(self) -> int:
        """
        Return cost of tool.
        """
        tool_to_cost = {
            Tool.red: 1,
            Tool.green: 1,
            Tool.blue: 1,
            Tool.sensor: 5,
            Tool.ppe: 5,
            Tool.disposer: 5,
            Tool.fire_extinguisher: 1,
        }
        return tool_to_cost[self]

    @property
    def area(self) -> Optional[int]:
        """
        Return area of effect of tool.
        """
        if self == Tool.fire_extinguisher:
            return 3 * 3


class WorldObj:
    """
    Class for representing persistent world objects.
    """

    def __init__(self, *args, **kwargs):
        self._children = set()

    def reset(self, *args, **kwargs):
        """
        Reset this object.
        """
        for child in self._children:
            if isinstance(child, WorldObj):
                child.reset(*args, **kwargs)

    def tick(self, dt: float = 1, **kwargs):
        """
        Advance time by a given interval.

        Parameters
        ----------
        dt : float, default=1
            Seconds per tick
        """
        for child in set(self._children):
            if isinstance(child, WorldObj):
                child.tick(dt=dt, **kwargs)

    def add_child(self, obj: 'WorldObj'):
        """
        Add a child object.

        Parameters
        ----------
        obj : WorldObj
            Child object
        """
        self._children.add(obj)

    def remove_child(self, obj: 'WorldObj'):
        """
        Remove a child object.

        Parameters
        ----------
        obj : WorldObj
            Child object
        """
        self._children.discard(obj)

    def clear_children(self):
        """
        Remove all child objects.
        """
        self._children.clear()


class Bomb(WorldObj):
    """
    Class for representing bombs.

    Each bomb has a fuse that will cause the bomb to explode on its own
    after a set amount of time.

    Each bomb has a `Color` sequence indicating the sequence needed for defusal.

    Agents can `inspect()` a bomb, revealing:
        * the state of the bomb
        * the fuse length
        * the remaining defusal sequence
        * dependency on other bomb (if any; see variants below)

    When an agent uses `apply_tool(tool)` the bomb becomes active, and a countdown starts.
    When the next correct tool in the sequence is used, the countdown restarts.
    If all the correct steps in the sequence are satisfied, the bomb is defused.
    If at any point the countdown reaches 0, the bomb explodes.

    Variants:
        * Chained Bomb
            Has a dependency on another bomb, which must be defused first
        * Fire Bomb
            Spreads fire after it explodes

    Attributes
    ----------
    id : int
        Unique identifier for this bomb
    location : tuple[int, int]
        The (x, z) location of the bomb
    region : Region
        The region this bomb is located in
    num_stages : int
        Total number of stages in the original bomb defusal sequence
    state : Bomb.BombState
        Current state of the bomb
    sequence : tuple[Color]
        Remaining bomb defusal sequence
    color : Color
        Current color in the bomb sequence
    dependency : Bomb, optional
        Bomb this bomb is dependent on
    value : float
        Current point value gained from this bomb
    is_fire_bomb : bool
        Whether this bomb is a fire bomb
    fuse : float
        Length of bomb fuse (in seconds)
    """
    value_per_stage = 10
    # countdown_length = 15
    countdown_length = None
    max_num_stages = 3
    dim = 8 + (max_num_stages + 1) * len(Color) # dimension of Bomb.to_vec()

    class BombState(enum.IntEnum):
        """
        Enumeration of bomb states.
        """
        inactive = 0 #: Initial bomb state, indicating no countdown is running
        active = enum.auto() #: Bomb is active and has a countdown ticking
        exploded = enum.auto() #: Bomb has exploded
        defused = enum.auto() #: Bomb has been completely defused

        def is_terminal(self) -> bool:
            """
            Return whether this state is terminal.
            """
            return self in {Bomb.BombState.exploded, Bomb.BombState.defused}


    def __init__(
        self,
        bomb_id: int,
        location: tuple[int, int],
        sequence: Iterable[Color],
        fuse: Optional[float] = None,
        dependency: Optional['Bomb'] = None,
        is_fire_bomb: bool = False,
        on_state_change: Callable[['Bomb'], Any] = (lambda bomb: None)):
        """
        Parameters
        ----------
        bomb_id : int
            Unique identifier for this bomb
        location : tuple[int, int]
            The (x, z) location of the bomb
        sequence : Iterable[Color]
            The bomb defusal sequence (e.g. (Color.red, Color.green, Color.red))
        fuse : float, optional
            The length of the fuse (in seconds)
        dependency : Bomb, optional
            Bomb this bomb is dependent on
        is_fire_bomb : bool, default=False
            Whether this is a fire bomb (i.e. starts fire on explosion)
        on_state_change : Callable(Bomb)
            Callback triggered whenever the bomb changes state
        """
        super().__init__()
        self._id = bomb_id
        self._location = location
        self._full_sequence = tuple(map(Color, sequence))
        self._fuse = fuse # remaining length of fuse (in seconds)
        self._dependency = dependency
        self._is_fire_bomb = is_fire_bomb
        self._on_state_change = on_state_change

        # Internal
        self._countdown = None # remaining seconds on countdown, or `None` if not active
        self._current_step = 0
        self._state = Bomb.BombState.inactive
        self._value = 0 # current point value gained from the bomb

    @property
    def id(self) -> int:
        """
        Unique identifier for this bomb.
        """
        return self._id

    @property
    def location(self) -> tuple[int, int]:
        """
        Return the (x, z) location of the bomb.
        """
        return self._location

    @property
    def region(self) -> Region:
        """
        Return the region this bomb is located in.
        """
        return Region.from_location(self._location)

    @property
    def num_stages(self) -> int:
        """
        Return the total number of stages in the original bomb defusal sequence.
        """
        return len(self._full_sequence)

    @property
    def state(self) -> BombState:
        """
        Return the current state of the bomb.
        """
        return self._state

    @state.setter
    def state(self, new_state: BombState):
        """
        Set the state of the bomb.

        Parameters
        ----------
        new_state : Bomb.BombState
            New state to set bomb to
        """
        # Adjust other bomb variables
        if new_state == Bomb.BombState.inactive:
            self._countdown = None
        elif new_state == Bomb.BombState.active:
            self._countdown = Bomb.countdown_length
        elif new_state == Bomb.BombState.exploded:
            self._countdown = None
            self._fuse = None
        elif new_state == Bomb.BombState.defused:
            self._countdown = None
            self._fuse = None
            self._current_step = self.num_stages

        # Change state variable and trigger callback
        if self._state != new_state:
            self._state = new_state
            self._on_state_change(self)

    @property
    def sequence(self) -> tuple[Color]:
        """
        Return the remaining defusal sequence.
        """
        return self._full_sequence[self._current_step:]

    @property
    def color(self) -> Optional[Color]:
        """
        Return the current color in the bomb sequence.
        """
        if self._current_step < len(self._full_sequence):
            return self._full_sequence[self._current_step]

    @property
    def dependency(self) -> Optional['Bomb']:
        """
        Return the bomb this bomb is dependent on.
        """
        return self._dependency

    @dependency.setter
    def dependency(self, bomb: Optional['Bomb']):
        """
        Set the bomb this bomb is dependent on.

        Parameters
        ----------
        bomb : Bomb
            Bomb this bomb is dependent on
        """
        self._dependency = bomb

    @property
    def value(self) -> float:
        """
        Return the current point value gained from this bomb.
        """
        return self._value

    @property
    def is_fire_bomb(self) -> bool:
        """
        Return whether this bomb is a fire bomb.
        """
        return self._is_fire_bomb

    @property
    def fuse(self) -> Optional[float]:
        """
        Return length of bomb fuse (in seconds).
        """
        return self._fuse

    def tick(self, dt: float = 1, **kwargs):
        """
        Advance time by a given interval.

        Parameters
        ----------
        dt : float, default=1
            Seconds per tick
        """
        super().tick(dt=dt, **kwargs)

        # Check dependency
        if self._dependency and self._dependency.state.is_terminal():
            self._dependency = None

        # Decrement fuse
        if self._fuse is not None:
            self._fuse -= dt
            if self._fuse <= 0:
                self.state = Bomb.BombState.exploded

        # Decrement countdown
        if self._countdown is not None:
            self._countdown -= dt
            if self._countdown <= 0 and self.state == Bomb.BombState.active:
                self.state = Bomb.BombState.exploded

    def inspect(self):
        """
        Agent inspects the bomb.
        """
        return self.state, self._fuse, self.sequence, self.dependency

    def apply_tool(self, tool: Tool):
        """
        Agent applies a tool to the bomb.

        Parameters
        ----------
        tool : Tool
            Tool to apply
        """
        # If dependency has not been dealt with, bomb explodes
        if self._dependency and not self._dependency.state.is_terminal():
            self.state = Bomb.BombState.exploded

        # Apply the tool to the bomb
        if tool == Tool.disposer:
            self._value = Bomb.value_per_stage
            self.state = Bomb.BombState.defused
        elif tool.color is not None and tool.color == self.color:
            # Correct color tool, move to the next step in the sequence
            self._current_step += 1
            if self._current_step < self.num_stages:
                self.state = Bomb.BombState.active
            else:
                self._value = Bomb.value_per_stage * self.num_stages
                self.state = Bomb.BombState.defused
        # else:
        #     # Incorrect tool was used, bomb explodes
        #     self.state = Bomb.BombState.exploded

    def to_vec(self, include_inspection: bool = False) -> np.ndarray:
        """
        Return a vector representation of this bomb.

        Parameters
        ----------
        include_inspection : bool
            Whether or not to include information from bomb inspection
        """
        x = UNKNOWN * np.ones(Bomb.dim)
        x[0] = True # indicator that bomb is present
        x[1] = (self.state == Bomb.BombState.active)
        x[2:4] = positional_encoding(hash(self.id))

        if self._countdown is not None:
            x[4] = self._countdown / Bomb.countdown_length

        if include_inspection:
            # Include fuse length
            x[5] = (self._fuse if self._fuse else 0) / 60

            # Include chain bomb dependency, if any
            if self._dependency:
                x[6:8] = positional_encoding(hash(self._dependency.id))

            # Include bomb sequence
            start, stop = self._current_step, len(self._full_sequence)
            sequence = np.zeros((Bomb.max_num_stages + 1, len(Color)))
            sequence[start:stop] = one_hot(self._full_sequence[start:], Color)

            # Additional feature for current color in sequence
            sequence[stop] = one_hot(self.color, Color) if self.color else UNKNOWN

            x[8:] = sequence.flatten()

        return x

    @staticmethod
    def random(
        bomb_id: int, location: tuple[int, int], fuse: Optional[float] = None,
        num_stages: Optional[int] = None, is_fire_bomb: Optional[bool] = None,
        on_state_change: Callable[['Bomb'], Any] = (lambda bomb: None),
        random: Optional[np.random.Generator] = None) -> 'Bomb':
        """
        Create a new randomly generated bomb.

        Parameters
        ----------
        bomb_id : int
            Unique identifier for the bomb to be created
        location : tuple[int, int]
            The (x, z) location of the bomb
        fuse : float, optional
            The length of the fuse (in seconds)
        num_stages : int, optional
            The desired sequence length for the bomb (determined randomly if not provided)
        is_fire_bomb : bool, optional
            Whether this is a fire bomb (determined randomly if not provided)
        on_state_change : Callable(Bomb), optional
            Callback triggered whenever the bomb changes state
        random : np.random.Generator, optional
            Random number generator
        """
        random = random if random else np.random.default_rng()

        if num_stages is None:
            num_stages = random.integers(1, Bomb.max_num_stages + 1)
        if is_fire_bomb is None:
            is_fire_bomb = (random.random() <= 0.2)

        sequence = random.permutation(len(Color))[:num_stages]

        return Bomb(
            bomb_id,
            location,
            sequence=sequence,
            fuse=fuse,
            is_fire_bomb=is_fire_bomb,
            on_state_change=on_state_change,
        )


class BombBeacon(WorldObj):
    """
    Class for representing bomb beacons.
    """

    def __init__(self, bomb: Bomb):
        super().__init__()
        self._bomb = bomb

    @property
    def bomb(self) -> Bomb:
        return self._bomb


class HelpBeacon(WorldObj):
    """
    Class for representing help/hazard beacons.
    """

    class MessageType(enum.IntEnum):
        """
        Messages types that can be displayed by help beacons.
        """
        red = 0
        green = enum.auto()
        blue = enum.auto()
        all_clear = enum.auto()
        fire = enum.auto()
        frozen = enum.auto()


    def __init__(self, message_type: MessageType):
        super().__init__()
        self._message_type = message_type

    @property
    def message_type(self) -> MessageType:
        return self._message_type

    @staticmethod
    def from_color(color: Color) -> 'HelpBeacon':
        color_to_message_type = {
            Color.red: HelpBeacon.MessageType.red,
            Color.green: HelpBeacon.MessageType.green,
            Color.blue: HelpBeacon.MessageType.blue,
        }
        return HelpBeacon(color_to_message_type[color])


class Fire(WorldObj):
    """
    Class for representing a spreading fire.

    Fires spread by igniting adjacent locations (above, below, left, and right).
    Ignition events will occur at regular intervals until a fixed number of
    cycles have occurred.

    Extinguished locations cannot be reignited by the same fire.
    """
    cycle_length = 15
    total_cycles = 6
    ignitable_blocks = {'ground', 'bomb_active', 'bomb_inactive'}

    def __init__(
        self,
        location: tuple[int, int],
        block_grid: BoundedGrid,
        node_grid: BoundedGrid,
        on_change: Callable[[], Any] = (lambda: None)):
        """
        Parameters
        ----------
        location : tuple[int, int]
            The initial (x, z) location of the fire
        block_grid : BoundedGrid[Node]
            Grid indicating the block type at each (x, z) location
        node_grid : BoundedGrid[Node]
            Grid indicating the node at each (x, z) location
        on_change : Callable()
            Callback triggered on ignition or extinguishment
        """
        super().__init__()
        self._block_grid = block_grid
        self._node_grid = node_grid
        self._on_change = on_change
        self._ignited_locations = set()
        self._extinguished_locations = set()
        self._num_ignitions = 0
        self._time_elapsed = 0
        self._last_ignition_time = 0

        # Initial ignition
        self._block_grid[location] = 'fire'
        self._node_grid[location].ignite(location)
        self._ignited_locations.add(location)
        self._on_change()

    def tick(self, dt: float = 1, **kwargs):
        """
        Advance time by a given interval.

        Parameters
        ----------
        dt : float, default=1
            Seconds per tick
        """
        super().tick(dt=dt, **kwargs)
        self._time_elapsed += dt

        if self._num_ignitions <= Fire.total_cycles:
            if self._last_ignition_time + Fire.cycle_length <= self._time_elapsed:
                self._last_ignition_time += 20
                self.ignite()

    def ignite(self):
        """
        Spread fire by igniting adjacent locations.
        """
        self._num_ignitions += 1

        ignited_locations = list(self._ignited_locations)
        for x, z in ignited_locations:
            for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                loc = (x + dx, z + dz)
                if self._block_grid.is_in_bounds(loc):
                    if self._block_grid[loc] in Fire.ignitable_blocks:
                        if loc not in self._extinguished_locations:
                            self._block_grid[loc] = 'fire'
                            self._ignited_locations.add(loc)
                            if self._node_grid[loc]:
                                self._node_grid[loc].ignite(loc)

        self._on_change()

    def extinguish(self, *locations: tuple[int, int]):
        """
        Extinguish fire at a given location.

        Parameters
        ----------
        locations : tuple[int, int]
            The (x, z) locations to extinguish
        """
        locations = self._ignited_locations.intersection(locations)
        for loc in locations:
            self._block_grid[loc] = 'ground'
            self._extinguished_locations.add(loc)
            self._ignited_locations.discard(loc)
            if self._node_grid:
                self._node_grid[loc].extinguish(loc)

        if locations:
            self._on_change()

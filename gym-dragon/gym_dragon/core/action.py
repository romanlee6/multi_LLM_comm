"""
Enumerations of actions in the environment.
"""
import enum
import numpy as np

from typing import Hashable, Iterable

from .agent import Agent
from .graph import Node
from .world import Tool



def _vectorized_dict(d: dict, max_key=None, dtype=object):
    """
    Dictionary mapping with fast numpy-style vectorized indexing.

    Parameters
    ----------
    d : dict
        Dictionary with non-negative integers as keys
    max_key : Optional[int]
        Maximum key
    dtype : type
        Value type
    """
    class _vectorize(np.vectorize):
        def __getitem__(self, key):
            return self(key)

    # If keys are all non-negative integers, just use a numpy array
    if all(isinstance(key, int) and key >= 0 for key in d):
        max_key = max_key if max_key else max(d.keys())
        mapping = np.empty(max_key + 1, dtype=dtype)
        mapping[list(d.keys())] = list(d.values())
        return mapping

    # Use np.vectorize
    return _vectorize(d.__getitem__, otypes=[dtype])



def ActionEnum(agents: Iterable[Agent], nodes: Iterable[Node]) -> enum.EnumMeta:
    """
    Returns an Enum metaclass for enumerating possible actions.

    Actions:
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

    Instance Methods:
        tool()
            Get the corresponding tool for a 'use_tool' Action
        node()
            Get the corresponding node for a 'go_to' Action

    Examples
    --------
    >>> Action = ActionEnum(
    ...     agents=(Agent(i) for i in range(3)),
    ...     nodes=(Node(c) for c in 'XYZ'),
    ... )

    >>> a = Action.go_to('Z')
    >>> a
    <Action.go_to_node_Z: 5>
    >>> a.node()
    Node('Z')

    >>> a = Action.use_tool(Tool.blue)
    >>> a
    <Action.use_blue_tool: 8>
    >>> a.tool()
    <Tool.blue: 2>
    """
    Action = enum.IntEnum('Action', {
        'unfreeze_agents': 0,
        'inspect_bomb': enum.auto(),
        'find_next_bomb': enum.auto(),
        'place_bomb_beacon': enum.auto(),
        'place_help_beacon': enum.auto(),
        'remove_bomb_beacon': enum.auto(),
        'remove_help_beacon': enum.auto(),
        **{f'go_to_node_{node.id}': enum.auto() for node in nodes},
        **{f'use_{tool.name}_tool': enum.auto() for tool in Tool},
    })

    # Action-to-object mappings
    action_node_map = {Action[f'go_to_node_{node.id}']: node for node in nodes}
    action_tool_map = {Action[f'use_{tool.name}_tool']: tool for tool in Tool}

    # Vectorized action-to-object mappings
    action_to_node = _vectorized_dict(action_node_map, max_key=max(Action))
    action_to_tool = _vectorized_dict(action_tool_map, max_key=max(Action))

    # Object-to-action mappings
    node_action_map = {node.id: action for action, node in action_node_map.items()}
    tool_action_map = {tool: action for action, tool in action_tool_map.items()}

    # Vectorized object-to-action mappings
    node_to_action = _vectorized_dict(node_action_map, dtype=int)
    tool_to_action = _vectorized_dict(tool_action_map, dtype=int)

    @staticmethod
    def go_to(node_id: Hashable) -> Action:
        """
        Return the `Action` going to the specified node.
        """
        a = node_to_action[node_id]
        return Action(a.item()) if a.shape == () else a

    @staticmethod
    def use_tool(tool: Tool) -> Action:
        """
        Return the `Action` using the given tool.
        """
        a = tool_to_action[tool]
        return Action(a.item()) if a.shape == () else a

    def node(action: Action) -> Node:
        """
        Get the corresponding node for a 'go_to' Action.
        """
        return action_to_node[action]

    def tool(action: Action) -> Tool:
        """
        Get the corresponding tool for a 'use_tool' Action.
        """
        return action_to_tool[action]

    # Dynamically add methods to `Action`
    setattr(Action, 'go_to', go_to)
    setattr(Action, 'use_tool', use_tool)
    setattr(Action, 'node', node)
    setattr(Action, 'tool', tool)

    return Action


def MiniActionEnum(agents: Iterable[Agent], nodes: Iterable[Node]) -> enum.EnumMeta:
    """
    Returns an Enum metaclass for enumerating possible actions.

    Actions:
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

    Instance Methods:
        tool()
            Get the corresponding tool for a 'use_tool' Action
        node()
            Get the corresponding node for a 'go_to' Action

    Examples
    --------
    >>> Action = ActionEnum(
    ...     agents=(Agent(i) for i in range(3)),
    ...     nodes=(Node(c) for c in 'XYZ'),
    ... )

    >>> a = Action.go_to('Z')
    >>> a
    <Action.go_to_node_Z: 5>
    >>> a.node()
    Node('Z')

    >>> a = Action.use_tool(Tool.blue)
    >>> a
    <Action.use_blue_tool: 8>
    >>> a.tool()
    <Tool.blue: 2>
    """
    Action = enum.IntEnum('Action', {
        'inspect_bomb': 0,
        **{f'go_to_node_{node.id}': enum.auto() for node in nodes},
        'use_red_tool':enum.auto(),
        'use_green_tool': enum.auto(),
        'use_blue_tool': enum.auto(),
    })

    # Action-to-object mappings
    action_node_map = {Action[f'go_to_node_{node.id}']: node for node in nodes}
    action_tool_map = {Action['use_red_tool']: Tool.red ,
                       Action['use_green_tool']: Tool.green ,
                       Action['use_blue_tool']: Tool.blue}

    # Vectorized action-to-object mappings
    action_to_node = _vectorized_dict(action_node_map, max_key=max(Action))
    action_to_tool = _vectorized_dict(action_tool_map, max_key=max(Action))

    # Object-to-action mappings
    node_action_map = {node.id: action for action, node in action_node_map.items()}
    tool_action_map = {tool: action for action, tool in action_tool_map.items()}

    # Vectorized object-to-action mappings
    node_to_action = _vectorized_dict(node_action_map, dtype=int)
    tool_to_action = _vectorized_dict(tool_action_map, dtype=int)

    @staticmethod
    def go_to(node_id: Hashable) -> Action:
        """
        Return the `Action` going to the specified node.
        """
        a = node_to_action[node_id]
        return Action(a.item()) if a.shape == () else a

    @staticmethod
    def use_tool(tool: Tool) -> Action:
        """
        Return the `Action` using the given tool.
        """
        a = tool_to_action[tool]
        return Action(a.item()) if a.shape == () else a

    def node(action: Action) -> Node:
        """
        Get the corresponding node for a 'go_to' Action.
        """
        return action_to_node[action]

    def tool(action: Action) -> Tool:
        """
        Get the corresponding tool for a 'use_tool' Action.
        """
        return action_to_tool[action]

    # Dynamically add methods to `Action`
    setattr(Action, 'go_to', go_to)
    setattr(Action, 'use_tool', use_tool)
    setattr(Action, 'node', node)
    setattr(Action, 'tool', tool)

    return Action

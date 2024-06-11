import json
import numpy as np

from collections import deque
from gym_dragon.utils import BoundedGrid
from pathlib import Path
from typing import Callable, Iterable



### Blocks

CLIMBABLE = {
    'birch_stairs', 'brick_stairs',
    'jungle_stairs', 'nether_brick_stairs', 'oak_stairs', 'quartz_stairs',
    'stone_brick_stairs', 'red_sandstone_stairs',
    'stone_slab', 'stone_slab2', 'wooden_slab',
}
DOORS = {'acacia_door', 'birch_door', 'dark_oak_door', 'spruce_door', 'wooden_door'}
GATES = {'birch_fence_gate', 'dark_oak_fence_gate', 'jungle_fence_gate'}
NON_SOLID = {'air', 'rail', 'red_flower', 'tallgrass', 'tnt', 'torch',
    'vine', 'waterlily', 'yellow_flower',
}
ROOFS = {
    'birch_stairs', 'double_stone_slab', 'double_wooden_slab', 'nether_brick_stairs',
    'planks', 'stone_slab', 'wooden_slab',
}



### Helper Functions

def get_dragon_info() -> tuple[tuple[int, int, int], BoundedGrid]:
    """
    Get start location and regions for the dragon map.
    """
    start_loc = (27, 51, 50)
    regions = BoundedGrid((-1, 52), (-1, 152))
    regions[:, :51] = 'forest'
    regions[:, 51:101] = 'village'
    regions[:, 101:] = 'desert'
    return start_loc, regions

def deltas(dim) -> np.ndarray:
    """
    Get the relative grid locations of the neighbors of a given grid location.

    Parameters
    ----------
    dim : int
        Dimension of the grid

    Examples
    --------
    >>> deltas(dim=2).tolist()
    [[1, 0], [0, 1], [-1, 0], [0, -1]]
    """
    return np.vstack([np.eye(dim, dtype=int), -np.eye(dim, dtype=int)])

def is_traversable(
    loc: tuple[int, int, int],
    blocks: BoundedGrid,
    regions: BoundedGrid,
    dy:int = 0) -> bool:
    """
    Check whether the given location is traversable.

    Parameters
    ----------
    loc : tuple[int, int, int]
        (x, y, z) location to check
    blocks : BoundedGrid
        3D BoundedGrid indicating the block type at each (x, y, z) location
    regions : BoundedGrid
        2D BoundedGrid indicating the region name at each (x, z) location
    dy : int, default=0
        The relative height difference between the given location and its neighbor
    """
    x, y, z = loc

    # Check bounds
    if not blocks.is_in_bounds(loc):
        return False

    # Ignore roofs in the village region
    if regions and regions[x, z] == 'village' and blocks[loc] in ROOFS:
        return False

    # Can only climb "climbable" blocks
    if dy > 0 and blocks[loc] not in CLIMBABLE:
        return False

    # Check for a solid block, with a non-solid block or door above it
    return (
        blocks[loc] not in NON_SOLID
        and blocks[x, y + 1, z] in {*NON_SOLID, *DOORS, *GATES}
    )

def traversable_neighbors(
    loc: tuple[int, int, int],
    blocks: BoundedGrid,
    regions: BoundedGrid) -> set[tuple[int, int, int]]:
    """
    Get traversable neighbors from a given location.

    Parameters
    ----------
    loc : tuple[int, int, int]
        (x, y, z) location
    blocks : BoundedGrid
        3D BoundedGrid indicating the block type at each (x, y, z) location
    regions : BoundedGrid
        2D BoundedGrid indicating the region name at each (x, z) location
    """
    x, y, z = loc
    neighbors = set()

    for dx, dz in deltas(dim=2):
        for dy in [-1, 0, 1]:
            neighbor = (x + dx, y + dy, z + dz)
            if is_traversable(neighbor, blocks, regions, dy=dy):
                neighbors.add(neighbor)

    return neighbors

def bfs(
    source: object,
    neighbors_fn: Callable[[object], set[object]],
    invalid: set = set()) -> set:
    """
    Find all nodes connected to the source.

    Parameters
    ----------
    source: object
        Node to start search from
    neighbors_fn : Callable(object) -> set[object]
        Function to get neighbors of a given node
    invalid : set, optional
        Nodes to consider invalid and return `None` if any are found
    """
    explored, queue = {source}, deque([source])
    while queue:
        v = queue.popleft()
        for u in neighbors_fn(v):
            if u in invalid:
                return None
            if u not in explored:
                explored.add(u)
                queue.append(u)

    return explored



### Maps

def get_map_3d(json_path: str) -> BoundedGrid:
    """
    Create 3D map from JSON file.

    Parameters
    ----------
    json_path : str
        Path to JSON file for the map

    Returns
    -------
    grid : BoundedGrid
        3D BoundedGrid indicating the block type at each (x, y, z) location
    """
    # Load map data from JSON
    with open(json_path, 'r') as map_file:
        map_data = json.load(map_file)

    # Get map bounds
    lower_bounds = [map_data['metadata']['lower_bound'][axis] for axis in ['x', 'y', 'z']]
    upper_bounds = [map_data['metadata']['upper_bound'][axis] for axis in ['x', 'y', 'z']]
    bounds = list(zip(lower_bounds, upper_bounds))

    # Get 3D map
    map_3d = BoundedGrid(*bounds, fill='air')
    for block in map_data['blocks']:
        location = (block['location']['x'], block['location']['y'], block['location']['z'])
        map_3d[location] = block['type']

    return map_3d

def get_map_2d(
    json_path: str,
    start_loc: tuple[int, int, int],
    regions: BoundedGrid = None,
    include_bombs=False) -> BoundedGrid:
    """
    Create 2D map from JSON file.

    Parameters
    ----------
    json_path : str
        Path to JSON file for the map
    start_loc : tuple[int, int, int]
        (x, y, z) location to start traversability search from
    regions : BoundedGrid, optional
        2D BoundedGrid indicating the region name at each (x, z) location
    include_bombs : bool, default=False
        Whether to include bombs in the map

    Returns
    -------
    grid : BoundedGrid
        2D BoundedGrid indicating the block type at each (x, z) location as one of:
            - 'ground' (traversable)
            - 'water' (traversable)
            - 'stepping_stone' (traversable)
            - 'obstacle' (non-traversable)
            - 'bomb'
    """
    map_3d = get_map_3d(json_path)
    map_2d = BoundedGrid(map_3d.bounds[0], map_3d.bounds[2], fill='obstacle')

    # Populate 2D map
    traversable_locations = bfs(
        source=start_loc,
        neighbors_fn=lambda v: traversable_neighbors(v, map_3d, regions),
    )
    for x, y, z in traversable_locations:
        if map_3d[x, y, z] == 'water':
            map_2d[x, z] = 'water'
        elif map_3d[x, y, z] == 'stone_slab':
            map_2d[x, z] = 'stepping_stone'
        else:
            map_2d[x, z] = 'ground'

    # Remove stepping stones that are not adjacent to water
    for x, _, z in traversable_locations:
        if map_2d[x, z] == 'stepping_stone':
            if not any (
                map_2d.is_in_bounds((x + dx, z + dz))
                and map_2d[x + dx, z + dz] == 'water' for dx, dz in deltas(dim=2)):
                map_2d[x, z] = 'ground'

    # Place bombs
    for x, y, z in map_3d.locations():
        if map_3d[x, y, z] == 'tnt':
            map_2d[x, z] = 'bomb' if include_bombs else 'ground'

    return map_2d

def get_dragon_2d(json_path: str) -> BoundedGrid:
    """
    Get 2D Dragon map.

    Parameters
    ----------
    json_path : str
        Path to JSON file for the map

    Returns
    -------
    grid : BoundedGrid
        2D BoundedGrid indicating the block type at each (x, z) location as one of:
            * 'ground' (traversable)
            * 'water' (traversable)
            * 'stepping_stone' (traversable)
            * 'obstacle' (non-traversable)
            * 'bomb'
    """
    start_loc, regions = get_dragon_info()
    return get_map_2d(
        json_path, start_loc=start_loc, regions=regions, include_bombs=False)



### Visualization

def visualize(grid: BoundedGrid, save_path=None):
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    block_to_color = {
        'obstacle': 'black',
        'water': 'blue',
        'bomb': 'red',
        'stepping_stone': 'gray',
        'ground': 'white',
    }
    colors = list(block_to_color.values())

    img = grid.numpy()[::-1] # bird's eye view
    img = np.vectorize(lambda x: colors.index(block_to_color.get(x)))(img)
    plt.axis('off')
    plt.imshow(img, cmap=mcolors.ListedColormap(colors))
    plt.legend(
        handles=[
            mpatches.Patch(color='white', label='Ground'),
            mpatches.Patch(color='black', label='Obstacle'),
            mpatches.Patch(color='blue', label='Water'),
            mpatches.Patch(color='gray', label='Stepping Stone'),
            mpatches.Patch(color='red', label='Bomb'),
        ],
        loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.5),
        handlelength=1, handleheight=1, facecolor='black', framealpha=0.1,
    )

    if save_path:
        plt.imsave(save_path, img, cmap=mcolors.ListedColormap(colors))

    plt.show()



if __name__ == '__main__':
    RESOURCES_DIR = Path(__file__).parent / '../resources'

    dragon_2d = get_dragon_2d(RESOURCES_DIR / 'Dragon_1.0_3D_Updated-fov_map.json')
    visualize(dragon_2d, save_path=None)

    # saturn_2d = get_map_2d(RESOURCES_DIR / 'Saturn_2.6_3D-fov_map.json', start_loc=(-2100, 60, 0))
    # saturn_2d = saturn_2d.view[:, -14:67] # adjust bounds for visualization
    # visualize(saturn_2d, save_path='saturn.png')

    # desert_2d = get_map_2d(RESOURCES_DIR / 'Desert_1.0_3D-fov_map.json', start_loc=(25, 51, 25))
    # desert_2d = desert_2d.view[:52, :52] # adjust bounds for visualization
    # visualize(desert_2d, save_path=None)

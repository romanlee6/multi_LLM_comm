import itertools
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import tqdm

from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

from maps import DOORS, traversable_neighbors, bfs, deltas, get_map_2d, get_map_3d, get_dragon_info
from gym_dragon.utils import BoundedGrid



### Constants

RESOURCES_DIR = Path(__file__).parent / '../resources'
INITIAL_DRAGON_CUT = {
    (26, 50), (27, 50),
    (26, 100), (27, 100),
    (8, 103),
    (32, 6),
    (40, 54), (44, 54), (48, 54),
    (49, 57), (46, 61),
    (45, 70),
    (43, 78),
    (47, 83),
    (46, 87), (48, 87),
    (39, 95), (43, 95), (48, 95),
    (41, 89),
    (33, 90),
    (31, 97),
    (32, 61), (33, 61),
    (20, 58),
    (13, 69),
    (9, 69),
    (9, 84),
    (3, 79), (3, 92), (3, 97),
    (20, 91),
    (36, 73), (36, 74), (36, 75), (36, 76), (36, 77), (36, 78),
}



### Misc

def unique(items: Iterable) -> list:
    """
    Return list of unique items.
    """
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out

def grid_neighbors(loc: tuple, grid: BoundedGrid, valid: set) -> set[tuple]:
    """
    Return neighboring grid locations.

    Parameters
    ----------
    loc : tuple
        (x, z) grid location
    grid : BoundedGrid
        2D BoundedGrid
    valid : set
        Valid neighboring grid values
    """
    x, z = loc
    return {
        (x + dx, z + dz)
        for dx, dz in deltas(dim=2)
        if grid.is_in_bounds((x + dx, z + dz))
        and grid[(x + dx, z + dz)] in valid
    }



### Graph

def get_graph(
    source: tuple,
    blocks: BoundedGrid,
    regions: BoundedGrid,
    exclude={'water'}) -> dict[tuple, set[tuple]]:
    """
    Create adjacency list representation of traversability graph.

    Parameters
    ----------
    source : tuple
        (x, y, z) location to start traversability search from
    blocks : BoundedGrid
        3D BoundedGrid indicating the block type at each (x, y, z) location
    regions : BoundedGrid
        2D BoundedGrid indicating the region name at each (x, z) location
    exclude : set[str], default={'water'}
        Block types to exclude from traversibility graph

    Returns
    -------
    graph : dict[tuple, set[tuple]]
        Adjacency list representation of traversable (x, z) locations
    """
    graph = defaultdict(set)
    explored, queue = {source}, deque([source])
    while queue:
        v = queue.popleft()
        for u in traversable_neighbors(v, blocks, regions):
            if {blocks[u], blocks[v]}.isdisjoint(exclude):
                graph[v[0], v[2]].add((u[0], u[2]))
                graph[u[0], u[2]].add((v[0], v[2]))
            if u not in explored:
                explored.add(u)
                queue.append(u)

    return graph

def remove_node(graph: dict, v: object) -> dict:
    """
    Remove a node from the graph.

    Parameters
    ----------
    graph : dict[object, set[object]]
        Adjacency list representation of graph
    v: object
        Node to remove
    """
    for u in graph[v]:
        graph[u].remove(v)
    del graph[v]
    return graph



### Supercut

def potential_cuts(graph: dict, k=1, diagonals=True) -> set[tuple]:
    """
    Generate potential cuts of k nodes that form a
    vertical, horizontal, or diagonal line.

    Parameters
    ----------
    graph : dict[tuple, set[tuple]]
        Adjacency list representation of graph
    k : int, default=1
        Number of nodes in each cut
    diagonals : bool, default=True
        Whether to consider diagonal cuts
    """
    for x, z in graph:
        groups = {
            tuple((x + dx, z) for dx in range(k)),
            tuple((x, z + dz) for dz in range(k)),
        }

        if diagonals:
            groups.add(tuple((x + delta, z + delta) for delta in range(k)))
            groups.add(tuple((x + delta, z - delta) for delta in range(k)))

        for group in groups:
            if all(v in graph for v in group):
                yield group

def max_k_cut(graph: dict, k=1, diagonals=True) -> tuple[set, int]:
    """
    Let a k-cut be a set of k contiguous nodes that, when removed,
    partition the graph into multiple connected components.

    Find the k-cut with the maximum number of nodes in its
    smallest connected component.

    Parameters
    ----------
    graph : dict[object, set[object]]
        Adjacency list representation of graph
    k : int, default=1
        Number of nodes in k-cut
    diagonals : bool, default=True
        Whether to consider diagonal cuts

    Returns
    -------
    cut : set
        Set of nodes in max k-cut
    size : int
        Size of smallest connected component created by max k-cut
    """
    cuts = {}
    for cut in list(potential_cuts(graph, k=k, diagonals=diagonals)):
        # Find connected components
        components = []
        v = list(cut)[0]
        for u in graph[v] - set(cut):
            c = bfs(
                u,
                neighbors_fn=lambda node: graph[node] - set(cut),
                invalid=graph[v] - {u, *cut},
            )
            components.append(c)

        # Check if cut is valid
        components = unique(components)
        if None in components or len(components) < 2:
            continue

        # Record size of smallest component
        cuts[cut] = min(len(c) for c in components)

    # Get the max cut
    if cuts:
        max_cut = max(cuts, key=cuts.get)
        return max_cut, cuts[max_cut]

    return set(), 0

def get_supercut(
    graph: dict, initial_cut=set(),
    max_k=50, threshold=4, diagonals=True, save_path=None) -> set:
    """
    Get a set of nodes to remove that "nicely" partition the graph
    into connected components.

    Parameters
    ----------
    graph : dict[object, set[object]]
        Adjacency list representation of graph
    initial_cut : set, default=set()
        Set of nodes to remove before running supercut algorithm
    max_k : int, default=50
        Maximum number of nodes in a single cut
    threshold : int, default=4
        Minimum cut size
    save_path : str, optional
        Path to save supercut results to disk
    """
    # Greedy Algorithm
    supercut = initial_cut
    for v in initial_cut:
        remove_node(graph, v)

    for _ in tqdm.tqdm(itertools.count()): # infinite loop
        cuts = [max_k_cut(graph, k=k, diagonals=diagonals) for k in range(1, max_k + 1)]
        cuts = [cut for cut in cuts if cut[1] >= threshold]
        if len(cuts) == 0:
            break

        # Remove nodes from cut and add them to supercut set
        nodes, size = max(cuts, key=lambda cut: cut[1] - len(cut[0])**2)
        for v in nodes:
            remove_node(graph, v)
            supercut.update(nodes)

        tqdm.tqdm.write(f'k={len(nodes)} cutsize={size}')

    # Save to disk
    if save_path:
        with open(save_path, 'wb') as file:
            pickle.dump(supercut, file)

    return supercut



### Supergraph

def get_supergraph(graph: dict, map_2d: BoundedGrid, supercut: set[tuple]
    ) -> tuple[BoundedGrid, dict[int, tuple[int, int]], set[tuple[int, int]]]:
    """
    Create supergraph where:
        * the supernodes are the connected components after removing supercut nodes
        * the superedges indicate traversability between supernodes

    Parameters
    ----------
    graph : dict[tuple, set[tuple]]
        Adjacency list representation of traversablity graph
    map_2d : BoundedGrid
        Grid indicating the block type of each (x, z) location
    supercut : set[tuple]
        Set of grid locations to remove to segment the grid into connected components

    Returns
    -------
    segmentation_grid : BoundedGrid
        Grid indicating the supernode index of each (x, z) location
    centroids : dict[int, tuple[int, int]]
        The (x, z) centroid for each supernode
    edges : set[tuple[int, int]]
        Set of superedges, where (i, j) indicates an edge between supernodes i & j
    """
    # Remove supercut nodes
    for node in supercut:
        if node in graph:
            remove_node(graph, node)

    # Get land components
    components = []
    for node in graph:
        if not components or node not in set.union(*components):
            components.append(bfs(node, neighbors_fn=graph.get))

    # Get water components
    get_water_neighbors = lambda x: grid_neighbors(x, map_2d, {'water'})
    for node in map_2d.locations():
        if map_2d[node] == 'water':
            if not components or node not in set.union(*components):
                components.append(bfs(node, neighbors_fn=get_water_neighbors))

    # Create segmentation grid
    segmentation_grid = BoundedGrid(*map_2d.bounds)
    for i, c in enumerate(components):
        for node in c:
            segmentation_grid[node] = i

    # Arbitrarily assign cut nodes to segments
    for x, z in supercut:
        for dx, dz in deltas(dim=2):
            if segmentation_grid.is_in_bounds((x + dx, z + dz)):
                if segmentation_grid[x + dx, z + dz] not in {None, 'water'}:
                    segmentation_grid[x, z] = segmentation_grid[x + dx, z + dz]
                    break

    # Get superedges
    edges = set()
    for x, z in segmentation_grid.locations():
        a = segmentation_grid[x, z]
        for dx, dz in deltas(dim=2):
            if segmentation_grid.is_in_bounds((x + dx, z + dz)):
                b = segmentation_grid[x + dx, z + dz]
                if a is not None and b is not None and a != b:
                    if map_2d[x, z] == 'water':
                        if map_2d[x + dx, z + dz] == 'stepping_stone':
                            edges.update([(a, b), (b, a)])
                        else:
                            edges.add((b, a)) # without stepping stone, can only ENTER water
                    elif map_2d[x + dx, z + dz] == 'water':
                        if map_2d[x, z] == 'stepping_stone':
                            edges.update([(a, b), (b, a)])
                        else:
                            edges.add((a, b)) # without stepping stone, can only ENTER water
                    else:
                        edges.update([(a, b), (b, a)])

    # Get supernode centroids
    centroids = dict(enumerate([
        tuple(np.around(np.array(list(c)).mean(axis=0)).astype(int))
        for c in components
    ]))

    return segmentation_grid, centroids, edges

def create_dragon_graph(
    map_json_path=RESOURCES_DIR / 'Dragon_1.0_3D_Updated-fov_map.json',
    threshold=30,
    cut_load_path=None, cut_save_path='cut_dragon.pkl',
    visualize=False, graph_save_path=None):
    """
    Create and visualize a supergraph for the Dragon map.

    Parameters
    ----------
    map_json_path : str
        Path to map json file
    threshold : int
        Minimum cut size in supercut algorithm
    cut_load_path : str
        Path to load previously calculated supercut result from
    cut_save_path : str
        Path to save supercut result to
    visualize : bool, default=False
        Whether to visualize the supergraph segmentation
    graph_save_path : str
        Path to save supergraph data to

    Returns
    -------
    dragon_2d : BoundedGrid
        2D grid of map blocks
    segmentation_grid : BoundedGrid
        Grid indicating the supernode number of each (x, z) location
    centroids : dict[int, tuple[int, int]]
        The (x, z) centroid for each supernode
    edges : set[tuple[int, int]]
        Set of superedges, where (i, j) indicates an edge between supernodes i & j
    """
    start_loc, regions = get_dragon_info()

    # Load graph
    dragon_3d = get_map_3d(map_json_path)
    graph = get_graph(start_loc, dragon_3d, regions)

    # Get supercut
    if cut_load_path and Path(cut_load_path).is_file():
        with open(cut_load_path, 'rb') as file:
            supercut = pickle.load(file)
    else:
        supercut = get_supercut(
            graph, initial_cut=INITIAL_DRAGON_CUT, diagonals=False, save_path=cut_save_path, threshold=threshold)

    # Get supergraph
    dragon_2d = get_map_2d(map_json_path, start_loc=start_loc, regions=regions)
    segmentation_grid, centroids, edges = get_supergraph(graph, dragon_2d, supercut)

    # Visualize
    if visualize:
        visualize_supercut(dragon_2d, supercut)
        visualize_graph(dragon_2d, centroids, edges)
        print(f'{len(centroids)} nodes', f'{len(edges)} directed edges')

    # Save to disk
    if graph_save_path:
        graph_data = {
            'block_grid': dragon_2d,
            'segmentation_grid': segmentation_grid,
            'centroids': centroids,
            'edges': edges,
        }
        with open(graph_save_path, 'wb') as file:
            pickle.dump(graph_data, file)

    return dragon_2d, segmentation_grid, centroids, edges



### Semantic Map

def create_semantic_map(
    input_path=RESOURCES_DIR / 'Dragon_1.0_3D_Updated-fov_map.json',
    output_path='semantic_map.json',
    threshold=30) -> dict:
    """
    Create a semantic map.

    Parameters
    ----------
    input_path : str
        Path to the input map JSON file
    output_path : str
        Path to the output JSON file for the semantic map
    threshold : int, default=30
        Minimum cut size for supercut algorithm
    """
    start_loc, regions = get_dragon_info()
    map_3d = get_map_3d(input_path)
    map_2d = get_map_2d(input_path, start_loc=start_loc, regions=regions)
    graph = get_graph(start_loc, map_3d, regions)

    # Get supercut
    cut_load_path = 'cut_dragon.pkl'
    if cut_load_path and Path(cut_load_path).is_file():
        with open(cut_load_path, 'rb') as file:
            supercut = pickle.load(file)
    else:
        supercut = get_supercut(
            graph,
            initial_cut=INITIAL_DRAGON_CUT,
            diagonals=False,
            save_path=None,
            threshold=threshold,
        )

    # Get supergraph
    segmentation_grid, centroids, edges = get_supergraph(graph, map_2d, supercut)

    # Initialize semantic map
    semantic_map = {'locations': [], 'connections': [], 'objects': []}

    # Add locations
    for i in segmentation_grid.unique(exclude={None}):
        region = regions[centroids[i]]
        room = {
            'id': f'{region.lower()}_{i}',
            'name': f'{region.capitalize()} {i}',
            'type': 'room',
            'bounds': {
                'type': 'xz_list',
                'coordinates': [[int(x), int(z)] for x, z in segmentation_grid.find(i)],
            }
        }
        semantic_map['locations'].append(room)
    
    # Add connections
    undirected_edges = [(i, j) for i, j in edges if i < j]
    for i, j in sorted(undirected_edges):
        border = []
        for x, z in segmentation_grid.find(i):
            for dx, dz in deltas(dim=2):
                if segmentation_grid.is_in_bounds((x + dx, z + dz)):
                    if segmentation_grid[x + dx, z + dz] == j:
                        border.append([x, z])
                        break

        border = sorted(border)
        border_type = None
        if all(loc[0] == border[0][0] for loc in border):
            border_type = 'line'
            border[-1][1] += 1
        elif all(loc[1] == border[0][1] for loc in border):
            border_type = 'line'
            border[-1][0] += 1

        if not any (DOORS.isdisjoint(map_3d[x, :, z]) for x, z in border):
            border_type = 'door'

        connection = {
            'id': f'c_{i}_{j}',
            'type': 'extension',
            'bounds': {
                'type': border_type,
                'coordinates': [
                    {'x': int(border[0][0]), 'z': int(border[0][1])},
                    {'x': int(border[-1][0]), 'z': int(border[-1][1])},
                ],
            },
            'connected_locations': [
                f'{regions[centroids[i]].lower()}_{i}',
                f'{regions[centroids[j]].lower()}_{j}',
            ],
        }
        semantic_map['connections'].append(connection)
    
    # Add objects
    semantic_map['connections']
    for obj_id, x, z in [('ls_1', -43, 8), ('ls_2', -43, 4), ('ls_3', -44, 6)]:
        obj = {
            'id': obj_id,
            'type': 'button',
            'bounds': {
                'type': 'block',
                'coordinates': [{'x': x, 'z': z}]
            },
            'facing': 'top_south',
        }

    with open(output_path, 'w') as file:
        json.dump(semantic_map, file, indent=2)
    
    return semantic_map



### Visualization

def visualize_supercut(grid: BoundedGrid, supercut: set, save_path=None):
    """
    Visualize the supercut over the map.
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    block_to_color = {
        'obstacle': 'black',
        'water': 'blue',
        'bomb': 'red',
        'stepping_stone': 'gray',
        'ground': 'white',
        'border': 'violet',
    }
    colors = list(block_to_color.values())

    # Fill in supercut locations
    grid = grid.copy()
    for x, z in supercut:
        grid[x, z] = 'border'

    # Plot grid overlayed with the borders given by the supercut
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
            mpatches.Patch(color='violet', label='Border'),
        ],
        loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.5),
        handlelength=1, handleheight=1, facecolor='black', framealpha=0.1,
    )

    # Save to disk
    if save_path:
        plt.imsave(save_path, img, cmap=mcolors.ListedColormap(colors))

    plt.show()

def visualize_graph(
    grid: BoundedGrid, centroids: dict, edges: set, upsample=8, dpi=96, save_path=None):
    """
    Visualize the graph resulting from a supercut.
    """
    # Create directed graph
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(centroids)
    directed_graph.add_edges_from(edges)

    # Plot directed graph
    fig, ax = plt.subplots(
        figsize=(grid.shape[1] * upsample // dpi, grid.shape[0] * upsample // dpi),
        dpi=dpi, frameon=False,
    )
    nx.draw_networkx(
        directed_graph,
        pos={node: (z, x) for node, (x, z) in centroids.items()},
        arrowsize=7.5, with_labels=True, node_size=100, font_size=6, ax=ax,
    )
    ax.set_xlim(grid.bounds[1][0], grid.bounds[1][1])
    ax.set_ylim(grid.bounds[0][0], grid.bounds[0][1])

    # Save to disk
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()



if __name__ == '__main__':
    create_semantic_map()
    create_dragon_graph(cut_load_path='cut_dragon.pkl', graph_save_path='dragon.pkl', visualize=True)

"""
Module for rendering and visualizing environments.
"""
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .core import Color
from .utils import BoundedGrid



### Colors & Labels

_rgb = lambda *x: tuple(c / 255 for c in x) # scale to within [0, 1]

AGENT_COLOR = {
    Color.red: _rgb(255, 0, 0),
    Color.green: _rgb(0, 255, 0),
    Color.blue: _rgb(0, 0, 255),
}

BLOCK_INFO = {
    'obstacle': ('Obstacle', _rgb(50, 50, 50)),
    'ground': ('Ground', _rgb(255, 255, 255)),
    'water': ('Water', _rgb(155, 200, 255)),
    'fire': ('Fire', _rgb(255, 174, 66)),
    'bomb_inactive': ('Inactive Bomb', _rgb(255, 180, 180)),
    'bomb_active': ('Active Bomb', _rgb(255, 0, 0)),
}



### Helper Functions

def _legend_circle(label: str, color: tuple[float]):
    """
    Create a legend entry for a circle.

    Parameters
    ----------
    label : str
        Label for legend entry
    color : tuple[float]
        RGB color of circle (values between 0 and 1)
    """
    return mlines.Line2D(
        [], [], mfc=color, label=label,
        mec=(0, 0, 0, 0), c=(0, 0, 0, 0), marker='o', ms=10,
    )



### Rendering

class Renderer:
    """
    Class for visualizing trajectories in a DragonEnv environment.

    Attributes
    ----------
    env : DragonEnv
        Environment to render
    """

    def __init__(self, env, figsize=(12, 4), overlay_graph=False, show_legend=True, **kwargs):
        """
        Parameters
        ----------
        env : DragonEnv
            Environment to render
        figsize : tuple[float, float], default=(12, 4)
            Figure size (width, height)
        overlay_graph : bool, default=False
            Whether to overlay the graph representation over the map
        show_legend : bool, default=True
            Whether to show the legend on the rendered figure
        """
        self.env = env
        self._figure, self._ax = None, None
        self._map = None
        self._figsize = figsize
        self._overlay_graph = overlay_graph
        self._show_legend = show_legend

    def render(self, block_grid: BoundedGrid, save_path=None, sleep=1e-6, **kwargs):
        """
        Render the current state of the environment.

        Parameters
        ----------
        block_grid : BoundedGrid
            Grid indicating the block type at each (x, z) location
        save_path : Optional[str]
            Path to save figure as image
        """
        # Create rendering figure if one does not already exist
        if not self._figure:
            self._figure, self._ax = plt.subplots(figsize=self._figsize)
            self.reset(block_grid)

        # Update agent locations
        locations = [agent.node.centroid for agent in self.env.agents.values()]
        overlapping = {loc for loc in locations if locations.count(loc) > 1}
        for agent in self.env.agents.values():
            if agent.node:
                x, z = agent.node.centroid
                self._agent_dots[agent.id].set_center((z, x))
                self._agent_dots[agent.id].set_alpha(0.5 if (x, z) in overlapping else 1)

        # Display figure
        plt.pause(sleep)

        # Save figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

    def update_map(self, block_grid: BoundedGrid):
        """
        Update the image of the map's block grid.

        Parameters
        ----------
        block_grid : BoundedGrid
            Grid indicating the block type at each (x, z) location
        """
        block_to_rgb = lambda b: BLOCK_INFO[b][1]
        img = np.vectorize(block_to_rgb)(block_grid.numpy()[::-1])
        img = np.array(img).transpose(1, 2, 0)
        if not self._map:
            self._map = self._ax.imshow(
                img, extent=(*block_grid.bounds[1], *block_grid.bounds[0]))
        else:
            self._map.set_data(img)

    def reset(self, block_grid: BoundedGrid):
        """
        Reset the rendering figure.

        Parameters
        ----------
        block_grid : BoundedGrid
            Grid indicating the block type at each (x, z) location
        """
        # Clear axes
        self._ax.clear()
        self._ax.set_xlim(*block_grid.bounds[1])
        self._ax.set_ylim(*block_grid.bounds[0])
        self._ax.axis(False)
        self._map = None

        # Draw map
        self.update_map(block_grid)

        # Draw agents
        self._agent_dots = {}
        for agent in self.env.agents.values():
            dot = mpatches.Circle((0, 0), radius=1, fc=AGENT_COLOR[agent.color], zorder=99)
            self._agent_dots[agent.id] = dot
            self._ax.add_patch(dot)

        # Draw graph
        if self._overlay_graph:
            directed_graph = nx.DiGraph()
            directed_graph.add_nodes_from(self.env.graph.nodes)
            directed_graph.add_edges_from(self.env.graph.edges)

            flip = lambda x, z: (z, x)
            nx.draw(
                directed_graph,
                pos={node.id: flip(*node.centroid) for node in self.env.graph.nodes.values()},
                arrows=False, node_size=10, node_color='silver', edge_color='silver',
                alpha=0.3, ax=self._ax,
            )

        # Draw legend
        if self._show_legend:
            handles = [mpatches.Patch(label=l, color=c) for l, c in BLOCK_INFO.values()]
            for color, c in AGENT_COLOR.items():
                handles.append(_legend_circle(f'{color.name.capitalize()} Player', c))

            plt.legend(
                handles=handles, handlelength=1, handleheight=1, loc='lower center',
                ncol=3, bbox_to_anchor=(0.5, -0.5), facecolor='black', framealpha=0.05,
            ).get_frame().set_boxstyle('square')

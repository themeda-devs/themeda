import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from polytorch.plots import format_fig

from .colours import LEVEL4_COLOURS

def plotly_discrete_colorscale(colors):
    """
    parameters:
      - colors : a list of plotly RGB colors
    
    returns:
      - list : a plotly discrete colorscale
    """
    if not colors:
        raise ValueError("colorscale must not be empty")
    if len(colors) == 1:
        return [[0, colors[0]], [1, colors[0]]]
    n_colors = len(colors)
    color_positions = np.linspace(0, 1, n_colors+1)
    discrete_colorscale = []
    for i, color in enumerate(colors):
        discrete_colorscale.extend([[color_positions[i], color], [color_positions[i + 1], color]])
    return discrete_colorscale


def plot_level4(array, show:bool=False):
    color_scale = plotly_discrete_colorscale(list(LEVEL4_COLOURS.values()))
    labels = list(LEVEL4_COLOURS.keys())
    tickvals = np.arange(len(labels))
    heatmap = go.Heatmap(
        z=array, 
        zmin=-0.5,
        zmax=len(labels)-0.5,
        colorscale=color_scale,
        colorbar=dict(thickness=25, tickvals=tickvals, ticktext=labels),
    )
    fig = go.Figure(data=[heatmap])
    format_fig(fig)

    if show:
        fig.show()

    return fig


def plot_level4_chiplet(chiplet:Path|str, **kwargs):
    chiplet = Path(chiplet)
    if not chiplet.exists():
        raise FileNotFoundError(f"Cannot find chiplet {chiplet}")
    data = np.load(chiplet)
    return plot_level4(data["data"], **kwargs)

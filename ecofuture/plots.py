import numpy as np
from pathlib import Path
import plotly.express as px
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


def plot_chiplet_location(chiplet:Path|str, **kwargs):
    chiplet = Path(chiplet)
    if not chiplet.exists():
        raise FileNotFoundError(f"Cannot find chiplet {chiplet}")
    data = np.load(chiplet)

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:3577", "latlon", always_xy=True)
    longitude, latitude = transformer.transform(data['position'][0],data['position'][1])

    fig = px.scatter_geo(
        lat=[latitude],
        lon=[longitude],
        projection="mercator",
        opacity=1.0,
        width=1200,
        height=600,
        # zoom=4,
    )
    # fig.update_geos(lataxis_range=list(df["lat"].quantile([0.01, 0.99])), lonaxis_range=list(df["lon"].quantile([0.01, 0.99])))
    fig.update_traces(marker=dict(size=2, symbol="square"))
    fig.update_layout(
            geo = dict(
                projection_scale=10, #this is kind of like zoom
                center=dict(lat=latitude, lon=longitude), # this will center on the point
            ))
    fig.update_traces(marker=dict(size=10), marker_color="red", marker_symbol="circle")

    format_fig(fig)

    return fig
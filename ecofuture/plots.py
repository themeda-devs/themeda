# -*- coding: future_typing -*-

import torch
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import wandb
import plotly.graph_objects as go
from fastcore.dispatch import typedispatch

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


def plot_level4(array, show:bool=False, title=""):
    color_scale = plotly_discrete_colorscale(list(LEVEL4_COLOURS.values()))
    labels = list(LEVEL4_COLOURS.keys())
    labels = [f"{i}: {label}" for i, label in enumerate(labels)]
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

    if title:
        fig.update_layout(title=title)

    if show:
        fig.show()

    return fig


def plot_level4_chiplet(chiplet:Path|str, **kwargs):
    chiplet = Path(chiplet)
    if not chiplet.exists():
        raise FileNotFoundError(f"Cannot find chiplet {chiplet}")
    data = np.load(chiplet)
    return plot_level4(data["data"], **kwargs)


def plot_chiplet_location(chiplet:Path|str, projection_scale:int=10):
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
    )

    fig.update_layout(
        geo = dict(
            projection_scale=projection_scale,
            center=dict(lat=latitude, lon=longitude),
        )
    )
    fig.update_traces(marker=dict(size=10), marker_color="red", marker_symbol="circle")

    format_fig(fig)

    return fig


@typedispatch
def wandb_process(x, y, samples, outs, preds):
    
    table = wandb.Table(columns=["Input", "Prediction"])
    index = 0

    wandb_log_dir = Path("wandb-images")
    wandb_log_dir.mkdir(parents=True, exist_ok=True)

    for sample_input, prediction in zip(samples, outs):
        input_filename = str(wandb_log_dir/f"input-{index}.png")
        prediction_filename = str(wandb_log_dir/f"prediction-{index}.png")
        
        plot_level4(sample_input[0][-1], title="Input").write_image(input_filename)
        # plot_level4(sample_target[0]).write_image(f"target-{index}.png")
        plot_level4(torch.argmax(prediction[0][-1], axis=0), title="Prediction").write_image(prediction_filename)

        table.add_data(
            wandb.Image(input_filename),
            # wandb.Image(f"target-{index}.png"),
            wandb.Image(prediction_filename),
        )
        index += 1
        
    return {"Land Cover": table}
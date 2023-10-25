from typing import List
import torch
from pathlib import Path
from torch import nn
from functools import partial
from fastai.data.core import DataLoaders
import torchapp as ta
from fastcore.foundation import mask2idxs
from fastai.data.block import DataBlock
from fastai.data.transforms import IndexSplitter 
from rich.console import Console
console = Console()
from enum import Enum
import dateutil.parser
from fastai.learner import Learner, load_learner

from dateutil import rrule
from rich.table import Table
from rich.box import SIMPLE
from torchapp.util import call_func

from fastai.data.block import TransformBlock

from polytorch import CategoricalData, ContinuousData, BinaryData, PolyLoss, CategoricalLossType, BinaryLossType, PolyData
from polytorch.metrics import categorical_accuracy, smooth_l1, binary_accuracy, binary_dice, binary_iou, generalized_dice, PolyMetric

from themeda_preproc.source import DataSourceName, is_data_source_continuous
from themeda_preproc.roi import ROIName
from themeda_preproc.chiplet_table import load_table
from themeda_preproc.summary_stats import load_stats
import torch.nn.functional as F

from .dataloaders import TPlus1Callback, get_chiplets_list, PredictPersistanceCallback, FutureDataLoader
from .models import ResNet, TemporalProcessorType, ThemedaModelUNet, ThemedaModel, ThemedaModelSimpleConv, PersistenceModel, ProportionsLSTMModel,ThemedaModelEResBlock
from .transforms import ChipletBlock, StaticChipletBlock, Normalize, make_binary
from .metrics import smooth_l1_rain, smooth_l1_tmax, kl_divergence_proportions, HierarchicalKLDivergence, HierarchicalCategoricalAccuracy
from .plots import wandb_process
from .util import get_land_cover_colours
from .loss import ProportionLoss
from .apps import Themeda

from torchviz import make_dot


if __name__ == "__main__":
    # Instantiate the Themeda class
    themeda_instance = Themeda()

    # Get the model
    model = themeda_instance.model(
        # Add any necessary arguments for the model method here
        # For example:
        embedding_size=16,
        hidden_size=64,
        kernel_size=15,
        temporal_processor_type="LSTM",
        # ... and so on
    )

    # Create a dummy input
    # Adjust the dimensions and number of inputs based on your actual model's requirements
    dummy_input = [torch.randn(2, 16, 160, 160) for _ in range(len(themeda_instance.inputs))]

    # Forward pass with dummy input
    output = model(*dummy_input)

    # Visualize
    dot = make_dot(output)
    dot.view()

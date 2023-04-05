from typing import List
import torch
from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as fa
from rich.console import Console
console = Console()
from enum import Enum
import dateutil.parser
from dateutil import rrule
import geojson

from .models import ResNet, TemporalProcessorType, EcoFutureModel

class Interval(Enum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"

    def __str__(self):
        return self.value

    def freq(self):
        return getattr(rrule, self.value)


def get_dates(start:str, end:str, interval:Interval, dayfirst:bool = True):
    start_date = dateutil.parser.parse(start, dayfirst=dayfirst)
    end_date = dateutil.parser.parse(end, dayfirst=dayfirst)

    return list(rrule.rrule(
        freq=interval.freq(),
        dtstart=start_date,
        until=end_date,
    ))


class DictionarySplitter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, objects):
        validation_indexes = mask2idxs(self.dictionary[object] for object in objects)
        return IndexSplitter(validation_indexes)(objects)


class EcoFuture(fa.TorchApp):
    """
    A model to forecast changes to ecosystems in Australia.
    """
    def dataloaders(
        self,
        inputs:List[str] = fa.Param(help="The input products to use from Digital Earth Australia."),
        outputs:List[str] = fa.Param(
            None, 
            help="The output products to use from Digital Earth Australia. " +
            "If empty then it uses the same as the input products",
        ),
        cache:Path=fa.Param("cache", help="The path to a directory with cached product outputs"),
        centres:Path=fa.Param(help="A GeoJSON file specifying the centers of the chips."),
        start:str=fa.Param(help="The start date."),
        end:str=fa.Param(help="The start date."),
        interval:Interval=fa.Param(Interval.MONTHLY, help="The start date."),
        width:int=128,
        height:int=None,
        batch_size:int = fa.Param(default=32, help="The batch size."),
        split:int=fa.Param(None, help="The cross-validation split to use.")
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which EcoFuture uses in training and prediction.
        Returns:
            DataLoaders: The DataLoaders object.
        """
        self.inputs = inputs
        self.outputs = outputs or inputs
        self.height = height or width

        dates = get_dates(start=start, end=end, interval=interval)

        # Read chips
        chips = []
        validation = dict()
        with open(centres, 'r') as f:
            chip_data = geojson.load(f)
            assert isinstance(chip_data, geojson.FeatureCollection)

            for feature in chip_data["features"]:
                assert isinstance(feature.geometry, geojson.Point)
                chip = Chip(
                    x=feature.geometry[0],
                    y=feature.geometry[1],
                    width=width,
                    height=height,
                    cache=cache,
                )
                chips.append(chip)
                if split is not None:
                    validation[chip] = feature.properties.get("split", None) == split
                else:
                    validation[chip] = feature.properties.get("validation", False)

        datablock = DataBlock(
            blocks=(
                ClipBlock(
                    products=inputs,
                    dates=dates[:-1]
                ),
                ClipBlock(
                    products=outputs,
                    dates=dates[1:]
                ),
            ),
            splitter=DictionarySplitter(validation),
        )

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=chips,
            bs=batch_size,
        )

        return dataloaders

    def model(
        self,
        encoder_resent:ResNet=ResNet.resnet18,
        temporal_processor_type:TemporalProcessorType=TemporalProcessorType.LSTM,
    ) -> nn.Module:
        """
        Creates a deep learning model for the EcoFuture to use.

        Returns:
            nn.Module: The created model.
        """
        return EcoFutureModel(
            in_channels=len(self.inputs), # Assumes only one channel per product
            out_channels=len(self.outputs),
            encoder_resent=encoder_resent,
            temporal_processor_type=temporal_processor_type,
        )

    def loss_func(self, l1:bool=fa.Param(default=False, help="Whether to use the L1 loss (Mean Absolute Loss). Otherwise the Mean Squared Error is used."),):
        """
        Returns the loss function to use with the model.
        By default the Mean Squared Error (MSE) is used.
        """
        if l1:
            return nn.L1Loss()              
        return nn.MSELoss()
        
    def output_results(
        self,
        results,
    ):
        return results
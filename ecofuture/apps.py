# -*- coding: future_typing -*-

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
from dateutil import rrule
from fastai.data.block import TransformBlock

from polytorch import CategoricalData, ContinuousData, BinaryData, PolyLoss
from polytorch.metrics import categorical_accuracy, smooth_l1, binary_accuracy, binary_dice, binary_iou
from polytorch.enums import BinaryDataLossType

from .dataloaders import TPlus1Callback, get_chiplets_list, PredictPersistanceCallback
from .models import ResNet, TemporalProcessorType, EcoFutureModelUNet, EcoFutureModel, EcoFutureModel1x1Conv
from .transforms import ChipletBlock
from .metrics import smooth_l1_rain, smooth_l1_tmax

class Interval(Enum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"

    def __str__(self):
        return self.value

    def freq(self):
        return getattr(rrule, self.value)


def get_dates(start:str, end:str, interval:Interval|str, dayfirst:bool = True):
    if isinstance(interval, str):
        interval = Interval[interval.upper()]

    start_date = dateutil.parser.parse(start, dayfirst=dayfirst)
    end_date = dateutil.parser.parse(end, dayfirst=dayfirst)

    return list(rrule.rrule(
        freq=interval.freq(),
        dtstart=start_date,
        until=end_date,
    ))


class AttributeSplitter:
    def __call__(self, objects):
        validation_indexes = mask2idxs(object.validation for object in objects)
        return IndexSplitter(validation_indexes)(objects)


class DictionarySplitter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, objects):
        validation_indexes = mask2idxs(self.dictionary[object] for object in objects)
        return IndexSplitter(validation_indexes)(objects)


class SetSplitter:
    def __init__(self, validation_set):
        self.validation_set = validation_set

    def __call__(self, objects):
        validation_indexes = mask2idxs(object in self.validation_set for object in objects)
        return IndexSplitter(validation_indexes)(objects)



class SubsetSplitter:
    def __init__(self, validation_set):
        self.validation_set = validation_set

        assert 1 <= validation_set <= 5

    def __call__(self, objects):
        validation_indexes = mask2idxs(object.subset == self.validation_set for object in objects)
        return IndexSplitter(validation_indexes)(objects)



class EcoFuture(ta.TorchApp):
    """
    A model to forecast changes to ecosystems in Australia.
    """
    def dataloaders(
        self,
        level4:Path=ta.Param("", help="The path to a directory with cached level 4 chiplets"),
        rain:Path=ta.Param("", help="The path to a directory with cached rain chiplets"),
        tmax:Path=ta.Param("", help="The path to a directory with cached tmax chiplets"),
        start:str=ta.Param("1988-01-01", help="The start date."),
        end:str=ta.Param("2018-01-01", help="The end date."),
        interval:Interval=ta.Param(Interval.YEARLY.value, help="The time interval to use."),
        max_chiplets:int=None,
        max_years:int=None,
        batch_size:int = ta.Param(default=1, help="The batch size."),
        validation_subset:int=1,
        predict_persistance:bool=False,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which EcoFuture uses in training and prediction.
        Returns:
            DataLoaders: The DataLoaders object.
        """
        dates = get_dates(start=start, end=end, interval=interval)        
        self.input_types = []
        splitter = SubsetSplitter(validation_subset)
        self.level4 = level4
        self.rain = rain
        self.tmax = tmax
        self.in_channels_continuous = bool(rain) + bool(tmax)

        blocks = []
        getters = []
        for directory in (level4, rain, tmax):
            if not directory:
                continue
            directory = Path(directory)
            if not directory.exists:
                raise FileNotFoundError(f"Cannot find directory {directory}")

            getters.append(ChipletBlock(base_dir=directory, dates=dates, max_years=max_years))
            blocks.append(TransformBlock)

            # hack
            if directory.name == "level4":
                self.input_types.append(CategoricalData(21))
            elif directory.name in ["rain", "tmax"]:
                self.input_types.append(ContinuousData())

        # hack
        self.output_types = self.input_types
        
        # hack
        self.predict_persistance = predict_persistance
        if predict_persistance and isinstance(self.output_types[0], CategoricalData):
            self.output_types = [BinaryData(loss_type=BinaryDataLossType.DICE)] + self.output_types[1:]

        assert len(getters) > 0, "At least one of level4, rain or tmax must be given a valid directory."

        chiplets = get_chiplets_list(getters[0].base_dir, max_chiplets)

        datablock = DataBlock(
            blocks=blocks,
            getters=getters,
            splitter=splitter,
            n_inp=len(blocks),
        )

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=chiplets,
            bs=batch_size,
        )

        return dataloaders

    def model(
        self,
        embedding_size:int=ta.Param(16, help="The number of embedding dimensions."),
        encoder_resent:ResNet=ResNet.resnet18.value,
        temporal_processor_type:TemporalProcessorType=ta.Param(TemporalProcessorType.GRU.value, case_sensitive=False),
        fastai_unet:bool=False,
        onebyone:bool=False,
        dropout:float=0.0,    
    ) -> nn.Module:
        """
        Creates a deep learning model for the EcoFuture to use.

        Returns:
            nn.Module: The created model.
        """
        if onebyone:
            ModelClass = EcoFutureModel1x1Conv
        else:
            ModelClass = EcoFutureModelUNet if fastai_unet else EcoFutureModel

        return ModelClass(
            input_types=self.input_types,
            output_types=self.output_types,
            embedding_size=embedding_size,
            encoder_resent=encoder_resent,
            temporal_processor_type=temporal_processor_type,
            dropout=dropout,
        )
    
    def extra_callbacks(self):
        callbacks = [TPlus1Callback()]
        self.predict_persistance = True # hack
        if self.predict_persistance:
            callbacks.append(PredictPersistanceCallback())
        return callbacks

    def loss_func(
        self, 
        # l1:bool=ta.Param(
        #     default=False, 
        #     help="Whether to use the L1 loss (Mean Absolute Loss) for continuous variables. "
        #         "Otherwise the Mean Squared Error (L2 loss) is used.",
        # ),
        # label_smoothing:float = ta.Param(
        #     default=0.0, 
        #     min=0.0,
        #     max=1.0,
        #     help="The amount of label smoothing to use.",
        # ), 
    ):
        return PolyLoss(data_types=self.output_types, feature_axis=2)
        
    def inference_dataloader(self, learner, base_dir:Path=None, max_chiplets:int=None, num_workers:int=None, **kwargs):
        self.inference_chiplets = get_chiplets_list(base_dir, max_chiplets)
        from .transforms import Chiplet # hack
        self.inference_chiplets = [Chiplet(subset=2, id="00010460.npz")]# hack
        dataloader = learner.dls.test_dl(self.inference_chiplets, num_workers=num_workers, **kwargs)
        return dataloader
    
    def output_results(
        self,
        results,
    ):
        for chiplet, item in zip(self.inference_chiplets, results[0][0]):
            
            for timestep, values in enumerate(item):
                predictions = torch.argmax(values, dim=0)
                # Save in same format as chiplet input?
                filename = f"level4.{chiplet.subset}.{chiplet.id}.{timestep}.pkl"
                print(f"saving to {filename}")
                torch.save(values, filename)

        return results

    def metrics(self):
        metrics = []
        if self.level4:
            metrics += [
                # partial(categorical_accuracy, data_index=0, feature_axis=2),
                partial(binary_accuracy, data_index=0, feature_axis=2), # hack
                partial(binary_dice, data_index=0, feature_axis=2), # hack
                partial(binary_iou, data_index=0, feature_axis=2), # hack
            ]
            
        if self.rain:
            metrics.append(smooth_l1_rain)

        if self.tmax:
            metrics.append(smooth_l1_tmax)
        
        return metrics

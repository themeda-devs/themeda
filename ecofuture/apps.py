# -*- coding: future_typing -*-

import random
import re
from pathlib import Path
from torch import nn
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

from .dataloaders import TPlus1Callback
from .models import ResNet, TemporalProcessorType, EcoFutureModel
from .transforms import CroppedChipBlock, CroppedChip, ChipletBlock, Chiplet
from .loss import MultiDatatypeLoss
from .metrics import accuracy

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
        chiplet_dir:Path=ta.Param("", help="The path to a directory with cached level 4 chiplets"),
        start:str=ta.Param("1988-01-01", help="The start date."),
        end:str=ta.Param("2018-01-01", help="The end date."),
        interval:Interval=ta.Param(Interval.YEARLY.value, help="The time interval to use."),
        max_chiplets:int=None,
        max_years:int=None,
        width:int=160,
        height:int=None,
        batch_size:int = ta.Param(default=1, help="The batch size."),
        validation_subset:int=1,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which EcoFuture uses in training and prediction.
        Returns:
            DataLoaders: The DataLoaders object.
        """
        chiplets = set()
        for path in Path(chiplet_dir).glob("*.npz"):
            # path is something like: ecofuture_chiplet_level4_1988_subset_1_00004207.npz
            chiplet_components = path.name.split("_")
            subset = int(chiplet_components[5])
            chiplet_id = chiplet_components[6]
            chiplets.add( Chiplet(subset=subset, id=chiplet_id) )
        chiplets = list(chiplets)

        if max_chiplets and len(chiplets) > max_chiplets:
            random.seed(42)
            chiplets = random.sample(chiplets, max_chiplets)

        dates = get_dates(start=start, end=end, interval=interval)        
        splitter = SubsetSplitter(validation_subset)

        datablock = DataBlock(
            blocks=(ChipletBlock(base_dir=chiplet_dir, dates=dates, max_years=max_years),),
            splitter=splitter,
        )

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=chiplets,
            bs=batch_size,
            # dl_type=TPlus1Dataloader,
            # dl_kwargs=[dict(after_batch=t_plus_one),dict(after_batch=t_plus_one)],
        )

        return dataloaders

    def model(
        self,
        encoder_resent:ResNet=ResNet.resnet18.value,
        temporal_processor_type:TemporalProcessorType=ta.Param(TemporalProcessorType.GRU.value, case_sensitive=False),
    ) -> nn.Module:
        """
        Creates a deep learning model for the EcoFuture to use.

        Returns:
            nn.Module: The created model.
        """
        categorical_counts = 22 # hack - the extra one is padding
        return EcoFutureModel(
            categorical_counts=[categorical_counts], # hack
            out_channels=categorical_counts, # hack
            encoder_resent=encoder_resent,
            temporal_processor_type=temporal_processor_type,
        )
    
    def extra_callbacks(self):
        return [TPlus1Callback()]

    def loss_func(
        self, 
        l1:bool=ta.Param(
            default=False, 
            help="Whether to use the L1 loss (Mean Absolute Loss) for continuous variables. "
                "Otherwise the Mean Squared Error (L2 loss) is used.",
        ),
        label_smoothing:float = ta.Param(
            default=0.0, 
            min=0.0,
            max=1.0,
            help="The amount of label smoothing to use.",
        ), 
    ):
        """
        Returns the loss function to use with the model.
        By default the Mean Squared Error (MSE) is used.
        """
        return MultiDatatypeLoss(l1=l1, label_smoothing=label_smoothing, ignore_index=21) # hack
        
    def output_results(
        self,
        results,
    ):
        return results

    def metrics(self):
        return [accuracy]

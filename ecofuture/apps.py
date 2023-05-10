import random
import re
from typing import List
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

from .dataloaders import TPlus1Dataloader
from .models import ResNet, TemporalProcessorType, EcoFutureModel
from .transforms import CroppedChipBlock, CroppedChip
from .loss import MultiDatatypeLoss

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


class EcoFuture(ta.TorchApp):
    """
    A model to forecast changes to ecosystems in Australia.
    """
    def dataloaders(
        self,
        # inputs:List[str] = ta.Param(help="The input products to use from Digital Earth Australia."),
        # outputs:List[str] = ta.Param(
        #     None, 
        #     help="The output products to use from Digital Earth Australia. " +
        #     "If empty then it uses the same as the input products",
        # ),
        # cache:Path=ta.Param("cache", help="The path to a directory with cached product outputs"),
        level4:Path=ta.Param("", help="The path to a directory with cached level 4 chips"),
        # centres:Path=ta.Param(help="A GeoJSON file specifying the centers of the chips."),
        start:str=ta.Param("1988-01-01", help="The start date."),
        end:str=ta.Param("2020-01-01", help="The start date."),
        interval:Interval=ta.Param(Interval.YEARLY.value, help="The start date."),
        width:int=200,
        height:int=None,
        batch_size:int = ta.Param(default=32, help="The batch size."),
        split:int=ta.Param(None, help="The cross-validation split to use."),
        validation_proportion:float=0.2,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which EcoFuture uses in training and prediction.
        Returns:
            DataLoaders: The DataLoaders object.
        """
        # self.inputs = inputs
        # self.outputs = outputs or inputs
        self.height = height or width
        self.width = width

        dates = get_dates(start=start, end=end, interval=interval)

        level4 = Path(level4)
        tiffs = level4.glob("*.tif")
        chips = set()
        for tiff in tiffs:
            m = re.match(r"(.*)_(\d{4})-(\d{2})-(\d{2})_level4\.tif", tiff.name)
            chip = m.group(1)
            chips.add(chip)

        validation = set()
        random.seed(42)
        cropped_chips = []
        for chip in chips:
            validation = random.random() < validation_proportion
            for x in range(0, 4_000, self.width):
                for y in range(0, 4_000, self.height):
                    cropped_chip = CroppedChip(
                        chip=chip,
                        validation=validation,
                        x=x,
                        y=y,
                        height=self.height,
                        width=self.width,
                    )
                    cropped_chips.append(cropped_chip)
        
        splitter = AttributeSplitter()

        datablock = DataBlock(
            blocks=(CroppedChipBlock(base_dir=level4, dates=dates),),
            splitter=splitter,
        )

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=cropped_chips,
            bs=batch_size,
            dl_type=TPlus1Dataloader,
        )

        return dataloaders

    def model(
        self,
        encoder_resent:ResNet=ResNet.resnet18.value,
        temporal_processor_type:TemporalProcessorType=TemporalProcessorType.LSTM.value,
    ) -> nn.Module:
        """
        Creates a deep learning model for the EcoFuture to use.

        Returns:
            nn.Module: The created model.
        """
        return EcoFutureModel(
            categorical_counts=[105], # hack
            out_channels=105,
            encoder_resent=encoder_resent,
            temporal_processor_type=temporal_processor_type,
        )

    def loss_func(
        self, 
        l1:bool=ta.Param(
            default=False, 
            help="Whether to use the L1 loss (Mean Absolute Loss) for continuous variables. "
                "Otherwise the Mean Squared Error (L2 loss) is used.",
        ),
    ):
        """
        Returns the loss function to use with the model.
        By default the Mean Squared Error (MSE) is used.
        """
        return MultiDatatypeLoss(l1=l1)
        
    def output_results(
        self,
        results,
    ):
        return results
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
from fastai.learner import Learner, load_learner

from dateutil import rrule
from rich.table import Table
from rich.box import SIMPLE
from torchapp.util import call_func

from fastai.data.block import TransformBlock

from polytorch import CategoricalData, ContinuousData, BinaryData, PolyLoss, CategoricalLossType, BinaryLossType
from polytorch.metrics import categorical_accuracy, smooth_l1, binary_accuracy, binary_dice, binary_iou, generalized_dice

from .dataloaders import TPlus1Callback, get_chiplets_list, PredictPersistanceCallback
from .models import ResNet, TemporalProcessorType, EcoFutureModelUNet, EcoFutureModel, EcoFutureModelSimpleConv, PersistenceModel
from .transforms import ChipletBlock, Normalize
from .metrics import smooth_l1_rain, smooth_l1_tmax, kl_divergence_proportions
from .colours import LEVEL4_COLOURS
from .plots import wandb_process

MEAN = {'rain': 1193.8077, 'tmax':32.6068}
STD = {'rain': 394.8365, 'tmax':1.4878}


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

            # hack
            if directory.name == "level4":
                self.input_types.append(
                      CategoricalData(21, loss_type=CategoricalLossType.CROSS_ENTROPY, labels=LEVEL4_COLOURS.keys(), colors=LEVEL4_COLOURS.values())
                )
                blocks.append(TransformBlock)

            elif directory.name in ["rain", "tmax"]:
                self.input_types.append(ContinuousData())
                mean = MEAN[directory.name]
                std = STD[directory.name]
                blocks.append(TransformBlock(type_tfms=Normalize(mean, std) ))

        # hack
        self.output_types = self.input_types
        
        # hack
        self.predict_persistance = predict_persistance
        if predict_persistance and isinstance(self.output_types[0], CategoricalData):
            self.output_types = [BinaryData(loss_type=BinaryLossType.DICE)] + self.output_types[1:]

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
        persistence:bool = ta.Param(False, help='Whether or not to use a basic persistence model.'),
        embedding_size:int=ta.Param(16, help="The number of embedding dimensions."),
        encoder_resent:ResNet=ResNet.resnet18.value,
        temporal_processor_type:TemporalProcessorType=ta.Param(TemporalProcessorType.GRU.value, case_sensitive=False),
        fastai_unet:bool=False,
        simple:bool=False,
        kernel_size:int=1,
        dropout:float=0.0,   
        hidden_size:int=0,     # only for simple conv 
    ) -> nn.Module:
        """
        Creates a deep learning model for the EcoFuture to use.

        Returns:
            nn.Module: The created model.
        """
        if persistence:
            return PersistenceModel(self.input_types)
        
        if simple:
            return EcoFutureModelSimpleConv(
                kernel_size=kernel_size,
                input_types=self.input_types,
                output_types=self.output_types,
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                temporal_processor_type=temporal_processor_type,
            )
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
    
    def extra_callbacks(
        self, 
        predict_persistance:bool=False, # hack
    ):
        callbacks = [TPlus1Callback()]
        self.predict_persistance = predict_persistance 
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
            if self.predict_persistance:
                metrics += [
                    partial(binary_accuracy, data_index=0, feature_axis=2),
                    partial(binary_dice, data_index=0, feature_axis=2),
                    partial(binary_iou, data_index=0, feature_axis=2),
                ]
            else:
                metrics += [
                    partial(categorical_accuracy, data_index=0, feature_axis=2),
                    partial(kl_divergence_proportions, data_index=0, feature_axis=2),
                    partial(generalized_dice, data_index=0, feature_axis=2),
                ]
            
        if self.rain:
            metrics.append(smooth_l1_rain)

        if self.tmax:
            metrics.append(smooth_l1_tmax)
        
        return metrics

    def validate(
        self,
        gpu: bool = ta.Param(True, help="Whether or not to use a GPU for processing if available."),
        persistence:bool = False,
        **kwargs,
    ):

        # Check if CUDA is available
        gpu = gpu and torch.cuda.is_available()

        # Create a dataloader for inference
        dataloaders = call_func(self.dataloaders, **kwargs)

        if persistence:
            learner = call_func(self.learner, **kwargs)
            learner.model = PersistenceModel(self.input_types)
        else:
            path = call_func(self.pretrained_local_path, **kwargs)

            try:
                learner = load_learner(path, cpu=not gpu)
            except Exception:
                import dill
                learner = load_learner(path, cpu=not gpu, pickle_module=dill)


        table = Table(title="Validation", box=SIMPLE)

        values = learner.validate(dl=dataloaders.valid, cbs=[TPlus1Callback()])
        names = [learner.recorder.loss.name] + [metric.name for metric in learner.metrics]
        result = {name: value for name, value in zip(names, values)}

        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for name, value in result.items():
            table.add_row(name, str(value))

        console.print(table)

        return result

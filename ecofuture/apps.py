# -*- coding: future_typing -*-

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
from polytorch.metrics import categorical_accuracy, smooth_l1, binary_accuracy, binary_dice, binary_iou, generalized_dice

from ecofuture_preproc.source import DataSourceName
from ecofuture_preproc.roi import ROIName
from ecofuture_preproc.chiplet_table import load_table

from .dataloaders import TPlus1Callback, get_chiplets_list, PredictPersistanceCallback, FutureDataLoader
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


def get_block(name:DataSourceName|str) -> TransformBlock:
    name = str(name)
    if name in ["rain", "tmax"]:
        return TransformBlock(type_tfms=Normalize(MEAN[name], STD[name]) )
    if name == "land_cover":
        return TransformBlock()

    raise NotImplementedError


def get_datatype(name:DataSourceName|str) -> PolyData:
    name = str(name)
    if name == "land_cover":
        labels = list(LEVEL4_COLOURS.keys())
        colours = list(LEVEL4_COLOURS.values())
        return CategoricalData(21, loss_type=CategoricalLossType.CROSS_ENTROPY, labels=labels, colors=colours)
    if name in ["rain", "tmax"]:
        return ContinuousData()

    raise NotImplementedError


class EcoFuture(ta.TorchApp):
    """
    A model to forecast changes to ecosystems in Australia.
    """
    def dataloaders(
        self,
        input:List[DataSourceName]=ta.Param(..., help="The input data types."),
        output:List[DataSourceName]=ta.Param(None, help="The output data types. If not given, then the outputs are the same as the inputs."),
        roi:ROIName=ta.Param("savanna", help="The Region of Interest."),
        base_dir:Path=ta.Param(..., help="The base directory for the preprocessed data.", envvar="ECOFUTURE_PREPROC_BASE_OUTPUT_DIR"),
        start_year:int=ta.Param(1988, help="The start date."),
        end_year:int=ta.Param(2018, help="The end year (inclusive)."),
        max_chiplets:int=None,
        max_years:int=None,
        batch_size:int = ta.Param(default=1, help="The batch size."),
        validation_subset:int=1,
        # predict_persistance:bool=False,
        pad_size:int = 0,
        base_size:int = 160,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which EcoFuture uses in training and prediction.
        Returns:
            DataLoaders: The DataLoaders object.
        """        
        self.inputs = [i if isinstance(i, DataSourceName) else DataSourceName[str(i).upper()] for i in input]
        self.outputs = [i if isinstance(i, DataSourceName) else DataSourceName[str(i).upper()] for i in output]

        assert len(self.inputs) > 0, "You must include at least one input."
        if len(self.outputs) == 0:
            self.outputs = self.inputs

        all_types = self.inputs + self.outputs

        self.input_types = [get_datatype(name) for name in self.inputs]
        self.output_types = [get_datatype(name) for name in self.outputs]
        # self.output_types = self.input_types # hack

        base_dir = Path(base_dir)
        assert base_dir.exists(), f"Base Dir {base_dir} does not exist"

        blocks = [get_block(name) for name in all_types]
        years = list(range(start_year, end_year + 1))
        getters = [
            ChipletBlock(
                name=name,
                years=years,
                roi=roi,
                base_size=base_size,
                pad_size=pad_size,
                base_dir=base_dir,
            )
            for name in all_types
        ] 

        table = load_table(
            roi_name=roi,
            base_output_dir=base_dir,
            pad_size_pix=pad_size,
        )

        indexes = table['index']
        splitter = IndexSplitter(table['subset_num'] == validation_subset)

        datablock = DataBlock(
            blocks=blocks,
            getters=getters,
            splitter=splitter,
            n_inp=len(self.input_types),
        )

        dataloaders = DataLoaders.from_dblock(
            datablock,
            source=indexes,
            bs=batch_size,
            dl_type=FutureDataLoader,
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
        num_conv_layers:int=1, #add multiple conv layers
        padding_mode:str="zeros",
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
                num_conv_layers=num_conv_layers,
                padding_mode=padding_mode,
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

    def loss_func(self):
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
        feature_axis = 2

        for data_index, output in enumerate(self.outputs):
            output = str(output)
            if output == "land_cover":
                metrics += [
                    partial(categorical_accuracy, data_index=data_index, feature_axis=feature_axis),
                    partial(kl_divergence_proportions, data_index=data_index, feature_axis=feature_axis),
                    partial(generalized_dice, data_index=data_index, feature_axis=feature_axis),
                ]

            elif output in ["rain", "tmax"]:
                metrics.append(PolyMetric(name=f"smooth_l1_{output}", feature_axis=feature_axis, data_index=data_index, function=F.smooth_l1_loss))
            else:
                raise ValueError(f"No metrics for output {output}")
        
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

        callbacks = [TPlus1Callback()]
        values = learner.validate(dl=dataloaders.valid, cbs=callbacks)
        names = [learner.recorder.loss.name] + [metric.name for metric in learner.metrics]
        result = {name: value for name, value in zip(names, values)}

        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for name, value in result.items():
            table.add_row(name, str(value))

        console.print(table)

        return result

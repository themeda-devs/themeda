# -*- coding: future_typing -*-

from typing import List
import torch
from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as ta
from fastcore.foundation import mask2idxs
from fastai.data.block import DataBlock
from fastai.data.transforms import IndexSplitter 
from rich.console import Console
import numpy as np
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
from polytorch.metrics import PolyMetric, CategoricalAccuracy

from themeda_preproc.source import DataSourceName, is_data_source_continuous
from themeda_preproc.roi import ROIName
from themeda_preproc.chiplet_table import load_table
from themeda_preproc.summary_stats import load_stats
import torch.nn.functional as F

from .enums import TemporalProcessorType
from .dataloaders import TPlus1Callback, FutureDataLoader
from .models import ThemedaModel, PersistenceModel, ProportionsLSTMModel
from .transforms import ChipletBlock, StaticChipletBlock, Normalize, make_binary
from .metrics import KLDivergenceProportions, HierarchicalKLDivergence, HierarchicalCategoricalAccuracy
from .plots import wandb_process
from .loss import ProportionLoss
from .land_cover import LandCoverData
from .callbacks import WriteResults


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


def get_block(name:DataSourceName|str, roi:ROIName, base_dir:Path) -> TransformBlock:
    if isinstance(name, str):
        name = DataSourceName[name.lower()]

    type_tfms = []
    if is_data_source_continuous(name):
        stats = load_stats(
            source_name=name,
            roi_name=roi,
            base_output_dir=base_dir,
        )
        type_tfms.append(Normalize(stats.mean, stats.sd))
    elif name in [DataSourceName.FIRE_SCAR_EARLY, DataSourceName.FIRE_SCAR_LATE]:
        type_tfms.append(make_binary)

    return TransformBlock(type_tfms=type_tfms)


def get_datatype(name:DataSourceName|str, emd_loss:bool, hierarchical_embedding:bool) -> PolyData:
    name = str(name)
    if name == "land_cover":
        return LandCoverData(emd_loss=emd_loss, hierarchical_embedding=hierarchical_embedding)
    elif name == "land_use":
        from themeda_preproc.land_use.labels import get_cmap

        colourmap = get_cmap()
        labels = [entry.label for entry in colourmap]
        colours = [entry.hex for entry in colourmap]
        return CategoricalData(len(labels), name=name, loss_type=CategoricalLossType.CROSS_ENTROPY, labels=labels, colors=colours)
    elif is_data_source_continuous(DataSourceName(name)):
        return ContinuousData(name=name)
    elif "fire" in name:
        return BinaryData(name=name)

    raise NotImplementedError


def get_chiplet_block( 
    name,
    years,
    roi,
    base_size,
    pad_size,
    base_dir,
):
    kwargs = dict(
        name=name,
        roi=roi,
        base_size=base_size,
        pad_size=pad_size,
        base_dir=base_dir,
    )

    if name == DataSourceName.SOIL_CLAY:
        return StaticChipletBlock(dataset_year=2021, n_years=len(years), **kwargs)
    elif name == DataSourceName.SOIL_ECE:
        return StaticChipletBlock(dataset_year=2014, n_years=len(years), **kwargs)
    elif name == DataSourceName.SOIL_DEPTH:
        return StaticChipletBlock(dataset_year=2019, n_years=len(years), **kwargs)
    elif name == DataSourceName.ELEVATION:
        return StaticChipletBlock(dataset_year=2011, n_years=len(years), **kwargs)

    return ChipletBlock(years=years, **kwargs)


class Themeda(ta.TorchApp):
    """
    A model to forecast changes to ecosystems in Australia.
    """
    def dataloaders(
        self,
        input:List[DataSourceName]=ta.Param(..., help="The input data types."),
        output:List[DataSourceName]=ta.Param(None, help="The output data types. If not given, then the outputs are the same as the inputs."),
        roi:ROIName=ta.Param("savanna", help="The Region of Interest."),
        base_dir:Path=ta.Param(..., help="The base directory for the preprocessed data.", envvar="THEMEDA_PREPROC_BASE_OUTPUT_DIR"),
        start_year:int=ta.Param(1988, help="The start year."),
        end_year:int=ta.Param(2018, help="The end year (inclusive)."),
        max_chiplets:int=None,
        max_years:int=None,
        batch_size:int = ta.Param(default=1, help="The batch size."),
        validation_subset:int=1,
        # predict_persistance:bool=False,
        pad_size:int = 0,
        base_size:int = 160,
        emd_loss:bool=False,
        hierarchical_embedding:bool=False,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Themeda uses in training and prediction.
        Returns:
            DataLoaders: The DataLoaders object.
        """        
        self.inputs = [i if isinstance(i, DataSourceName) else DataSourceName[str(i).upper()] for i in input]
        self.outputs = [i if isinstance(i, DataSourceName) else DataSourceName[str(i).upper()] for i in output]

        if isinstance(roi, str):
            roi = ROIName[roi.upper()]
            
        assert len(self.inputs) > 0, "You must include at least one input."
        if len(self.outputs) == 0:
            self.outputs = self.inputs

        all_types = self.inputs + self.outputs

        self.input_types = [get_datatype(name, emd_loss=emd_loss, hierarchical_embedding=hierarchical_embedding) for name in self.inputs]
        self.output_types = [get_datatype(name, emd_loss=emd_loss, hierarchical_embedding=hierarchical_embedding) for name in self.outputs]
        # self.output_types = self.input_types # hack
        base_dir = Path(base_dir)
        assert base_dir.exists(), f"Base Dir {base_dir} does not exist"

        blocks = [get_block(name, roi, base_dir) for name in all_types]
        years = list(range(start_year, end_year + 1))
        if max_years:
            years = years[-max_years:]
        getters = [
            get_chiplet_block(
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

        if max_chiplets:
            table = table.sample(max_chiplets, seed=42)

        indexes = table['index']
        splitter = IndexSplitter(mask2idxs(table['subset_num'] == validation_subset))

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
        cnn_kernel:int=15,
        cnn_size:int=64,
        cnn_layers:int=1,
        padding_mode:str="reflect",
        temporal_processor_type:TemporalProcessorType=ta.Param(TemporalProcessorType.LSTM.value, case_sensitive=False),
        temporal_layers:int=2,
        temporal_size:int=32,
        transformer_heads:int=8,
        transformer_layers:int=4,
        transformer_positional_encoding:bool=True,
    ) -> nn.Module:
        """
        Creates a deep learning model for the Themeda to use.

        Returns:
            nn.Module: The created model.
        """
        if persistence:
            return PersistenceModel(self.input_types)
        
        return ThemedaModel(
            input_types=self.input_types,
            output_types=self.output_types,
            embedding_size=embedding_size,
            cnn_kernel=cnn_kernel,
            cnn_size=cnn_size,
            cnn_layers=cnn_layers,
            padding_mode=padding_mode,
            temporal_processor_type=temporal_processor_type,
            temporal_layers=temporal_layers,
            temporal_size=temporal_size,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_positional_encoding=transformer_positional_encoding,
        )
    
    def extra_callbacks(self, **kwargs):
        callbacks = [TPlus1Callback()]
        return callbacks

    def loss_func(self):
        return PolyLoss(data_types=self.output_types, feature_axis=2)
        
    def inference_callbacks(self, probabilities:Path=None, argmax:Path=None):
        assert (probabilities or argmax), f"Please give a path to output the results as probabilities or argmax."
        return [WriteResults(probabilities=probabilities, argmax=argmax)]        

    def inference_dataloader(
        self, 
        learner, 
        roi:ROIName=ta.Param("savanna", help="The Region of Interest."),
        base_dir:Path=ta.Param(..., help="The base directory for the preprocessed data.", envvar="THEMEDA_PREPROC_BASE_OUTPUT_DIR"),
        start_year:int=ta.Param(1988, help="The start year."),
        end_year:int=ta.Param(2018, help="The end year (inclusive)."),
        max_chiplets:int=None,
        max_years:int=None,
        batch_size:int = ta.Param(default=1, help="The batch size."),
        pad_size:int = 0,
        base_size:int = 160,
        # subset:int=ta.Param(None),
    ):
        input = [data_type.name for data_type in learner.model.embedding.input_types]
        output = [data_type.name for data_type in learner.model.output_types]
        dataloaders = self.dataloaders(
            input=input,
            output=output,
            roi=roi,
            base_dir=base_dir,
            start_year=start_year,
            end_year=end_year,
            max_chiplets=max_chiplets,
            max_years=max_years,
            batch_size=batch_size,
            pad_size=pad_size,
            base_size=base_size,
        )
        total_count = dataloaders.train.n+dataloaders.valid.n
        items = np.arange(total_count)
        dataloader = dataloaders.test_dl(items)
        # dataloader = dataloaders.valid.new(dataloaders.items) # Give this data load all the items, regardless of partition
        dataloader.pad_size = pad_size
        dataloader.base_size = base_size

        learner.dl = dataloader

        return dataloader

    def __call__(
        self, 
        gpu: bool = ta.Param(True, help="Whether or not to use a GPU for processing if available."), 
        **kwargs
    ):
        # overriding the call method to set with_preds=False
        # This should be fixed in torchapp so that kwargs to get_preds is set with a method

        # Check if CUDA is available
        gpu = gpu and torch.cuda.is_available()

        # Open the exported learner from a pickle file
        path = call_func(self.pretrained_local_path, **kwargs)
        learner = self.learner_obj = load_learner(path, cpu=not gpu)

        # Create a dataloader for inference
        dataloader = call_func(self.inference_dataloader, learner, **kwargs)

        inference_callbacks = call_func(self.inference_callbacks, **kwargs)

        results = learner.get_preds(
            dl=dataloader, 
            reorder=False, 
            with_decoded=False, 
            act=self.activation(), 
            cbs=inference_callbacks,
            with_preds=False,
        )

    def metrics(self):
        metrics = []
        feature_axis = 2

        for data_index, output in enumerate(self.outputs):
            output = str(output)
            output_datasource = DataSourceName[output.upper()]

            if output == "land_cover":
                metrics += [
                    CategoricalAccuracy(name=f"{output}_accuracy", data_index=data_index, feature_axis=feature_axis),
                    KLDivergenceProportions(name=f"{output}_kl", data_index=data_index, feature_axis=feature_axis),
                    HierarchicalCategoricalAccuracy(name=f"{output}_level0_accuracy", data_index=data_index, feature_axis=feature_axis),
                    HierarchicalKLDivergence(name=f"{output}_level0_kl", data_index=data_index, feature_axis=feature_axis),
                ]
            elif output == "land_use":
                metrics += [
                    CategoricalAccuracy(name=f"{output}_accuracy", data_index=data_index, feature_axis=feature_axis),
                    KLDivergenceProportions(name=f"{output}_kl", data_index=data_index, feature_axis=feature_axis),
                ]
            elif is_data_source_continuous(output_datasource):
                metrics.append(PolyMetric(name=f"smooth_l1_{output}", feature_axis=feature_axis, data_index=data_index, function=F.smooth_l1_loss))
            elif output_datasource in [DataSourceName.FIRE_SCAR_EARLY, DataSourceName.FIRE_SCAR_LATE]:
                print(f"no metric for {output}")
                continue
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


class ThemedaProportionsApp(Themeda):
    def model(
        self,
        hidden_size:int=256, 
        num_layers:int=4, 
        dropout:float=ta.Param(0.5, help="The amount of dropout to use."),
    ):
        return ProportionsLSTMModel(
            input_types=self.input_types,
            output_types=self.output_types,            
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        
    def loss_func(self):
        return ProportionLoss(output_types=self.output_types)
        
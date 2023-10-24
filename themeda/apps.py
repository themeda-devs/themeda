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

from .dataloaders import TPlus1Callback, get_chiplets_list, PredictPersistanceCallback, FutureDataLoader
from .models import ResNet, TemporalProcessorType, ThemedaModelUNet, ThemedaModel, ThemedaModelSimpleConv, PersistenceModel, ProportionsLSTMModel
from .transforms import ChipletBlock, StaticChipletBlock, Normalize, make_binary
from .metrics import KLDivergenceProportions, HierarchicalKLDivergence, HierarchicalCategoricalAccuracy
from .plots import wandb_process
from .loss import ProportionLoss
from .land_cover import LandCoverData



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


def get_datatype(name:DataSourceName|str, emd_loss:bool, hierarchical_embeddding:bool=False) -> PolyData:
    name = str(name)
    if name == "land_cover":
        return LandCoverData(emd_loss=emd_loss, hierarchical_embeddding=hierarchical_embeddding)
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
        base_dir:Path=ta.Param(..., help="The base directory for the preprocessed data.", envvar="Themeda_PREPROC_BASE_OUTPUT_DIR"),
        start_year:int=ta.Param(1988, help="The start date."),
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
            table = table.sample(max_chiplets)

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
        simple:bool=True,
        kernel_size:int=1,
        dropout:float=0.0,   
        hidden_size:int=0,     # only for simple conv 
        num_conv_layers:int=1, #add multiple conv layers
        padding_mode:str="reflect",
    ) -> nn.Module:
        """
        Creates a deep learning model for the Themeda to use.

        Returns:
            nn.Module: The created model.
        """
        if persistence:
            return PersistenceModel(self.input_types)
        
        if simple:
            return ThemedaModelSimpleConv(
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
            ModelClass = ThemedaModelUNet if fastai_unet else ThemedaModel

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
        
# -*- coding: future_typing -*-

import random
from typing import List
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import rasterio
from torch import Tensor
from fastai.data.transforms import DisplayedTransform
import numpy as np

from themeda_preproc.chiplets import load_chiplets, chiplets_reader

@dataclass
class CroppedChip():
    chip:str
    validation:bool
    x:int
    y:int
    width:int
    height:int

    def get_file(self, base_dir:Path, date:datetime) -> Path:
        date_str = date.strftime('%Y-%m-%d')
        return base_dir/f"{self.chip}_{date_str}_level4.tif"

    def get_tensor_at_date(self, base_dir:Path, date:datetime) -> Tensor:
        file = self.get_file(base_dir, date)
        if not file.exists():
            return None
            
        with rasterio.open(file) as img:
            data = img.read(1)

        # TODO Map values
        
        t = torch.as_tensor(data[self.x:self.x+self.width, self.y:self.y+self.height]).unsqueeze(0)
        return t

    def get_tensor(self, base_dir:Path, dates:List[datetime]):
        tensors = [self.get_tensor_at_date(base_dir=base_dir, date=date) for date in dates]
        return torch.cat( [t for t in tensors if t is not None] )


@dataclass
class Chiplet:
    subset:int
    id:str

    def __hash__(self):
        return hash(f"{self.subset}_{self.id}")


class ChipletBlock():
    def __init__(
        self,    
        name,
        years,
        roi,
        base_size,
        pad_size,
        base_dir,
    ):
        self.name = name
        self.years = years
        self.roi = roi
        self.base_size = base_size
        self.pad_size = pad_size
        self.base_dir = base_dir

        assert len(years) > 1
        
        chiplets_dict = {}
        for year in years:
            try:
                chiplets_for_year = load_chiplets(
                    source_name=name,
                    year=year,
                    roi_name=roi,
                    base_size_pix=base_size,
                    pad_size_pix=pad_size,
                    base_output_dir=base_dir,
                )
                chiplets_dict[year] = chiplets_for_year
            except FileNotFoundError as err:
                pass
        
        # make sure that the number of chiplets found is greater than one
        # If it is one, then it should be using the StaticChipletBlock which is faster to load
        assert len(chiplets_dict) > 1, f"ChipletBlock for {self.name} only found one year's data."

        # if the number of chiplets isn't available per year, then get the closest year
        if len(chiplets_dict) < len(years):
            closest_chiplets = {}
            available_years = list(chiplets_dict.keys())
            for year in years:
                closest_year = min(available_years, key=lambda x: abs(year - x))
                closest_chiplets[year] = chiplets_dict[closest_year]
            chiplets_dict = closest_chiplets

        if len(chiplets_dict) == 0:
            raise ValueError(f"Error reading chiplets for {name}")

        self.chiplets = list(chiplets_dict.values())

    def __call__(self, index:int):  
        pixels = self.base_size + self.pad_size * 2
        # shape of data will be:
        # (years, y, x)
        dtype = torch.int if self.chiplets[0].dtype == np.uint8 else torch.float16
        data = torch.empty( (len(self.chiplets), pixels, pixels), dtype=dtype )
        
        for i, chiplets_for_year in enumerate(self.chiplets):
            data[i] = torch.as_tensor(np.array(chiplets_for_year[index,:,:], copy=False))
        
        return data

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["chiplets"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.chiplets = []


class StaticChipletBlock():
    """ A Chiplet Block for data sources which only are given for a single year and are deemed constant. """
    def __init__(
        self,    
        name,
        dataset_year:int,
        n_years:int,
        roi,
        base_size,
        pad_size,
        base_dir,
    ):
        self.name = name
        self.dataset_year = dataset_year
        self.n_years = n_years
        self.roi = roi
        self.base_size = base_size
        self.pad_size = pad_size
        self.base_dir = base_dir

        self.chiplets_for_year = load_chiplets(
            source_name=name,
            year=dataset_year,
            roi_name=roi,
            base_size_pix=base_size,
            pad_size_pix=pad_size,
            base_output_dir=base_dir,
        )
        
    def __call__(self, index:int):  
        data = torch.as_tensor(np.array(self.chiplets_for_year[index,:,:], copy=False))
        data = data.expand(self.n_years,-1,-1)
        return data

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["chiplets_for_year"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.chiplets_for_year = None


class Normalize(DisplayedTransform):
    order = 99
    
    def __init__(self, mean=None, std=None): 
        self.mean = mean
        self.std = std

    def encodes(self, x): 
        return (x-self.mean) / self.std
    
    def decodes(self, x):
        return x * self.std + self.mean


def make_binary(x):
    return x > 0


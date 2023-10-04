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

from ecofuture_preproc.chiplets import load_chiplets, chiplets_reader

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
        self.kwargs = dict(
            source_name=name,
            roi_name=roi,
            base_size_pix=base_size,
            pad_size_pix=pad_size,
            base_output_dir=base_dir,
        )
        # self.chiplets = [
        #     load_chiplets(
        #         source_name=name,
        #         year=year,
        #         roi_name=roi,
        #         base_size_pix=base_size,
        #         pad_size_pix=pad_size,
        #         base_output_dir=base_dir,
        #     )
        #     for year in years
        # ]
    
    def __call__(self, index):  
        arrays = []

        for year in self.years:
            with chiplets_reader(year=year, **self.kwargs) as chiplets:
                print(year, index)
                chiplet_data = np.array(chiplets[index,:,:])
                # nans = np.isnan(chiplet_data)
                # if nans.any():
                #     chiplet_data = np.nan_to_num(chiplet_data, nan=chiplet_data[~np.isnan(chiplet_data)].mean())

                chiplet_data = np.expand_dims(chiplet_data, axis=0)

                arrays.append(chiplet_data)

        data = torch.tensor(np.concatenate(arrays))
        del arrays

        if isinstance(data, torch.ByteTensor):
            data = data.int()
        
        # if torch.isnan(data).any():
        #     raise ValueError(f"NaN in {self.name}, {self.years}, index={index}")

        return data


class ChipletBlockOLD():
    def __init__(self, base_dir:Path, dates:List[datetime], pad_value:int=0, max_years:int=0, pad:bool=True): # hack
        super().__init__()
        # super().__init__(item_tfms=[self.tuple_to_tensor])
        self.dates = dates
        self.base_dir = Path(base_dir)
        self.pad_value = pad_value
        self.max_years = max_years
        self.time_dims = min(self.max_years, len(self.dates)) if self.max_years else len(self.dates)
        self.pad = pad

    def get_paths(self, item:Chiplet):
        paths = [
            self.base_dir/f"ecofuture_chiplet_{self.base_dir.name}_{date.strftime('%Y')}_subset_{item.subset}_{item.id}" 
            for date in self.dates
        ]

        # filter for paths that exist
        paths = [path for path in paths if path.exists()]
        
        if len(paths) > self.time_dims:
            start = random.randint(0,len(paths)-self.time_dims)
            end = start + self.time_dims
            paths = paths[start:end]

        return paths

    # def tuple_to_tensor(self, item:Chiplet):

    def get_position(self, item:Chiplet):
        paths = self.get_paths(item)
        assert len(paths) > 0
        path = paths[0]
        data = np.load(path, allow_pickle=True)
        return data["position"]
    
    def __call__(self, item:Chiplet):   
        self.pad = False # hack
        arrays = []
        for path in self.get_paths(item):
            data = np.load(path, allow_pickle=True)
            arrays.append(torch.as_tensor(data["data"]).unsqueeze(0))

        assert len(arrays) != 0, f"Number of timesteps is zero for chiplet {item} \nPaths:\n{self.get_paths(item)}"

        if self.pad and len(arrays) < self.time_dims:
            arrays.extend( [torch.full_like(arrays[0], self.pad_value)]* (self.time_dims-len(arrays)))
            assert len(arrays) == self.time_dims

        data = torch.cat(arrays)
        if isinstance(data, torch.ByteTensor):
            data = data.int()
        return data


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


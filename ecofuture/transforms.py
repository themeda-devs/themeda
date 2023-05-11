# -*- coding: future_typing -*-

from typing import List
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from fastai.data.block import TransformBlock
import rasterio
from torch import Tensor


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


# @dataclass
# class Chip():
#     x:float
#     y:float
#     width:int
#     height:int
#     cache:Path

#     def get_file(self, product:str, time:datetime) -> Path:
#         date_str = time.strftime('%Y-%m-%d')
#         return self.cache/f"{product}-{self.x}-{self.y}-{self.width}-{self.height}-{date_str}.pt"

#     def get_data(self, product:str, time:datetime) -> Path:
#         return torch.load(self.get_file(product, time))


class CroppedChipBlock(TransformBlock):
    def __init__(self, base_dir:Path, dates:List[datetime]):
        super().__init__(item_tfms=[self.chip_to_tensor])
        self.dates = dates
        self.base_dir = base_dir

    def chip_to_tensor(self, c: CroppedChip):
        return c.get_tensor(base_dir=self.base_dir, dates=self.dates)

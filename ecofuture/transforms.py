# -*- coding: future_typing -*-

from typing import List
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from fastai.data.block import TransformBlock
import rasterio
from torch import Tensor
from typing import Dict


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
    def __init__(self, base_dir:Path, dates:List[datetime], map:Dict=None):
        super().__init__(item_tfms=[self.chip_to_tensor])
        self.dates = dates
        self.base_dir = base_dir
        self.map = { # hack
            0: 0,
            14: 1,
            15: 2,
            16: 3,
            17: 4,
            18: 4,
            27: 5,
            28: 6,
            29: 7,
            30: 8,
            31: 8,
            32: 9,
            33: 10,
            34: 11,
            35: 12,
            36: 12,
            63: 13,
            64: 13,
            65: 13,
            66: 14,
            67: 14,
            68: 14,
            69: 15,
            70: 15,
            71: 15,
            72: 15,
            73: 15,
            74: 15,
            75: 15,
            76: 15,
            77: 15,
            78: 16,
            79: 16,
            80: 16,
            81: 16,
            82: 16,
            83: 16,
            84: 16,
            85: 16,
            86: 16,
            87: 16,
            88: 16,
            89: 16,
            90: 16,
            91: 16,
            92: 16,
            93: 17,
            94: 18,
            95: 12,
            96: 12,
            97: 18,
            98: 19,
            99: 19,
            100: 19,
            101: 19,
            102: 19,
            103: 20,
            104: 20
        }

    def chip_to_tensor(self, c: CroppedChip):
        t = c.get_tensor(base_dir=self.base_dir, dates=self.dates)
        
        if self.map:
            # adapted from https://stackoverflow.com/a/16993364
            unique,inverse = torch.unique(t,return_inverse = True)
            mapped = [self.map[x.item()] for x in unique]
            t = torch.tensor(mapped)[inverse].reshape(t.shape)

        return t.long()

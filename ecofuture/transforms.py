import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from fastcore.transform import Transform
from fastai.data.block import TransformBlock


@dataclass
class Chip():
    x:float
    y:float
    width:int
    height:int
    cache:Path

    def get_file(self, product:str, time:datetime) -> Path:
        date_str = time.strftime('%Y-%m-%d')
        return self.cache/f"{product}-{self.x}-{self.y}-{self.width}-{self.height}-{date_str}.pt"

    def get_data(self, product:str, time:datetime) -> Path:
        return torch.load(self.get_file(product, time))


def ChipBlock(TransformBlock):
    def __init__(self, dates, products):
        super().__init__(item_tfms=[self.chip_to_tensor])
        self.dates = dates
        self.products = products

    def point_to_tensor(self, point: Point):
        point

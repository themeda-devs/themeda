from pathlib import Path
import random
from fastai.data.core import TfmdDL
from fastai.data.load import _loaders, to_device
from fastai.callback.core import Callback

from .transforms import Chiplet

class TPlus1Callback(Callback):
    def before_batch(self):
        self.learn.xb = tuple(x[:,:-1] for x in self.xb)
        self.learn.yb = tuple(x[:,1:] for x in self.xb)


def get_chiplets_list(chiplet_dir:Path, max_chiplets:int=0):
    chiplets = set()
    for path in Path(chiplet_dir).glob("*.npz"):
        # path is something like: ecofuture_chiplet_level4_1988_subset_1_00004207.npz
        chiplet_components = path.name.split("_")
        subset = int(chiplet_components[5])
        chiplet_id = chiplet_components[6]
        chiplets.add( Chiplet(subset=subset, id=chiplet_id) )
    chiplets = list(chiplets)

    if max_chiplets and len(chiplets) > max_chiplets:
        random.seed(42)
        chiplets = random.sample(chiplets, max_chiplets)

    return chiplets

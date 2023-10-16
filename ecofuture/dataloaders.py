from pathlib import Path
import random
from fastai.data.core import TfmdDL
from fastai.data.load import _loaders, to_device
from fastai.callback.core import Callback
from fastai.torch_core import batch_to_samples

from .transforms import Chiplet

class TPlus1Callback(Callback):
    def before_batch(self):
        xb = self.xb
        yb = self.yb or xb # use inputs if no y batch given
        self.learn.xb = tuple(x[:,:-1] for x in xb)
        self.learn.yb = tuple(y[:,1:] for y in yb)

        


class FutureDataLoader(TfmdDL):
    def _decode_batch(self, b, max_n=9, full=True):
        """ 
        Does not decode samples from batch due to issue with TPlus1Callback.
        
        This is only a problem when visualising when logging with W&B.
        """
        samples = batch_to_samples(b, max_n=max_n)
        return samples


class PredictPersistanceCallback(Callback):
    def before_batch(self):
        persistance = self.learn.yb[0] == self.learn.xb[0] # check if the t+1 value is the same as the value at time t
        self.learn.yb = (persistance.int(),) + (self.learn.yb[1:])


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

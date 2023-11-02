from pathlib import Path
import numpy as np
from fastai.callback.core import Callback, CancelBatchException, CancelEpochException

class WriteResults(Callback):
    def __init__(self, path:Path):
        self.path = path

    def before_epoch(self):
        base_size = self.learn.dl.base_size
        n_items = self.learn.dl.n
        n_classes = self.model.output_types[0].category_count
        shape = (n_items,n_classes,base_size,base_size)

        self.chiplets = np.memmap(
            filename=self.path,
            dtype=np.float16,
            mode="w+",
            shape=shape,
        )
        self.current_index = 0

    def after_batch(self): 
        land_cover_prediction = self.learn.pred[0]
        land_cover_prediction_final_year = land_cover_prediction[:,-1]
        land_cover_probabilities = land_cover_prediction_final_year.softmax(dim=2)
        batch_size = len(land_cover_probabilities)
        end_point = self.current_index + batch_size
        self.chiplets[self.current_index:end_point, ...] = np.array(land_cover_probabilities.cpu(), dtype=self.chiplets.dtype)
        self.current_index += batch_size

    def after_epoch(self):
        self.chiplets.flush()
        self.chiplets._mmap.close()
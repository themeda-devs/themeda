from pathlib import Path
import numpy as np
from fastai.callback.core import Callback, CancelBatchException, CancelEpochException

class WriteResults(Callback):
    def __init__(self, probabilities:Path, argmax:Path):
        self.probabilities = probabilities
        self.argmax = argmax

        assert self.probabilities or self.argmax

    def before_epoch(self):
        base_size = self.learn.dl.base_size

        n_items = self.learn.dl.n
        n_classes = self.model.output_types[0].category_count
        if self.probabilities:
            shape = (n_items,n_classes,base_size,base_size)
            self.probabilities_chiplets = np.memmap(
                filename=self.probabilities,
                dtype=np.float16,
                mode="w+",
                shape=shape,
            )
        
        if self.argmax:
            shape = (n_items,base_size,base_size)
            self.argmax_chiplets = np.memmap(
                filename=self.argmax,
                dtype=np.uint8,
                mode="w+",
                shape=shape,
            )
        self.current_index = 0

    def after_batch(self): 
        pad_size = self.learn.dl.pad_size
        land_cover_prediction = self.learn.pred[0]
        land_cover_prediction_final_year = land_cover_prediction[:,-1,:,pad_size:-pad_size,pad_size:-pad_size]
        # after indexing in to the final year, the feature axis should 1 instead of 2
        feature_axis = 1
        assert land_cover_prediction_final_year.shape[feature_axis] == 23
        land_cover_probabilities = land_cover_prediction_final_year.softmax(dim=feature_axis)
        assert len(land_cover_probabilities.shape) == 4
        batch_size = len(land_cover_probabilities)
        end_point = self.current_index + batch_size
        if self.probabilities:
            self.probabilities_chiplets[self.current_index:end_point, ...] = np.array(land_cover_probabilities.cpu(), dtype=self.probabilities_chiplets.dtype)
        if self.argmax:
            self.argmax_chiplets[self.current_index:end_point, ...] = np.array(land_cover_probabilities.argmax(dim=feature_axis).cpu(), dtype=self.argmax_chiplets.dtype)
        self.current_index += batch_size

    def after_epoch(self):
        if self.probabilities:
            self.probabilities_chiplets.flush()
            self.probabilities_chiplets._mmap.close()
        if self.argmax:
            self.argmax_chiplets.flush()
            self.argmax_chiplets._mmap.close()

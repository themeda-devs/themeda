from pathlib import Path
import torch
from polytorch.metrics import categorical_accuracy, smooth_l1, get_predictions_target_for_index, PolyMetric
import torch.nn.functional as F
from torch import Tensor
from attrs import define, field
from typing import Optional

from .util import get_land_cover_column
    
        
def accuracy(predictions, *targets):
    if not isinstance(predictions, tuple):
        predictions = (predictions,)

    assert len(predictions) == len(targets)

    for prediction, target in zip(predictions, targets):
        prediction = torch.argmax(prediction, dim=2) # only for categorical
        correct = prediction == target
        
        return correct.float().mean() # hack for only single input
    

def smooth_l1_rain(predictions, *targets):
    return smooth_l1(predictions, *targets, data_index=1, feature_axis=2)


def smooth_l1_tmax(predictions, *targets):
    return smooth_l1(predictions, *targets, data_index=2, feature_axis=2)


def kl_divergence_proportions_tensors(my_predictions:Tensor, my_targets:Tensor, feature_axis=-1, softmax:bool=True):
    n_classes = my_predictions.shape[feature_axis]
    averaging_axes = [-1,-2] # the axes for the pixels

    if softmax:
        my_predictions = my_predictions.softmax(dim=feature_axis)
    
    my_predictions_proportions = my_predictions.mean(dim=averaging_axes)
    log_proportions = my_predictions_proportions.log()

    # Get the distribution of targets on the chiplet for each timestep
    batch_size = my_targets.shape[0]
    timesteps = my_targets.shape[1]
    reshaped_targets = my_targets.view(batch_size, timesteps, -1).long()  # reshape the pixels into one axis
    target_counts = torch.zeros(batch_size, timesteps, n_classes, dtype=torch.float32, device=my_targets.device)
    target_counts.scatter_add_(-1, reshaped_targets, torch.ones_like(reshaped_targets, dtype=torch.float32))
    my_targets_proportions = target_counts / target_counts.sum(dim=-1, keepdim=True)

    return F.kl_div(log_proportions, my_targets_proportions, reduction='none')


# def kl_divergence_proportions(predictions, *targets, data_index=None, feature_axis=-1):
#     my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
#     return kl_divergence_proportions_tensors(my_predictions, my_targets, feature_axis=feature_axis)

@define
class WritableMetric(PolyMetric):
    output_dir: Optional[Path] = None
    output_path: Optional[Path] = None

    def __attrs_post_init__(self):
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(exist_ok=True, parents=True)
            self.output_path = self.output_dir/f"{self.name}.csv"
            with self.output_path.open("w") as f:
                print("year_index", "result", file=f, sep=",")

    def write_result(self, result):
        if getattr(self, 'output_path', None):
            with self.output_path.open("a") as f:
                for sample in range(result.shape[0]):
                    for year_index in range(result.shape[1]):
                        print(year_index, result[sample,year_index].float().mean().item(), file=f, sep=",")


@define
class KLDivergenceProportions(WritableMetric):
    def calc(self, predictions, targets):
        result = kl_divergence_proportions_tensors(
            predictions,
            targets, 
            feature_axis=self.feature_axis,
            softmax=True,
        )
        self.write_result(result)
        return result.mean()


@define
class CategoricalAccuracy(WritableMetric):
    def calc(self, predictions, targets):
        predictions = torch.max(predictions, dim=self.feature_axis).indices # should be argmax but there is an issue using MPS
        result = (predictions == targets).float()
        self.write_result(result)
        return result.mean()



@define
class HierarchicalCategoricalAccuracy(WritableMetric):
    mapping_tensor: Tensor = field(init=False)
    n_classes: int = field(init=False)
    
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        level0_codes = [int(code) for code in get_land_cover_column("LCNS_lev0")]
        self.mapping_tensor = torch.tensor(level0_codes, dtype=torch.int64)
        self.n_classes = self.mapping_tensor.max() + 1

    def map_to_level0(self, predictions, targets):
        shape = list(predictions.shape)
        shape[self.feature_axis] = self.n_classes

        self.mapping_tensor = self.mapping_tensor.to(predictions.device)

        prediction_probabilities = predictions.softmax(dim=self.feature_axis)
        probabilities_level0 = torch.zeros(shape, dtype=prediction_probabilities.dtype, device=predictions.device)
        probabilities_level0.scatter_add_(self.feature_axis, self.mapping_tensor.view(1,1,-1,1,1).expand( shape[0],shape[1],-1,shape[3], shape[4] ), prediction_probabilities)

        targets_level0 = torch.gather(
            self.mapping_tensor.view(1,1,1,-1).expand(targets.shape[0],targets.shape[1],targets.shape[3],-1), 
            -1, 
            targets.long(),
        )

        return probabilities_level0, targets_level0

    def calc(self, predictions, targets):
        probabilities_level0, targets_level0 = self.map_to_level0(predictions, targets)
        predictions_level0 = torch.argmax(probabilities_level0, dim=self.feature_axis)

        correct = predictions_level0 == targets_level0
        self.write_result(correct)
        return correct.float().mean()


@define
class HierarchicalKLDivergence(HierarchicalCategoricalAccuracy):
    def calc(self, predictions, targets):
        probabilities_level0, targets_level0 = self.map_to_level0(predictions, targets)

        result = kl_divergence_proportions_tensors(
            probabilities_level0, 
            targets_level0, 
            feature_axis=self.feature_axis, 
            softmax=False,
        )
        self.write_result(result)
        return result.mean()



@define
class CategoricalAccuracyFinalYear(CategoricalAccuracy):
    def calc(self, predictions, targets):
        return super().calc(predictions[:,-1:], targets[:,-1:])


@define
class KLDivergenceProportionsFinalYear(KLDivergenceProportions):
    def calc(self, predictions, targets):
        return super().calc(predictions[:,-1:], targets[:,-1:])


@define
class HierarchicalCategoricalAccuracyFinalYear(HierarchicalCategoricalAccuracy):
    def calc(self, predictions, targets):
        return super().calc(predictions[:,-1:], targets[:,-1:])



@define
class HierarchicalKLDivergenceFinalYear(HierarchicalKLDivergence):
    def calc(self, predictions, targets):
        return super().calc(predictions[:,-1:], targets[:,-1:])



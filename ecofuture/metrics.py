import torch
from polytorch.metrics import categorical_accuracy, smooth_l1, get_predictions_target_for_index
import torch.nn.functional as F
from torch import Tensor

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


def kl_divergence_proportions_tensors(my_predictions:Tensor, my_targets:Tensor, feature_axis=-1, averaging_axes=None):
    n_classes = my_predictions.shape[feature_axis]
    batch_size = my_predictions.shape[0]
    averaging_axes = averaging_axes or tuple(x for x in range(1,len(my_predictions.shape)) if x != feature_axis)

    my_prediction_probabilities = my_predictions.softmax(dim=feature_axis)
    my_predictions_proportions = my_prediction_probabilities.mean(dim=averaging_axes)
    log_proportions = my_predictions_proportions.log()

    permutations = list(range(len(my_predictions.shape)-1))
    permutations.insert(feature_axis, len(my_predictions.shape)-1)
    one_hot_targets = F.one_hot(my_targets.long(), n_classes).permute(*permutations).float()
    my_targets_proportions = one_hot_targets.mean(dim=averaging_axes)

    return F.kl_div(log_proportions, my_targets_proportions, reduction='sum')/batch_size


def kl_divergence_proportions(predictions, *targets, data_index=None, feature_axis=-1, averaging_axes=None):
    my_predictions, my_targets = get_predictions_target_for_index(predictions, *targets, data_index=data_index, feature_axis=feature_axis)
    return kl_divergence_proportions_tensors(my_predictions, my_targets, feature_axis=feature_axis, averaging_axes=averaging_axes)

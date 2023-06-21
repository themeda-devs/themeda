import torch
from polytorch.metrics import categorical_accuracy, smooth_l1


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
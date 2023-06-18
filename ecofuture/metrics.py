import torch

def accuracy(predictions, *targets):
    if not isinstance(predictions, tuple):
        predictions = (predictions,)

    assert len(predictions) == len(targets)

    for prediction, target in zip(predictions, targets):
        prediction = torch.argmax(prediction, dim=2) # only for categorical
        correct = prediction == target
        
        return correct.float().mean() # hack for only single input
import torch

def accuracy(prediction, target):
    prediction = torch.argmax(prediction, dim=2)
    correct = prediction == target
    return correct.float().mean()
from typing import List
import torch
from torch import nn
import torch.nn.functional as F

from .models import OrdinalTensor


class MultiDatatypeLoss(nn.Module):
    def __init__(
        self,
        l1:bool=False,        
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.l1 = l1    

    def forward(self, predictions, *targets):
        if not isinstance(predictions, tuple):
            predictions = (predictions,)

        assert len(predictions) == len(targets)
        loss = 0.0
        
        for prediction, target in zip(predictions, targets):
            if isinstance(target, OrdinalTensor):
                # TODO Earth Mover Loss
                prediction = prediction.permute(0, 2, 1, 3, 4) # softmax over axis 1
                target_loss = F.cross_entropy(prediction, target.int(), reduction="none")
            elif torch.is_floating_point(target):
                if self.l1:
                    target_loss = F.l1_loss(prediction, reduction="none")
                else:
                    target_loss = F.mse_loss(prediction, reduction="none")
            else:
                # TODO Focal Loss
                prediction = prediction.permute(0, 2, 1, 3, 4) # softmax over axis 1
                target_loss = F.cross_entropy(prediction, target.int(), reduction="none")
            
            loss += target_loss

        return loss.mean()
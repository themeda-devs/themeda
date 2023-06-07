from typing import List
import torch
from torch import nn
import torch.nn.functional as F

from .models import OrdinalTensor


class MultiDatatypeLoss(nn.Module):
    def __init__(
        self,
        l1:bool=False,        
        label_smoothing:float = 0.0,
        ignore_index:int = -100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.l1 = l1    
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, predictions, *targets):
        if not isinstance(predictions, tuple):
            predictions = (predictions,)

        assert len(predictions) == len(targets)
        loss = 0.0

        for prediction, target in zip(predictions, targets):
            if isinstance(target, OrdinalTensor):
                # TODO Earth Mover Loss
                prediction = prediction.permute(0, 2, 1, 3, 4) # softmax over axis 1
                target_loss = F.cross_entropy(prediction, target.long(), reduction="none")
            elif torch.is_floating_point(target):
                if self.l1:
                    target_loss = F.l1_loss(prediction, reduction="none")
                else:
                    target_loss = F.mse_loss(prediction, reduction="none")
                    # TODO Huber loss? or smooth l1 loss
            else:
                # TODO Focal Loss
                prediction = prediction.permute(0, 2, 1, 3, 4) # softmax over axis 1
                target_loss = F.cross_entropy(
                    prediction, 
                    target.long(), 
                    reduction="none", 
                    label_smoothing=self.label_smoothing,
                    ignore_index=self.ignore_index,
                )
            
            loss += target_loss

        return loss.mean()

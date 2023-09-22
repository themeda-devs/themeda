from torch import nn
from torch.nn import functional as F
from polytorch import CategoricalData

class ProportionLoss(nn.Module):
    def __init__(self, output_types):
        super().__init__()
        assert len(output_types) == 1
        assert isinstance(output_types[0], CategoricalData)
        
        self.output_size = output_types[0].category_count     
           
    def forward(self, prediction, targets):
        # aggregate over pixels in target   
        one_hot_targets = F.one_hot(targets.long(), self.output_size).float()
        my_targets_proportions = one_hot_targets.mean(dim=[-3,-2])            
        
        # take last timestep (maybe change?)
        my_targets_proportions = my_targets_proportions[:,1]
        
        assert prediction.shape == my_targets_proportions.shape
        # assert len(my_targets_proportions.shape) == 3
        
        return F.l1_loss(prediction, my_targets_proportions)

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .util import get_land_cover_column
from torch import Tensor

class LandCoverMapper():    
    def __init__(self):
        level0_codes = [int(code) for code in get_land_cover_column("LCNS_lev0")]
        self.mapping_tensor = torch.tensor(level0_codes, dtype=torch.int64)
        self.n_classes = self.mapping_tensor.max() + 1
        self.n_major_classes = len(set(level0_codes))

    def __call__(self, data:Tensor):
        """ 
        Gets Major Category for Land Cover data

        If float data then it should be a probability distribution over land cover categories and the result is the summed probabilties.

        If integer data, then this is the index of the LCNS category.
        """
        # If given distribution
        if data.is_floating_point():
            assert data.shape[2] == self.n_classes
            raise NotImplementedError()
        
        assert len(data.shape) == 4

        return torch.gather(
            self.mapping_tensor.view(1,1,1,-1).expand(data.shape[0],data.shape[1],data.shape[3],-1), 
            -1, 
            data.long(),
        )


class LandCoverEmbedding(nn.Module):
    def __init__(
        self, 
        embedding_size:int,
        bias:bool=True,
        mean:float|None=None,
        stdev:float|None=None,
        device=None, 
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)    
        
        self.mapper = LandCoverMapper()

        self.embedding_size = embedding_size
        self.mean = mean
        self.stdev = stdev

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((self.n_major_classes, embedding_size,), **factory_kwargs), requires_grad=True)
        self.bias = Parameter(torch.empty((self.n_major_classes, embedding_size,), **factory_kwargs), requires_grad=True)
        self.distances = torch.as_tensor([
            0,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            4,
            0,
            1,
            2,
            3,
            0,
            0,
            0,
            1,
            2
        ])

        self.reset_parameters()

    def forward(self, input):
        # Get Major Categories
        level0 = self.mapper(input)

        # If given distribution
        if input.is_floating_point():
            raise NotImplementedError()
        else:
            level0_bias = F.embedding(level0, self.bias)
            level0_vector = F.embedding(level0, self.vectors)
            distance = torch.gather(self.distances, 0, input.flatten())

        return level0_bias + distance * level0_vector

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.vectors)
        torch.nn.init.constant_(self.bias, 0.0)

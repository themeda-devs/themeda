
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import Tensor
from polytorch import CategoricalData, CategoricalLossType
from .util import get_land_cover_colours
from .util import get_land_cover_column


class LandCoverMapper():    
    def __init__(self):
        level0_codes = [int(code) for code in get_land_cover_column("LCNS_lev0")]
        self.level0_codes = level0_codes
        self.mapping_tensor = torch.tensor(level0_codes, dtype=torch.int64)
        self.n_classes = len(level0_codes)
        self.n_major_classes = len(set(level0_codes))
        

    def __call__(self, data:Tensor):
        """ 
        Gets Major Category for Land Cover data

        If float data then it should be a probability distribution over land cover categories and the result is the summed probabilties.

        If integer data, then this is the index of the LCNS category.
        """
        feature_axis = 2

        # make sure the mapping tensor is on the same device
        self.mapping_tensor = self.mapping_tensor.to(data.device)

        # If given distribution, then sum the probabilities for each major category
        if data.is_floating_point():
            assert len(data.shape) == 5, f"Expected 5 dimensions, got {len(data.shape)}"
            assert data.shape[feature_axis] == self.n_classes, f"Expected {self.n_classes} classes, got {data.shape[feature_axis]}"
            shape = list(data.shape)
            shape[feature_axis] = self.n_major_classes

            probabilities_level0 = torch.zeros(shape, dtype=data.dtype, device=data.device)
            probabilities_level0.scatter_add_(
                feature_axis, 
                self.mapping_tensor.view(1,1,-1,1,1).expand( shape[0],shape[1],-1,shape[3], shape[4] ), 
                data,
            )

            return probabilities_level0
        
        # If given integer data, then just get the major category
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
        device=None, 
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)    
        
        self.mapper = LandCoverMapper()

        self.embedding_size = embedding_size

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weights = Parameter(torch.empty((self.mapper.n_major_classes, embedding_size,), **factory_kwargs), requires_grad=True)
        self.bias = Parameter(torch.empty((self.mapper.n_major_classes, embedding_size,), **factory_kwargs), requires_grad=True)
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
            level0_bias = level0.permute(0,1,3,4,2) @ self.bias

            # expand weights with codes
            all_weights = torch.gather(
                self.weights.T,
                -1, 
                self.mapper.mapping_tensor.view(1,-1).expand(self.weights.shape[-1], -1),
            ).T
            all_weights = self.distances.unsqueeze(1) *  all_weights

            return level0_bias + input.permute(0,1,3,4,2) @ all_weights
        else:
            level0_bias = F.embedding(level0, self.bias)
            level0_weights = F.embedding(level0, self.weights)
            distance = torch.gather(
                self.distances.view(1,1,1,-1).expand(input.shape[0],input.shape[1],input.shape[3],-1), 
                -1, 
                input.long(),
            ).unsqueeze(-1)

            return level0_bias + distance * level0_weights

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weights)
        torch.nn.init.constant_(self.bias, 0.0)


class LandCoverData(CategoricalData):
    def __init__(self):
        colours_dict = get_land_cover_colours()
        labels = list(colours_dict.keys())
        colours = list(colours_dict.values())
        return super().__init__(
            len(labels), 
            name="land_cover", 
            loss_type=CategoricalLossType.CROSS_ENTROPY, 
            labels=labels, 
            colors=colours
        )

    def embedding_module(self, embedding_size:int) -> LandCoverEmbedding:
        return LandCoverEmbedding(embedding_size)
    

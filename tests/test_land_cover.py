from themeda.land_cover import LandCoverMapper
import torch
from torch.nn import functional as F


def test_land_cover_mapper_ints():
    mapper = LandCoverMapper()
    categories = torch.arange(23)
    categories = categories.view(-1,1,1,1) # shape is (batch, timesteps, y, x)
    result = mapper(categories)
    assert result.shape == (23, 1, 1, 1)
    assert result.min() == 0
    assert result.max() == 8
    assert (result.squeeze() == torch.tensor([0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 7, 8])).all()


def test_land_cover_mapper_one_hot():
    mapper = LandCoverMapper()
    categories = torch.arange(23)
    categories = F.one_hot(categories).view(23,1,23,1,1).float() # shape is (batch, timesteps, classes, y, x)
    result = mapper(categories)
    assert result.shape == (23, 1, 9, 1, 1)
    assert (result.squeeze() == F.one_hot(torch.tensor([0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 7, 8]))).all()
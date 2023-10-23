from themeda.land_cover import LandCoverMapper, LandCoverEmbedding
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


def test_land_cover_embedding_ints():
    embedding = LandCoverEmbedding(9)
    categories = torch.arange(23)
    categories = categories.view(-1,1,1,1) # shape is (batch, timesteps, y, x)
    categories = categories.expand(23,2,3,5) # expand to simulate more data
    categories = torch.cat([categories, categories], dim=0) # double so that the batch size isn't confused with number of classes

    # hack the weights to be the identity matrix
    embedding.weights.data = torch.eye(9)
    embedding.bias.data = torch.arange(9).unsqueeze(1).expand(-1,9) * 100.0

    result = embedding(categories)
    assert result.shape == (46, 2, 3, 5, 9)
    assert torch.allclose(
        result[:23,0,0,0], 
        torch.tensor([
            [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [100., 100., 100., 100., 100., 100., 100., 100., 100.],
            [100., 101., 100., 100., 100., 100., 100., 100., 100.],
            [100., 102., 100., 100., 100., 100., 100., 100., 100.],
            [100., 103., 100., 100., 100., 100., 100., 100., 100.],
            [200., 200., 200., 200., 200., 200., 200., 200., 200.],
            [200., 200., 201., 200., 200., 200., 200., 200., 200.],
            [200., 200., 202., 200., 200., 200., 200., 200., 200.],
            [200., 200., 203., 200., 200., 200., 200., 200., 200.],
            [300., 300., 300., 300., 300., 300., 300., 300., 300.],
            [300., 300., 300., 301., 300., 300., 300., 300., 300.],
            [300., 300., 300., 302., 300., 300., 300., 300., 300.],
            [300., 300., 300., 303., 300., 300., 300., 300., 300.],
            [300., 300., 300., 304., 300., 300., 300., 300., 300.],
            [400., 400., 400., 400., 400., 400., 400., 400., 400.],
            [400., 400., 400., 400., 401., 400., 400., 400., 400.],
            [400., 400., 400., 400., 402., 400., 400., 400., 400.],
            [400., 400., 400., 400., 403., 400., 400., 400., 400.],
            [500., 500., 500., 500., 500., 500., 500., 500., 500.],
            [600., 600., 600., 600., 600., 600., 600., 600., 600.],
            [700., 700., 700., 700., 700., 700., 700., 700., 700.],
            [700., 700., 700., 700., 700., 700., 700., 701., 700.],
            [800., 800., 800., 800., 800., 800., 800., 800., 802.]],
        ),
    )


def test_land_cover_embedding_one_hot():
    embedding = LandCoverEmbedding(9)
    categories = torch.arange(23)
    categories = F.one_hot(categories).view(23,1,23,1,1).float() # shape is (batch, timesteps, classes, y, x)
    categories = categories.expand(23,2,23,3,5) # expand to simulate more data
    categories = torch.cat([categories, categories], dim=0) # double so that the batch size isn't confused with number of classes

    # hack the weights to be the identity matrix
    embedding.weights.data = torch.eye(9)
    embedding.bias.data = torch.arange(9).unsqueeze(1).expand(-1,9) * 100.0

    result = embedding(categories)
    assert result.shape == (46, 2, 3, 5, 9)

    assert torch.allclose(
        result[:23,0,0,0], 
        torch.tensor([
            [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [100., 100., 100., 100., 100., 100., 100., 100., 100.],
            [100., 101., 100., 100., 100., 100., 100., 100., 100.],
            [100., 102., 100., 100., 100., 100., 100., 100., 100.],
            [100., 103., 100., 100., 100., 100., 100., 100., 100.],
            [200., 200., 200., 200., 200., 200., 200., 200., 200.],
            [200., 200., 201., 200., 200., 200., 200., 200., 200.],
            [200., 200., 202., 200., 200., 200., 200., 200., 200.],
            [200., 200., 203., 200., 200., 200., 200., 200., 200.],
            [300., 300., 300., 300., 300., 300., 300., 300., 300.],
            [300., 300., 300., 301., 300., 300., 300., 300., 300.],
            [300., 300., 300., 302., 300., 300., 300., 300., 300.],
            [300., 300., 300., 303., 300., 300., 300., 300., 300.],
            [300., 300., 300., 304., 300., 300., 300., 300., 300.],
            [400., 400., 400., 400., 400., 400., 400., 400., 400.],
            [400., 400., 400., 400., 401., 400., 400., 400., 400.],
            [400., 400., 400., 400., 402., 400., 400., 400., 400.],
            [400., 400., 400., 400., 403., 400., 400., 400., 400.],
            [500., 500., 500., 500., 500., 500., 500., 500., 500.],
            [600., 600., 600., 600., 600., 600., 600., 600., 600.],
            [700., 700., 700., 700., 700., 700., 700., 700., 700.],
            [700., 700., 700., 700., 700., 700., 700., 701., 700.],
            [800., 800., 800., 800., 800., 800., 800., 800., 802.]],
        ),
    )


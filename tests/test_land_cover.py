from themeda.land_cover import LandCoverMapper, LandCoverEmbedding, LandCoverData
import torch
from torch.nn import functional as F
import numpy as np

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


def test_land_cover_default_loss():
    data = LandCoverData()
    targets = torch.cat([torch.arange(23),torch.arange(23)])
    targets = targets.view(-1,1,1,1).expand(-1,2,3,5)
    predictions = F.one_hot(targets).permute(0,1,4,2,3).float()

    loss = data.calculate_loss(predictions, targets, feature_axis=2)
    assert loss.shape == (46, 2, 3, 5)
    assert 2.2 <= loss.mean() <= 2.3

    loss = data.calculate_loss(predictions*100, targets, feature_axis=2)
    assert loss.shape == (46, 2, 3, 5)
    assert loss.mean() <= 0.1


def test_land_cover_emd_loss():
    data = LandCoverData(emd_loss=True)
    targets = torch.cat([torch.arange(23),torch.arange(23)])
    targets = targets.view(-1,1,1,1).expand(-1,2,3,5)
    predictions = F.one_hot(targets).permute(0,1,4,2,3).float()

    loss = data.calculate_loss(predictions, targets, feature_axis=2)
    assert loss.shape == (46, 2, 3, 5)
    assert 6.48 <= loss.mean() <= 6.49

    loss = data.calculate_loss(predictions*100, targets, feature_axis=2)
    assert loss.shape == (46, 2, 3, 5)
    assert loss.mean() <= 0.1

    def get_target(index):
        return torch.tensor([index]).view(-1,1,1,1)

    def get_prediction(index, value:float=1.0):
        return F.one_hot(torch.tensor([index]).view(-1,1,1,1), num_classes=23).permute(0,1,4,2,3).float() * value

    def assert_loss(prediction, target, result, value:float=1.0):
        loss = data.calculate_loss(get_prediction(prediction, value=value), get_target(target), feature_axis=2).mean()
        assert np.allclose(loss, result, atol=0.01)

    for i in range(23):
        assert_loss(i, i, 0.0, value=100.0)
    
    # Check Target 1 (Cultivated closed)
    assert_loss(0, 1, 7.8474, value=5.0)
    assert_loss(1, 1, 0.9272, value=5.0)
    assert_loss(2, 1, 1.7922, value=5.0)
    assert_loss(3, 1, 2.6572, value=5.0)
    assert_loss(4, 1, 3.5223, value=5.0)
    for i in range(5,23):
        assert_loss(i, 1, 7.8474, value=5.0)
    
    # Check Target 2 (Cultivated open 40)
    assert_loss(0, 2, 7.8357, value=5.0)
    assert_loss(1, 2, 1.7805, value=5.0)
    assert_loss(2, 2, 0.9154, value=5.0)
    assert_loss(3, 2, 1.7805, value=5.0)
    assert_loss(4, 2, 2.6455, value=5.0)
    for i in range(5,23):
        assert_loss(i, 2, 7.8357, value=5.0)

    # Check Target 7 (Woody open 15)
    assert_loss(5, 7, 2.6455, value=5.0)
    assert_loss(6, 7, 1.7805, value=5.0)
    assert_loss(7, 7, 0.9154, value=5.0)
    assert_loss(8, 7, 1.7805, value=5.0)
    for i in set(range(9,23))|set(range(0,5)):
        assert_loss(i, 7, 7.8357, value=5.0)


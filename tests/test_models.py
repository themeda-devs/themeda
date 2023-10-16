import torch
from themeda import models
from torch import nn

from polytorch import ContinuousData

def test_encoder_shape():
    batch_size = 10
    n_products = 5
    height = width = 128

    model = models.ResnetSpatialEncoder(
        in_channels=n_products,
        average_channels=True,
    )
    
    x = torch.zeros( (batch_size, n_products, height, width) )
    encoded, initial, l1, l2, l3, l4 = model(x)
    assert initial.shape == x.shape
    assert l1.shape == (batch_size, 64, height//4, width//4)    
    assert l2.shape == (batch_size, 128, height//8, width//8)    
    assert l3.shape == (batch_size, 256, height//16, width//16)    
    assert l4.shape == (batch_size, 512, height//32, width//32)    
    assert encoded.shape == (batch_size, 512)


def test_encoder_shape_time_distributed():
    batch_size = 10
    embedding_dims = 5
    timesteps = 3
    height = width = 128

    model = models.ResnetSpatialEncoder(
        in_channels=embedding_dims,
        average_channels=True,
    )
    
    x = torch.zeros( (batch_size, timesteps, embedding_dims, height, width) )
    encoded, initial, l1, l2, l3, l4 = model(x)
    assert initial.shape == x.shape
    assert l1.shape == (batch_size, timesteps, 64, height//4, width//4)    
    assert l2.shape == (batch_size, timesteps, 128, height//8, width//8)    
    assert l3.shape == (batch_size, timesteps, 256, height//16, width//16)    
    assert l4.shape == (batch_size, timesteps, 512, height//32, width//32)    
    assert encoded.shape == (batch_size, timesteps, 512)


def test_temporal_processor_lstm():
    batch_size = 10
    timesteps = 3
    height = width = 128

    model = models.ThemedaModel(
        input_types=[ContinuousData()],
        temporal_processor_type="LSTM",
        decoder_type=None,
    )
    
    x = torch.zeros( (batch_size, timesteps, height, width) )
    result = model(x)
    assert isinstance(model.temporal_processor, nn.LSTM)
    assert result.shape == (batch_size, timesteps, 512)


def test_temporal_processor_gru():
    batch_size = 10
    n_products = 1
    timesteps = 3
    height = width = 128

    model = models.ThemedaModel(
        input_types=[ContinuousData()],
        temporal_processor_type="gru",
        decoder_type=None,
    )
    
    x = torch.zeros( (batch_size, timesteps, height, width) )
    result = model(x)
    assert isinstance(model.temporal_processor, nn.GRU)
    assert result.shape == (batch_size, timesteps, 512)


def test_unet_decoder():
    batch_size = 10
    n_products = 1
    out_products = 2
    timesteps = 3
    height = width = 128

    model = models.ThemedaModel(
        input_types=[ContinuousData()],
        output_types=[ContinuousData(),ContinuousData()],
        temporal_processor_type="gru",
        decoder_type="unet",
    )
    
    x = torch.zeros( (batch_size, timesteps, height, width) )
    result = model(x)
    assert isinstance(result, tuple)
    assert len(result) == out_products
    assert isinstance(model.decoder, models.UNetDecoder)
    assert result[0].shape == (batch_size, timesteps, 1, height, width)



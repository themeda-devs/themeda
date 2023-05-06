import torch
from ecofuture import models
from torch import nn

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
    n_products = 1
    timesteps = 3
    height = width = 128

    model = models.EcoFutureModel(
        in_channels_continuous=n_products,
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

    model = models.EcoFutureModel(
        in_channels_continuous=n_products,
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

    model = models.EcoFutureModel(
        in_channels_continuous=n_products,
        out_channels=out_products,
        temporal_processor_type="gru",
        decoder_type="unet",
    )
    
    x = torch.zeros( (batch_size, timesteps, height, width) )
    result = model(x)
    assert isinstance(model.decoder, models.UNetDecoder)
    assert result.shape == (batch_size, timesteps, out_products, height, width)


def test_embedding():
    batch_size = 10
    timesteps = 3
    height = width = 128
    embedding_dim = 8

    embedding = models.MultiDatatypeEmbedding(embedding_dim=embedding_dim, in_channels_continuous=2, categorical_counts=[5,4,3])
    continuous0 = torch.zeros( (batch_size, timesteps, height, width) )
    continuous1 = torch.zeros( (batch_size, timesteps, height, width) )

    categorical0 = torch.randint( low=0, high=5, size=(batch_size, timesteps, height, width) )
    categorical1 = torch.randint( low=0, high=4, size=(batch_size, timesteps, height, width) )
    categorical2 = torch.randint( low=0, high=3, size=(batch_size, timesteps, height, width) )

    x = embedding(categorical0, continuous0, categorical1, continuous1, categorical2)
    assert x.shape == (batch_size, timesteps, embedding_dim, height, width)


def test_ordinal_embedding():
    embedding_dim = 8
    batch_size = 10
    timesteps = 3
    height = width = 128
    category_count = 5
    
    embedding = models.OrdinalEmbedding(embedding_dim=embedding_dim, category_count=category_count)
    ordinal = torch.randint( low=0, high=category_count, size=(batch_size, timesteps, height, width) )

    x = embedding(ordinal)
    assert x.shape == (batch_size, timesteps, height, width, embedding_dim)

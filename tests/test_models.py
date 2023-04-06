import torch
from ecofuture import models

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
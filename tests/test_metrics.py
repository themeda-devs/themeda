import torch
from themeda.metrics import accuracy

def test_accuracy():
    batch_size = 1
    timesteps = 10
    output_dims = 10
    height = width = 20

    prediction = torch.zeros( (batch_size, timesteps, output_dims, height, width), dtype=float )
    target = torch.zeros( (batch_size, timesteps, height, width), dtype=int )
    
    for i in range(output_dims):
        prediction[:,i,i,:,:] = 1
        target[:,i,:,:] = i

    assert accuracy(prediction, target) > 0.99

    prediction[:,0,0,:,:] = -1

    assert 0.89 < accuracy(prediction, target) < 0.91
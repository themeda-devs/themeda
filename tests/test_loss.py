import torch
from ecofuture.loss import MultiDatatypeLoss
import torch.nn.functional as F


def test_multidatatypeloss_categorical():
    batch_size = 1
    timesteps = 10
    output_dims = 9
    height = width = 20

    breakpoint()
    prediction = torch.zeros( (batch_size, timesteps, output_dims, height, width), dtype=float ) # hack
    target = torch.zeros( (batch_size, timesteps, height, width), dtype=int )

    # prediction = torch.zeros( (batch_size, output_dims, timesteps), dtype=float ) # hack
    # target = torch.zeros( (batch_size, timesteps), dtype=int )

    # loss = F.cross_entropy(prediction, target)
    # breakpoint()
    loss_module = MultiDatatypeLoss()

    loss = loss_module(prediction, target)
    assert 2.1 < loss < 2.2

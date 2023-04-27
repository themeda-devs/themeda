import torch
from torch import nn
import torchvision.models as visionmodels
from enum import Enum


class ResNet(Enum):
    resnet18 = "resnet18"
    resnet34 = "resnet34"
    resnet50 = "resnet50"
    resnet101 = "resnet101"
    resnet152 = "resnet152"

    def __str__(self):
        return self.value


class TemporalProcessorType(Enum):
    LSTM = "LSTM"
    GRU = "GRU"
    TransformerEncoder = "TransformerEncoder"

    def __str__(self):
        return self.value


class ResnetSpatialEncoder(nn.Module):
    def __init__(
        self,
        in_channels:int,
        encoder_resent:ResNet|str=ResNet.resnet18,
        average_channels:bool = True,
        weights:str = "DEFAULT",
        **kwargs,
    ):
        super().__init__(**kwargs)
        encoder_resent = str(encoder_resent)

        assert hasattr(visionmodels, encoder_resent)
        self.resnet = getattr(visionmodels, encoder_resent)(weights=weights)
        assert isinstance(self.resnet, visionmodels.resnet.ResNet)
        self.average_channels = average_channels

        # Modify input channels weights
        new_weights = self.resnet.conv1.weight.data.sum(dim=1, keepdim=True)/in_channels
        new_weights = new_weights.repeat( 1, in_channels, 1, 1 )
        params = {attr:getattr(self.resnet.conv1, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
        params['bias'] = bool(self.resnet.conv1.bias)
        params['in_channels'] = in_channels

        self.resnet.conv1 = nn.Conv2d( **params )
        self.resnet.conv1.data = new_weights

    def forward(self, x):
        initial = x

        time_distributed = (len(x.shape) == 5)
        if time_distributed:
            # Adapted from https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
            batch_size = x.shape[0]
            timesteps = x.shape[1]
            new_shape = (batch_size * timesteps,) + x.shape[2:]
            x = x.contiguous().view(new_shape)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        l1 = x = self.resnet.layer1(x)
        l2 = x = self.resnet.layer2(x)
        l3 = x = self.resnet.layer3(x)
        l4 = x = self.resnet.layer4(x)

        if self.average_channels:
            x = self.resnet.avgpool(x).squeeze()

        if time_distributed:
            # returns samples, timesteps, output_size
            x = x.contiguous().view( (batch_size, timesteps, -1) + x.shape[2:] )  
            l1 = l1.contiguous().view( (batch_size, timesteps, -1) + l1.shape[2:] )  
            l2 = l2.contiguous().view( (batch_size, timesteps, -1) + l2.shape[2:] )  
            l3 = l3.contiguous().view( (batch_size, timesteps, -1) + l3.shape[2:] )  
            l4 = l4.contiguous().view( (batch_size, timesteps, -1) + l4.shape[2:] )  

        return x, initial, l1, l2, l3, l4


class EcoFutureModel(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        encoder_resent:ResNet|str=ResNet.resnet18,
        encoder_weights:str="DEFAULT",
        temporal_processor_type:TemporalProcessorType|str=TemporalProcessorType.LSTM,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spatial_encoder = ResnetSpatialEncoder(in_channels=in_channels, encoder_resent=encoder_resent, weights=encoder_weights)

        # self.temporal_processor = 
        # self.decoder = 


    def forward(self, x:torch.Tensor):
        # This could be nn.Sequential but this might be helpful for debugging
        spatially_encoded_layers = self.spatial_encoder(x)
        temporally_processed = self.temporal_processor(spatially_encoded_layers)
        return self.decoder(temporally_processed)


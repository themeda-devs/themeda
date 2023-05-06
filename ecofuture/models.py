from typing import List
from torch import Tensor
import torch
from torch import nn
import torchvision.models as visionmodels
from enum import Enum
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class OrdinalTensor(torch.Tensor):
    pass


class OrdinalEmbedding(nn.Module):
    def __init__(
        self,
        category_count,
        embedding_dim,
        bias:bool=True,
        device=None, 
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)    
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.embedding_dim = embedding_dim
        self.distance_scores = Parameter(torch.empty((category_count-1,), **factory_kwargs), requires_grad=True)
        self.weight = Parameter(torch.empty((embedding_dim,), **factory_kwargs), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty((embedding_dim,), **factory_kwargs), requires_grad=True)
        else:
            self.bias = Parameter(torch.zeros((embedding_dim,), **factory_kwargs), requires_grad=False)

    def forward(self, x):
        distances = torch.cumsum(F.softmax(self.distance_scores, dim=0), dim=0)
        
        # prepend zero
        distances = torch.cat([torch.zeros((1,), device=distances.device, dtype=distances.dtype), distances])
        
        distance = torch.gather(distances, 0, x.flatten())
        embedded = self.bias + distance.unsqueeze(1) * self.weight.unsqueeze(0)
        embedded = embedded.reshape(x.shape + (-1,))

        return embedded


class MultiDatatypeEmbedding(nn.Module):
    def __init__(
        self,
        in_channels_continuous:int=0,
        categorical_counts:List[int]|None = None,
        ordinal_counts:List[int]|None = None,
        embedding_dim:int=16,        
        **kwargs,
    ):
        super().__init__(**kwargs)    
        self.embedding_dim = embedding_dim

        self.in_channels_continuous = in_channels_continuous
        if in_channels_continuous:
            self.embeddings_continuous = nn.Embedding(in_channels_continuous, embedding_dim)
            self.bias_continuous = nn.Embedding(in_channels_continuous, embedding_dim)
        
        self.categorical_counts = categorical_counts
        if self.categorical_counts:
            self.embeddings_categorical = nn.ModuleList([nn.Embedding(count, embedding_dim) for count in self.categorical_counts])

        self.ordinal_counts = ordinal_counts
        if self.ordinal_counts:
            self.embeddings_ordinal = nn.ModuleList([OrdinalEmbedding(count, embedding_dim) for count in self.ordinal_counts])

        assert self.in_channels_continuous or self.categorical_counts or self.ordinal_counts

    def forward(self, *inputs):
        batch_size, timesteps, height, width = inputs[0].shape
        x = torch.zeros( (batch_size, timesteps, self.embedding_dim, height, width) ) 
        categorical_index = 0
        ordinal_index = 0
        continuous_index = torch.as_tensor(0)
        for input in inputs:
            if isinstance(input, OrdinalTensor):
                embedding = self.embeddings_ordinal[ordinal_index](input)
                ordinal_index += 1
            elif torch.is_floating_point(input):
                embedding = input.unsqueeze(-1) * self.embeddings_continuous(continuous_index) + self.bias_continuous(continuous_index)
                continuous_index += 1
            else:
                embedding = self.embeddings_categorical[categorical_index](input)
                categorical_index += 1
            
            embedding = embedding.permute(0, 1, 4, 2, 3)
            x += embedding
        return x


def time_distributed_combine(x):
    batch_size = x.shape[0]
    timesteps = 0
    time_distributed = (len(x.shape) == 5)
    if time_distributed:
        # Adapted from https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
        timesteps = x.shape[1]
        new_shape = (batch_size * timesteps,) + x.shape[2:]
        x = x.contiguous().view(new_shape)

    return x, time_distributed, batch_size, timesteps

@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.

    Taken from https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
                            ]
    return encoder_layer, decoder_layer


def Conv(*args, dim:int, **kwargs):
    if dim == 2:
        return nn.Conv2d(*args, **kwargs)
    if dim == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"dimension {dim} not supported")    
    

def BatchNorm(*args, dim:int, **kwargs):
    if dim == 2:
        return nn.BatchNorm2d(*args, **kwargs)
    if dim == 3:
        return nn.BatchNorm3d(*args, **kwargs)
    raise ValueError(f"dimension {dim} not supported")    
    

def ConvTranspose(*args, dim:int, **kwargs):
    if dim == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    if dim == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    raise ValueError(f"dimension {dim} not supported")  

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
    TRANSFORMER = "TRANSFORMER"
    NONE = "NONE"

    def __str__(self):
        return self.value


class DecoderType(Enum):
    UNET = "UNET"
    DIFFUSION = "DIFFUSION"
    NONE = "NONE"

    def __str__(self):
        return self.value


class ResBlock(nn.Module):
    """ 
    Based on
        https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448 
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    def __init__(self, dim:int, in_channels:int, out_channels:int, downsample:bool, kernel_size:int=3):
        super().__init__()
        
        # calculate padding so that the output is the same as a kernel size of 1 with zero padding
        # this is required to be calculated becaues padding="same" doesn't work with a stride
        padding = (kernel_size - 1)//2 
        
        if downsample:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, dim=dim)
            self.shortcut = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=2, dim=dim), 
                BatchNorm(out_channels, dim=dim)
            )
        else:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dim=dim)
            self.shortcut = nn.Sequential()

        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dim=dim)
        self.bn1 = BatchNorm(out_channels, dim=dim)
        self.bn2 = BatchNorm(out_channels, dim=dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + shortcut
        return self.relu(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        dim:int,
        in_channels:int,
        out_channels:int, 
        resblock_kernel_size:int = 3,
        upsample_kernel_size:int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = ConvTranspose(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=upsample_kernel_size, stride=2, dim=dim)

        self.block1 = ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, downsample=False, dim=dim, kernel_size=resblock_kernel_size)
        # self.block2 = ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, downsample=False, dim=dim, kernel_size=resblock_kernel_size)

    def forward(self, x: Tensor, shortcut: Tensor=None) -> Tensor:
        x = self.upsample(x)
        
        if shortcut is not None:
            # crop upsampled tensor in case the size is different from the shortcut connection
            x, shortcut = autocrop(x, shortcut)
            x += shortcut

        x = self.block1(x)
        # x = self.block2(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels:int,
        initial:int,
        out_channels:int,
        kernel_size:int=3,
        final_upsample_dims:int=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.upblock3 = UpBlock(
            dim=2, 
            in_channels=in_channels, 
            out_channels=in_channels//2, 
            resblock_kernel_size=kernel_size
        )
        self.upblock2 = UpBlock(
            dim=2, 
            in_channels=in_channels//2, 
            out_channels=in_channels//4, 
            resblock_kernel_size=kernel_size
        )
        self.upblock1 = UpBlock(
            dim=2, 
            in_channels=in_channels//4, 
            out_channels=in_channels//8, 
            resblock_kernel_size=kernel_size
        )
        self.upblock0b = UpBlock(
            dim=2, 
            in_channels=in_channels//8, 
            out_channels=in_channels//16, 
            resblock_kernel_size=kernel_size
        )
        self.upblock0a = UpBlock(
            dim=2, 
            in_channels=in_channels//16, 
            out_channels=in_channels//32, 
            resblock_kernel_size=kernel_size
        )
        self.final_layer = Conv(
            in_channels=final_upsample_dims+initial, 
            out_channels=out_channels, 
            kernel_size=1,
            stride=1,
            dim=2,
        )
        
    def forward(self, initial, l1, l2, l3, x):
        x, time_distributed, batch_size, timesteps = time_distributed_combine(x)
        l1, _, _, _ = time_distributed_combine(l1)
        l2, _, _, _ = time_distributed_combine(l2)
        l3, _, _, _ = time_distributed_combine(l3)
        initial, _, _, _ = time_distributed_combine(initial)

        x = self.upblock3(x,l3)
        x = self.upblock2(x,l2)
        x = self.upblock1(x,l1)
        x = self.upblock0b(x)
        x = self.upblock0a(x)

        x = torch.cat([initial,x], dim=1)

        x = self.final_layer(x)
        if time_distributed:
            # returns samples, timesteps, output_size
            x = x.contiguous().view( (batch_size, timesteps, -1) + x.shape[2:] )  

        return x


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


        x, time_distributed, batch_size, timesteps = time_distributed_combine(x)

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
        in_channels_continuous:int=0,        
        categorical_counts:List[int]|None = None,
        out_channels:int|None=None,
        embedding_dim:int=16,        
        encoder_resent:ResNet|str=ResNet.resnet18,
        encoder_weights:str="DEFAULT",
        temporal_processor_type:TemporalProcessorType|str=TemporalProcessorType.LSTM,
        decoder_type:DecoderType|str=DecoderType.UNET,
        temporal_layers:int=2,
        temporal_size:int=512,
        temporal_bias:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        categorical_counts = categorical_counts or []
        self.embedding = MultiDatatypeEmbedding(
            embedding_dim=embedding_dim,
            categorical_counts=categorical_counts,
            in_channels_continuous=in_channels_continuous,
        )
        
        # If the number of output channels isn't explicitly given then assume that the outputs match the inputs
        if not out_channels:
            out_channels = in_channels_continuous + sum(categorical_counts)
        self.out_channels = out_channels

        self.spatial_encoder = ResnetSpatialEncoder(
            in_channels=embedding_dim, 
            encoder_resent=encoder_resent, 
            weights=encoder_weights,
        )

        # Set up the temporal processor
        temporal_dims = 512
        rnn_kwargs = dict(
            batch_first=True, 
            bidirectional=False, 
            input_size=temporal_dims,
            num_layers=temporal_layers,
            hidden_size=temporal_size,
            bias = temporal_bias,
        )
        temporal_processor_type = str(temporal_processor_type).upper()
        if temporal_processor_type == "NONE":
            self.temporal_processor = None
        elif temporal_processor_type == "LSTM":
            self.temporal_processor = nn.LSTM(**rnn_kwargs)
        elif temporal_processor_type == "GRU":
            self.temporal_processor = nn.GRU(**rnn_kwargs)
        else:
            raise ValueError(f"Cannot recognize temporal processor type {temporal_processor_type}")

        decoder_type = str(decoder_type).upper()
        if decoder_type == "NONE":
            self.decoder = None
        elif decoder_type == "UNET":
            self.decoder = UNetDecoder(in_channels=temporal_dims, out_channels=out_channels, initial=embedding_dim)
        else:
            raise ValueError(f"Cannot recognize decoder type {decoder_type}")

    def forward(self, *inputs):
        # Embedding
        x = self.embedding(*inputs)

        # Spatial Encoding
        x, initial, l1, l2, l3, l4 = self.spatial_encoder(x)
        
        # Temporal processing
        temporally_processed = self.temporal_processor(x)
        if isinstance(temporally_processed, tuple):
            temporally_processed = temporally_processed[0]
                
        l4 = l4 + temporally_processed.unsqueeze(dim=3).unsqueeze(dim=4).expand(l4.shape)

        if self.decoder is None:
            return temporally_processed

        # Decoding
        return self.decoder(initial, l1, l2, l3, l4)


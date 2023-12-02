# broken # -*- coding: future_typing -*-

import math
from typing import List
from torch import Tensor
import torch
from torch import nn
import torchvision.models as visionmodels
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from polytorch import PolyEmbedding, PolyData, split_tensor, total_size, CategoricalData

from torchvision.models import resnet18
from fastai.vision.learner import create_unet_model

from .enums import TemporalProcessorType, DecoderType, ResNet


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

def spatial_combine(x):
    batch_size = x.shape[0]
    width = 0
    height = 0
    assert len(x.shape) == 5

    # Adapted from https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    height = x.shape[-2]
    width = x.shape[-1]
    timesteps = x.shape[1]
    features = x.shape[2]
    new_shape = (batch_size * height * width, timesteps, features,)
    x = x.permute(0,3,4,1,2).contiguous().view(new_shape)

    return x, batch_size, height, width, timesteps, features


class PositionalEncoding(nn.Module):
    """ Derived from https://pytorch.org/tutorials/beginner/transformer_tutorial.html """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return x + self.pe[:x.size(1)].permute(1,0,2)



class SelfAttention(nn.Module):
    def __init__(self, dim, in_channels, num_heads:int=1, padding_mode:str="reflect") -> None:
        """
        Arguments:
            dim:
                the dimension of the image. Value should be 2 or 3
            in_channels:
                the number of channel of the image the module is self-attented to
            num_heads:
                the number of heads used in the self attntion module
        """
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm = BatchNorm(in_channels, dim=dim)
        self.qkv_generator = Conv(in_channels, in_channels * 3, kernel_size=1, stride =1, bias=False, padding_mode=padding_mode, dim=dim)
        self.output = Conv(in_channels, in_channels, kernel_size=1, padding_mode=padding_mode, dim=dim)

        if dim == 2:
            self.attn_mask_eq = "bnchw, bncyx -> bnhwyx"
            self.attn_value_eq = "bnhwyx, bncyx -> bnchw"
        elif dim == 3:
            self.attn_mask_eq = "bncdhw, bnczyx -> bndhwzyx"
            self.attn_value_eq = "bndhwzyx, bnczyx -> bncdhw"


    def forward(self, x):

        head_dim = x.shape[1] // self.num_heads

        normalised_x = self.norm(x)

        # compute query key value vectors
        qkv = self.qkv_generator(normalised_x).view(x.shape[0], self.num_heads, head_dim * 3, *x.shape[2:])
        query, key, value = qkv.chunk(3, dim=2) # split qkv along the head_dim axis

        # compute attention mask
        attn_mask = torch.einsum(self.attn_mask_eq, query, key) / math.sqrt(x.shape[1])
        attn_mask = attn_mask.view(x.shape[0], self.num_heads, *x.shape[2:], -1)
        attn_mask = torch.softmax(attn_mask, -1)
        attn_mask = attn_mask.view(x.shape[0], self.num_heads, *x.shape[2:], *x.shape[2:])

        #compute attntion value
        attn_value = torch.einsum(self.attn_value_eq, attn_mask, value)
        attn_value = attn_value.view(*x.shape)

        return x + self.output(attn_value)


def Conv(*args, dim:int, **kwargs):
    if dim == 2:
        return nn.Conv2d(*args, **kwargs)
    if dim == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"dimension {dim} not supported")    
    

def BatchNorm(*args, dim:int, **kwargs):
    return nn.Identity()
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


class ResBlock(nn.Module):
    """ 
    Based on
        https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448 
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        downsample: bool,
        padding_mode: str = "reflect",
        kernel_size: int = 3,
        position_emb_dim: int = None,
        use_affine: bool = False,
        use_attn: bool=False
    ):
        super().__init__()
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.affine = use_affine
        self.use_attn = use_attn

        # calculate padding so that the output is the same as a kernel size of 1 with zero padding
        # this is required to be calculated becaues padding="same" doesn't work with a stride
        padding = (kernel_size - 1)//2

        # position_emb_dim is used as an idicator for incorporating position information or not
        self.position_emb_dim = position_emb_dim
        if position_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
                dim=dim,
                embedding_dim=position_emb_dim,
                image_channels=out_channels,
                use_affine=use_affine
            )

        if downsample:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, padding_mode=padding_mode, dim=dim)
            self.shortcut = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=2, padding_mode=padding_mode, dim=dim), 
                BatchNorm(out_channels, dim=dim)
            )
        else:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=padding_mode, dim=dim)
            self.shortcut = nn.Sequential()

        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=padding_mode, dim=dim)
        self.bn1 = BatchNorm(out_channels, dim=dim)
        self.bn2 = BatchNorm(out_channels, dim=dim)
        self.relu = nn.ReLU(inplace=True)

        if use_attn:
            self.attn = SelfAttention(dim=dim, in_channels=out_channels, padding_mode=padding_mode)

    def forward(self, x: Tensor, position_emb: Tensor = None):
        input = x
        shortcut = self.shortcut(x)
        # print('shortcut max', shortcut.max())
        x = self.relu(self.bn1(self.conv1(x)))

        # print('block 1 max', x.max())
        # incorporate position information only if position_emb is provided and noise_func exist
        if position_emb is not None and self.position_emb_dim is not None:
            x  = self.noise_func (x, position_emb)

        x = self.relu(self.bn2(self.conv2(x)))
        # print('block 2 max', x.max())

        x = self.relu(x + shortcut)

        if self.use_attn:
            x = self.attn(x)

        # if not torch.isfinite(x).all():
        #     breakpoint()

        return x

class UpBlock(nn.Module):
    def __init__(
        self,
        dim:int,
        in_channels:int,
        out_channels:int, 
        padding_mode:str = "reflect",
        resblock_kernel_size:int = 3,
        upsample_kernel_size:int = 2,
        position_emb_dim: int = None,
        use_affine: bool = False,
        use_attn: bool = False
    ):
        super().__init__()
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.position_emb_dim = position_emb_dim
        self.use_affine = use_affine
        self.use_attn = use_attn

        self.upsample = ConvTranspose(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=upsample_kernel_size,
            stride=2,
            dim=dim
        )

        self.block1 = ResBlock(
            dim=dim,
            padding_mode=padding_mode,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            downsample=False,
            kernel_size=resblock_kernel_size,
            position_emb_dim=position_emb_dim,
            use_affine=use_affine,
            use_attn=use_attn
        )
        # self.block2 = ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, downsample=False, dim=dim, kernel_size=resblock_kernel_size)

    def forward(self, x: Tensor, shortcut: Tensor, position_emb: Tensor = None) -> Tensor:
        if not self.upsample(x).isfinite().all():
            breakpoint()
        x = self.upsample(x)
        # crop upsampled tensor in case the size is different from the shortcut connection
        x, shortcut = autocrop(x, shortcut)
        
        if not x.isfinite().all():
            breakpoint()
        if not shortcut.isfinite().all():
            breakpoint()
            
        """ should be concatenation, is there a reason for this implementation """
        x += shortcut

        if not self.block1(x, position_emb).isfinite().all():
            breakpoint()

        x = self.block1(x, position_emb)
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
        dropout:float=0.0,
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
        self.dropout = nn.Dropout(dropout)
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

        x = self.dropout(x)
        l1 = self.dropout(l1)
        l2 = self.dropout(l2)
        l3 = self.dropout(l3)
        initial = self.dropout(initial)

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


class ThemedaModelUNet(nn.Module):
    def __init__(
        self,
        input_types:List[PolyData],
        output_types:List[PolyData] = None,
        embedding_size:int=16,        
        encoder_resent:ResNet|str=ResNet.resnet18,
        encoder_weights:str="DEFAULT",
        temporal_processor_type:TemporalProcessorType|str=TemporalProcessorType.LSTM,
        decoder_type:DecoderType|str=DecoderType.UNET,
        temporal_layers:int=2,
        temporal_size:int=512,
        temporal_bias:bool=True,
        dropout:float=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding = PolyEmbedding(
            input_types=input_types,
            embedding_size=embedding_size,
            feature_axis=2,
        )

        output_types = output_types or input_types
        self.output_types = output_types
        out_channels = total_size(output_types)
        
        self.spatial_encoder = ResnetSpatialEncoder(
            in_channels=embedding_size, 
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
            self.temporal_processor = nn.Identity()
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
            self.decoder = UNetDecoder(in_channels=temporal_dims, out_channels=out_channels, initial=embedding_size, dropout=dropout)
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
        decoded = self.decoder(initial, l1, l2, l3, l4)

        # Split into tuple
        return split_tensor(decoded, self.output_types, feature_axis=2)


class ThemedaModelUNetFastAI(nn.Module):
    def __init__(
        self,
        input_types=List[PolyData],
        output_types=List[PolyData],
        embedding_size:int=16,        
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
        self.embedding = PolyEmbedding(
            input_types=input_types,
            embedding_size=embedding_size,
            feature_axis=2,
        )
        self.output_types = output_types
        out_channels = total_size(output_types)

        self.unet = create_unet_model(
            arch=resnet18, 
            n_out=out_channels, 
            img_size=(160,160), 
            pretrained=True, 
            cut=None, 
            n_in=embedding_size
        )

    def forward(self, *inputs):
        # Embedding
        x = self.embedding(*inputs)

        x, time_distributed, batch_size, timesteps = time_distributed_combine(x)

        decoded = self.unet(x)

        decoded = decoded.contiguous().view( (batch_size, timesteps, -1) + x.shape[2:] )  

        # Split into tuple
        return split_tensor(decoded, self.output_types, feature_axis=2)


class ThemedaModel(nn.Module):
    def __init__(
        self,
        input_types=List[PolyData],
        output_types=List[PolyData],
        embedding_size:int=16,
        cnn_size:int=0,  
        cnn_layers:int=1,
        cnn_kernel:int=1,    
        temporal_processor_type:TemporalProcessorType=TemporalProcessorType.NONE,
        temporal_size:int=32,
        temporal_layers:int=2,
        temporal_bias:bool=True,
        padding_mode:str="zeros",
        transformer_heads:int=8,
        transformer_layers:int=4,
        transformer_positional_encoding:bool=True,
        **kwargs,
    ):
        super().__init__()
        self.embedding = PolyEmbedding(
            input_types=input_types,
            embedding_size=embedding_size,
            feature_axis=2,
        )
        self.output_types = output_types
        out_channels = total_size(output_types)

        current_size = embedding_size

        self.cnn_size = cnn_size
        if cnn_size and cnn_layers:
            self.spatial_processor = nn.Sequential()
            for _ in range(cnn_layers):
                self.spatial_processor.append(nn.Conv2d(
                    current_size,
                    cnn_size,
                    kernel_size=cnn_kernel,
                    padding="same",
                    padding_mode=padding_mode,
                ))
                self.spatial_processor.append(nn.ReLU())
                current_size = cnn_size
        else:
            self.spatial_processor = None

        # Temporal Processor
        rnn_kwargs = dict(
            batch_first=True, 
            bidirectional=False, 
            input_size=current_size,
            num_layers=temporal_layers,
            hidden_size=temporal_size,
            bias = temporal_bias,
        )
        if isinstance(temporal_processor_type, str):
            temporal_processor_type = TemporalProcessorType[temporal_processor_type.upper()]

        self.positional_encoding = PositionalEncoding(current_size) if transformer_positional_encoding else None

        self.temporal_processor_type = temporal_processor_type
        if temporal_processor_type == TemporalProcessorType.NONE:
            self.temporal_processor = nn.Identity()
        elif temporal_processor_type == TemporalProcessorType.LSTM:
            self.temporal_processor = nn.LSTM(**rnn_kwargs)
            current_size = temporal_size
        elif temporal_processor_type == TemporalProcessorType.GRU:
            self.temporal_processor = nn.GRU(**rnn_kwargs)
            current_size = temporal_size
        elif temporal_processor_type == temporal_processor_type.TRANSFORMER:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=current_size, 
                nhead=transformer_heads,
                batch_first=True,
                # bias=temporal_bias, # only allowed in later versions of pytorch
                dim_feedforward=temporal_size,
            )
            self.temporal_processor = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=transformer_layers,
            )
        else:
            raise ValueError(f"Cannot recognize temporal processor type {temporal_processor_type}")

        self.final_conv = nn.Conv2d(
            current_size,
            out_channels,
            kernel_size=1,
        )

    def forward(self, *inputs):        
        # Embedding 
        # All the different data sources are combined here and embedded in the feature space
        x = self.embedding(*inputs)

        if self.spatial_processor:
            x, time_distributed, batch_size, timesteps = time_distributed_combine(x)
            x = self.spatial_processor(x)
            x = x.contiguous().view( (batch_size, timesteps, -1) + x.shape[2:] )  

        if self.temporal_processor_type != TemporalProcessorType.NONE:
            x, batch_size, height, width, timesteps, _ = spatial_combine(x)

            if self.temporal_processor_type == TemporalProcessorType.TRANSFORMER:
                if self.positional_encoding:
                    x = self.positional_encoding(x)
                
                mask = nn.Transformer.generate_square_subsequent_mask(
                    timesteps,
                    # dtype=x.dtype, # for later versions of torch
                    device=x.device,
                ).type(x.dtype)
                
                x = self.temporal_processor(x, mask=mask, is_causal=True)
            else:
                x = self.temporal_processor(x)

            if isinstance(x, tuple):
                x = x[0]
            x = F.relu(x) # is this necessary?
            x = x.contiguous().view( (batch_size, height, width, timesteps, -1) ).permute(0,3,4,1,2)
            

        x, time_distributed, batch_size, timesteps = time_distributed_combine(x)
        prediction = self.final_conv(x)
        prediction = prediction.contiguous().view( (batch_size, timesteps, -1) + x.shape[2:] )  

        return split_tensor(prediction, self.output_types, feature_axis=2)


class PersistenceModel(nn.Module):
    def __init__(
        self,
        input_types:List[PolyData],
        **kwargs,
    ):
        super().__init__()
        self.input_types = input_types
        self.true_logit_value = 100.0

    def forward(self, *inputs):
        results = []
        for input, datatype in zip(inputs, self.input_types):
            if isinstance(datatype, CategoricalData):
                results.append(
                    F.one_hot(input.long(), num_classes=datatype.category_count).permute(0,1,4,2,3) * self.true_logit_value
                )
            else:
                results.append(input)

        return tuple(results)
    
    
class ProportionsLSTMModel(nn.Module):
    def __init__(
        self, 
        input_types:List[PolyData],        
        output_types:List[PolyData],        
        hidden_size, 
        num_layers, 
        dropout
    ):
        super().__init__()
        assert len(input_types) == 1
        assert isinstance(input_types[0], CategoricalData)
        assert len(output_types) == 1
        assert isinstance(output_types[0], CategoricalData)
        
        self.input_size = input_types[0].category_count
        output_size = output_types[0].category_count
        
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, pixel_input):
        # aggregate over pixels          
        one_hot = F.one_hot(pixel_input.long(), self.input_size).float()
        x = one_hot.mean(dim=[-3,-2])          
        
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last output from the sequence
        return out 
    


class DownBlock(nn.Module):
    def __init__(
        self,
        dim:int,
        padding_mode: str = "reflect",
        in_channels:int = 1,
        downsample:bool = True,
        growth_factor:float = 2.0,
        kernel_size:int = 3,
        position_emb_dim:int = None,
        use_affine:bool = False,
        use_attn:bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.position_emb_dim = position_emb_dim
        self.use_affine = use_affine
        self.use_attn = use_attn
        self.padding_mode = padding_mode

        if downsample:
            self.out_channels = int(growth_factor*self.out_channels)

        self.block1 = ResBlock(
            dim=dim,
            padding_mode=padding_mode,
            in_channels=in_channels,
            out_channels=self.out_channels,
            downsample=downsample,
            kernel_size=kernel_size,
            position_emb_dim=position_emb_dim,
            use_affine=use_affine,
            use_attn=use_attn,
        )
        self.block2 = ResBlock(
            dim=dim,
            padding_mode=padding_mode,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            downsample=False,
            kernel_size=kernel_size,
            position_emb_dim=position_emb_dim,
            use_affine=use_affine,
            use_attn=use_attn,
        )

    def forward(self, x: Tensor, position_emb: Tensor = None) -> Tensor:
        x1 = self.block1(x, position_emb)
        # if not torch.isfinite(x1).all():
        #     breakpoint()
        x2 = self.block2(x1, position_emb)
        # if not torch.isfinite(x2).all():
        #     breakpoint()
        return x2


class ThemedaUNet(nn.Module):
    def __init__(
        self,
        input_types:List[PolyData],
        output_types:List[PolyData] = None,
        embedding_size:int=16,        
        padding_mode: str = "reflect",
        # initial_features:int = 64,
        growth_factor:float = 2.0,
        kernel_size:int = 3,
        layers:int = 4,
        attn_layers=(3,),
        position_emb_dim:int = None,
        use_affine:bool = False,
    ):
        super().__init__()
            
        self.embedding = PolyEmbedding(
            input_types=input_types,
            embedding_size=embedding_size,
            feature_axis=2,
        )
        self.output_types = output_types
        out_channels = total_size(output_types)

        current_num_features = embedding_size
        
        self.downblock_layers = nn.ModuleList()
        for layer_idx in range(layers):
            downblock = DownBlock(
                dim=2, 
                padding_mode=padding_mode,
                in_channels=current_num_features,
                downsample=True,
                growth_factor=growth_factor, 
                kernel_size=kernel_size,
                position_emb_dim=position_emb_dim,
                use_affine=use_affine,
                use_attn = (layer_idx in attn_layers),
            )
            self.downblock_layers.append(downblock)
            current_num_features = downblock.out_channels

        self.upblock_layers = nn.ModuleList()
        for downblock in reversed(self.downblock_layers):
            upblock = UpBlock(
                dim=2, 
                padding_mode=padding_mode,
                in_channels=downblock.out_channels,
                out_channels=downblock.in_channels,
                resblock_kernel_size=kernel_size,
                position_emb_dim=position_emb_dim,
                use_affine=use_affine,
                use_attn=downblock.use_attn
            )
            self.upblock_layers.append(upblock)
            current_num_features = upblock.out_channels

        # self.final_upsample_dims = self.upblock_layers[-1].out_channels//2
        # self.final_upsample = ConvTranspose(
        #     in_channels=self.upblock_layers[-1].out_channels, 
        #     out_channels=self.final_upsample_dims, 
        #     kernel_size=2, 
        #     stride=2,
        #     dim=2,
        # )

        self.prediction_layer = Conv(
            padding_mode=padding_mode,
            in_channels=current_num_features, 
            out_channels=out_channels, 
            kernel_size=1,
            stride=1,
            dim=2,
        )

    def forward(self, *inputs):
        # Embedding
        embedded = self.embedding(*inputs)
        
        # Encoder
        x, time_distributed, batch_size, timesteps = time_distributed_combine(embedded)
        encoded_list = []
        for downblock in self.downblock_layers:
            encoded_list.append(x)
            x = downblock(x)

        # Decoder
        for encoded, upblock in zip(reversed(encoded_list), self.upblock_layers):
            x = upblock(x, encoded)

            if not x.isfinite().all():
                breakpoint()


        prediction = self.prediction_layer(x)
        prediction = prediction.contiguous().view( (batch_size, timesteps, -1) + x.shape[2:] )  

        if not prediction.isfinite().all():
            breakpoint()

        return split_tensor(prediction, self.output_types, feature_axis=2)

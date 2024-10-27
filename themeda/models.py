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
from .convlstm import Conv_LSTM

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


def Conv(*args, dim:int, **kwargs):
    if dim == 2:
        return nn.Conv2d(*args, **kwargs)
    if dim == 3:
        return nn.Conv3d(*args, **kwargs)
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
    ):
        super().__init__()
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

        # calculate padding so that the output is the same as a kernel size of 1 with zero padding
        # this is required to be calculated becaues padding="same" doesn't work with a stride
        padding = (kernel_size - 1)//2

        if downsample:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, padding_mode=padding_mode, dim=dim)
            self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=2, padding_mode=padding_mode, dim=dim)
        else:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=padding_mode, dim=dim)
            self.shortcut = nn.Identity()

        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=padding_mode, dim=dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor):
        shortcut = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(x + shortcut)

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
    ):
        super().__init__()
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.out_channels = out_channels

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
        )
        # self.block2 = ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, downsample=False, dim=dim, kernel_size=resblock_kernel_size)

    def forward(self, x: Tensor, shortcut: Tensor) -> Tensor:
        x = self.upsample(x)
        # crop upsampled tensor in case the size is different from the shortcut connection
        x, shortcut = autocrop(x, shortcut)
                    
        """ should be concatenation, is there a reason for this implementation """
        x += shortcut

        x = self.block1(x)
        # x = self.block2(x)
        return x


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
        output_types:List[PolyData],
        **kwargs,
    ):
        super().__init__()
        # This model can only predict a single categorical class and assumes the first input is that class
        assert isinstance(input_types[0], CategoricalData)
        assert isinstance(output_types[0], CategoricalData)
        assert output_types[0].category_count == input_types[0].category_count
        self.category_count = output_types[0].category_count

        self.logit_value = torch.nn.Parameter(data=torch.full((1,), 10.0), requires_grad=True)

    def forward(self, *inputs):
        input = inputs[0]
        result = F.one_hot(input.long(), num_classes=self.category_count).permute(0,1,4,2,3) * self.logit_value
        return result
    
    
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
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
        )
        self.block2 = ResBlock(
            dim=dim,
            padding_mode=padding_mode,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            downsample=False,
            kernel_size=kernel_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.block1(x)
        x2 = self.block2(x1)
        return x2


class ThemedaUNet(nn.Module):
    def __init__(
        self,
        input_types:List[PolyData],
        output_types:List[PolyData] = None,
        embedding_size:int=16,        
        padding_mode: str = "reflect",
        temporal_processor_type:TemporalProcessorType|str=TemporalProcessorType.LSTM,
        temporal_bias:bool=True,
        temporal_layers:int=1,
        transformer_heads:int=8,
        transformer_layers:int=4,
        # initial_features:int = 64,
        growth_factor:float = 2.0,
        kernel_size:int = 3,
        layers:int = 4,
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
            )
            self.downblock_layers.append(downblock)
            current_num_features = downblock.out_channels

        if isinstance(temporal_processor_type, str):
            temporal_processor_type = TemporalProcessorType[temporal_processor_type.upper()]

        self.temporal_processor_type = temporal_processor_type
        self.temporal_processors = nn.ModuleList()
        for downblock in self.downblock_layers:
            rnn_kwargs = dict(
                batch_first=True, 
                bidirectional=False, 
                input_size=downblock.in_channels,
                hidden_size=downblock.in_channels,
                num_layers=temporal_layers,
                bias = temporal_bias,
            )
            if temporal_processor_type == TemporalProcessorType.NONE:
                temporal_processor = nn.Identity()
            elif temporal_processor_type == TemporalProcessorType.LSTM:
                temporal_processor = nn.LSTM(**rnn_kwargs)
            elif temporal_processor_type == TemporalProcessorType.GRU:
                temporal_processor = nn.GRU(**rnn_kwargs)
            elif temporal_processor_type == temporal_processor_type.TRANSFORMER:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=downblock.out_channels, 
                    nhead=transformer_heads,
                    batch_first=True,
                    # bias=temporal_bias, # only allowed in later versions of pytorch
                    dim_feedforward=downblock.out_channels,
                )
                temporal_processor = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=transformer_layers,
                )
            else:
                raise ValueError(f"Cannot recognize temporal processor type {temporal_processor_type}")
            
            self.temporal_processors.append(temporal_processor)

        self.upblock_layers = nn.ModuleList()
        for downblock in reversed(self.downblock_layers):
            upblock = UpBlock(
                dim=2, 
                padding_mode=padding_mode,
                in_channels=downblock.out_channels,
                out_channels=downblock.in_channels,
                resblock_kernel_size=kernel_size,
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

        if self.temporal_processor_type != TemporalProcessorType.NONE:
            temporal_encoded_list = []
            for encoded, temporal_processor in zip(encoded_list, self.temporal_processors):
                encoded_shape = encoded.shape
                encoded = encoded.contiguous().view( (batch_size, timesteps, -1) + encoded.shape[2:] )  
                encoded, batch_size, height, width, timesteps, _ = spatial_combine(encoded)

                if self.temporal_processor_type == TemporalProcessorType.TRANSFORMER:
                    if self.positional_encoding:
                        encoded = self.positional_encoding(encoded)
                    
                    mask = nn.Transformer.generate_square_subsequent_mask(
                        timesteps,
                        # dtype=encoded.dtype, # for later versions of torch
                        device=encoded.device,
                    ).type(encoded.dtype)
                    
                    encoded = temporal_processor(encoded, mask=mask, is_causal=True)
                else:
                    encoded = temporal_processor(encoded)
                if isinstance(encoded, tuple):
                    encoded = encoded[0]
                encoded = encoded.contiguous().view( (batch_size, height, width, timesteps, -1) ).permute(0,3,4,1,2)
                encoded, time_distributed, batch_size, timesteps = time_distributed_combine(encoded)

                assert encoded.shape == encoded_shape

                temporal_encoded_list.append(encoded)
            del encoded_list
            encoded_list = temporal_encoded_list

        # Decoder
        for encoded, upblock in zip(reversed(encoded_list), self.upblock_layers):
            x = upblock(x, encoded)

        prediction = self.prediction_layer(x)
        prediction = prediction.contiguous().view( (batch_size, timesteps, -1) + x.shape[2:] )  

        return split_tensor(prediction, self.output_types, feature_axis=2)


class ThemedaConvLSTM(nn.Module):
    def __init__(
        self,
        input_types:List[PolyData],
        output_types:List[PolyData] = None,
        kernel_size:int = 5,
        layers:int = 3,
        hidden_dims:int = 20,
        dilation_rate:int = 1,
        img_width:int = None,
        img_height:int = None,
        memory_kernel_size:int = 5,
        peephole:bool = True,
        layer_norm_flag:bool = False,
    ):
        """ 
        Defaults from https://github.com/rudolfwilliam/satellite_image_forecasting/blob/master/config/ConvLSTM.json
        """
        super().__init__()
            
        self.output_types = output_types
        out_channels = total_size(output_types)
        embedding_size = out_channels
        self.embedding = PolyEmbedding(
            input_types=input_types,
            embedding_size=embedding_size,
            feature_axis=2,
        )

        self.convlstm = Conv_LSTM(
            input_dim=embedding_size,
            big_mem=True,
            output_dim=out_channels,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            memory_kernel_size=memory_kernel_size,
            num_layers=layers,
            dilation_rate=dilation_rate,
            img_height=img_height,
            img_width=img_width,
            peephole=peephole,
            layer_norm_flag=layer_norm_flag,
        )

    def forward(self, *inputs):
        embedded = self.embedding(*inputs)
        embedded = embedded.permute(0,2,3,4,1)
        # (b, c, w, h, t)
        preds, pred_deltas, baselines = self.convlstm(embedded)

        preds = preds.permute(0,4,1,2,3)

        return split_tensor(preds, self.output_types, feature_axis=2)


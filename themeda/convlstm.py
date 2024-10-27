""" Code adapted from https://github.com/rudolfwilliam/satellite_image_forecasting/

@article{kladny2024enhanced,
      title={Enhanced prediction of vegetation responses to extreme drought using deep learning and Earth observation data},
      author={Kladny, Klaus-Rudolf and Milanta, Marco and Mraz, Oto and Hufkens, Koen and Stocker, Benjamin D.},
      journal={Ecological Informatics},
      volume={80},
      pages={102474},
      year={2024},
      publisher={Elsevier}
}

https://www.sciencedirect.com/science/article/pii/S1574954124000165

In particular:
- https://github.com/rudolfwilliam/satellite_image_forecasting/blob/master/drought_impact_forecasting/models/utils/utils.py
- https://github.com/rudolfwilliam/satellite_image_forecasting/blob/master/drought_impact_forecasting/models/LSTM_model.py
- https://github.com/rudolfwilliam/satellite_image_forecasting/blob/master/drought_impact_forecasting/models/model_parts/Conv_LSTM.py
- https://github.com/rudolfwilliam/satellite_image_forecasting/blob/master/config/ConvLSTM.json
"""

import torch.nn as nn
import torch

class Conv_LSTM_Cell(nn.Module):
    def __init__(self, input_dim, h_channels, big_mem, kernel_size, memory_kernel_size, dilation_rate, layer_norm_flag, img_width, img_height, peephole):
        super(Conv_LSTM_Cell, self).__init__()
        self.input_dim = input_dim
        self.h_channels = h_channels
        self.c_channels = h_channels if big_mem else 1
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.layer_norm_flag = layer_norm_flag
        self.img_width = img_width
        self.img_height = img_height
        self.peephole = peephole

        self.conv_cc = nn.Conv2d(self.input_dim + self.h_channels, self.h_channels + 3 * self.c_channels, 
                                 dilation=dilation_rate, kernel_size=kernel_size,
                                 bias=True, padding='same', padding_mode='reflect')
        if self.peephole:
            self.conv_ll = nn.Conv2d(self.c_channels, self.h_channels + 2 * self.c_channels, 
                                     dilation=dilation_rate, kernel_size=memory_kernel_size,
                                     bias=False, padding='same', padding_mode='reflect')
        
        if self.layer_norm_flag:
            self.layer_norm = nn.InstanceNorm2d(self.input_dim + self.h_channels, affine=True)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        if self.layer_norm_flag:
            combined = self.layer_norm(combined)

        combined_conv = self.conv_cc(combined) 
        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, [self.c_channels, self.c_channels, self.c_channels, self.h_channels], dim=1)
        if self.peephole:
            combined_memory = self.conv_ll(c_cur)
            ll_i, ll_f, ll_o = torch.split(combined_memory, [self.c_channels, self.c_channels, self.h_channels], dim=1)
            i = torch.sigmoid(cc_i + ll_i)
            f = torch.sigmoid(cc_f + ll_f)
            o = torch.sigmoid(cc_o + ll_o)
        else:
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o) 

        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.h_channels, height, width, device=self.conv_cc.weight.device),  
                torch.zeros(batch_size, self.c_channels, height, width, device=self.conv_cc.weight.device))

class Conv_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, big_mem, kernel_size, memory_kernel_size, dilation_rate,
                 img_width, img_height, layer_norm_flag=True, num_layers=1, peephole=True):
        super(Conv_LSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)

        self.input_dim = input_dim
        self.h_channels = self._extend_for_multilayer(hidden_dims, num_layers - 1)
        self.h_channels.append(output_dim)
        self.big_mem = big_mem
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.memory_kernel_size = memory_kernel_size
        self.dilation_rate = dilation_rate
        self.layer_norm_flag = layer_norm_flag
        self.img_width = img_width
        self.img_height = img_height
        self.peephole = peephole

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.h_channels[i - 1]
            cur_layer_norm_flag = self.layer_norm_flag if i != 0 else False

            cell_list.append(Conv_LSTM_Cell(input_dim=cur_input_dim,
                                            h_channels=self.h_channels[i],
                                            big_mem=self.big_mem,
                                            layer_norm_flag=cur_layer_norm_flag,
                                            img_width=self.img_width,
                                            img_height=self.img_height,
                                            kernel_size=self.kernel_size,
                                            memory_kernel_size=self.memory_kernel_size,
                                            dilation_rate=dilation_rate,
                                            peephole=self.peephole))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):
        b, _, width, height, T = input_tensor.size()
        hs = [torch.zeros((b, self.h_channels[i], height, width, T + 1), device=self._get_device(), dtype=input_tensor.dtype) for i in range(self.num_layers)]
        cs = [torch.zeros((b, self.h_channels[i] if self.big_mem else 1, height, width, T + 1), device=self._get_device(), dtype=input_tensor.dtype) for i in range(self.num_layers)]

        pred_deltas = torch.zeros((b, self.h_channels[-1], height, width, T), device=self._get_device(), dtype=input_tensor.dtype)
        baselines = torch.zeros((b, self.h_channels[-1], height, width, T), device=self._get_device(), dtype=input_tensor.dtype)

        # Set the baselines for the subsequent timesteps using the previous timestep values
        baselines[..., 1:] = input_tensor[..., :T-1].clone()

        for t in range(T):
            input_t = input_tensor[..., t]
            h0, c0 = self.cell_list[0](input_tensor=input_t, cur_state=[hs[0][..., t].detach().clone(), cs[0][..., t].detach().clone()])
            hs[0][..., t + 1] = h0.clone()
            cs[0][..., t + 1] = c0.clone()

            for i in range(1, self.num_layers):
                h, c = self.cell_list[i](input_tensor=hs[i - 1][..., t + 1], cur_state=[hs[i][..., t].detach().clone(), cs[i][..., t].detach().clone()])
                hs[i][..., t + 1] = h.clone()
                cs[i][..., t + 1] = c.clone()

            pred_deltas[..., t] = hs[-1][..., t + 1]
        
        preds = pred_deltas + input_tensor

        return preds, pred_deltas, baselines

    def _get_device(self):
        return self.cell_list[0].conv_cc.weight.device

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, int) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, rep):
        if not isinstance(param, list):
            if rep > 0:
                param = [param] * rep
            else:
                return []
        return param

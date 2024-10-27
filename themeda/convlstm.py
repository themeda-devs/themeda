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


def mean_cube(cube, mask_channel=False):
    # cube is the input cube (note that the time is always the last coordinate)
    # cannels is the list of channels we compute the avarage on
    # mask_channel whether we include the data quality mask (if we include the mask channel it should be the last one)
    '''
    cube dimensions are:
        b, c, w, h, t
    '''
    if not mask_channel:
        return torch.mean(cube, dim=-1)
    else:
        channels = cube.shape[1]
        # mask which data is cloudy and shouldn't be used for averaging
        mask = torch.repeat_interleave(1 - cube[:, -1:, :, :, :], channels - 1, axis=1)

        masked_cube = mask * cube[:, :-1, :, :, :] 
        avg_cube = torch.sum(masked_cube, dim=-1) / torch.sum(mask, dim = -1)
        return torch.nan_to_num(avg_cube, nan = 0)

def last_frame(cube, mask_channel=4):
    # note that by default the last channel will be the mask
    T = cube.shape[-1]
    # 1 = good quality, 0 = bad quality (in the flipped version)
    mask = 1 - cube[:, mask_channel:mask_channel + 1, :, :, T - 1]
    missing = cube[:, mask_channel:mask_channel + 1, :, :, T - 1]  # 1 = is missing, 0 = is already assigned
    new_cube = cube[:, :-1, :, :, T - 1] * mask

    t = T - 1
    while (torch.min(mask) == 0 and t >= 0):
        mask = missing * (1 - cube[:, mask_channel:mask_channel + 1, :, :, t])
        new_cube += cube[:, :-1, :, :, t] * mask
        missing = missing * (1 - mask)
        t -= 1
    return new_cube


def zeros(cube, mask_channel = 4):
    return cube[:, :-1, :, :, 0]*0


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

        self.conv_cc = nn.Conv2d(self.input_dim + self.h_channels, self.h_channels + 3*self.c_channels, dilation=dilation_rate, kernel_size=kernel_size,
                                     bias=True, padding='same', padding_mode='reflect')
        if self.peephole:
            self.conv_ll = nn.Conv2d(self.c_channels, self.h_channels + 2*self.c_channels, dilation=dilation_rate, kernel_size=memory_kernel_size,
                                        bias=False, padding='same', padding_mode='reflect')
        
        if self.layer_norm_flag:
            self.layer_norm = nn.InstanceNorm2d(self.input_dim + self.h_channels, affine=True)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        # apply layer normalization
        if self.layer_norm_flag:
            combined = self.layer_norm(combined)

        combined_conv = self.conv_cc(combined) # h_channel + 3 * c_channel 

        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, [self.c_channels, self.c_channels, self.c_channels, self.h_channels], dim=1)
        if self.peephole:
            combined_memory = self.conv_ll(c_cur)  # h_channel + 2 * c_channel  # NO BIAS HERE
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
        if self.h_channels == self.c_channels:
            h_next = o * torch.tanh(c_next)
        elif self.c_channels == 1:
            h_next = o * torch.tanh(c_next).repeat([1, self.h_channels, 1, 1])

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.h_channels, height, width, device=self.conv_cc.weight.device),  
                torch.zeros(batch_size, self.c_channels, height, width, device=self.conv_cc.weight.device))


class Conv_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, big_mem, kernel_size, memory_kernel_size, dilation_rate,
                    img_width, img_height, layer_norm_flag=True, baseline="last_frame", num_layers=1, peephole=True):
        """
        Can be initialized both with and without peephole connections. This Conv_LSTM works in a delta prediction 
        fashion, i.e. predicts the deviation to a given baseline.

        Parameters:
            input_dim: Number of channels in input
            output_dim: Number of channels in the output
            hidden_dim: Number of channels in the hidden outputs (should be a number or a list of num_layers - 1)
            kernel_size: Size of kernel in convolutions (Note: Will do same padding)
            memory_kernel_size: Size of kernel in convolutions when the memory influences the output
            dilation_rate: Size of holes in convolutions
            img_width: Width of the image in pixels
            img_height: Height of the image in pixels
            layer_norm_flag: Whether to perform layer normalization
            baseline: What baseline function to use
            num_layers: Number of LSTM layers stacked on each other
            peephole: Whether to include peephole connections or not
        Input:
            A tensor of shape (b, c, w, h, t)
        Output:
            The residual from the baseline
        """
        super(Conv_LSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)

        self.input_dim = input_dim                                                  # n of channels in input pics
        self.h_channels = self._extend_for_multilayer(hidden_dims, num_layers - 1)  # n of hidden channels   
        self.h_channels.append(output_dim)                                          # n of channels in output pics
        self.big_mem = big_mem                                                      # true means c = h, false c = 1. 
        self.num_layers = num_layers                                                # n of channels that go through hidden layers
        self.kernel_size = kernel_size     
        self.memory_kernel_size = memory_kernel_size                                # n kernel size (no magic here)
        self.dilation_rate = dilation_rate
        self.layer_norm_flag = layer_norm_flag
        self.img_width = img_width
        self.img_height = img_height
        self.baseline = baseline
        self.peephole = peephole

        cell_list = []
        for i in range(0, self.num_layers):
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

    def forward(self, input_tensor, non_pred_feat=None, prediction_count=1):
        """
        Parameters
        ----------
        input_tensor:
            (b - batch_size, h - height, w - width, c - channel, t - time)
            5-D Tensor either of shape (b, c, w, h, t)
        non_pred_feat:
            non-predictive features for future frames
        baseline:
            baseline computed on the input variables. Only needed for prediction_count > 1.
        Returns
        -------
        pred_deltas
        """

        baseline = eval(self.baseline + "(input_tensor[:, :, :, :, :], 4)")
        b, _, width, height, T = input_tensor.size()
        hs = []
        cs = []

        for i in range(self.num_layers):
            h, c = self.cell_list[i].init_hidden(b,(height,width))
            hs.append(h)
            cs.append(c)

        pred_deltas = torch.zeros((b, self.h_channels[-1], height, width, prediction_count), device = self._get_device())
        preds = torch.zeros((b, self.h_channels[-1], height, width, prediction_count), device = self._get_device())
        baselines = torch.zeros((b, self.h_channels[-1], height, width, prediction_count), device = self._get_device())

        
        # iterate over the past
        for t in range(T):
            hs[0], cs[0] = self.cell_list[0](input_tensor=input_tensor[..., t], cur_state=[hs[0], cs[0]])
            for i in range(1, self.num_layers):
                hs[i], cs[i] = self.cell_list[i](input_tensor=hs[i - 1], cur_state=[hs[i], cs[i]])

        baselines[..., 0] = baseline
        pred_deltas[..., 0] = hs[-1]
        preds[..., 0] = pred_deltas[..., 0] + baselines[..., 0]
        
        # add a mask to prediction
        if prediction_count > 1:
            non_pred_feat = torch.cat((torch.zeros((non_pred_feat.shape[0],
                                                    1,
                                                    non_pred_feat.shape[2],
                                                    non_pred_feat.shape[3],
                                                    non_pred_feat.shape[4]), device=non_pred_feat.device), non_pred_feat), dim = 1)

            # iterate over the future
            for t in range(1, prediction_count):
                # glue together with non_pred_data
                prev = torch.cat((preds[..., t-1], non_pred_feat[..., t-1]), axis=1)

                hs[0], cs[0] = self.cell_list[0](input_tensor=prev, cur_state=[hs[0], cs[0]])
                for i in range(1, self.num_layers):
                    hs[i], cs[i] = self.cell_list[i](input_tensor=hs[i-1], cur_state=[hs[i], cs[i]])

                pred_deltas[..., t] = hs[-1]

                if self.baseline == "mean_cube":
                    baselines[..., t] = (preds[..., t-1] + (baselines[..., t-1] * (T + t)))/(T + t + 1)
                if self.baseline == "zeros":
                    pass
                else:
                    baselines[..., t]  = preds[..., t-1]

                preds[..., t] = pred_deltas[..., t] + baselines[..., t]

        return preds, pred_deltas, baselines

    def _get_device(self):
        return self.cell_list[0].conv_cc.weight.device

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                isinstance(kernel_size, int) or
                # lists are currently not supported for Peephole_Conv_LSTM
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
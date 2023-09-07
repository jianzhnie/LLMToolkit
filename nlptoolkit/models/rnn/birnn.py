'''
Author: jianzhnie
Date: 2021-12-23 16:42:56
LastEditTime: 2021-12-23 16:42:56
LastEditors: jianzhnie
Description:

'''

import torch
import torch.nn as nn

from .gru import NaiveGRUCell
from .lstm import NaiveLSTMCell
from .rnn import NaiveRNNTanhCell


class BidirRecurrentModel(nn.Module):
    """
    Bidirectional Recurrent Model that supports LSTM, GRU, and RNN variants.

    Args:
        mode (str): The RNN cell type ('LSTM', 'GRU', 'RNN_TANH', or 'RNN_RELU').
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        num_layers (int): The number of RNN layers to stack.
        bidirectional (bool): Whether or not to include bias terms in the RNN cells.


    Example usage:
    ```python
        model = BidirRecurrentModel(mode='LSTM', input_size=64, hidden_size=128, num_layers=2, bias=True, output_size=10)
        input_data = torch.randn(32, 10, 64)
        # Batch size of 32, sequence length of 10, input size of 64
        output = model(input_data)
    ```
    """
    def __init__(self,
                 mode: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bidirectional: bool = True):
        super(BidirRecurrentModel, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if mode == 'LSTM':
            Cell = NaiveLSTMCell
        elif mode == 'GRU':
            Cell = NaiveGRUCell
        elif mode == 'RNN':
            Cell = NaiveRNNTanhCell
        else:
            raise ValueError('Invalid RNN mode selected.')

        self.rnn_model = nn.ModuleList(
            [Cell(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, input: torch.Tensor, hidden=None) -> torch.Tensor:
        """
        Perform the forward pass of the Bidirectional RNN model.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, sequence, input_size).
            hx (torch.Tensor, optional): Initial hidden state tensor of shape (num_layers, batch, hidden_size).
                If not provided, it is initialized with zeros.

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_size).
        """
        bs, seq_len, _ = input.size()

        # 初始化正向和反向的隐藏状态
        if hidden is None:
            if self.bidirectional:
                h_x_fwd, h_x_bwd = (
                    torch.zeros(bs, self.num_layers,
                                self.hidden_size).to(input.device),
                    torch.zeros(bs, self.num_layers,
                                self.hidden_size).to(input.device),
                )
                if self.mode == 'LSTM':
                    h_x_bwd = (h_x_fwd, h_x_fwd)
                    h_x_bwd = (h_x_bwd, h_x_bwd)
            else:
                h_x_fwd = torch.zeros(bs, self.hidden_size).to(input.device)
                if self.mode == 'LSTM':
                    h_x_fwd = (h_x_fwd, h_x_fwd)
        else:
            if self.bidirectional:
                h_x_fwd, h_x_bwd = hidden
                if self.mode == 'LSTM':
                    h_x_bwd = (h_x_fwd, h_x_fwd)
                    h_x_bwd = (h_x_bwd, h_x_bwd)
            else:
                h_x_fwd = hidden
                if self.mode == 'LSTM':
                    h_x_fwd = (h_x_fwd, h_x_fwd)

        fwd_outputs = []
        bwd_outputs = []
        for t in range(seq_len):
            x_t = input[:, t, :]
            for layer_idx in range(self.num_layers):
                rnn_cell = self.rnn_model[layer_idx]

                hy_fwd = rnn_cell(x_t, h_x_fwd)
                fwd_outputs.append(
                    hy_fwd[0] if isinstance(hy_fwd, tuple) else hy_fwd)

                if self.bidirectional:
                    x_t_bwd = input[:, -(t + 1), :]
                    hy_bwd = rnn_cell(x_t_bwd, h_x_bwd)

                    bwd_outputs.insert(
                        0, hy_bwd[0] if isinstance(hy_bwd, tuple) else hy_bwd)

        # 合并正向和反向的隐藏状态
        if self.bidirectional:
            combined_outputs = [
                torch.cat((fwd, bwd), dim=-1)
                for fwd, bwd in zip(fwd_outputs, bwd_outputs)
            ]
        else:
            combined_outputs = bwd_outputs

        # 将所有时间步的输出堆叠在一起
        combined_outputs = torch.cat(combined_outputs,
                                     dim=0).transpose(0, 1).contiguous()

        # 此时combined_outputs的形状为(seq_len, batch, hidden_size*2)（如果是双向RNN）

        # 可以根据需要添加额外的处理或返回其它形式的输出
        return combined_outputs

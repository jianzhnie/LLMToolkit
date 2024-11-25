from typing import Tuple, Union

import torch
import torch.nn as nn
from gru import NaiveGRUCell
from rnn import NaiveRNNTanhCell


class BiRNNModel(nn.Module):
    """Bidirectional Recurrent Model that supports custom LSTM, GRU, and RNN
    cell variants.

    Args:
        mode (str): The RNN cell type ('LSTM', 'GRU', 'RNN').
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        num_layers (int): The number of RNN layers to stack.
        bidirectional (bool): Whether or not to use bidirectional RNN cells.

    Example usage:
    ```python
        model = BiRNNModel(mode='LSTM', input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        input_data = torch.randn(32, 10, 64)
        # Batch size of 32, sequence length of 10, input size of 64
        output, hidden_states = model(input_data)
    ```
    """

    def __init__(self,
                 mode: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bidirectional: bool = True):
        super(BiRNNModel, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Select the appropriate RNN cell based on the mode
        if mode == 'GRU':
            Cell = NaiveGRUCell
        elif mode == 'RNN':
            Cell = NaiveRNNTanhCell
        else:
            raise ValueError('Invalid RNN mode selected.')

        # Create lists of custom RNN cells for forward and backward passes
        self.rnn_model_fwd = nn.ModuleList()
        self.rnn_model_bwd = nn.ModuleList()

        for i in range(num_layers):
            # Initialize forward pass cells
            input_size_fwd = input_size if i == 0 else hidden_size if bidirectional else hidden_size
            self.rnn_model_fwd.append(Cell(input_size_fwd, hidden_size))

            # Initialize backward pass cells
            input_size_bwd = input_size if i == 0 else hidden_size if bidirectional else hidden_size
            self.rnn_model_bwd.append(Cell(input_size_bwd, hidden_size))

    def forward(
        self,
        input: torch.Tensor,
        hidden: Union[None, Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform the forward pass of the Bidirectional RNN model.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, sequence, input_size).
            hidden (Tuple[torch.Tensor], optional): Initial hidden state tensor of shape (num_layers * 2, batch, hidden_size)
                for bidirectional, or (num_layers, batch, hidden_size) for unidirectional.
                If not provided, it is initialized with zeros.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Tuple containing:
                - output (torch.Tensor): Output tensor of shape (batch, sequence, hidden_size * 2) if bidirectional,
                  or (batch, sequence, hidden_size) if not bidirectional.
                - hidden_states (Tuple[torch.Tensor, torch.Tensor]): Tuple containing forward and backward hidden states
                  of shape (num_layers * 2, batch, hidden_size) for bidirectional, or (num_layers, batch, hidden_size)
                  for unidirectional.
        """
        bs, seq_len, _ = input.size()

        # Initialize forward and backward hidden states
        if hidden is None:
            num_directions = 2 if self.bidirectional else 1
            hidden_states = torch.zeros(bs, num_directions * self.num_layers,
                                        self.hidden_size).to(input.device)
        else:
            hidden_states = hidden

        # Initialize lists to collect forward and backward outputs
        fwd_outputs = []
        bwd_outputs = []

        # Process input sequence through time steps
        for t in range(seq_len):
            x_t_fwd = input[:, t, :]
            x_t_bwd = input[:, -t, :]  # Reverse the input for backward pass

            # Process input through each RNN layer
            for layer_idx in range(self.num_layers):
                # Forward pass
                rnn_cell_fwd = self.rnn_model_fwd[layer_idx]
                h_t_fwd = hidden_states[:, layer_idx]
                hy_fwd = rnn_cell_fwd(x_t_fwd, h_t_fwd)
                hidden_states[:, layer_idx] = hy_fwd
                x_t_fwd = hy_fwd

                # Backward pass
                rnn_cell_bwd = self.rnn_model_bwd[layer_idx]
                h_t_bwd = hidden_states[:, self.num_layers + layer_idx]
                hy_bwd = rnn_cell_bwd(x_t_bwd, h_t_bwd)
                hidden_states[:, self.num_layers + layer_idx] = hy_bwd
                x_t_bwd = hy_bwd

            fwd_outputs.append(hy_fwd)
            bwd_outputs.append(hy_bwd)

        # Stack the outputs across time steps and transpose for correct dimensions
        outputs = [
            torch.cat((fwd, bwd), -1)
            for fwd, bwd in zip(fwd_outputs, reversed(bwd_outputs))
        ]
        outputs = torch.cat(outputs, dim=0).reshape(bs, seq_len, -1)
        return outputs, hidden_states


if __name__ == '__main__':
    input_data = torch.randn(32, 10, 128)
    print(input_data.shape)
    rnn_model = BiRNNModel(mode='RNN',
                           input_size=128,
                           hidden_size=256,
                           num_layers=2)
    # Batch size of 32, sequence length of 10, input size of 64
    outputs, hidden_state = rnn_model(input_data)
    print(outputs.shape, hidden_state.shape)

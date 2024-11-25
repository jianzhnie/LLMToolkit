import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class NaiveGRUCell(nn.Module):
    """A custom implementation of a GRU (Gated Recurrent Unit) layer.

    - 𝐑𝑡=𝜎(𝐗𝑡𝐖𝑥𝑟+𝐇𝑡−1𝐖ℎ𝑟+𝐛𝑟),
    - 𝐙𝑡=𝜎(𝐗𝑡𝐖𝑥𝑧+𝐇𝑡−1𝐖ℎ𝑧+𝐛𝑧),
    - 𝐇̃𝑡=tanh(𝐗𝑡𝐖𝑥ℎ+(𝐑𝑡⊙𝐇𝑡−1)𝐖ℎℎ+𝐛ℎ)
    - 𝐇𝑡=𝐙𝑡⊙𝐇𝑡−1+(1−𝐙𝑡)⊙𝐇̃𝑡.


    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.

    Attributes:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        W_xz (nn.Parameter): Weight matrix for the update gate.
        W_hz (nn.Parameter): Weight matrix for the hidden-to-hidden update gate.
        b_z (nn.Parameter): Bias for the update gate.
        W_xr (nn.Parameter): Weight matrix for the reset gate.
        W_hr (nn.Parameter): Weight matrix for the hidden-to-hidden reset gate.
        b_r (nn.Parameter): Bias for the reset gate.
        W_xh (nn.Parameter): Weight matrix for the candidate hidden state.
        W_hh (nn.Parameter): Weight matrix for the hidden-to-hidden candidate hidden state.
        b_h (nn.Parameter): Bias for the candidate hidden state.

    Reference:
        https://d2l.ai
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate parameters (z_gate)
        self.W_xz = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))

        # Reset gate parameters (r_gate)
        self.W_xr = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.Tensor(hidden_size))

        # Candidate hidden state parameters (h_tilda)
        self.W_xh = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and biases with uniform random values.

        Weight initialization follows the Xavier initialization scheme.

        Returns:
            None
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self,
                input: torch.Tensor,
                hx: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the GRU cell.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, input_size).
            hx (torch.Tensor, optional): The initial hidden state tensor of shape (batch_size, hidden_size).
                If not provided, it is initialized as zeros.

        Returns:
            hy (torch.Tensor): The final hidden state tensor of shape (batch_size, hidden_size).
        """
        if hx is None:
            hx = input.new_zeros(input.size(0),
                                 self.hidden_size).to(input.device)

        # Reset gate
        reset_gate = torch.sigmoid(
            torch.mm(input, self.W_xr) + torch.mm(hx, self.W_hr) + self.b_r)

        # Update gate
        update_gate = torch.sigmoid(
            torch.mm(input, self.W_xz) + torch.mm(hx, self.W_hz) + self.b_z)

        # Candidate hidden state
        h_tilda = torch.tanh(
            torch.mm(input, self.W_xh) + torch.mm(reset_gate * hx, self.W_hh) +
            self.b_h)

        # Hidden state
        hy = update_gate * hx + (1 - update_gate) * h_tilda

        return hy


class GRULayer(nn.Module):
    """A custom implementation of a GRU layer.

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.

    Reference:
        https://github.com/piEsposito/pytorch-lstm-by-hand
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(GRULayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # GRU cell
        self.gru_cell = NaiveGRUCell(input_size, hidden_size)

    def forward(
        self,
        input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the GRU layer.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, sequence_size, input_size).
            hidden (torch.Tensor, optional): The initial hidden state tensor of shape (batch_size, hidden_size).
                If not provided, it is initialized as zeros.

        Returns:
            outputs (torch.Tensor): The sequence of hidden states of shape (batch_size, sequence_size, hidden_size).
            hy (torch.Tensor): The final hidden state tensor of shape (batch_size, hidden_size).
        """
        bs, seq_len, _ = input.size()

        if hidden is None:
            # Initialize hidden state with zeros
            hidden = torch.zeros(bs, self.hidden_size).to(input.device)

        outputs = []

        # Forward pass through RNN layers through time
        for t in range(seq_len):
            x_t = input[:, t, :]

            # GRU cell forward pass
            hidden = self.gru_cell(x_t, hidden)
            outputs.append(hidden)

        outputs = torch.cat(outputs, dim=0).view(bs, seq_len, self.hidden_size)

        return outputs, hidden


class MultiLayerGRU(nn.Module):
    """Multi-layer GRU model implemented using multiple GRULayer layers.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        num_layers (int): The number of GRU layers to stack.

    Example usage:
    ```python
        gru_model = MultiLayerGRU(input_size=64, hidden_size=128, num_layers=2)
        input_data = torch.randn(32, 10, 64)
        # Batch size of 32, sequence length of 10, input size of 64
        output, layer_hidden_states = gru_model(input_data)
    ```
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(MultiLayerGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a list of GRU cells for each layer
        self.rnn_model = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # The first layer takes the input
                self.rnn_model.append(NaiveGRUCell(input_size, hidden_size))
            else:
                # The other layers take the hidden state of the previous layer
                self.rnn_model.append(NaiveGRUCell(hidden_size, hidden_size))

    def forward(
        self,
        input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass of the multi-layer GRU model.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, sequence, input_size).
            hidden (torch.Tensor, optional): Initial hidden state tensor of shape (batch, num_layers, hidden_size).
                If not provided, it is initialized with zeros.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - output (torch.Tensor): Output tensor of shape (batch, sequence, hidden_size).
                - layer_hidden_states (torch.Tensor): Hidden state tensor of shape (batch, num_layers, hidden_size)
                    at the final layer.
        """
        bs, seq_len, _ = input.size()

        if hidden is None:
            # Initialize hidden state with zeros
            hidden = torch.zeros(bs, self.num_layers,
                                 self.hidden_size).to(input.device)
        else:
            _, num_layers, _ = hidden.size()
            assert num_layers == self.num_layers, 'Number of layers mismatch'

        outputs = []

        for t in range(seq_len):
            x_t = input[:, t, :]

            # Forward pass through each RNN layer
            for layer_idx in range(self.num_layers):
                rnn_cell = self.rnn_model[layer_idx]
                # Pass the input through the current RNN layer
                hy = rnn_cell(x_t, hidden[:, layer_idx])
                hidden[:, layer_idx] = hy
                x_t = hy  # Update input for the next layer

            # Store output
            outputs.append(hy)

        # Concatenate and stack the hidden states
        outputs = torch.cat(outputs, dim=0).view(bs, seq_len, self.hidden_size)
        return outputs, hidden


if __name__ == '__main__':
    input_data = torch.randn(32, 10, 128)
    print(input_data.shape)
    rnn_model = MultiLayerGRU(input_size=128, hidden_size=256, num_layers=2)
    # Batch size of 32, sequence length of 10, input size of 64
    outputs, hidden_state = rnn_model(input_data)
    print(outputs.shape, hidden_state.shape)

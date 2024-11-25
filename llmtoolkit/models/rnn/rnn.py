from typing import List, Tuple

import torch
import torch.nn as nn


class NaiveRNNTanhCell(nn.Module):
    """A custom implementation of a simple RNN Cell layer using tanh
    activation.

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.

    Attributes:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        W_xh (nn.Parameter): Weight matrix for input-to-hidden connections.
        W_hh (nn.Parameter): Weight matrix for hidden-to-hidden connections.
        b_xh (nn.Parameter): Bias for input-to-hidden connections.
        b_hh (nn.Parameter): Bias for hidden state.

    Example usage:
        rnn_cell = RNNTanhCell(input_size=64, hidden_size=128)
        input_data = torch.randn(32, 64)  # Batch size of 32, input size of 64
        hidden_state = torch.zeros(32, 128)  # Initial hidden state
        new_hidden_state = rnn_cell(input_data, hidden_state)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Hidden layer parameters
        self.W_xh = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_xh = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and biases with uniform random values.

        Weight initialization follows the Xavier initialization scheme.

        Returns:
            None
        """
        stdv = 1.0 / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32))
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
            self,
            input: torch.Tensor,
            hidden: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the RNN layer.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, input_size).
            hidden (torch.Tensor, optional): The initial hidden state tensor of shape (batch_size, hidden_size).
                If not provided, it is initialized with zeros.

        Returns:
            out (torch.Tensor): The output tensor of shape (batch_size, hidden_size).
            hy (torch.Tensor): The updated hidden state tensor of shape (batch_size, hidden_size).
        """
        if hidden is None:
            hidden = input.new_zeros(input.size(0), self.hidden_size)

        # Compute the RNN cell's output
        output = torch.mm(input, self.W_xh) + self.b_xh + torch.mm(
            hidden, self.W_hh) + self.b_hh

        # Another way to compute the RNN cell's output
        # combined = torch.cat((input, hidden), dim=1)
        # output = torch.mm(combined, self.W_xh) + self.b_xh

        output = torch.tanh(output)
        return output


class RNNTanhCell(nn.Module):
    """A simple RNN cell module using tanh activation.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Use nn.Linear to simplify the code
        self.ih = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, input: torch.Tensor,
                hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the RNN cell.

        Args:
            input: Input tensor of shape (batch_size, input_size)
            hidden: Initial hidden state of shape (batch_size, hidden_size)

        Returns:
            hy: Next hidden state of shape (batch_size, hidden_size)
        """
        if hidden is None:
            hidden = input.new_zeros(input.size(0), self.hidden_size)
        hy = torch.tanh(self.ih(input) + self.hh(hidden))
        return hy


class RNNLayer(nn.Module):
    """Custom RNNBase model implemented using RNNTanhCell.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        rnn_cells (nn.ModuleList): List of RNN cells.

    Methods:
        forward(input, hidden): Perform the forward pass of the RNN model.

    Example usage:
        rnn_model = RNNBase(input_size=64, hidden_size=128)
        input_data = torch.randn(32, 10, 64)
        # Batch size of 32, sequence length of 10, input size of 64
        output, final_hidden_state = rnn_model(input_data)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # RNN cell Model
        self.rnn_cell = RNNTanhCell(input_size, hidden_size)

    def forward(
            self,
            input: torch.Tensor,
            hidden: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass of the RNN model.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, sequence, input_size).
            hidden (torch.Tensor, optional): Initial hidden state tensor of shape (batch, hidden_size).
                Defaults to None, in which case it is initialized with zeros.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - output (torch.Tensor): Output tensor of shape (batch, sequence, hidden_size).
                - final_hidden (torch.Tensor): Final hidden state tensor of shape (batch, hidden_size).
        """
        bs, seq_len, _ = input.size()

        if hidden is None:
            # Initialize hidden state with zeros
            hidden = torch.zeros(bs, self.hidden_size).to(input.device)

        output = []
        # Forward pass through RNN layers
        for t in range(seq_len):
            x_t = input[:, t, :]
            # Pass the input through the current RNN layer
            hidden = self.rnn_cell(x_t, hidden)
            output.append(hidden)

        # Cat the output
        output = torch.cat(output, dim=0).view(bs, seq_len, self.hidden_size)
        return output, hidden


class MultiLayerRNN(nn.Module):
    """Multi-layer RNN model implemented using multiple RNNTanhCell layers.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        num_layers (int): The number of RNN layers to stack.

    Example usage:
    ```python
        rnn_model = MultiLayerRNN(input_size=64, hidden_size=128, num_layers=2)
        input_data = torch.randn(32, 10, 64)
        # Batch size of 32, sequence length of 10, input size of 64
        output, final_hidden_state = rnn_model(input_data)
    ```
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(MultiLayerRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a list of RNN cells for each layer
        self.rnn_model = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # The first layer takes the input
                self.rnn_model.append(RNNTanhCell(input_size, hidden_size))
            else:
                # The other layers take the hidden state of the previous layer
                self.rnn_model.append(RNNTanhCell(hidden_size, hidden_size))

    def forward(
        self,
        input: torch.Tensor,
        hidden: List[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Perform the forward pass of the multi-layer RNN model.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, sequence, input_size).
            hidden (List[torch.Tensor], optional): Initial hidden state tensor for each layer.
                If not provided, it is initialized with zeros for all layers.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Tuple containing:
                - output (torch.Tensor): Output tensor of shape (batch, sequence, hidden_size).
                - final_hidden (List[torch.Tensor]): Final hidden state tensor for each layer.
        """
        bs, seq_len, _ = input.size()

        if hidden is None:
            # Initialize hidden state with zeros
            hidden = torch.zeros(bs, self.num_layers,
                                 self.hidden_size).to(input.device)
        else:
            _, num_layers, _ = hidden.size()
            assert num_layers == self.num_layers, 'Number of layers mismatch'

        # Initialize a list to collect the hidden states at each layer
        outputs = []

        # Forward pass through RNN layers
        for t in range(seq_len):
            x_t = input[:, t, :]

            # Forward pass through each RNN layer
            for layer_idx in range(self.num_layers):
                # Pass the input through the current RNN layer
                rnn_cell = self.rnn_model[layer_idx]
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
    rnn_model = MultiLayerRNN(input_size=128, hidden_size=256, num_layers=2)
    # Batch size of 32, sequence length of 10, input size of 64
    outputs, hidden_state = rnn_model(input_data)
    print(outputs.shape, hidden_state.shape)

    rnn = nn.RNN(128, 256, 2)
    input_data = input_data.permute(1, 0, 2)
    outputs, hidden_state = rnn(input_data)
    print(outputs.shape, hidden_state.shape)

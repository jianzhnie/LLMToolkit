import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class Chomp1d(nn.Module):
    """Remove the elements on the right.

    Args:
        chomp_size (int): The number of elements removed.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """The TCN block, consists of dilated causal conv, relu and residual block.
    See the Figure 1(b) in https://arxiv.org/pdf/1803.01271.pdf for more
    details.

    Args:
        n_inputs ([int]): The number of channels in the input tensor.
        n_outputs ([int]): The number of filters.
        kernel_size ([int]): The filter size.
        stride ([int]): The stride size.
        dilation ([int]): The dilation size.
        padding ([int]): The size of zeros to be padded.
        dropout (float, optional): Probability of dropout the units. Defaults to 0.2.
    """

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout=0.2):

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs,
                      n_outputs,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation))
        # Chomp1d is used to make sure the network is causal.
        # We pad by (k-1)*d on the two sides of the input for convolution,
        # and then use Chomp1d to remove the (k-1)*d output elements on the right.
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs,
                      n_outputs,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1, self.conv2, self.chomp2,
                                 self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    r"""
    A `TCNEncoder` takes as input a sequence of vectors and returns a
    single vector, which is the last one time step in the feature map.
    The input to this encoder is of shape `(batch_size, num_tokens, input_size)`,
    and the output is of shape `(batch_size, num_channels[-1])` with a receptive
    filed:

    .. math::

        receptive filed = 2 * \sum_{i=0}^{len(num\_channels)-1}2^i(kernel\_size-1).

    Temporal Convolutional Networks is a simple convolutional architecture. It outperforms canonical recurrent networks
    such as LSTMs in many tasks. See https://arxiv.org/pdf/1803.01271.pdf for more details.

    Args:
        input_size (int): The number of expected features in the input (the last dimension).
        num_channels (list): The number of channels in different layer.
        kernel_size (int): The kernel size. Defaults to 2.
        dropout (float): The dropout probability. Defaults to 0.2.
    """

    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNEncoder, self).__init__()
        self.input_size = input_size
        self.output_dim = num_channels[-1]

        layers = nn.ModuleList()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout=dropout))

        self.network = nn.Sequential(*layers)

    def get_input_dim(self):
        """Returns the dimension of the vector input for each element in the
        sequence input to a `TCNEncoder`.

        This is not the shape of the input tensor, but the last element of that
        shape.
        """
        return self.input_size

    def get_output_dim(self):
        """Returns the dimension of the final vector output by this
        `TCNEncoder`.

        This is not the shape of the returned tensor, but the last element of
        that shape.
        """
        return self.output_dim

    def forward(self, inputs):
        r"""
        TCNEncoder takes as input a sequence of vectors and returns a
        single vector, which is the last one time step in the feature map.
        The input to this encoder is of shape `(batch_size, num_tokens, input_size)`,
        and the output is of shape `(batch_size, num_channels[-1])` with a receptive
        filed:

        .. math::

            receptive filed = 2 * \sum_{i=0}^{len(num\_channels)-1}2^i(kernel\_size-1).

        Args:
            inputs (Tensor): The input tensor with shape `[batch_size, num_tokens, input_size]`.

        Returns:
            Tensor: Returns tensor `output` with shape `[batch_size, num_channels[-1]]`.
        """
        inputs_t = inputs.transpose([0, 2, 1])
        output = self.network(inputs_t).transpose([2, 0, 1])[-1]
        return output

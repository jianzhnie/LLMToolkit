import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    r"""
    A RNNEncoder takes as input a sequence of vectors and returns a
    single vector, which is a combination of multiple `paddle.nn.RNN
    <https://www.paddlepaddle.org.cn/documentation/docs/en/api
    /paddle/nn/layer/rnn/RNN_en.html>`__ subclass.
    The input to this encoder is of shape `(batch_size, num_tokens, input_size)`,
    The output is of shape `(batch_size, hidden_size * 2)` if RNN is bidirection;
    If not, output is of shape `(batch_size, hidden_size)`.

    Paddle's RNN have two outputs: the hidden state for every time step at last layer,
    and the hidden state at the last time step for every layer.
    If `pooling_type` is not None, we perform the pooling on the hidden state of every time
    step at last layer to create a single vector. If None, we use the hidden state
    of the last time step at last layer as a single output (shape of `(batch_size, hidden_size)`);
    And if direction is bidirection, the we concat the hidden state of the last forward
    rnn and backward rnn layer to create a single vector (shape of `(batch_size, hidden_size * 2)`).

    Args:
        input_size (int):
            The number of expected features in the input (the last dimension).
        hidden_size (int):
            The number of features in the hidden state.
        num_layers (int, optional):
            Number of recurrent layers.
            E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN,
            with the second RNN taking in outputs of the first RNN and computing the final results.
            Defaults to 1.
        direction (str, optional):
            The direction of the network. It can be "forward" and "bidirect"
            (it means bidirection network). If "biderect", it is a birectional RNN,
            and returns the concat output from both directions. Defaults to "forward"
        dropout (float, optional):
            If non-zero, introduces a Dropout layer on the outputs of each RNN layer
            except the last layer, with dropout probability equal to dropout.
            Defaults to 0.0.
        pooling_type (str, optional):
            If `pooling_type` is None, then the RNNEncoder will return the hidden state
            of the last time step at last layer as a single vector.
            If pooling_type is not None, it must be one of "sum", "max" and "mean".
            Then it will be pooled on the RNN output (the hidden state of every time
            step at last layer) to create a single vector.
            Defaults to `None`.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.pooling_type = pooling_type

        self.rnn_layer = nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                batch_first=True,
                                bidirectional=bidirectional,
                                **kwargs)

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `RNNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self.input_size

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `RNNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        if self.bidirectional:
            return self.hidden_size * 2
        else:
            return self.hidden_size

    def forward(self, inputs):
        r"""
        RNNEncoder takes the a sequence of vectors and and returns a
        single vector, which is a combination of multiple RNN layers.
        The input to this encoder is of shape `(batch_size, num_tokens, input_size)`.
        The output is of shape `(batch_size, hidden_size * 2)` if RNN is bidirection;
        If not, output is of shape `(batch_size, hidden_size)`.

        Args:
            inputs (Tensor): Shape as `(batch_size, num_tokens, input_size)`.
                Tensor containing the features of the input sequence.
            sequence_length (Tensor): Shape as `(batch_size)`.
                The sequence length of the input sequence.

        Returns:
            Tensor: Returns tensor `output`, the hidden state at the last time step for every layer.
            Its data type is `float` and its shape is `[batch_size, hidden_size]`.

        """
        encoded_text, last_hidden = self.rnn_layer(inputs)
        if not self.pooling_type:
            # We exploit the `last_hidden` (the hidden state at the last time step for every layer)
            # to create a single vector.
            # If rnn is not bidirection, then output is the hidden state of the last time step
            # at last layer. Output is shape of `(batch_size, hidden_size)`.
            # If rnn is bidirection, then output is concatenation of the forward and backward hidden state
            # of the last time step at last layer. Output is shape of `(batch_size, hidden_size * 2)`.
            if not self.bidirectional:
                output = last_hidden[-1, :, :]
            else:
                output = torch.cat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), dim=1)
        else:
            # We exploit the `encoded_text` (the hidden state at the every time step for last layer)
            # to create a single vector. We perform pooling on the encoded text.
            # The output shape is `(batch_size, hidden_size * 2)` if use bidirectional RNN,
            # otherwise the output shape is `(batch_size, hidden_size * 2)`.
            if self.pooling_type == 'sum':
                output = torch.sum(encoded_text, dim=1)
            elif self.pooling_type == 'max':
                output = torch.max(encoded_text, dim=1)
            elif self.pooling_type == 'mean':
                output = torch.mean(encoded_text, dim=1)
            else:
                raise RuntimeError(
                    'Unexpected pooling type %s .'
                    'Pooling type must be one of sum, max and mean.' %
                    self.pooling_type)
        return output


class RNNModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 rnn_hidden_size=198,
                 bidirectional=False,
                 rnn_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=emb_dim,
                                     padding_idx=padding_idx)
        self.rnn_encoder = RNNEncoder(emb_dim,
                                      rnn_hidden_size,
                                      num_layers=rnn_layers,
                                      bidirectional=bidirectional,
                                      dropout=dropout_rate,
                                      pooling_type=pooling_type)
        self.fc = nn.Linear(self.rnn_encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens, num_directions*rnn_hidden_size)
        # num_directions = 2 if direction is 'bidirect'
        # if not, num_directions = 1
        text_repr = self.rnn_encoder(embedded_text)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = torch.tanh(self.fc(text_repr))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits

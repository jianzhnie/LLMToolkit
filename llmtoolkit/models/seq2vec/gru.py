import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    r"""
    A GRUEncoder takes as input a sequence of vectors and returns a
    single vector, which is a combination of multiple `paddle.nn.GRU
    <https://www.paddlepaddle.org.cn/documentation/docs/en/api
    /paddle/nn/layer/rnn/GRU_en.html>`__ subclass.
    The input to this encoder is of shape `(batch_size, num_tokens, input_size)`,
    The output is of shape `(batch_size, hidden_size * 2)` if GRU is bidirection;
    If not, output is of shape `(batch_size, hidden_size)`.

    Paddle's GRU have two outputs: the hidden state for every time step at last layer,
    and the hidden state at the last time step for every layer.
    If `pooling_type` is not None, we perform the pooling on the hidden state of every time
    step at last layer to create a single vector. If None, we use the hidden state
    of the last time step at last layer as a single output (shape of `(batch_size, hidden_size)`);
    And if direction is bidirection, the we concat the hidden state of the last forward
    gru and backward gru layer to create a single vector (shape of `(batch_size, hidden_size * 2)`).

    Args:
        input_size (int):
            The number of expected features in the input (the last dimension).
        hidden_size (int):
            The number of features in the hidden state.
        num_layers (int, optional):
            Number of recurrent layers.
            E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU,
            with the second GRU taking in outputs of the first GRU and computing the final results.
            Defaults to 1.
        direction (str, optional):
            The direction of the network. It can be "forward" and "bidirect"
            (it means bidirection network). If "bidirect", it is a birectional GRU,
            and returns the concat output from both directions.
            Defaults to "forward".
        dropout (float, optional):
            If non-zero, introduces a Dropout layer on the outputs of each GRU layer
            except the last layer, with dropout probability equal to dropout.
            Defaults to 0.0.
        pooling_type (str, optional):
            If `pooling_type` is None, then the GRUEncoder will return the hidden state of
            the last time step at last layer as a single vector.
            If pooling_type is not None, it must be one of "sum", "max" and "mean".
            Then it will be pooled on the GRU output (the hidden state of every time
            step at last layer) to create a single vector.
            Defaults to `None`
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

        self.gru_layer = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                batch_first=True,
                                bidirectional=bidirectional,
                                **kwargs)

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `GRUEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self.input_size

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `GRUEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        if self.bidirectional:
            return self.hidden_size * 2
        else:
            return self.hidden_size

    def forward(self, inputs):
        r"""
        GRUEncoder takes the a sequence of vectors and and returns a single vector,
        which is a combination of multiple GRU layers. The input to this
        encoder is of shape `(batch_size, num_tokens, input_size)`,
        The output is of shape `(batch_size, hidden_size * 2)` if GRU is bidirection;
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
        encoded_text, last_hidden = self.gru_layer(inputs)
        if not self.pooling_type:
            # We exploit the `last_hidden` (the hidden state at the last time step for every layer)
            # to create a single vector.
            # If gru is not bidirection, then output is the hidden state of the last time step
            # at last layer. Output is shape of `(batch_size, hidden_size)`.
            # If gru is bidirection, then output is concatenation of the forward and backward hidden state
            # of the last time step at last layer. Output is shape of `(batch_size, hidden_size * 2)`.
            if not self.bidirectional:
                output = last_hidden[-1, :, :]
            else:
                output = torch.cat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), dim=1)
        else:
            # We exploit the `encoded_text` (the hidden state at the every time step for last layer)
            # to create a single vector. We perform pooling on the encoded text.
            # The output shape is `(batch_size, hidden_size * 2)` if use bidirectional GRU,
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


class GRUModel(nn.Layer):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 gru_hidden_size=198,
                 bidirectional=False,
                 gru_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=emb_dim,
                                     padding_idx=padding_idx)
        self.gru_encoder = GRUEncoder(emb_dim,
                                      gru_hidden_size,
                                      num_layers=gru_layers,
                                      bidirectional=bidirectional,
                                      dropout=dropout_rate,
                                      pooling_type=pooling_type)
        self.fc = nn.Linear(self.gru_encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens, num_directions*gru_hidden_size)
        # num_directions = 2 if direction is 'bidirect'
        # if not, num_directions = 1
        text_repr = self.gru_encoder(embedded_text)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = torch.tanh(self.fc(text_repr))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits

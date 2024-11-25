import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    r"""
    A `CNNEncoder` takes as input a sequence of vectors and returns a
    single vector, a combination of multiple convolution layers and max pooling layers.
    The input to this encoder is of shape `(batch_size, num_tokens, emb_dim)`,
    and the output is of shape `(batch_size, ouput_dim)` or `(batch_size, len(ngram_filter_sizes) * num_filter)`.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filter. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is `len(ngram_filter_sizes) * num_filter`.  This then gets
    (optionally) projected down to a lower dimensional output, specified by `output_dim`.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to `A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification <https://arxiv.org/abs/1510.03820>`__ ,
    Zhang and Wallace 2016, particularly Figure 1.

    Args:
        emb_dim(int):
            The dimension of each vector in the input sequence.
        num_filter(int):
            This is the output dim for each convolutional layer, which is the number of "filters"
            learned by that layer.
        ngram_filter_sizes(Tuple[int], optional):
            This specifies both the number of convolutional layers we will create and their sizes.  The
            default of `(2, 3, 4, 5)` will have four convolutional layers, corresponding to encoding
            ngrams of size 2 to 5 with some number of filters.
        conv_layer_activation(Layer, optional):
            Activation to use after the convolution layers.
            Defaults to `paddle.nn.Tanh()`.
        output_dim(int, optional):
            After doing convolutions and pooling, we'll project the collected features into a vector of
            this size.  If this value is `None`, we will just return the result of the max pooling,
            giving an output of shape `len(ngram_filter_sizes) * num_filter`.
            Defaults to `None`.
    """

    def __init__(self,
                 emb_dim,
                 num_filter,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activation=nn.Tanh(),
                 output_dim=None,
                 **kwargs):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_filter = num_filter
        self.ngram_filter_sizes = ngram_filter_sizes
        self.activation = conv_layer_activation
        self.output_dim = output_dim

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1,
                      out_channels=self.num_filter,
                      kernel_size=(i, self.emb_dim),
                      **kwargs) for i in self.ngram_filter_sizes
        ])

        maxpool_output_dim = self.num_filter * len(self.ngram_filter_sizes)
        if self.output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim,
                                              self.output_dim)
        else:
            self.projection_layer = None
            self.output_dim = maxpool_output_dim

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `CNNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self.emb_dim

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `CNNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self.output_dim

    def forward(self, inputs, mask=None):
        r"""
        The combination of multiple convolution layers and max pooling layers.

        Args:
            inputs (Tensor):
                Shape as `(batch_size, num_tokens, emb_dim)` and dtype as `float32` or `float64`.
                Tensor containing the features of the input sequence.
            mask (Tensor, optional):
                Shape should be same as `inputs` and dtype as `int32`, `int64`, `float32` or `float64`.
                Its each elements identify whether the corresponding input token is padding or not.
                If True, not padding token. If False, padding token.
                Defaults to `None`.

        Returns:
            Tensor:
                Returns tensor `result`.
                If output_dim is None, the result shape is of `(batch_size, output_dim)` and
                dtype is `float`; If not, the result shape is of `(batch_size, len(ngram_filter_sizes) * num_filter)`.

        """
        if mask is not None:
            inputs = inputs * mask

        # Shape: (batch_size, 1, num_tokens, emb_dim) = (N, seq_len, dembed_dim)
        inputs = inputs.unsqueeze(1)

        # If output_dim is None, result shape of (batch_size, len(ngram_filter_sizes) * num_filter));
        # else, result shape of (batch_size, output_dim).
        convs_out = [
            self.activation(conv(inputs)).squeeze(2) for conv in self.convs
        ]
        maxpool_out = [
            F.adaptive_max_pool1d(t, output_size=1).squeeze(2)
            for t in convs_out
        ]
        result = torch.concat(maxpool_out, axis=1)

        if self.projection_layer is not None:
            result = self.projection_layer(result)
        return result


class CNNModel(nn.Module):
    """This class implements the Convolution Neural Network model. At a high
    level, the model starts by embedding the tokens and running them through a
    word embedding. Then, we encode these epresentations with a `CNNEncoder`.
    The CNN has one convolution layer for each ngram filter size. Each
    convolution operation gives out a vector of size num_filter. The number of
    times a convolution layer will be used.

    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).
    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 num_filter=128,
                 ngram_filter_sizes=(3, ),
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size,
                                     emb_dim,
                                     padding_idx=padding_idx)
        self.encoder = CNNEncoder(emb_dim=emb_dim,
                                  num_filter=num_filter,
                                  ngram_filter_sizes=ngram_filter_sizes)
        self.fc = nn.Linear(self.encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, len(ngram_filter_sizes)*num_filter)
        encoder_out = self.encoder(embedded_text)
        encoder_out = torch.tanh(encoder_out)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = self.fc(encoder_out)
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits


class TextCNNModel(nn.Module):
    """This class implements the Text Convolution Neural Network model. At a
    high level, the model starts by embedding the tokens and running them
    through a word embedding. Then, we encode these epresentations with a
    `CNNEncoder`. The CNN has one convolution layer for each ngram filter size.
    Each convolution operation gives out a vector of size num_filter. The
    number of times a convolution layer will be used.

    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).
    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 num_filter=128,
                 ngram_filter_sizes=(1, 2, 3),
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size,
                                     emb_dim,
                                     padding_idx=padding_idx)
        self.encoder = CNNEncoder(emb_dim=emb_dim,
                                  num_filter=num_filter,
                                  ngram_filter_sizes=ngram_filter_sizes)
        self.fc = nn.Linear(self.encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, len(ngram_filter_sizes)*num_filter)
        encoder_out = self.encoder(embedded_text)
        encoder_out = torch.tanh(encoder_out)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = torch.tanh(self.fc(encoder_out))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits

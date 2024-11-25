import torch
import torch.nn as nn

__all__ = ['BoWEncoder']


class BoWEncoder(nn.Module):
    r"""
    A `BoWEncoder` takes as input a sequence of vectors and returns a
    single vector, which simply sums the embeddings of a sequence across the time dimension.
    The input to this encoder is of shape `(batch_size, num_tokens, emb_dim)`,
    and the output is of shape `(batch_size, emb_dim)`.

    Args:
        emb_dim(int):
            The dimension of each vector in the input sequence.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `BoWEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self.emb_dim

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `BoWEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self.emb_dim

    def forward(self, inputs, mask=None):
        r"""
        It simply sums the embeddings of a sequence across the time dimension.

        Args:
            inputs (Tensor):
                Shape as `(batch_size, num_tokens, emb_dim)` and dtype as `float32` or `float64`.
                The sequence length of the input sequence.
            mask (Tensor, optional):
                Shape same as `inputs`.
                Its each elements identify whether the corresponding input token is padding or not.
                If True, not padding token. If False, padding token.
                Defaults to `None`.

        Returns:
            Tensor:
                Returns tensor `summed`, the result vector of BagOfEmbedding.
                Its data type is same as `inputs` and its shape is `[batch_size, emb_dim]`.
        """
        if mask is not None:
            inputs = inputs * mask

        # Shape: (batch_size, embedding_dim)
        summed = inputs.sum(dim=1)
        return summed


class BoWModel(nn.Module):
    """This class implements the Bag of Words Classification Network model to
    classify texts.

    At a high level, the model starts by embedding the tokens and running them
    through a word embedding. Then, we encode these representations with a
    `BoWEncoder`. Lastly, we take the output of the encoder to create a final
    representation, which is passed through some feed-forward layers to output
    a logits (`output_layer`).
    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 hidden_size=128):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size,
                                     emb_dim,
                                     padding_idx=padding_idx)
        self.bow_encoder = BoWEncoder(emb_dim)
        self.fc1 = nn.Linear(self.bow_encoder.get_output_dim(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs, seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(inputs)

        # Shape: (batch_size, embedding_dim)
        summed = self.bow_encoder(embedded_text)
        encoded_text = torch.tanh(summed)

        # Shape: (batch_size, hidden_size)
        output = torch.tanh(self.fc1(encoded_text))
        # Shape: (batch_size, fc_hidden_size)
        output = torch.tanh(self.fc2(output))
        return output

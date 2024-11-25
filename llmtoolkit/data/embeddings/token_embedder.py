import math

import torch.nn as nn
from torch import Tensor


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedder(nn.Module):
    """Word Embedding layer of Transformer.

    This layer automatically constructs a 2D embedding matrix based on the
    input the size of vocabulary (`vocab_size`) and the size of each embedding
    vector (`emb_dim`). This layer lookups embeddings vector of ids provided
    by input `word`.

    After the embedding, those weights are multiplied by `sqrt(d_model)` which is
    `sqrt(emb_dim)` in the interface.

    .. math::

        Out = embedding(word) * sqrt(emb_dim)

    Args:
        vocab_size (int):
            The size of vocabulary.
        emb_dim (int):
            Dimensionality of each embedding vector.
        bos_id (int, optional):
            The start token id and also is used as padding id. Defaults to 0.
    """

    def __init__(self, vocab_size: int, emb_dim: int):
        super(TokenEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, tokens: Tensor):
        r"""
        Computes word embedding.

        Args:
            word (Tensor):
                The input ids which indicates the sequences' words with shape
                `[batch_size, sequence_length]` whose data type can be
                int or int64.

        Returns:
            Tensor:
                The (scaled) embedding tensor of shape
                `(batch_size, sequence_length, emb_dim)` whose data type can be
                float32 or float64.
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_dim)

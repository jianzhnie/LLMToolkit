import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """This layer produces sinusoidal positional embeddings of any length.
    While in `forward()` method, this layer lookups embeddings vector of ids
    provided by input `pos`.

    Args:
        emb_dim (int):
            The size of each embedding vector.
        max_length (int):
            The maximum length of sequences.
    """

    def __init__(self, emb_dim, max_length):
        super(PositionalEmbedding, self).__init__()
        self.emb_dim = emb_dim

        self.pos_encoder = nn.Embedding(num_embeddings=max_length,
                                        embedding_dim=emb_dim)

    def forward(self, pos):
        r"""
        Computes positional embedding.

        Args:
            pos (Tensor):
                The input position ids with shape `[batch_size, sequence_length]` whose
                data type can be int or int64.

        Returns:
            Tensor:
                The positional embedding tensor of shape
                `(batch_size, sequence_length, emb_dim)` whose data type can be
                float32 or float64.
        """
        pos_emb = self.pos_encoder(pos)
        return pos_emb

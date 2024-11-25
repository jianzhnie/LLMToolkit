import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.

    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/embed_size))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/embed_size))
        \text{where pos is the word position and i is the embed idx)

    Args:
        embed_size: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).

    Examples:
        >>> pos_encoder = PositionalEncoding(embed_size)
    """

    def __init__(self,
                 embed_size: int = 512,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        if embed_size % 2 != 0:
            raise ValueError(
                'Cannot use sin/cos postional encoding with odd dim (got dim ={:d}'
                .format(embed_size))

        self.embed_size = embed_size
        self.dropout = nn.Dropout(p=dropout)

        # torch.Size([max_len, 1])
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        # torch.Size([embed_size//2])
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() *
            (-math.log(10000.0) / embed_size))

        # torch.Size([max_len, embed_size])
        pos_embedding = torch.zeros(max_len, embed_size)
        # 偶数位置编码
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置编码
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        # [max_len, embed_size] ===> [max_len, 1, embed_size]
        pos_embedding = pos_embedding.unsqueeze(0).transpose(0, 1)
        # 不对位置编码求梯度
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        """
        Args:
            token_embedding: Tensor, shape  [seq_len, batch_size, embedding_dim]
        """
        # 输入的词向量与位置编码相加
        pos_embed = token_embedding + self.pos_embedding[:token_embedding.
                                                         size(0), :]
        return self.dropout(pos_embed)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(15, 10))
    vocab_size = 1000
    batch_size = 32
    seq_len = 512
    embed_size = 128
    drop_out = 0
    max_len = 5000
    pe = PositionalEncoding(embed_size=embed_size, dropout=drop_out)
    x = torch.from_numpy(
        np.random.randint(1,
                          vocab_size,
                          size=(batch_size, seq_len, embed_size)))
    print(x.shape)
    x = x.transpose(0, 1)
    print(x.shape)
    y = pe.forward(x)
    print(y.shape)

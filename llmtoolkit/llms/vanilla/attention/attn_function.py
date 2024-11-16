import math
from typing import Optional

import torch
import torch.nn as nn

from llmtoolkit.losses.mask_softmax import masked_softmax


def transpose_qkv(inputs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Transpose input tensor inputs for multi-head attention.

    Args:
        inputs (torch.Tensor): Input tensor.
        num_heads (int): Number of attention heads.

    Returns:
        torch.Tensor: Transposed tensor.
    """
    # inputs shape: (batch_size, num_queries or num_key_value_pairs, num_hiddens)
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], num_heads, -1)
    # inputs shape: (batch_size, num_queries or num_key_value_pairs, num_heads, num_hiddens / num_heads)
    inputs = inputs.permute(0, 2, 1, 3)
    # inputs shape: (batch_size, num_heads, num_queries or num_key_value_pairs, num_hiddens / num_heads)
    inputs = inputs.reshape(-1, inputs.shape[2], inputs.shape[3])
    # inputs shape: (batch_size * num_heads, num_queries or num_key_value_pairs, num_hiddens / num_heads)
    return inputs


def transpose_output(inputs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Transpose and reshape output tensor from multi-head attention.

    Args:
        inputs (torch.Tensor): Output tensor.
        num_heads (int): Number of attention heads.

    Returns:
        torch.Tensor: Transposed and reshaped tensor.
    """
    # inputs shape: (batch_size * num_heads, num_queries or num_key_value_pairs, num_hiddens / num_heads)
    inputs = inputs.reshape(-1, num_heads, inputs.shape[1], inputs.shape[2])
    # inputs shape: (batch_size, num_heads, num_queries or num_key_value_pairs, num_hiddens / num_heads)
    inputs = inputs.permute(0, 2, 1, 3)
    # inputs shape: (batch_size, num_queries or num_key_value_pairs, num_heads, num_hiddens / num_heads)
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
    # inputs shape: (batch_size, num_queries or num_key_value_pairs, num_hiddens)
    return inputs


class AdditiveAttention(nn.Module):
    """Additive Attention mechanism.

    Args:
        key_size (int): Size of the key vectors.
        query_size (int): Size of the query vectors.
        num_hiddens (int): Number of hidden units in the attention mechanism.
        dropout (float): Dropout probability for regularization.


    Methods:
        forward(queries, keys, values, valid_lens):
            Perform additive attention and return the attention-weighted values.
    """

    def __init__(self, key_size: int, query_size: int, num_hiddens: int,
                 dropout: float):
        """Initialize the AdditiveAttention module.

        Args:
            key_size (int): Size of the key vectors.
            query_size (int): Size of the query vectors.
            num_hiddens (int): Number of hidden units in the attention mechanism.
            dropout (float): Dropout probability for regularization.
        """
        super(AdditiveAttention, self).__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute additive attention.

        Args:
            queries (torch.Tensor): The query tensor of shape (batch_size, num_queries, d).
            keys (torch.Tensor): The key tensor of shape (batch_size, num_key_value_pairs, d).
            values (torch.Tensor): The value tensor of shape (batch_size, num_key_value_pairs, value_dimension).
            valid_lens (Optional[torch.Tensor]): An optional tensor of shape (batch_size,) or (batch_size, num_queries).

        Returns:
            Tensor: The attention-weighted output tensor.
        """
        queries, keys = self.W_q(queries), self.W_k(keys)

        # queries shape: (batch_size, num_queries, num_hiddens)
        # keys shape: (batch_size, num_key_value_pairs, num_hiddens)
        # Broadcast the queries and keys to calculate the attention scores
        # queries shape: (batch_size, num_queries, 1, num_hiddens)
        # keys shape: (batch_size, 1, num_key_value_pairs, num_hiddens)
        # features shape: (batch_size, num_queries, num_key_value_pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # Calculate attention scores and apply masking if valid_lens is provided
        # scores shape: (batch_size, num_queries, num_key_value_pairs, 1)
        # scores shape: (batch_size, num_queries, num_key_value_pairs)
        scores = self.w_v(features).squeeze(-1)
        # Calculate the attention-weighted values
        p_attn = masked_softmax(scores, valid_lens)
        # Apply dropout to the attention weights
        p_attn = self.dropout(p_attn)
        # output shape: (batch_size, num_queries, value_dimension)
        output = torch.bmm(p_attn, values)
        self.attention_weights = p_attn
        return output


class DotProductAttention(nn.Module):
    """Scaled Dot Product Attention.

    Args:
        dropout (float): Dropout probability for regularization.

    Methods:
        forward(queries, keys, values, valid_lens=None):
            Perform scaled dot product attention and return the attention-weighted values.
    """

    def __init__(self, dropout: float):
        """Initialize the DotProductAttention module.

        Args:
            dropout (float): Dropout probability for regularization.
        """
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            queries (torch.Tensor): The query tensor of shape (batch_size, num_queries, d).
            keys (torch.Tensor): The key tensor of shape (batch_size, num_key_value_pairs, d).
            values (torch.Tensor): The value tensor of shape (batch_size, num_key_value_pairs, value_dimension).
            valid_lens (Optional[torch.Tensor]): An optional tensor of shape (batch_size,) or (batch_size, num_queries).

        Returns:
            Tensor: The attention-weighted output tensor.
        """
        d = queries.shape[-1]

        # Compute attention scores using dot product
        # quries shape: (batch_size, num_queries, d)
        # keys shape: (batch_size, num_key_value_pairs, d)
        # scores shape: (batch_size, num_queries, num_key_value_pairs)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        # Calculate attention weights and apply dropout
        p_attn = masked_softmax(scores, valid_lens)
        p_attn = self.dropout(p_attn)

        # Calculate the attention-weighted values
        output = torch.bmm(p_attn, values)
        # outputs: (batch_size, num_queries, value_dimension)
        return output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer.

    1. 为了避免计算代价和参数代价的大幅增长， 我们设定 𝑝_𝑞=𝑝_𝑘=𝑝_𝑣=𝑝_𝑜/ℎ$ 。
    2. 值得注意的是，如果我们将查询、键和值的线性变换的输出数量设置为  𝑝_𝑞ℎ=𝑝_𝑘ℎ=𝑝_𝑣ℎ=𝑝_𝑜 ， 则可以并行计算 ℎ 个头。
    3. 在下面的实现中，𝑝_𝑜 是通过参数num_hiddens指定的。

    Args:
        key_size (int): Size of the key vectors.
        query_size (int): Size of the query vectors.
        value_size (int): Size of the value vectors.
        num_hiddens (int): Size of the hidden vectors.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability for attention scores.
        bias (bool, optional): Whether to include bias terms in linear transformations.
    """

    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        num_heads: int,
        dropout: float,
        bias: Optional[bool] = False,
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the multi-head attention layer.

        Args:
            queries (torch.Tensor): Query vectors. Shape: [batch_size, num_queries, query_size]
            keys (torch.Tensor): Key vectors.  Shape: [batch_size, num_key_value_pairs, key_size]
            values (torch.Tensor): Value vectors.  Shape: [batch_size, num_key_value_pairs, value_size]
            valid_lens (torch.Tensor, optional): Valid sequence lengths for masking. Shape: [batch_size,]

        Returns:
            torch.Tensor: Output of the multi-head attention layer.
        """
        # Linear transformations for queries, keys, and values
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        # queries shape: (batch_size * num_heads, num_queries, num_hiddens / num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        # keys shape: (batch_size * num_heads, num_key_value_pairs, num_hiddens / num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # values shape: (batch_size * num_heads, num_key_value_pairs, num_hiddens / num_heads)

        if valid_lens is not None:
            # Repeat valid_lens to match the shape of transformed queries, keys, and values
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        # (batch_size * num_heads, num_queries or num_key_value_pairs, num_hiddens / num_heads)
        output = transpose_output(output, self.num_heads)
        # output shape: (batch_size, num_queries, num_hiddens)
        output = self.W_o(output)
        # output shape: (batch_size, num_queries, num_hiddens)
        return output


if __name__ == '__main__':
    queries = torch.normal(0, 1, (2, 1, 20))
    keys = torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10,
                                                           4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    attention1 = AdditiveAttention(key_size=2,
                                   query_size=20,
                                   num_hiddens=8,
                                   dropout=0)
    res = attention1(queries, keys, values, valid_lens)
    print(res.shape)  # torch.Size([2, 1, 4])

    queries = torch.normal(0, 1, (2, 1, 2))
    attention2 = DotProductAttention(dropout=0)
    results = attention2(queries, keys, values, valid_lens)
    print(results.shape)

    # D2l.ai  MultiHeadAttentionD2L
    num_hiddens, num_heads = 100, 5
    attention3 = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                    num_hiddens, num_heads, 0.5)
    batch_size = 2
    num_queries = 4
    num_kvpairs = 6
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    res = attention3(X, Y, Y, valid_lens)
    print(res.shape)

import math
from typing import Callable, List, Optional, Tuple

import packaging.version
import torch
import torch.nn as nn
from torch import Tensor

# Check PyTorch version for flex_attention availability
TORCH_VERSION = packaging.version.parse(torch.__version__)
REQUIRED_VERSION = packaging.version.parse(
    '2.2.0')  # Minimum version for flex_attention

if TORCH_VERSION >= REQUIRED_VERSION:
    from torch.nn.attention.flex_attention import (create_block_mask,
                                                   flex_attention)
else:
    raise ImportError(
        f'flex_attention requires PyTorch >= 2.2.0, but found version {torch.__version__}'
    )
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


class MultiHeadAttentionV0(nn.Module):
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
        super().__init__()
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


class CausalAttention(nn.Module):
    """Single-head causal attention implementation.

    Attributes:
        d_out (int): Output dimension for attention head
        W_query (nn.Linear): Query projection matrix
        W_key (nn.Linear): Key projection matrix
        W_value (nn.Linear): Value projection matrix
        dropout (nn.Dropout): Dropout layer for attention weights
        mask (Tensor): Causal mask to prevent attending to future tokens
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize the causal attention module.

        Args:
            d_in: Input dimension
            d_out: Output dimension
            context_length: Maximum sequence length
            dropout: Dropout probability for attention weights
            qkv_bias: Whether to include bias terms in QKV projections
        """
        super().__init__()

        self.d_out = d_out
        self.scale = 1.0 / math.sqrt(
            d_out)  # Scaling factor for attention scores

        # QKV projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Create causal mask: upper triangular matrix of ones
        mask = torch.triu(torch.ones(context_length, context_length),
                          diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x: Tensor) -> Tensor:
        """Compute causal attention over input sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Tensor of shape (batch_size, seq_len, d_out) containing attended values
        """
        batch_size, seq_len, _ = x.shape

        # Compute QKV projections
        queries = self.W_query(x)  # (batch_size, seq_len, d_out)
        keys = self.W_key(x)  # (batch_size, seq_len, d_out)
        values = self.W_value(x)  # (batch_size, seq_len, d_out)

        # Compute scaled dot-product attention scores
        attn_scores = torch.bmm(queries, keys.transpose(
            1, 2))  # (batch_size, seq_len, seq_len)
        attn_scores = attn_scores * self.scale

        # Apply causal mask
        attn_scores.masked_fill_(self.mask[:seq_len, :seq_len].bool(),
                                 float('-inf'))

        # Compute attention weights with softmax
        attn_weights = torch.softmax(attn_scores,
                                     dim=-1)  # (batch_size, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # Compute weighted sum of values
        seq_vec = torch.bmm(attn_weights,
                            values)  # (batch_size, seq_len, d_out)

        return seq_vec


class MultiHeadAttentionV1(nn.Module):
    """Multi-head attention implementation that uses multiple parallel
    attention heads.

    Attributes:
        heads (nn.ModuleList): List of attention heads
        out_proj (nn.Linear): Output projection layer
        num_heads (int): Number of attention heads
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize multi-head attention module.

        Args:
            d_in: Input dimension
            d_out: Output dimension per head
            context_length: Maximum sequence length
            num_heads: Number of attention heads
            dropout: Dropout probability
            qkv_bias: Whether to include bias in QKV projections
        """
        super().__init__()

        self.num_heads = num_heads
        head_dim = d_out

        # Create parallel attention heads
        self.heads = nn.ModuleList([
            CausalAttention(
                d_in=d_in,
                d_out=head_dim,
                context_length=context_length,
                dropout=dropout,
                qkv_bias=qkv_bias,
            ) for _ in range(num_heads)
        ])

        # Output projection
        self.out_proj = nn.Linear(head_dim * num_heads, d_in)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize the output projection using Xavier uniform
        initialization."""
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Compute multi-head attention over input sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Tensor of shape (batch_size, seq_len, d_in) containing attended values
        """
        # Process each attention head in parallel
        head_outputs: List[Tensor] = [head(x) for head in self.heads]

        # Concatenate head outputs along feature dimension
        multi_head_out = torch.cat(head_outputs, dim=-1)

        # Project back to input dimension
        return self.out_proj(multi_head_out)


class MultiHeadAttentionV2(nn.Module):
    """Memory-efficient implementation of multi-head causal attention.

    This implementation processes all attention heads in parallel by cleverly
    reshaping the input tensors, resulting in better GPU utilization and
    faster computation compared to processing heads separately.

    Attributes:
        d_out (int): Total output dimension across all heads
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        scale (float): Scaling factor for attention scores
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize the multi-head attention module.

        Args:
            d_in: Input embedding dimension
            d_out: Total output dimension (must be divisible by num_heads)
            seq_len: Maximum sequence length
            num_heads: Number of attention heads
            dropout: Dropout probability
            qkv_bias: Whether to include bias terms in QKV projections

        Raises:
            AssertionError: If d_out is not divisible by num_heads
            ValueError: If any dimension is invalid
        """
        super().__init__()

        if d_out % num_heads != 0:
            raise ValueError(
                f'Output dimension ({d_out}) must be divisible by num_heads ({num_heads})'
            )

        if d_out <= 0 or num_heads <= 0:
            raise ValueError(
                f'Invalid dimensions: d_out={d_out}, num_heads={num_heads}')

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # QKV projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Create causal mask
        mask = torch.triu(torch.ones(context_length, context_length),
                          diagonal=1)
        self.register_buffer('mask', mask)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize the parameters using Xavier uniform initialization."""
        gain = 1.0 / math.sqrt(2.0)

        # Initialize QKV projections
        nn.init.xavier_uniform_(self.W_query.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_key.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_value.weight, gain=gain)

        # Initialize output projection
        nn.init.xavier_uniform_(self.out_proj.weight)

        # Initialize biases if present
        if self.W_query.bias is not None:
            nn.init.zeros_(self.W_query.bias)
        if self.W_key.bias is not None:
            nn.init.zeros_(self.W_key.bias)
        if self.W_value.bias is not None:
            nn.init.zeros_(self.W_value.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _reshape_for_attention(self, x: Tensor, batch_size: int,
                               seq_len: int) -> Tensor:
        """Reshape input tensor for parallel attention computation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_out)
            batch_size: Batch size
            seq_len: Number of tokens

        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        return x.view(batch_size, seq_len, self.num_heads,
                      self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Compute multi-head attention over input sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Tensor of shape (batch_size, seq_len, d_out) containing attended values

        Raises:
            RuntimeError: If input tensor has incorrect shape
        """
        if x.dim() != 3:
            raise RuntimeError(
                f'Expected 3D tensor (batch_size, seq_len, d_in), got shape {x.shape}'
            )

        batch_size, seq_len, d_in = x.shape

        # Compute QKV projections
        queries = self.W_query(x)  # (batch_size, seq_len, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape for parallel attention computation
        queries = self._reshape_for_attention(queries, batch_size, seq_len)
        keys = self._reshape_for_attention(keys, batch_size, seq_len)
        values = self._reshape_for_attention(values, batch_size, seq_len)

        # Compute scaled dot-product attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attn_scores = attn_scores * self.scale

        # Apply causal mask
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attn_scores.masked_fill_(mask_bool, float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context_vec = torch.matmul(attn_weights, values)

        # Reshape back to original dimensions
        context_vec = (context_vec.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_out))

        # Apply output projection
        return self.out_proj(context_vec)

    def extra_repr(self) -> str:
        """Return a string with extra representation information."""
        return (f'd_out={self.d_out}, num_heads={self.num_heads}, '
                f'head_dim={self.head_dim}, dropout={self.dropout.p}')


class MultiHeadAttentionCombinedQKV(nn.Module):
    """Multi-Head Attention module with combined QKV projection.

    This implementation follows the "Attention is All You Need" paper but combines
    the Q, K, V projections into a single linear transformation for efficiency.

    Attributes:
        num_heads (int): Number of attention heads
        seq_len (int): Maximum sequence length
        head_dim (int): Dimension of each attention head
        qkv (nn.Linear): Combined projection for queries, keys, and values
        proj (nn.Linear): Output projection
        dropout (nn.Dropout): Dropout layer
        mask (torch.Tensor): Causal attention mask
    """

    def __init__(
            self,
            d_in: int,  # Input dimension
            d_out: int,  # Output dimension
            num_heads: int,  # Number of attention heads
            context_length: int,  # Maximum sequence length
            dropout: float = 0.0,  # Dropout probability
            qkv_bias: bool = False,  # Whether to use bias in QKV projection
    ) -> None:
        """Initialize the Multi-Head Attention module.

        Args:
            config (MultiHeadAttentionConfig): Configuration object containing all parameters

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()

        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.scale = self.head_dim**-0.5  # Scaling factor for attention scores

        # Combined projection for Q, K, V
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Create causal attention mask
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multi-Head Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Combined QKV projection
        qkv = self.qkv(x)  # Shape: (b, seq_len, 3 * embed_dim)

        # Reshape and separate Q, K, V
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # Shape: (3, batch_size, num_heads, seq_len, head_dim)
        queries, keys, values = qkv.unbind(0)

        # Compute attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        # Shape: (b, num_heads, seq_len, seq_len)

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:seq_len, :seq_len], float('-inf'))

        # Compute attention weights with scaled dot product
        attn_weights = torch.softmax(attn_scores * self.scale, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context_vec = torch.matmul(attn_weights, values)
        # Shape: (b, num_heads, seq_len, head_dim)

        # Reshape and project output
        context_vec = context_vec.transpose(1, 2).contiguous()
        # Shape: (b, seq_len, num_heads, head_dim)
        context_vec = context_vec.view(batch_size, seq_len, embed_dim)
        # Shape: (b, seq_len, embed_dim)

        return self.proj(context_vec)


class MultiHeadAttentionEinsum(nn.Module):
    """Multi-Head Attention implementation using Einstein summation (einsum)
    operations.

    This implementation separates Q, K, V projections and uses einsum for efficient
    matrix multiplication. It includes proper initialization and optional bias terms.

    Mathematical formulation:
    - Q = x @ W_q + b_q
    - K = x @ W_k + b_k
    - V = x @ W_v + b_v
    - Attention(Q, K, V) = softmax(QK^T/√d_k)V

    Attributes:
        d_out (int): Output dimension
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        W_query (nn.Parameter): Query projection weight matrix
        W_key (nn.Parameter): Key projection weight matrix
        W_value (nn.Parameter): Value projection weight matrix
        bias_q (Optional[nn.Parameter]): Query projection bias
        bias_k (Optional[nn.Parameter]): Key projection bias
        bias_v (Optional[nn.Parameter]): Value projection bias
        out_proj (nn.Linear): Output projection layer
        dropout (nn.Dropout): Dropout layer
        mask (torch.Tensor): Causal attention mask
    """

    def __init__(
            self,
            d_in: int,  # Input dimension
            d_out: int,  # Output dimension
            context_length: int,  # Maximum sequence length
            num_heads: int,  # Number of attention heads
            dropout: float,  # Dropout probability
            qkv_bias: bool = False,  # Whether to use bias in Q, K, V projections
    ) -> None:
        """Initialize the Multi-Head Attention module with einsum
        implementation.

        Args:
            config (MHAConfig): Configuration object containing all parameters

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()
        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.scale = self.head_dim**-0.5  # Pre-compute scaling factor

        # Initialize projection matrices
        self.W_query = nn.Parameter(torch.empty(d_out, d_in))
        self.W_key = nn.Parameter(torch.empty(d_out, d_in))
        self.W_value = nn.Parameter(torch.empty(d_out, d_in))

        # Initialize biases if requested
        if qkv_bias:
            self.bias_q = nn.Parameter(torch.empty(d_out))
            self.bias_k = nn.Parameter(torch.empty(d_out))
            self.bias_v = nn.Parameter(torch.empty(d_out))
        else:
            self.register_parameter('bias_q', None)
            self.register_parameter('bias_k', None)
            self.register_parameter('bias_v', None)

        # Output projection and dropout
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Create causal mask
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the parameters using Kaiming initialization."""
        # Initialize weights
        for weight in [self.W_query, self.W_key, self.W_value]:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # Initialize biases if they exist
        if self.bias_q is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_query)
            bound = 1 / math.sqrt(fan_in)
            for bias in [self.bias_q, self.bias_k, self.bias_v]:
                nn.init.uniform_(bias, -bound, bound)

    def _compute_qkv(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Query, Key, and Value matrices using einsum.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Q, K, V tensors
        """
        # Calculate Q, K, V using einsum
        Q = torch.einsum('bnd,di->bni', x, self.W_query)
        K = torch.einsum('bnd,di->bni', x, self.W_key)
        V = torch.einsum('bnd,di->bni', x, self.W_value)

        # Add biases if they exist
        if self.bias_q is not None:
            Q = Q + self.bias_q
            K = K + self.bias_k
            V = V + self.bias_v

        return Q, K, V

    def _reshape_for_attention(self, tensor: torch.Tensor, batch_size: int,
                               seq_len: int) -> torch.Tensor:
        """Reshape tensor for multi-head attention computation.

        Args:
            tensor (torch.Tensor): Input tensor
            batch_size (int): Batch size
            seq_len (int): Number of tokens

        Returns:
            torch.Tensor: Reshaped tensor
        """
        return tensor.view(batch_size, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multi-Head Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        Q, K, V = self._compute_qkv(x)

        # Reshape for multi-head attention
        Q = self._reshape_for_attention(Q, batch_size, seq_len)
        K = self._reshape_for_attention(K, batch_size, seq_len)
        V = self._reshape_for_attention(V, batch_size, seq_len)

        # Compute scaled dot-product attention
        scores = torch.einsum('bhnd,bhmd->bhnm', Q, K) * self.scale

        # Apply causal mask
        scores = scores.masked_fill(
            self.mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(1).expand(
                batch_size, self.num_heads, seq_len, seq_len).bool(),
            float('-inf'),
        )

        # Apply attention
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = torch.einsum('bhnm,bhmd->bhnd', attn_weights, V)

        # Reshape and project output
        context_vec = (context_vec.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_out))

        return self.out_proj(context_vec)


class MultiHeadAttentionPyTorch(nn.Module):
    """Multi-Head Attention implementation using PyTorch's built-in
    scaled_dot_product_attention.

    This implementation leverages PyTorch's optimized attention computation which includes
    flash attention when available on supported hardware. It provides automatic causal
    masking and efficient memory usage.

    Features:
    - Combined QKV projection for efficiency
    - Automatic causal masking using is_causal flag
    - Flash attention support on compatible hardware
    - Conditional dropout during training

    Attributes:
        num_heads (int): Number of attention heads
        context_length (int): Maximum sequence length
        head_dim (int): Dimension of each attention head
        d_out (int): Output dimension
        qkv (nn.Linear): Combined projection for queries, keys, and values
        proj (nn.Linear): Output projection
        dropout (float): Dropout probability
    """

    def __init__(
            self,
            d_in: int,  # Input dimension
            d_out: int,  # Output dimension
            context_length: int,  # Maximum sequence length
            num_heads: int,  # Number of attention heads
            dropout: float,  # Dropout probability
            qkv_bias: bool = False,  # Whether to use bias in Q, K, V projections
    ) -> None:
        """Initialize the Multi-Head Attention module with PyTorch's
        implementation.

        Args:
            config (MHAConfig): Configuration object containing all parameters

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()

        assert d_out % num_heads == 0, (
            f'Output dimension ({d_out}) must be divisible by '
            f'number of heads ({num_heads})')

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.dropout = dropout

        # Combined QKV projection
        self.qkv = nn.Linear(in_features=d_in,
                             out_features=3 * d_out,
                             bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(in_features=d_out, out_features=d_out)

    def _split_qkv(
            self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the concatenated QKV tensor into separate Q, K, V tensors.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Separated Q, K, V tensors
        """
        batch_size, seq_len, _ = x.shape

        # Project and reshape QKV
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Separate Q, K, V
        return tuple(qkv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multi-Head Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out)
        """
        batch_size, seq_len, _ = x.shape

        # Get queries, keys, and values
        queries, keys, values = self._split_qkv(x)

        # Use training-aware dropout
        dropout_p = self.dropout if self.training else 0.0

        # Compute attention using PyTorch's optimized implementation
        context_vec = nn.functional.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            attn_mask=None,  # Not needed when is_causal=True
            dropout_p=dropout_p,
            is_causal=True,  # Enable automatic causal masking
        )

        # Reshape and project output
        context_vec = (context_vec.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_out))

        return self.proj(context_vec)


class MultiHeadAttentionPyTorchNoFlash(nn.Module):
    """Multi-Head Attention implementation using PyTorch's
    scaled_dot_product_attention with explicit mask handling for cases where
    flash attention is not available.

    This implementation uses an explicit attention mask instead of the is_causal flag,
    making it suitable for environments where flash attention is not supported or
    when more control over the attention mask is needed.

    Features:
    - Combined QKV projection for efficiency
    - Explicit causal masking
    - Automatic mask size adjustment for variable sequence lengths
    - Conditional dropout during training

    Attributes:
        num_heads (int): Number of attention heads
        context_length (int): Maximum sequence length
        head_dim (int): Dimension of each attention head
        d_out (int): Output dimension
        qkv (nn.Linear): Combined projection for queries, keys, and values
        proj (nn.Linear): Output projection
        dropout (float): Dropout probability
        mask (torch.Tensor): Causal attention mask
    """

    def __init__(
            self,
            d_in: int,  # Input dimension
            d_out: int,  # Output dimension
            num_heads: int,  # Number of attention heads
            context_length: int,  # Maximum sequence length
            dropout: float = 0.0,  # Dropout probability
            qkv_bias: bool = False,  # Whether to use bias in QKV projection
    ) -> None:
        """Initialize the Multi-Head Attention module with explicit mask
        handling.

        Args:
            config (MHAConfig): Configuration object containing all parameters

        Raises:
            AssertionError: If d_out is not divisible by num_heads
        """
        super().__init__()

        assert d_out % num_heads == 0, (
            f'Output dimension ({d_out}) must be divisible by '
            f'number of heads ({num_heads})')

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.dropout = dropout

        # Combined QKV projection
        self.qkv = nn.Linear(in_features=d_in,
                             out_features=3 * d_out,
                             bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(in_features=d_out, out_features=d_out)

        # Create and register causal mask
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1).bool(),
        )

    def _split_qkv(
            self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the concatenated QKV tensor into separate Q, K, V tensors.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Separated Q, K, V tensors
        """
        batch_size, seq_len, _ = x.shape

        # Project and reshape QKV
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Separate Q, K, V
        return tuple(qkv)

    def _get_attention_mask(self, seq_len: int) -> torch.Tensor:
        """Get the appropriate attention mask for the current sequence length.

        Args:
            seq_len (int): Current sequence length

        Returns:
            torch.Tensor: Boolean attention mask
        """
        if self.context_length >= seq_len:
            return self.mask[:seq_len, :seq_len]
        return self.mask[:self.context_length, :self.context_length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multi-Head Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out)
        """
        batch_size, seq_len, _ = x.shape

        # Get queries, keys, and values
        queries, keys, values = self._split_qkv(x)

        # Use training-aware dropout
        dropout_p = self.dropout if self.training else 0.0

        # Get appropriate attention mask
        attn_mask = self._get_attention_mask(seq_len)

        # Compute attention using PyTorch's implementation
        context_vec = nn.functional.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,  # Using explicit mask instead
        )

        # Reshape and project output
        context_vec = (context_vec.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_out))

        return self.proj(context_vec)


class MHAPyTorchClass(nn.Module):
    """Enhanced Multi-Head Attention implementation with causal masking.

    This module implements multi-head attention with:
    - Configurable input/output dimensions
    - Causal masking for auto-regressive models
    - Optional layer normalization
    - Proper shape validation
    - Optional residual connections

    Args:
        d_in (int): Input dimension
        d_out (int): Output dimension
        num_heads (int): Number of attention heads
        context_length (int): Maximum sequence length
        dropout (float, optional): Dropout probability. Defaults to 0.0
        qkv_bias (bool, optional): Enable bias for Q,K,V projections. Defaults to False
        need_weights (bool, optional): Return attention weights. Defaults to True
        use_layer_norm (bool, optional): Apply layer normalization. Defaults to True
        use_residual (bool, optional): Add residual connection. Defaults to True
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        need_weights: bool = True,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        if d_out % num_heads != 0:
            raise ValueError(
                f'd_out ({d_out}) must be divisible by num_heads ({num_heads})'
            )

        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.need_weights = need_weights
        self.use_residual = use_residual

        # Input projection if dimensions differ
        self.input_proj = nn.Linear(d_in,
                                    d_out) if d_in != d_out else nn.Identity()

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(
            d_out) if use_layer_norm else nn.Identity()

        # Output projection
        self.proj = nn.Linear(d_out, d_out)

        # Register causal mask buffer
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1).bool(),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in)
            key_padding_mask (Optional[torch.Tensor]): Mask for padded tokens

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output tensor and optional attention weights
        """
        batch_size, seq_len, _ = x.shape

        if seq_len > self.context_length:
            raise ValueError(
                f'Input sequence length ({seq_len}) exceeds maximum context length ({self.context_length})'
            )

        # Project input if dimensions differ
        x = self.input_proj(x)

        # Prepare causal mask for current sequence length
        attn_mask = self.mask[:seq_len, :seq_len]

        # Apply multi-head attention
        attn_output, attn_weights = self.multihead_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.need_weights,
        )

        # Apply output projection
        output = self.proj(attn_output)

        # Add residual connection if enabled
        if self.use_residual:
            output = output + x

        # Apply layer normalization
        output = self.layer_norm(output)

        return output, attn_weights if self.need_weights else None


def create_causal_mask(
    batch_size: Optional[int],
    num_heads: Optional[int],
    query_length: int,
    key_length: int,
) -> Callable:
    """Creates a causal attention mask function.

    Args:
        batch_size: Optional batch size (can be None for dynamic batching)
        num_heads: Optional number of attention heads (can be None)
        query_length: Maximum query sequence length
        key_length: Maximum key sequence length

    Returns:
        Callable: Function that generates causal masks
    """

    def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        """Determines if position q_idx can attend to position kv_idx.

        Args:
            b: Batch index
            h: Head index
            q_idx: Query position index
            kv_idx: Key/Value position index

        Returns:
            bool: True if attention is allowed, False otherwise
        """
        return q_idx >= kv_idx

    return causal_mask


class FlexMultiHeadAttention(nn.Module):
    """Multi-Head Attention implementation using PyTorch's flex_attention.

    This implementation uses the experimental flex_attention module for potentially
    better performance on certain hardware configurations.

    Args:
        d_in (int): Input dimension
        d_out (int): Output dimension (must be divisible by num_heads)
        num_heads (int): Number of attention heads
        context_length (int): Maximum sequence length
        dropout (float, optional): Dropout probability. Defaults to 0.0
        qkv_bias (bool, optional): Use bias in QKV projections. Defaults to False

    Raises:
        ValueError: If d_out is not divisible by num_heads
        ImportError: If PyTorch version doesn't support flex_attention
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        if d_out % num_heads != 0:
            raise ValueError(
                f'Output dimension ({d_out}) must be divisible by '
                f'number of heads ({num_heads})')

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.dropout = dropout

        # QKV projection
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(d_out, d_out)

        # Create causal attention mask
        # Note: create_block_mask doesn't support buffer registration yet
        self.block_mask = create_block_mask(
            create_causal_mask(None, None, context_length, context_length),
            B=None,
            H=None,
            Q_LEN=context_length,
            KV_LEN=context_length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the flex attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out)

        Raises:
            ValueError: If input sequence length exceeds context_length
        """
        batch_size, seq_length, _ = x.shape

        # Validate sequence length
        if seq_length > self.context_length:
            raise ValueError(
                f'Input sequence length ({seq_length}) exceeds maximum '
                f'context length ({self.context_length})')

        # Project input to Q, K, V
        # Shape: (batch_size, seq_length, 3 * d_out)
        qkv = self.qkv(x)

        # Reshape to separate heads
        # Shape: (batch_size, seq_length, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads,
                       self.head_dim)

        # Permute to get separate Q, K, V tensors
        # Shape: (3, batch_size, num_heads, seq_length, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv

        # Get appropriate attention mask for current sequence length
        if self.context_length >= seq_length:
            attn_mask = self.block_mask[:seq_length, :seq_length]
        else:
            attn_mask = self.block_mask[:self.context_length, :self.
                                        context_length]

        # Apply flex attention
        # Shape: (batch_size, num_heads, seq_length, head_dim)
        context_vec = flex_attention(
            queries,
            keys,
            values,
            block_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Combine heads
        # Shape: (batch_size, seq_length, d_out)
        context_vec = (context_vec.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_out))

        # Final projection
        output = self.proj(context_vec)

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
    attention3 = MultiHeadAttentionV0(num_hiddens, num_hiddens, num_hiddens,
                                      num_hiddens, num_heads, 0.5)
    batch_size = 2
    num_queries = 4
    num_kvpairs = 6
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    res = attention3(X, Y, Y, valid_lens)
    print(res.shape)

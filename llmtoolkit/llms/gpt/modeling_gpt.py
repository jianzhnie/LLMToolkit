import math
import os
import sys
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import tiktoken
import torch
import torch.nn as nn
from torch import Tensor

sys.path.append(os.getcwd())
from llmtoolkit.llms.vanilla.attention import \
    MultiHeadAttentionV2 as MultiHeadAttention


class LayerNorm(nn.Module):
    """Custom Layer Normalization implementation.

    Applies Layer Normalization over the last dimension of the input.
    Computes mean and variance statistics across the last dimension only.

    Args:
        emb_dim (int): The dimension to normalize over (usually embedding dimension)
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5
        elementwise_affine (bool, optional): Whether to learn affine parameters. Defaults to True
        device (Union[torch.device, str], optional): Device to place parameters on. Defaults to None
        dtype (torch.dtype, optional): Data type of parameters. Defaults to None

    Shape:
        - Input: (*, emb_dim), where * represents any number of leading dimensions
        - Output: Same shape as input

    Examples:
        >>> layer = LayerNorm(768)
        >>> x = torch.randn(32, 64, 768)  # (batch_size, seq_len, emb_dim)
        >>> output = layer(x)
        >>> print(output.shape)
        torch.Size([32, 64, 768])
    """

    def __init__(
        self,
        emb_dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Union[torch.device, str, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            factory_kwargs = {'device': device, 'dtype': dtype}
            self.scale = nn.Parameter(torch.ones(emb_dim, **factory_kwargs))
            self.shift = nn.Parameter(torch.zeros(emb_dim, **factory_kwargs))
        else:
            self.register_parameter('scale', None)
            self.register_parameter('shift', None)

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization to the input.

        Args:
            x (Tensor): Input tensor of shape (*, emb_dim)

        Returns:
            Tensor: Normalized tensor of the same shape

        Raises:
            ValueError: If the last dimension of input doesn't match emb_dim
        """
        if x.size(-1) != self.emb_dim:
            raise ValueError(
                f'Expected last dimension of input to be {self.emb_dim}, '
                f'but got {x.size(-1)}')

        # Calculate mean and variance along last dimension
        # keepdim=True preserves the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False uses the simpler form of variance calculation
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the input
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation if enabled
        if self.elementwise_affine:
            return self.scale * norm_x + self.shift
        return norm_x

    def extra_repr(self) -> str:
        """Returns a string containing extra information about the module.

        This appears in str(module) and repr(module).
        """
        return (f'emb_dim={self.emb_dim}, '
                f'eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine}')

    def get_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate normalization statistics for the input.

        Useful for debugging or analyzing the normalization behavior.

        Args:
            x (Tensor): Input tensor of shape (*, emb_dim)

        Returns:
            Tuple[Tensor, Tensor]: Mean and variance tensors
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return mean, var


class GELUApproximation(str, Enum):
    """Enumeration of available GELU approximation methods."""

    TANH = 'tanh'
    SIGMOID = 'sigmoid'
    NONE = 'none'


class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function.

    This implementation provides multiple approximation methods for the GELU function:
    1. Tanh approximation (faster, slightly less accurate)
    2. Sigmoid approximation (fast, less accurate)
    3. Exact computation (slower, most accurate)

    The original paper: https://arxiv.org/abs/1606.08415

    Args:
        approximate (Union[str, GELUApproximation], optional): Approximation method.
            Choose from "tanh", "sigmoid", or "none". Defaults to "tanh".
        device (Union[torch.device, str, None], optional): Device to place constants on.
            Defaults to None.
        dtype (torch.dtype, optional): Data type for constants. Defaults to None.

    Examples:
        >>> gelu = GELU()
        >>> x = torch.randn(100)
        >>> output = gelu(x)

        # With different approximation methods:
        >>> gelu_exact = GELU(approximate="none")
        >>> gelu_sigmoid = GELU(approximate="sigmoid")
    """

    def __init__(
        self,
        approximate: Union[str, GELUApproximation] = 'tanh',
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        # Validate and store approximation method
        if isinstance(approximate, str):
            approximate = GELUApproximation(approximate.lower())
        self.approximate = approximate

        # Constants for tanh approximation
        if approximate == GELUApproximation.TANH:
            self.register_buffer(
                'tanh_const',
                torch.tensor(math.sqrt(2.0 / math.pi),
                             device=device,
                             dtype=dtype),
            )
            self.register_buffer(
                'tanh_scale', torch.tensor(0.044715,
                                           device=device,
                                           dtype=dtype))

        # Constants for sigmoid approximation
        elif approximate == GELUApproximation.SIGMOID:
            self.register_buffer(
                'sig_const',
                torch.tensor(1.702, device=device, dtype=dtype),  # ≈ sqrt(2/π)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU activation function.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Output tensor with GELU activation applied
        """
        if self.approximate == GELUApproximation.TANH:
            return self._tanh_approximation(x)
        elif self.approximate == GELUApproximation.SIGMOID:
            return self._sigmoid_approximation(x)
        else:
            return self._exact_gelu(x)

    def _tanh_approximation(self, x: Tensor) -> Tensor:
        """GELU using tanh approximation (faster).

        This is the standard approximation used in most implementations.
        """
        return (0.5 * x *
                (1.0 + torch.tanh(self.tanh_const *
                                  (x + self.tanh_scale * torch.pow(x, 3)))))

    def _sigmoid_approximation(self, x: Tensor) -> Tensor:
        """GELU using sigmoid approximation (fastest).

        This is a simpler approximation that's slightly less accurate but
        faster.
        """
        return x * torch.sigmoid(self.sig_const * x)

    def _exact_gelu(self, x: Tensor) -> Tensor:
        """Exact GELU computation (slower but most accurate)."""
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def extra_repr(self) -> str:
        """Returns a string containing extra information about the module."""
        return f'approximate="{self.approximate.value}"'


class FeedForward(nn.Module):
    """A feed-forward neural network module implementing a Multi-Layer
    Perceptron (MLP) with GELU activation, commonly used in transformer
    architectures.

    The network consists of two linear transformations with a GELU activation in between:
        x -> Linear -> GELU -> Linear -> output

    The first linear layer expands the dimension by a factor of 4, and the second layer
    projects it back to the original dimension.

    Args:
        config (Dict[str, Union[int, float]]): Configuration dictionary containing:
            - emb_dim (int): The embedding dimension/input dimension

    Attributes:
        layers (nn.Sequential): Sequential container of the feed-forward layers

    Raises:
        KeyError: If 'emb_dim' is not found in the config dictionary
        ValueError: If 'emb_dim' is not a positive integer
    """

    def __init__(self, config: Dict[str, Union[int, float]]) -> None:
        super().__init__()

        # Validate config parameters
        if 'emb_dim' not in config:
            raise KeyError("Config must contain 'emb_dim' key")

        emb_dim = config['emb_dim']
        if not isinstance(emb_dim, int) or emb_dim <= 0:
            raise ValueError("'emb_dim' must be a positive integer")

        # Compute the hidden dimension (4x expansion)
        hidden_dim = 4 * emb_dim

        # Build the feed-forward network
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            GELU(),  # Using nn.GELU() instead of custom GELU class
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the feed-forward network.

        Args:
            x (Tensor): Input tensor of shape (..., emb_dim) where ... represents
                       any number of leading dimensions

        Returns:
            Tensor: Output tensor of shape (..., emb_dim), same shape as input

        Raises:
            RuntimeError: If input tensor's last dimension doesn't match emb_dim
        """
        # Validate input dimensions
        expected_dim = self.layers[0].in_features
        if x.size(-1) != expected_dim:
            raise RuntimeError(
                f"Expected input tensor's last dimension to be {expected_dim}, "
                f'but got {x.size(-1)}')

        return self.layers(x)

    def extra_repr(self) -> str:
        """Returns a string with extra information about the module.

        Returns:
            str: String containing input and output dimensions
        """
        in_features = self.layers[0].in_features
        return f'in_features={in_features}, out_features={in_features}'


class TransformerBlock(nn.Module):
    """A standard Transformer block implementing the architecture from
    "Attention Is All You Need" with pre-layer normalization, multi-head self-
    attention, and feed-forward networks.

    The block consists of two sub-layers:
    1. Multi-Head Self-Attention with pre-LayerNorm and residual connection
    2. Feed-Forward Network with pre-LayerNorm and residual connection

    Each sub-layer follows the pattern:
    LayerNorm -> Sublayer -> Dropout -> Residual Connection

    Args:
        config (Dict[str, Union[int, float, bool]]): Configuration dictionary containing:
            - emb_dim (int): Embedding dimension
            - context_length (int): Maximum sequence length
            - n_heads (int): Number of attention heads
            - drop_rate (float): Dropout probability
            - qkv_bias (bool): Whether to use bias in QKV projections

    Attributes:
        att (MultiHeadAttention): Multi-head self-attention layer
        ff (FeedForward): Feed-forward network layer
        norm1 (LayerNorm): Layer normalization before attention
        norm2 (LayerNorm): Layer normalization before feed-forward
        drop_shortcut (nn.Dropout): Dropout for residual connections

    Raises:
        KeyError: If required configuration parameters are missing
        ValueError: If parameter values are invalid
    """

    required_config_keys = {
        'emb_dim': int,
        'context_length': int,
        'n_heads': int,
        'drop_rate': float,
        'qkv_bias': bool,
    }

    def __init__(self, config: Dict[str, Union[int, float, bool]]) -> None:
        super().__init__()

        # Validate config parameters
        self._validate_config(config)

        # Initialize layers
        self.att = MultiHeadAttention(
            d_in=config['emb_dim'],
            d_out=config['emb_dim'],
            context_length=config['context_length'],
            num_heads=config['n_heads'],
            dropout=config['drop_rate'],
            qkv_bias=config['qkv_bias'],
        )

        self.ff = FeedForward(config)

        # Layer normalization layers (using eps=1e-5 as per standard practice)
        self.norm1 = LayerNorm(config['emb_dim'], eps=1e-5)
        self.norm2 = LayerNorm(config['emb_dim'], eps=1e-5)

        # Dropout for residual connections
        self.drop_shortcut = nn.Dropout(p=config['drop_rate'])

        # Save config for later reference
        self.config = config

    def _validate_config(self, config: Dict[str, Union[int, float,
                                                       bool]]) -> None:
        """Validates the configuration dictionary.

        Args:
            config: Configuration dictionary to validate

        Raises:
            KeyError: If required keys are missing
            ValueError: If parameter values are invalid
        """
        # Check for required keys
        for key, expected_type in self.required_config_keys.items():
            if key not in config:
                raise KeyError(f'Missing required config key: {key}')
            if not isinstance(config[key], expected_type):
                raise ValueError(
                    f"Config key '{key}' must be of type {expected_type}")

        # Validate specific parameter values
        if config['emb_dim'] <= 0:
            raise ValueError('emb_dim must be positive')
        if config['context_length'] <= 0:
            raise ValueError('context_length must be positive')
        if config['n_heads'] <= 0:
            raise ValueError('n_heads must be positive')
        if not 0 <= config['drop_rate'] <= 1:
            raise ValueError('drop_rate must be between 0 and 1')
        if config['emb_dim'] % config['n_heads'] != 0:
            raise ValueError('emb_dim must be divisible by n_heads')

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the Transformer block.

        Applies self-attention followed by a feed-forward network, with layer
        normalization, dropout, and residual connections.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_length, emb_dim]

        Returns:
            Tensor: Output tensor of same shape as input

        Raises:
            RuntimeError: If input tensor dimensions don't match configuration
        """
        # Validate input dimensions
        if x.size(-1) != self.config['emb_dim']:
            raise RuntimeError(
                f"Expected input dimension {self.config['emb_dim']}, "
                f'but got {x.size(-1)}')
        if x.size(1) > self.config['context_length']:
            raise RuntimeError(
                f'Input sequence length {x.size(1)} exceeds maximum '
                f"context length {self.config['context_length']}")

        # 1. Self-attention block with residual connection
        shortcut = x
        x = self.norm1(x)  # Pre-layer normalization
        x = self.att(x)  # Multi-head self-attention
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        # 2. Feed-forward block with residual connection
        shortcut = x
        x = self.norm2(x)  # Pre-layer normalization
        x = self.ff(x)  # Feed-forward network
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        return x

    def extra_repr(self) -> str:
        """Returns a string with extra information about the module.

        Returns:
            str: String containing key configuration parameters
        """
        return (f'emb_dim={self.config["emb_dim"]}, '
                f'n_heads={self.config["n_heads"]}, '
                f'context_length={self.config["context_length"]}, '
                f'drop_rate={self.config["drop_rate"]}, '
                f'qkv_bias={self.config["qkv_bias"]}')


class GPTModel(nn.Module):
    """A GPT-style autoregressive transformer model for language modeling.

    This implementation follows the architecture described in the GPT papers,
    using token embeddings, positional embeddings, and a stack of transformer blocks,
    followed by a final layer normalization and output projection.

    Architecture:
        1. Token Embeddings
        2. Positional Embeddings
        3. Embedding Dropout
        4. N Transformer Blocks
        5. Final Layer Normalization
        6. Output Projection to Vocabulary

    Args:
        config (Dict[str, Union[int, float, bool]]): Configuration dictionary containing:
            - vocab_size (int): Size of the vocabulary
            - emb_dim (int): Dimension of embeddings and hidden states
            - context_length (int): Maximum sequence length
            - n_layers (int): Number of transformer blocks
            - drop_rate (float): Dropout probability
            Additional parameters required by TransformerBlock

    Attributes:
        tok_emb (nn.Embedding): Token embedding layer
        pos_emb (nn.Embedding): Positional embedding layer
        drop_emb (nn.Dropout): Embedding dropout layer
        trf_blocks (nn.Sequential): Stack of transformer blocks
        final_norm (LayerNorm): Final layer normalization
        out_head (nn.Linear): Output projection to vocabulary

    Raises:
        KeyError: If required configuration parameters are missing
        ValueError: If parameter values are invalid
    """

    required_config_keys = {
        'vocab_size': int,
        'emb_dim': int,
        'context_length': int,
        'n_layers': int,
        'drop_rate': float,
    }

    def __init__(self, config: Dict[str, Union[int, float, bool]]) -> None:
        super().__init__()

        # Validate configuration
        self._validate_config(config)
        self.config = config

        # Token and position embeddings
        self.tok_emb = nn.Embedding(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['emb_dim'],
            padding_idx=None,  # Set if you have a padding token
        )

        self.pos_emb = nn.Embedding(num_embeddings=config['context_length'],
                                    embedding_dim=config['emb_dim'])

        # Initialize embeddings using scaled normal distribution
        self._init_embeddings()

        # Embedding dropout
        self.drop_emb = nn.Dropout(p=config['drop_rate'])

        # Stack of transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])])

        # Final layer norm and output projection
        self.final_norm = LayerNorm(config['emb_dim'], eps=1e-5)
        self.out_head = nn.Linear(
            in_features=config['emb_dim'],
            out_features=config['vocab_size'],
            bias=False,  # Following GPT-2 architecture
        )

        # Tie weights between token embedding and output projection
        self.out_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _validate_config(self, config: Dict[str, Union[int, float,
                                                       bool]]) -> None:
        """Validates the configuration dictionary.

        Args:
            config: Configuration dictionary to validate

        Raises:
            KeyError: If required keys are missing
            ValueError: If parameter values are invalid
        """
        # Check for required keys
        for key, expected_type in self.required_config_keys.items():
            if key not in config:
                raise KeyError(f'Missing required config key: {key}')
            if not isinstance(config[key], expected_type):
                raise ValueError(
                    f"Config key '{key}' must be of type {expected_type}")

        # Validate specific parameter values
        if config['vocab_size'] <= 0:
            raise ValueError('vocab_size must be positive')
        if config['emb_dim'] <= 0:
            raise ValueError('emb_dim must be positive')
        if config['context_length'] <= 0:
            raise ValueError('context_length must be positive')
        if config['n_layers'] <= 0:
            raise ValueError('n_layers must be positive')
        if not 0 <= config['drop_rate'] <= 1:
            raise ValueError('drop_rate must be between 0 and 1')

    def _init_embeddings(self) -> None:
        """Initialize embedding layers using scaled normal distribution."""
        std = 0.02  # Following GPT-2 initialization
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=std)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for linear layers and layer normalization.

        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            input_ids (Tensor): Token ids of shape [batch_size, seq_len]
            position_ids (Optional[Tensor]): Custom position ids of shape [batch_size, seq_len]
            attention_mask (Optional[Tensor]): Attention mask of shape [batch_size, seq_len]

        Returns:
            Tensor: Logits of shape [batch_size, seq_len, vocab_size]

        Raises:
            RuntimeError: If input dimensions don't match configuration
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Validate input dimensions
        if seq_len > self.config['context_length']:
            raise RuntimeError(
                f'Input sequence length {seq_len} exceeds maximum '
                f"context length {self.config['context_length']}")

        # Get token embeddings
        token_embeddings = self.tok_emb(
            input_ids)  # [batch_size, seq_len, emb_dim]

        # Get position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device)
        pos_embeddings = self.pos_emb(position_ids)  # [seq_len, emb_dim]

        # Combine embeddings and apply dropout
        x = token_embeddings + pos_embeddings  # Broadcasting handles batch dimension
        x = self.drop_emb(x)

        # Pass through transformer blocks
        x = self.trf_blocks(x)

        # Apply final normalization and output projection
        x = self.final_norm(x)
        logits = self.out_head(x)  # [batch_size, seq_len, vocab_size]

        return logits

    def extra_repr(self) -> str:
        """Returns a string with extra information about the module.

        Returns:
            str: String containing key configuration parameters
        """
        return (f'vocab_size={self.config["vocab_size"]}, '
                f'emb_dim={self.config["emb_dim"]}, '
                f'context_length={self.config["context_length"]}, '
                f'n_layers={self.config["n_layers"]}, '
                f'drop_rate={self.config["drop_rate"]}')

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """Generates text tokens autoregressively.

        Args:
            input_ids (Tensor): Starting token ids of shape [batch_size, seq_len]
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k (Optional[int]): Number of highest probability tokens to keep for top-k sampling

        Returns:
            Tensor: Generated token ids of shape [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            context = input_ids[:, -self.config['context_length']:]

            # Get predictions
            logits = self(context)
            logits = logits[:,
                            -1, :] / temperature  # Consider only the last token

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Get probabilities and sample
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class GPTConfig:
    """Configuration class for GPT model parameters.

    Provides preset configurations and validation.
    """

    # Pre-defined model configurations
    CONFIGS = {
        '124M': {
            'vocab_size': 50257,  # GPT-2 vocabulary size
            'context_length': 1024,  # Maximum sequence length
            'emb_dim': 768,  # Embedding dimension
            'n_heads': 12,  # Number of attention heads
            'n_layers': 12,  # Number of transformer layers
            'drop_rate': 0.1,  # Dropout probability
            'qkv_bias': False,  # Query-Key-Value projection bias
        }
        # Add other model sizes as needed (e.g., "355M", "774M", "1.4B")
    }

    @classmethod
    def get_config(
            cls,
            model_size: str = '124M') -> Dict[str, Union[int, float, bool]]:
        """Get configuration for specified model size.

        Args:
            model_size: Size of the model ("124M", "355M", etc.)

        Returns:
            Dictionary containing model configuration

        Raises:
            ValueError: If model_size is not recognized
        """
        if model_size not in cls.CONFIGS:
            raise ValueError(f'Unknown model size: {model_size}. '
                             f'Available sizes: {list(cls.CONFIGS.keys())}')
        return cls.CONFIGS[model_size].copy()


def main(seed=42) -> None:
    """Main function to demonstrate GPT model text generation."""
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # Get model configuration
    config = GPTConfig.get_config('124M')

    # Initialize model
    model = GPTModel(config)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    # Generation parameters
    prompt = 'Hello, I am'
    max_new_tokens = 50
    temperature = 0.7
    top_k = 50

    # Generate text
    # Encode input text
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded).unsqueeze(0).to(device)

    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode and return generated text
    generated_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    # Print results
    print('\nGenerated Text:')
    print('-' * 50)
    print(generated_text)
    print('-' * 50)


if __name__ == '__main__':
    main()

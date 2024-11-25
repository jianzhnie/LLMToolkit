import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Transformer(nn.Module):
    """A transformer model based on the paper "Attention Is All You Need" by
    Vaswani et al. (2017).

    Args:
        d_model (int): The number of expected features in the encoder/decoder inputs (default=512).
        nhead (int): The number of heads in the multi-head attention models (default=8).
        num_encoder_layers (int): The number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers (int): The number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward (int): The dimension of the feedforward network model (default=2048).
        dropout (float): The dropout value (default=0.1).
        activation (str): The activation function of encoder/decoder intermediate layer, 'relu' or 'gelu' (default='relu').
        custom_encoder (Optional[nn.Module]): Custom encoder module (default=None).
        custom_decoder (Optional[nn.Module]): Custom decoder module (default=None).

    Examples:
        >>> transformer_model = Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 custom_encoder: Optional[nn.Module] = None,
                 custom_decoder: Optional[nn.Module] = None) -> None:
        super(Transformer, self).__init__()

        # Define the encoder
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead,
                                                    dim_feedforward, dropout,
                                                    activation)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer,
                                              num_encoder_layers, encoder_norm)

        # Define the decoder
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead,
                                                    dim_feedforward, dropout,
                                                    activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer,
                                              num_decoder_layers, decoder_norm)

        # Initialize parameters
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Perform a forward pass through the Transformer model.

        Args:
            src (Tensor): The sequence to the encoder (required).
            tgt (Tensor): The sequence to the decoder (required).
            src_mask (Optional[Tensor]): The additive mask for the src sequence (optional).
            tgt_mask (Optional[Tensor]): The additive mask for the tgt sequence (optional).
            memory_mask (Optional[Tensor]): The additive mask for the encoder output (optional).
            src_key_padding_mask (Optional[Tensor]): The ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask (Optional[Tensor]): The ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask (Optional[Tensor]): The ByteTensor mask for memory keys per batch (optional).

        Returns:
            Tensor: The output tensor of shape (T, N, E).

        Shape:
            - src: (S, N, E).
            - tgt: (T, N, E).
            - src_mask: (S, S).
            - tgt_mask: (T, T).
            - memory_mask: (T, S).
            - src_key_padding_mask: (N, S).
            - tgt_key_padding_mask: (N, T).
            - memory_key_padding_mask: (N, S).

        Note: [src/tgt/memory]_mask ensures that position i is allowed to attend to the unmasked
        positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
        while the zero positions will be unchanged. If a BoolTensor is provided, positions with `True`
        are not allowed to attend while `False` values will be unchanged. If a FloatTensor
        is provided, it will be added to the attention weight.
        [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
        the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
        positions will be unchanged. If a BoolTensor is provided, the positions with the
        value of `True` will be ignored while the position with the value of `False` will be unchanged.
        """
        if src.size(1) != tgt.size(1):
            raise RuntimeError('The batch number of src and tgt must be equal')

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError(
                'The feature number of src and tgt must be equal to d_model')

        memory = self.encoder(src,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt,
                              memory,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generate a square mask for the sequence.

        Args:
            sz (int): The size of the square mask.

        Returns:
            Tensor: The generated mask tensor.

        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        """Initialize parameters in the transformer model using Xavier
        initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerEncoder(nn.Module):
    """A stack of N encoder layers in a Transformer.

    Args:
        encoder_layer (nn.Module): An instance of the TransformerEncoderLayer class (required).
        num_layers (int): The number of sub-encoder-layers in the encoder (required).
        norm (nn.Module, optional): The layer normalization component (optional).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self,
                 encoder_layer: nn.Module,
                 num_layers: int,
                 norm: nn.Module = None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the input through the encoder layers in turn.

        Args:
            src (Tensor): The sequence to the encoder (required).
            mask (Tensor, optional): The mask for the src sequence (optional).
            src_key_padding_mask (Tensor, optional): The mask for the src keys per batch (optional).

        Returns:
            Tensor: Output tensor after applying the encoder layers.

        Shape:
            Input:
                - src: (sequence_length, batch_size, d_model)
                - mask: (sequence_length, sequence_length)
                - src_key_padding_mask: (batch_size, sequence_length)

            Output:
                - Output tensor with the same shape as input src.
        """
        output = src

        for mod in self.layers:
            output = mod(output,
                         src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """A stack of N decoder layers in a Transformer.

    Args:
        decoder_layer (nn.Module): An instance of the TransformerDecoderLayer class (required).
        num_layers (int): The number of sub-decoder-layers in the decoder (required).
        norm (nn.Module, optional): The layer normalization component (optional).

    Examples::
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self,
                 decoder_layer: nn.Module,
                 num_layers: int,
                 norm: nn.Module = None):
        super(TransformerDecoder, self).__init__()

        # Stack multiple decoder layers
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the inputs (and mask) through the decoder layers in turn.

        Args:
            tgt (Tensor): The sequence to the decoder (required).
            memory (Tensor): The sequence from the last layer of the encoder (required).
            tgt_mask (Tensor, optional): The mask for the target sequence (optional).
            memory_mask (Tensor, optional): The mask for the memory sequence (optional).
            tgt_key_padding_mask (Tensor, optional): The mask for the target keys per batch (optional).
            memory_key_padding_mask (Tensor, optional): The mask for the memory keys per batch (optional).

        Returns:
            Tensor: Output tensor after applying the decoder layers.

        Shape:
            Input:
                - tgt: (sequence_length_tgt, batch_size, d_model)
                - memory: (sequence_length_memory, batch_size, d_model)
                - tgt_mask: (sequence_length_tgt, sequence_length_tgt)
                - memory_mask: (sequence_length_tgt, sequence_length_memory)
                - tgt_key_padding_mask: (batch_size, sequence_length_tgt)
                - memory_key_padding_mask: (batch_size, sequence_length_memory)

            Output:
                - Output tensor with the same shape as input tgt.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output,
                         memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def _get_clones(self, module, N):
        """Create N identical copies of the given module.

        Args:
            module (nn.Module): Module to be copied.
            N (int): Number of copies.

        Returns:
            nn.ModuleList: List of copied modules.
        """
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoderLayer(nn.Module):
    """A single layer of the Transformer encoder.

    TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model (int): The number of expected features in the input (required).
        nhead (int): The number of heads in the multi-head attention models (required).
        dim_feedforward (int): The dimension of the feedforward network model (default=2048).
        dropout (float): The dropout value (default=0.1).
        activation (str): The activation function of the intermediate layer, either 'relu' or 'gelu' (default='relu').

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super(TransformerEncoderLayer, self).__init__()

        # Self-attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward neural network
        self.ffn = PositionWiseFFN(
            ffn_num_input=d_model,
            ffn_num_hiddens=dim_feedforward,
            ffn_num_outputs=d_model,
            activation=activation,
        )

        # Add && Layer normalization
        self.addnorm1 = AddNorm(d_model=d_model, dropout=dropout)
        self.addnorm2 = AddNorm(d_model=d_model, dropout=dropout)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            src (Tensor): The sequence to the encoder layer (required).
            src_mask (Tensor, optional): The mask for the src sequence (optional).
            src_key_padding_mask (Tensor, optional): The mask for the src keys per batch (optional).

        Returns:
            Tensor: Output tensor after applying the encoder layer.

        Shape:
            Input:
                - src: (sequence_length, batch_size, d_model)
                - src_mask: (sequence_length, sequence_length)
                - src_key_padding_mask: (batch_size, sequence_length)

            Output:
                - Output tensor with the same shape as input src.
        """
        # Self-attention for the src sequence
        src_y = self.self_attn(src,
                               src,
                               src,
                               attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        # Add & Layer normalization
        src = self.addnorm1(src, src_y)
        # Feedforward neural network
        src_y = self.ffn(src)
        # Add & Layer normalization
        src = self.addnorm2(src, src_y)
        return src


class TransformerDecoderLayer(nn.Module):
    """A single layer of the Transformer decoder.

    Args:
        d_model (int): The number of expected features in the input (required).
        nhead (int): The number of heads in the multi-head attention models (required).
        dim_feedforward (int): The dimension of the feedforward network model (default=2048).
        dropout (float): The dropout value (default=0.1).
        activation (str): The activation function of the intermediate layer, either 'relu' or 'gelu' (default='relu').

    Examples::
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super(TransformerDecoderLayer, self).__init__()

        # Self-attention mechanism for the target sequence
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Multi-head attention mechanism between target and memory (encoder output)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout)

        # Feedforward neural network
        self.ffn = PositionWiseFFN(d_model, dim_feedforward, d_model, dropout,
                                   activation)

        # Add & Layer normalization
        self.addnorm1 = AddNorm(d_model=d_model, dropout=dropout)
        self.addnorm2 = AddNorm(d_model=d_model, dropout=dropout)
        self.addnorm3 = AddNorm(d_model=d_model, dropout=dropout)

        # Activation function
        self.activation = _get_activation_fn(activation)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt (Tensor): The sequence to the decoder layer (required).
            memory (Tensor): The sequence from the last layer of the encoder (required).
            tgt_mask (Tensor, optional): The mask for the target sequence (optional).
            memory_mask (Tensor, optional): The mask for the memory sequence (optional).
            tgt_key_padding_mask (Tensor, optional): The mask for the target keys per batch (optional).
            memory_key_padding_mask (Tensor, optional): The mask for the memory keys per batch (optional).

        Returns:
            Tensor: Output tensor after applying the decoder layer.

        Shape:
            Input:
                - tgt: (sequence_length_tgt, batch_size, d_model)
                - memory: (sequence_length_memory, batch_size, d_model)
                - tgt_mask: (sequence_length_tgt, sequence_length_tgt)
                - memory_mask: (sequence_length_tgt, sequence_length_memory)
                - tgt_key_padding_mask: (batch_size, sequence_length_tgt)
                - memory_key_padding_mask: (batch_size, sequence_length_memory)

            Output:
                - Output tensor with the same shape as input tgt.
        """
        # Self-attention for the target sequence
        tgt2 = self.self_attn(tgt,
                              tgt,
                              tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # Add & Layer normalization
        tgt = self.addnorm1(tgt, tgt2)

        # Multi-head attention between target and memory
        tgt2 = self.multihead_attn(tgt,
                                   memory,
                                   memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # Add & Layer normalization
        tgt = self.addnorm2(tgt, tgt2)
        # Feedforward neural network
        tgt2 = self.ffn(tgt)
        # Add & Layer normalization
        tgt = self.addnorm3(tgt, tgt2)
        return tgt


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization.

    Args:
        d_model (int or list): The expected size of the input features.
        dropout (float): Dropout probability.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        ln (nn.LayerNorm): Layer normalization layer.
    """

    def __init__(self, d_model, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, src_x: torch.Tensor,
                src_y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AddNorm layer.

        Args:
            src_x (torch.Tensor): Input tensor.
            src_y (torch.Tensor): Output tensor from a sublayer.

        Returns:
            torch.Tensor: Output of the AddNorm layer.
        """
        # Apply dropout to the output tensor src_y, add it to the input tensor src_x, and normalize the result.
        src = self.dropout(src_y)
        return self.ln(src_x + src)


class PositionWiseFFN(nn.Module):
    """Position-wise Feedforward Neural Network (FFN) layer.

    Args:
        ffn_num_input (int): Number of input features.
        ffn_num_hiddens (int): Number of hidden units in the FFN.
        ffn_num_outputs (int): Number of output features.
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self,
                 ffn_num_input: int,
                 ffn_num_hiddens: int,
                 ffn_num_outputs: int,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.activation = _get_activation_fn(activation)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PositionWiseFFN layer.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the FFN layer.
        """
        # Apply the feedforward neural network: Linear -> ReLU -> Linear
        outputs = self.dense1(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.dense2(outputs)
        return outputs


def _get_clones(module, N):
    """Create N identical copies of the given module.

    Args:
        module (nn.Module): Module to be copied.
        N (int): Number of copies.

    Returns:
        nn.ModuleList: List of copied modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str):
    """Get the activation function based on the provided string.

    Args:
        activation (str): Activation function name, either 'relu' or 'gelu'.

    Returns:
        Callable: Activation function.
    """
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    else:
        raise ValueError(
            f"Activation '{activation}' not supported. Choose 'relu' or 'gelu'."
        )


if __name__ == '__main__':
    encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
    src = torch.rand(10, 32, 512)
    out = encoder_layer(src)
    print(out.shape)

    decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
    memory = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)
    out = decoder_layer(tgt, memory)
    print(out.shape)

    transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
    memory = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)
    out = transformer_decoder(tgt, memory)
    print(out.shape)

    transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
    src = torch.rand(10, 32, 512)
    out = transformer_encoder(src)
    print(out.shape)

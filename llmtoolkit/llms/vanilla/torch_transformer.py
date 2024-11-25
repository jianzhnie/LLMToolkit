import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (Dropout, LayerNorm, Linear, Module, MultiheadAttention,
                      TransformerEncoder)

from llmtoolkit.data.embeddings import (PositionalEmbedding,
                                        PositionalEncoding, TokenEmbedding)


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


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src,
                              src,
                              src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu'):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 nhead,
                                                 dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt,
                              tgt,
                              tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoderModel(nn.Module):
    """The nn.TransformerEncoder consists of multiple layers of
    nn.TransformerEncoderLayer.

    Along with the input sequence, a square attention mask is required because
    the self-attention layers in nn.TransformerEncoder are only allowed to
    attend the earlier positions in the sequence. For the language  modeling
    task, any tokens on the future positions should be masked. To produce a
    probability distribution over output words, the output of the
    nn.TransformerEncoder model is passed through a linear layer followed by a
    log-softmax function.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        d_ffn: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_ffn,
                                                 dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.fc = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.token_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.fc(output)
        return output


class TransformerModel(nn.Module):
    """The Transformer model.

    Args:
        src_vocab_size (int):
            The size of source vocabulary.
        tgt_vocab_size (int):
            The size of target vocabulary.
        max_length (int):
            The maximum length of input sequences.
        num_encoder_layers (int):
            The number of sub-layers to be stacked in the encoder.
        num_decoder_layers (int):
            The number of sub-layers to be stacked in the decoder.
        n_head (int):
            The number of head used in multi-head attention.
        d_model (int):
            The dimension for word embeddings, which is also the last dimension of
            the input and output of multi-head attention, position-wise feed-forward
            networks, encoder and decoder.
        dim_feedforward (int):
            Size of the hidden layer in position-wise feed-forward networks.
        dropout (float):
            Dropout rates. Used for pre-process, activation and inside attention.
        weight_sharing (bool):
            Whether to use weight sharing.
        bos_id (int, optional):
            The start token id and also be used as padding id. Defaults to 0.
        eos_id (int, optional):
            The end token id. Defaults to 1.
    """

    def __init__(
        self,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
        max_length: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        bos_id: int = 0,
        eos_id: int = 1,
    ):
        super(TransformerModel, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.dim_feedforward = dim_feedforward
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout

        self.src_word_embedding = TokenEmbedding(vocab_size=src_vocab_size,
                                                 emb_dim=d_model)
        self.src_pos_embedding = PositionalEmbedding(emb_dim=d_model,
                                                     max_length=max_length)

        self.tgt_word_embedding = TokenEmbedding(vocab_size=tgt_vocab_size,
                                                 emb_dim=d_model)
        self.tgt_pos_embedding = PositionalEmbedding(emb_dim=d_model,
                                                     max_length=max_length)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
        )

        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, src_word, tgt_word):
        r"""
        The Transformer forward methods. The input are source/target sequences, and
        returns logits.

        Args:
            src_word (Tensor):
                The ids of source sequences words. It is a tensor with shape
                `[batch_size, source_sequence_length]` and its data type can be
                int or int64.
            tgt_word (Tensor):
                The ids of target sequences words. It is a tensor with shape
                `[batch_size, target_sequence_length]` and its data type can be
                int or int64.

        Returns:
            Tensor:
                Output tensor of the final layer of the model whose data
                type can be float32 or float64 with shape
                `[batch_size, sequence_length, vocab_size]`.
        """
        src_max_len = src_word.shape[-1]
        tgt_max_len = tgt_word.shape[-1]
        src_slf_attn_mask = (
            torch.cast(src_word == self.bos_id,
                       dtype=torch.get_default_dtype()).unsqueeze([1, 2]) *
            -1e4)
        src_slf_attn_mask.stop_gradient = True
        tgt_slf_attn_mask = self.transformer.generate_square_subsequent_mask(
            tgt_max_len)
        tgt_slf_attn_mask.stop_gradient = True
        tgt_src_attn_bias = src_slf_attn_mask

        src_pos = torch.cast(
            src_word != self.bos_id, dtype=src_word.dtype) * torch.arange(
                start=0, end=src_max_len, dtype=src_word.dtype)
        tgt_pos = torch.cast(
            tgt_word != self.bos_id, dtype=src_word.dtype) * torch.arange(
                start=0, end=tgt_max_len, dtype=tgt_word.dtype)

        src_word_emb = self.src_word_embedding(src_word)
        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_word_emb + src_pos_emb
        enc_input = (F.dropout(src_emb, p=self.dropout, training=self.training)
                     if self.dropout else src_emb)

        tgt_word_emb = self.tgt_word_embedding(tgt_word)
        tgt_pos_emb = self.tgt_pos_embedding(tgt_pos)
        tgt_emb = tgt_word_emb + tgt_pos_emb
        dec_input = (F.dropout(tgt_emb, p=self.dropout, training=self.training)
                     if self.dropout else tgt_emb)

        dec_output = self.transformer(
            enc_input,
            dec_input,
            src_mask=src_slf_attn_mask,
            tgt_mask=tgt_slf_attn_mask,
            memory_mask=tgt_src_attn_bias,
        )

        predict = self.linear(dec_output)

        return predict

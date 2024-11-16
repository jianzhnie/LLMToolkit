import sys

import torch
import torch.nn as nn

sys.path.append('../../')
from llmtoolkit.data.embeddings import PositionalEncoding
from llmtoolkit.llms.vanilla import Transformer


class Seq2SeqTransformer(nn.Module):
    """Sequence-to-Sequence Transformer model for machine translation.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads in the transformer.
        num_layers (int): Number of transformer encoder/decoder layers.
        dim_feedforward (int): Dimensionality of the hidden layer in the feedforward network.
        dropout (float, optional): Dropout probability. Default is 0.1.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()

        # Source word embedding and positional encoding
        self.src_word_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, dropout)

        # Target word embedding and positional encoding
        self.tgt_word_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.tgt_pos_encoding = PositionalEncoding(d_model, dropout)

        # Transformer model
        self.transformer: Transformer = Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Fully connected layer for output
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Seq2SeqTransformer.

        Args:
            src (torch.Tensor): Source input tensor.
            tgt (torch.Tensor): Target input tensor.
            src_mask (torch.Tensor): Source mask tensor.
            tgt_mask (torch.Tensor): Target mask tensor.
            src_padding_mask (torch.Tensor): Source padding mask tensor.
            tgt_padding_mask (torch.Tensor): Target padding mask tensor.
            memory_key_padding_mask (torch.Tensor): Memory key padding mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Embedding and positional encoding for source
        src_word_emb = self.src_word_embedding(src)
        src_pos_emb = self.src_pos_encoding(src)
        src_emb = src_word_emb + src_pos_emb

        # Embedding and positional encoding for target
        tgt_word_emb = self.tgt_word_embedding(tgt)
        tgt_pos_emb = self.tgt_pos_encoding(tgt)
        tgt_emb = tgt_word_emb + tgt_pos_emb

        # Transformer forward pass
        seq2seq_outputs = self.transformer.forward(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Linear layer for output
        outputs = self.fc(seq2seq_outputs)

        return outputs

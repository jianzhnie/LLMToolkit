import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

sys.path.append('../../../')
from llmtoolkit.llms.vanilla.attention import AdditiveAttention
from llmtoolkit.models.seq2seq.rnn_mt import RNNEncoder


class RNNAttentionDecoder(nn.Module):
    """RNN-based decoder with Bahdanau attention for sequence-to-sequence
    learning.

    带有Bahdanau注意力的循环神经网络解码器。

    首先，初始化解码器的状态，需要下面的输入：

    - 编码器在所有时间步的最终层隐状态，将作为注意力的键和值；

    - 上一时间步的编码器全层隐状态，将作为初始化解码器的隐状态；

    - 编码器有效长度（排除在注意力池中填充词元）。

    在每个解码时间步骤中，解码器上一个时间步的最终层隐状态将用作查询。 因此，注意力输出和输入嵌入都连结为循环神经网络解码器的输入。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        """Initialize the RNN Attention Decoder.

        Args:
            vocab_size (int): Size of the target vocabulary.
            embed_size (int): Size of word embeddings.
            hidden_size (int): Size of the decoder's hidden states.
            num_layers (int): Number of GRU layers in the decoder.
            dropout (float, optional): Dropout probability (default: 0.0).
        """
        super(RNNAttentionDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Initialize the embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Initialize the GRU cell
        self.rnn = nn.GRU(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Initialize the Bahdanau attention module
        self.attention = AdditiveAttention(hidden_size, hidden_size,
                                           hidden_size, dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        inputs: Tensor,
        encoder_outputs: Tensor,
        encoder_hidden_states: Tensor,
        encoder_valid_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the RNN Attention Decoder.

        Args:
            inputs (Tensor): Target sequences (word indices) for decoding,
                             shape [seq_len, batch_size].
            encoder_outputs (Tensor): Encoder outputs, shape
                                      [seq_len, batch_size, hidden_size].
            encoder_hidden_states (Tensor): Hidden states of the encoder's GRU,
                                            shape [num_layers, batch_size, hidden_size].
            encoder_valid_lens (Tensor): Valid lengths of the encoder input sequences,
                                         shape [batch_size].

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing:
                - outputs (Tensor): Decoder outputs, shape
                                   [batch_size, seq_len, vocab_size].
                - decoder_hidden_state (Tensor): Final decoder hidden state,
                                                shape [num_layers, batch_size, hidden_size].
        """
        # Transpose inputs for batch processing
        inputs = inputs.permute(1, 0)
        embedded = self.embedding(inputs)
        # embedded shape: [seq_len, batch_size, embed_size]

        outputs = []
        seq_len = embedded.shape[0]

        for idx in range(seq_len):
            input = embedded[idx]
            # input shape: [batch_size, embed_size]

            query = torch.unsqueeze(encoder_hidden_states[-1], dim=1)
            # query shape: [batch_size, 1, hidden_size]

            context = self.attention(query, encoder_outputs, encoder_outputs,
                                     encoder_valid_lens)
            # context shape: [batch_size, 1, hidden_size]

            input = torch.unsqueeze(input, dim=1)
            # input shape: [batch_size, 1, embed_size]

            concatenated_features = torch.cat((context, input), dim=-1)
            concatenated_features = concatenated_features.permute(1, 0, 2)
            # concatenated_features shape: [batch_size, 1, embed_size + hidden_size]

            decoder_outputs, decoder_hidden_state = self.rnn(
                concatenated_features, encoder_hidden_states)
            # decoder_outputs shape: [1, batch_size, hidden_size]
            # decoder_hidden_state shape: [num_layers, batch_size, hidden_size]

            outputs.append(decoder_outputs)

        # Concatenate and process decoder outputs
        outputs = torch.cat(outputs, dim=0)
        # outputs shape: [seq_len, batch_size, hidden_size]
        outputs = self.fc_out(outputs)
        # outputs shape: [seq_len, batch_size, vocab_size]
        outputs = outputs.permute(1, 0, 2)
        # outputs shape: [batch_size, seq_len, vocab_size]

        return outputs, decoder_hidden_state


class RNNeq2SeqModel(nn.Module):
    """The Seq2Seq model with attention for sequence-to-sequence learning.

    Args:
        src_vocab_size (int): Source vocabulary size.
        trg_vocab_size (int): Target vocabulary size.
        embed_size (int): Size of word embeddings.
        hidden_size (int): Size of hidden states in the encoder and decoder.
        num_layers (int): Number of RNN layers in the encoder and decoder.
        dropout (float, optional): Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize the encoder and decoder
        self.encoder = RNNEncoder(src_vocab_size, embed_size, hidden_size,
                                  num_layers, dropout)
        self.decoder = RNNAttentionDecoder(trg_vocab_size, embed_size,
                                           hidden_size, num_layers, dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Seq2Seq model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target input tensor of shape (batch_size, tgt_seq_len).

        Returns:
            dec_outputs (torch.Tensor): Decoder outputs tensor of shape (batch_size, tgt_seq_len, trg_vocab_size).
        """
        # Encode the source sequence
        enc_outputs, enc_state = self.encoder(src)
        # enc_state shape: [num_layers, batch_size, hidden_size]
        # enc_outputs shape: [src_seq_len, batch_size, hidden_size]

        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs shape: [batch_size, src_seq_len, hidden_size]

        # Decode the target sequence
        dec_outputs, _ = self.decoder(tgt, enc_outputs, enc_state, None)

        return dec_outputs


if __name__ == '__main__':
    src_vocab_size = 10
    tgt_vocab_size = 20
    embed_size = 8
    hiddens_size = 16
    num_layers = 2
    batch_size = 4
    max_seq_len = 5
    encoder = RNNEncoder(src_vocab_size, embed_size, hiddens_size, num_layers)
    input = torch.LongTensor([[1, 2, 4, 5, 3], [4, 3, 2, 9, 2],
                              [1, 2, 3, 4, 4], [4, 3, 2, 1, 6]])

    target = torch.LongTensor([[1, 3, 4, 5, 3], [4, 3, 2, 9, 2],
                               [1, 2, 3, 4, 4], [4, 3, 2, 1, 6]])
    # input: [batch_size, max_seq_len]
    enc_outputs, enc_state = encoder(input)
    # enc_outputs: [max_seq_len, batch_size, hiddens_size]
    # enc_state: [num_layers, batch_size, hiddens_size]
    print(enc_outputs.shape, enc_state.shape)

    decoder = RNNAttentionDecoder(tgt_vocab_size, embed_size, hiddens_size,
                                  num_layers)
    enc_outputs = enc_outputs.permute(1, 0, 2)
    dec_outputs, decoder_hidden_state = decoder(target, enc_outputs, enc_state,
                                                None)
    print(dec_outputs.shape, decoder_hidden_state.shape)

    print('seq2seq')
    seq2seq = RNNeq2SeqModel(src_vocab_size, tgt_vocab_size, embed_size,
                             hiddens_size, num_layers)

    output = seq2seq(input, target)
    print(output.shape)

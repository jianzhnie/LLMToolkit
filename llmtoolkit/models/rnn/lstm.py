import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

WEIGHT_INIT_RANGE = 0.1

START_TAG = '<START>'
STOP_TAG = '<STOP>'


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(
        torch.sum(torch.exp(vec - max_score_broadcast)))


class NaiveLSTMCell(nn.Module):
    """
    A custom implementation of an LSTM layer.

    ```
    - I𝑡=𝜎(𝐗𝑡𝐖𝑥i+𝐇𝑡−1𝐖ℎi+𝐛i),
    - F𝑡=𝜎(𝐗𝑡𝐖𝑥f+𝐇𝑡−1𝐖ℎf+𝐛f),
    - O𝑡=𝜎(𝐗𝑡𝐖𝑥o+𝐇𝑡−1𝐖ℎo+𝐛o),
    - 𝐂̃_𝑡=tanh(𝐗𝑡𝐖𝑥c+𝐇𝑡−1𝐖ℎc+𝐛c),
    - 𝐂̃𝑡=F𝑡@C𝑡-1 + I𝑡@𝐂̃_𝑡
    - H𝑡=O𝑡@tanh(𝐂̃𝑡)
    ```


    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.

    Attributes:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        W_xi (nn.Parameter): Weight matrix for the input gate.
        W_hi (nn.Parameter): Weight matrix for the hidden-to-hidden input gate.
        b_i (nn.Parameter): Bias for the input gate.
        W_xf (nn.Parameter): Weight matrix for the forget gate.
        W_hf (nn.Parameter): Weight matrix for the hidden-to-hidden forget gate.
        b_f (nn.Parameter): Bias for the forget gate.
        W_xc (nn.Parameter): Weight matrix for the candidate memory cell.
        W_hc (nn.Parameter): Weight matrix for the hidden-to-hidden candidate memory cell.
        b_c (nn.Parameter): Bias for the candidate memory cell.
        W_xo (nn.Parameter): Weight matrix for the output gate.
        W_ho (nn.Parameter): Weight matrix for the hidden-to-hidden output gate.
        b_o (nn.Parameter): Bias for the output gate.

    Reference:
        https://github.com/piEsposito/pytorch-lstm-by-hand
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate
        self.W_xi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate
        self.W_xf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Candidate memory cell
        self.W_xc = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate
        self.W_xo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and biases with uniform random values.

        Weight initialization follows the Xavier initialization scheme.

        Returns:
            None
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self,
        input: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the LSTM cell.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, input_size).
            hidden (tuple, optional): The initial hidden state tensor of shape (batch_size, hidden_size)
                and cell state tensor of shape (batch_size, hidden_size). If not provided, they are initialized as zeros.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - hy (torch.Tensor): The hidden state tensor of shape (batch_size, hidden_size).
                - cy (torch.Tensor): The cell state tensor of shape (batch_size, hidden_size).
        """
        bs = input.shape[0]

        if hidden is None:
            h_x, c_x = (
                torch.zeros(bs, self.hidden_size).to(input.device),
                torch.zeros(bs, self.hidden_size).to(input.device),
            )
        else:
            h_x, c_x = hidden
            assert h_x.shape == c_x.shape
            assert h_x.shape[1] == self.hidden_size

        # Input gate
        input_gate = torch.sigmoid(input @ self.W_xi + h_x @ self.W_hi +
                                   self.b_i)

        # Forget gate
        forget_gate = torch.sigmoid(input @ self.W_xf + h_x @ self.W_hf +
                                    self.b_f)

        # Output gate
        output_gate = torch.sigmoid(input @ self.W_xo + h_x @ self.W_ho +
                                    self.b_o)

        # Candidate memory cell
        C_tilda = torch.tanh(input @ self.W_xc + h_x @ self.W_hc + self.b_c)

        # Update cell state
        cy = forget_gate * c_x + input_gate * C_tilda

        # Update hidden state
        hy = output_gate * torch.tanh(cy)

        return hy, cy


class LSTMLayer(nn.Module):
    """A custom implementation of an LSTM layer.

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.

    Reference:
        https://github.com/piEsposito/pytorch-lstm-by-hand
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM cell
        self.lstm_cell = NaiveLSTMCell(input_size, hidden_size)

    def forward(
        self,
        input: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the LSTM layer.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, sequence_size, input_size).
            hidden (tuple, optional): The initial hidden state tensor of shape (batch_size, hidden_size)
                and cell state tensor of shape (batch_size, hidden_size). If not provided, they are initialized as zeros.

        Returns:
            outputs (torch.Tensor): The sequence of hidden states of shape (batch_size, sequence_size, hidden_size).
            (h_x, c_x) (tuple): The final hidden state tensor of shape (batch_size, hidden_size)
                and cell state tensor of shape (batch_size, hidden_size).
        """
        bs, seq_len, _ = input.size()

        if hidden is None:
            h_x, c_x = (
                torch.zeros(bs, self.hidden_size).to(input.device),
                torch.zeros(bs, self.hidden_size).to(input.device),
            )
        else:
            h_x, c_x = hidden

        outputs = []

        for t in range(seq_len):
            x_t = input[:, t, :]

            # LSTM cell forward pass
            hy, cy = self.lstm_cell(x_t, (h_x, c_x))

            h_x, c_x = hy, cy

            outputs.append(hy)

        outputs = torch.cat(outputs, dim=0).view(bs, seq_len, self.hidden_size)

        return outputs, (h_x, c_x)


class MultiLayerLSTM(nn.Module):
    """A custom implementation of a multi-layer LSTM model.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int, optional): Number of LSTM layers (default is 1).

    Reference:
        https://github.com/piEsposito/pytorch-lstm-by-hand
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(MultiLayerLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a list of LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # The first layer takes the input
                self.lstm_layers.append(NaiveLSTMCell(input_size, hidden_size))
            else:
                # The other layers take the hidden state of the previous layer
                self.lstm_layers.append(NaiveLSTMCell(hidden_size,
                                                      hidden_size))

    def forward(
        self,
        input: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the multi-layer LSTM.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, sequence_size, input_size).
            hidden (tuple, optional): Initial hidden state tensor of shape (num_layers, batch_size, hidden_size)
                and cell state tensor of shape (num_layers, batch_size, hidden_size). Default is None.

        Returns:
            outputs (torch.Tensor): Sequence of hidden states, shape (batch_size, sequence_size, hidden_size).
            (h_x, c_x) (tuple): Final hidden state tensor of shape (num_layers, batch_size, hidden_size)
                and cell state tensor of shape (num_layers, batch_size, hidden_size).
        """
        batch_size, seq_len, _ = input.size()

        if hidden is None:
            # Initialize hidden and cell states as zeros
            h_x, c_x = (
                torch.zeros(self.num_layers, batch_size,
                            self.hidden_size).to(input.device),
                torch.zeros(self.num_layers, batch_size,
                            self.hidden_size).to(input.device),
            )
        else:
            h_x, c_x = hidden

        outputs = []

        for t in range(seq_len):
            x_t = input[:, t, :]

            # Forward pass through each LSTM layer
            for layer_idx in range(self.num_layers):
                lstm_cell = self.lstm_layers[layer_idx]
                # hidden and cell states of the current layer
                h_t, c_t = h_x[layer_idx], c_x[layer_idx]
                # LSTM cell forward pass
                h_x[layer_idx], c_x[layer_idx] = lstm_cell(x_t, (h_t, c_t))
                # Update input for the next layer
                x_t = h_x[layer_idx]

            # Store the output for this time step
            outputs.append(x_t)

        # Stack outputs and reshape to match the input shape
        outputs = torch.stack(outputs, dim=1)

        return outputs, (h_x, c_x)


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(
                    1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([
            torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags
        ])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


class LSTMPostag(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTMPostag, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.output = nn.Linear(hidden_dim, num_class)
        self.init_weights()

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        x_pack = pack_padded_sequence(embeddings,
                                      lengths,
                                      batch_first=True,
                                      enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        outputs = self.output(hidden)
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs

    def init_weights(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param,
                                   a=-WEIGHT_INIT_RANGE,
                                   b=WEIGHT_INIT_RANGE)


if __name__ == '__main__':
    input_data = torch.randn(32, 10, 128)
    print(input_data.shape)
    rnn_model = MultiLayerLSTM(input_size=128, hidden_size=256, num_layers=2)
    # Batch size of 32, sequence length of 10, input size of 64
    outputs, hidden_state = rnn_model(input_data)
    print(outputs.shape, hidden_state[0].shape)

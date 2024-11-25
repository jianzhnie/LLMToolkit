import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Highway(nn.Module):

    def __init__(self, input_dim, num_layers, activation=F.relu):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        self.activation = activation
        for layer in self.layers:
            # set bias in the gates to be positive
            # such that the highway layer will be biased towards the input part
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        curr_inputs = inputs
        for layer in self.layers:
            projected_inputs = layer(curr_inputs)
            hidden = self.activation(projected_inputs[:, 0:self.input_dim])
            gate = torch.sigmoid(projected_inputs[:, self.input_dim:])
            curr_inputs = gate * curr_inputs + (1 - gate) * hidden
        return curr_inputs


class ConvTokenEmbedder(nn.Module):

    def __init__(self,
                 vocab_c,
                 char_embedding_dim,
                 char_conv_filters,
                 num_highways,
                 output_dim,
                 pad='<pad>'):
        super(ConvTokenEmbedder, self).__init__()
        self.vocab_c = vocab_c

        self.char_embeddings = nn.Embedding(len(vocab_c),
                                            char_embedding_dim,
                                            padding_idx=vocab_c[pad])
        self.char_embeddings.weight.data.uniform_(-0.25, 0.25)

        self.convolutions = nn.ModuleList()
        for kernel_size, out_channels in char_conv_filters:
            conv = nn.Conv1d(in_channels=char_embedding_dim,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             bias=True)
            self.convolutions.append(conv)

        self.num_filters = sum(f[1] for f in char_conv_filters)
        self.num_highways = num_highways
        self.highways = Highway(self.num_filters,
                                self.num_highways,
                                activation=F.relu)

        self.projection = nn.Linear(self.num_filters, output_dim, bias=True)

    def forward(self, inputs):
        """
        inputs: batch_size, seq_len, token_len ===>
        inputs: (batch_size*seq_len) * token_len
        char_embeds: (batch_size*seq_len) * token_len * char_embedding_dim ==>
        char_embeds: (batch_size*seq_len) * char_embedding_dim * token_len
        Conv Layers:
            - (batch_size*seq_len) * channel1 * dim1
                ==> (batch_size*seq_len) * channel1
            - (batch_size*seq_len) * channel2 * dim2
                ==> (batch_size*seq_len) * channel2
            - (batch_size*seq_len) * channel3 * dim3
                ==> (batch_size*seq_len) * channel3
            ...

        conv_hiddens: Concat & reshape
            - batch_size * seq_len *  (dim1 + dim2+ ... + dim_n)
        """

        batch_size, seq_len, token_len = inputs.shape
        inputs = inputs.view(batch_size * seq_len, -1)
        char_embeds = self.char_embeddings(inputs)
        char_embeds = char_embeds.transpose(1, 2)

        conv_hiddens = []
        for i in range(len(self.convolutions)):
            conv_hidden = self.convolutions[i](char_embeds)
            conv_hidden, _ = torch.max(conv_hidden, dim=-1)
            conv_hidden = F.relu(conv_hidden)
            conv_hiddens.append(conv_hidden)

        token_embeds = torch.cat(conv_hiddens, dim=-1)
        token_embeds = self.highways(token_embeds)
        token_embeds = self.projection(token_embeds)
        token_embeds = token_embeds.view(batch_size, seq_len, -1)

        return token_embeds


class ELMoLstmEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout_prob=0.0):
        super(ELMoLstmEncoder, self).__init__()

        # set projection_dim==input_dim for ELMo usage
        self.projection_dim = input_dim
        self.num_layers = num_layers

        self.forward_layers = nn.ModuleList()
        self.backward_layers = nn.ModuleList()
        self.forward_projections = nn.ModuleList()
        self.backward_projections = nn.ModuleList()

        lstm_input_dim = input_dim
        for _ in range(num_layers):
            forward_layer = nn.LSTM(lstm_input_dim,
                                    hidden_dim,
                                    num_layers=1,
                                    batch_first=True)
            forward_projection = nn.Linear(hidden_dim,
                                           self.projection_dim,
                                           bias=True)

            backward_layer = nn.LSTM(lstm_input_dim,
                                     hidden_dim,
                                     num_layers=1,
                                     batch_first=True)
            backward_projection = nn.Linear(hidden_dim,
                                            self.projection_dim,
                                            bias=True)

            lstm_input_dim = self.projection_dim

            self.forward_layers.append(forward_layer)
            self.forward_projections.append(forward_projection)
            self.backward_layers.append(backward_layer)
            self.backward_projections.append(backward_projection)

    def forward(self, inputs, lengths):
        batch_size, seq_len, input_dim = inputs.shape
        rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        for i in range(lengths.shape[0]):
            rev_idx[i, :lengths[i]] = torch.arange(lengths[i] - 1, -1, -1)
        rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
        rev_idx = rev_idx.to(inputs.device)
        rev_inputs = inputs.gather(1, rev_idx)

        forward_inputs, backward_inputs = inputs, rev_inputs
        stacked_forward_states, stacked_backward_states = [], []

        for layer_index in range(self.num_layers):
            # Transfer `lengths` to CPU to be compatible with latest PyTorch versions.
            packed_forward_inputs = pack_padded_sequence(forward_inputs,
                                                         lengths.cpu(),
                                                         batch_first=True,
                                                         enforce_sorted=False)
            packed_backward_inputs = pack_padded_sequence(backward_inputs,
                                                          lengths.cpu(),
                                                          batch_first=True,
                                                          enforce_sorted=False)

            # forward
            forward_layer = self.forward_layers[layer_index]
            packed_forward, _ = forward_layer(packed_forward_inputs)
            forward = pad_packed_sequence(packed_forward, batch_first=True)[0]
            forward = self.forward_projections[layer_index](forward)
            stacked_forward_states.append(forward)

            # backward
            backward_layer = self.backward_layers[layer_index]
            packed_backward, _ = backward_layer(packed_backward_inputs)
            backward = pad_packed_sequence(packed_backward,
                                           batch_first=True)[0]
            backward = self.backward_projections[layer_index](backward)
            # convert back to original sequence order using rev_idx
            stacked_backward_states.append(backward.gather(1, rev_idx))

            forward_inputs, backward_inputs = forward, backward

        # stacked_forward_states: [batch_size, seq_len, projection_dim] * num_layers
        # stacked_backward_states: [batch_size, seq_len, projection_dim] * num_layers
        return stacked_forward_states, stacked_backward_states


class BiLM(nn.Module):
    """多层双向语言模型。"""

    def __init__(self, configs, vocab_w, vocab_c):
        super(BiLM, self).__init__()
        self.dropout_prob = configs['dropout_prob']
        self.num_classes = len(vocab_w)

        self.token_embedder = ConvTokenEmbedder(vocab_c,
                                                configs['char_embedding_dim'],
                                                configs['char_conv_filters'],
                                                configs['num_highways'],
                                                configs['projection_dim'])

        self.encoder = ELMoLstmEncoder(configs['projection_dim'],
                                       configs['hidden_dim'],
                                       configs['num_layers'])

        self.classifier = nn.Linear(configs['projection_dim'],
                                    self.num_classes)

    def forward(self, inputs, lengths):
        token_embeds = self.token_embedder(inputs)
        token_embeds = F.dropout(token_embeds, self.dropout_prob)
        forward, backward = self.encoder(token_embeds, lengths)

        return self.classifier(forward[-1]), self.classifier(backward[-1])

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.token_embedder.state_dict(),
                   os.path.join(path, 'token_embedder.pth'))
        torch.save(self.encoder.state_dict(),
                   os.path.join(path, 'encoder.pth'))
        torch.save(self.classifier.state_dict(),
                   os.path.join(path, 'classifier.pth'))

    def load_pretrained(self, path):
        self.token_embedder.load_state_dict(
            torch.load(os.path.join(path, 'token_embedder.pth')))
        self.encoder.load_state_dict(
            torch.load(os.path.join(path, 'encoder.pth')))
        self.classifier.load_state_dict(
            torch.load(os.path.join(path, 'classifier.pth')))

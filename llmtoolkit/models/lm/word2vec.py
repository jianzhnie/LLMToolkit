import torch
import torch.nn as nn
import torch.nn.functional as F


class NGramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(NGramLanguageModel, self).__init__()
        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 线性变换：词嵌入层->隐含层
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        # 线性变换：隐含层->输出层
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        # 使用ReLU激活函数
        self.activate = nn.ReLU(inplace=True)
        self.init_weights()

    def forward(self, inputs):
        # batch * context_size * embedding_dim ==> batch * (context_size * embedding_dim)
        embeds = self.embeddings(inputs)
        embeds = embeds.view((inputs.shape[0], -1))
        hidden = self.activate(self.linear1(embeds))
        output = self.linear2(hidden)
        # 根据输出层（logits）计算概率分布并取对数，以便于计算对数似然
        # 这里采用PyTorch库的log_softmax实现
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' not in name:
                torch.nn.init.uniform_(param, a=-0.1, b=0.1)


class CBOWLanguageModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim=128,
                 hidden_dim=128,
                 padding_idx=0):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,
                                       embedding_dim,
                                       padding_idx=padding_idx)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        # 使用ReLU激活函数
        self.activate = nn.ReLU(inplace=True)
        self.init_weights()

    def forward(self, inputs):
        # Shape: (batch_size, seq_len, embedding_dim)
        embeds = self.embeddings(inputs)
        # 计算隐含层：对上下文词向量求平均
        # Shape: (batch_size, hidden_size)
        hidden = embeds.mean(dim=1)
        out = self.activate(self.linear1(hidden))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' not in name:
                torch.nn.init.uniform_(param, a=-0.1, b=0.1)


class SkipGramModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        self.init_weights()

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = self.output(embeds)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' not in name:
                torch.nn.init.uniform_(param, a=-0.1, b=0.1)


class SkipGramNegativeSamplingModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegativeSamplingModel, self).__init__()
        # 词嵌入
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 上下文嵌入
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward_w(self, words):
        w_embeds = self.w_embeddings(words)
        return w_embeds

    def forward_c(self, contexts):
        c_embeds = self.c_embeddings(contexts)
        return c_embeds

    def forward(self, words, contexts, neg_contexts):
        word_embeds = self.forward_w(words)
        context_embeds = self.forward_c(contexts)
        neg_context_embeds = self.forward_c(neg_contexts)

        return word_embeds, context_embeds, neg_context_embeds

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' not in name:
                torch.nn.init.uniform_(param, a=-0.1, b=0.1)


class RNNLM(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0):
        super(RNNLM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedder = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout)

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        x_emb = self.embedder(inputs)
        x_emb = self.dropout(x_emb)
        output, _ = self.rnn(x_emb)
        output = self.dropout(output)
        output = self.fc(output)
        log_probs = F.log_softmax(output, dim=2)
        return log_probs

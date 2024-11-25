import torch
import torch.nn as nn


class GloveModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(GloveModel, self).__init__()
        # 词嵌入及偏置向量
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w_biases = nn.Embedding(vocab_size, 1)
        # 上下文嵌入及偏置向量
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.c_biases = nn.Embedding(vocab_size, 1)

    def forward_w(self, words):
        w_embeds = self.w_embeddings(words)
        w_biases = self.w_biases(words)
        return w_embeds, w_biases

    def forward_c(self, contexts):
        c_embeds = self.c_embeddings(contexts)
        c_biases = self.c_biases(contexts)
        return c_embeds, c_biases

    def forward(self, words, contexts):

        w_embeds = self.w_embeddings(words)
        w_biases = self.w_biases(words)

        c_embeds = self.c_embeddings(contexts)
        c_biases = self.c_biases(contexts)
        return w_embeds, w_biases, c_embeds, c_biases

    def init_weights(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param, a=-0.1, b=0.1)

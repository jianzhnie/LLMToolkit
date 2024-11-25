import torch
import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter,
                 num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim,
                                num_filter,
                                filter_size,
                                padding=1)
        self.activate = nn.ReLU()
        self.linear = nn.Linear(num_filter, num_class)

    def forward(self, inputs):
        """
        inputs: batch * seq_length
        step1 : embedding, batch * seq_length ==>  batch * seq_length * embedding_dim
                    ==> batch *  embedding_dim * seq_length
        step2 : conv1d   batch *  embedding_dim * seq_length ==>  batch *  embedding_dim * seq_length
        step3 : relu()
        step4 : Linear()
        """
        embedding = self.embedding(inputs)
        convolution = self.activate(self.conv1d(embedding.permute(0, 2, 1)))
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
        outputs = self.linear(pooling.squeeze(dim=2))
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs


class TextCNN(nn.Module):

    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这个嵌入层不需要训练
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat(
            (self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)
        # 每个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs
        ],
                             dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


class CNNEncoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 num_filter,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activation=nn.Tanh(),
                 output_dim=None,
                 **kwargs):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filter = num_filter
        self.ngram_filter_sizes = ngram_filter_sizes
        self.activation = conv_layer_activation
        self.output_dim = output_dim

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=self.num_filter,
                      kernel_size=(i, self.embedding_dim),
                      **kwargs) for i in self.ngram_filter_sizes
        ])

        maxpool_output_dim = self.num_filter * len(self.ngram_filter_sizes)
        if self.output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim,
                                              self.output_dim)
        else:
            self.projection_layer = None
            self.output_dim = maxpool_output_dim

    def get_input_dim(self):
        r"""
        Returns the dimension of the vector input for each element in the sequence input
        to a `CNNEncoder`. This is not the shape of the input tensor, but the
        last element of that shape.
        """
        return self.embedding_dim

    def get_output_dim(self):
        r"""
        Returns the dimension of the final vector output by this `CNNEncoder`.  This is not
        the shape of the returned tensor, but the last element of that shape.
        """
        return self.output_dim

    def forward(self, inputs, mask=None):
        if mask is not None:
            inputs = inputs * mask

        embedding = self.embedding(inputs)
        embedding = embedding.permute(0, 2, 1)
        # If output_dim is None, result shape of (batch_size, len(ngram_filter_sizes) * num_filter));
        # else, result shape of (batch_size, output_dim).
        convs_out = [self.activation(conv(inputs)) for conv in self.convs]
        maxpool_out = [
            F.adaptive_max_pool1d(t, output_size=1) for t in convs_out
        ]
        result = torch.concat(maxpool_out, axis=1)

        if self.projection_layer is not None:
            result = self.projection_layer(result)
        return result

# Defined in Section 4.6.8
import sys

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from llmtoolkit.data.embeddings.pos_encding import PositionalEncoding
from llmtoolkit.data.utils.utils import length_to_mask, load_sentence_polarity

sys.path.append('../../')


class TransformerDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


class Transformer(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_class,
                 dim_feedforward=512,
                 num_head=2,
                 num_layers=2,
                 dropout=0.1,
                 max_len=128,
                 activation: str = 'relu'):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout,
                                                     max_len)
        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head,
                                                   dim_feedforward, dropout,
                                                   activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 输出层
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        # 与LSTM 处理情况类似， 输入数据是 batch * seq_length
        # 需要转换成  seq_length * batch 的格式
        inputs = torch.transpose(inputs, 0, 1)
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)
        # 根据序列长度生成 Padding Mask 矩阵
        attention_mask = length_to_mask(lengths)
        hidden_states = self.transformer(hidden_states,
                                         src_key_padding_mask=attention_mask)
        # idx == 0 is for classification
        # 取第一个标记位置的输出作为分类层的输出
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=-1)
        return log_probs


if __name__ == '__main__':

    embedding_dim = 128
    hidden_dim = 128
    num_class = 2
    batch_size = 32
    num_epoch = 5

    # 加载数据
    train_data, test_data, vocab = load_sentence_polarity()
    train_dataset = TransformerDataset(train_data)
    test_dataset = TransformerDataset(test_data)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   collate_fn=collate_fn,
                                   shuffle=True)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1,
                                  collate_fn=collate_fn,
                                  shuffle=False)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(len(vocab), embedding_dim, hidden_dim, num_class)
    model.to(device)  # 将模型加载到GPU中（如果已经正确安装）

    # 训练过程
    nll_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_data_loader, desc=f'Training Epoch {epoch}'):
            inputs, lengths, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss: {total_loss:.2f}')

    # 测试过程
    acc = 0
    for batch in tqdm(test_data_loader, desc='Testing'):
        inputs, lengths, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs, lengths)
            acc += (output.argmax(dim=1) == targets).sum().item()

    # 输出在测试集上的准确率
    print(f'Acc: {acc / len(test_data_loader):.2f}')

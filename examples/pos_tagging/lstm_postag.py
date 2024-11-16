# Defined in Section 4.7.2

import sys

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from llmtoolkit.data.utils.utils import load_treebank
from llmtoolkit.models.rnn.lstm import LSTMPostag

sys.path.append('../../')


class LstmDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = [torch.tensor(ex[1]) for ex in examples]
    inputs = pad_sequence(inputs,
                          batch_first=True,
                          padding_value=vocab['<pad>'])
    targets = pad_sequence(targets,
                           batch_first=True,
                           padding_value=vocab['<pad>'])
    return inputs, lengths, targets, inputs != vocab['<pad>']


if __name__ == '__main__':
    embedding_dim = 128
    hidden_dim = 256
    batch_size = 32
    num_epoch = 5

    # 加载数据
    train_data, test_data, vocab, pos_vocab = load_treebank()
    train_dataset = LstmDataset(train_data)
    test_dataset = LstmDataset(test_data)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   collate_fn=collate_fn,
                                   shuffle=True)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1,
                                  collate_fn=collate_fn,
                                  shuffle=False)

    num_class = len(pos_vocab)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMPostag(len(vocab), embedding_dim, hidden_dim, num_class)
    model.to(device)  # 将模型加载到GPU中（如果已经正确安装）

    # 训练过程
    nll_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_data_loader, desc=f'Training Epoch {epoch}'):
            inputs, lengths, targets, mask = [x.to(device) for x in batch]
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs[mask], targets[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss: {total_loss:.2f}')

    # 测试过程
    acc = 0
    total = 0
    for batch in tqdm(test_data_loader, desc='Testing'):
        inputs, lengths, targets, mask = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs, lengths)
            acc += (output.argmax(dim=-1) == targets)[mask].sum().item()
            total += mask.sum().item()

    # 输出在测试集上的准确率
    print(f'Acc: {acc / total:.2f}')

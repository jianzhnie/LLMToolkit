import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from llmtoolkit.data.utils.utils import (get_loader, load_reuters,
                                         save_pretrained)
from llmtoolkit.datasets.nlmdataset import RNNlmDataset
from llmtoolkit.models.lm.word2vec import RNNLM

sys.path.append('../../')

if __name__ == '__main__':
    embedding_dim = 32
    context_size = 2
    hidden_dim = 64
    batch_size = 8
    num_epoch = 1

    # 读取文本数据，构建FFNNLM训练数据集（n-grams）
    corpus, vocab = load_reuters()
    dataset = RNNlmDataset(corpus, vocab)
    data_loader = get_loader(dataset, batch_size)

    # 负对数似然损失函数，忽略pad_token处的损失
    nll_loss = nn.NLLLoss(ignore_index=dataset.pad)
    # 构建RNNLM，并加载至device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNLM(len(vocab), embedding_dim, hidden_dim)
    model.to(device)
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            log_probs = model(inputs)
            loss = nll_loss(log_probs.view(-1, log_probs.shape[-1]),
                            targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss: {total_loss:.2f}')

    save_pretrained(vocab, model.embedder.weight.data, 'rnnlm.vec')

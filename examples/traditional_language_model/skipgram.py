'''
Author: jianzhnie
Date: 2022-01-05 15:25:53
LastEditTime: 2022-03-07 16:17:53
LastEditors: jianzhnie
Description:

'''
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

sys.path.append('../../')
from llmtoolkit.datasets.nlmdataset import SkipGramDataset, Word2VecDataset
from llmtoolkit.models.lm import SkipGramModel
from llmtoolkit.utils.data_utils import (get_loader, load_ptb_data,
                                         save_pretrained)

sys.path.append('../../')

if __name__ == '__main__':

    embedding_dim = 64
    context_size = 2
    hidden_dim = 128
    batch_size = 1024
    num_epoch = 10

    # 读取文本数据，构建Skip-gram模型训练数据集
    ptb_data = load_ptb_data('../../data/ptb')
    word2vec = Word2VecDataset(ptb_data)
    vocab = word2vec.vocab
    dataset = SkipGramDataset(ptb_data, vocab, context_size=context_size)
    data_loader = get_loader(dataset, batch_size)

    nll_loss = nn.NLLLoss()
    # 构建Skip-gram模型，并加载至device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkipGramModel(len(vocab), embedding_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
            inputs, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            log_probs = model(inputs)
            loss = nll_loss(log_probs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss: {total_loss:.2f}')

    # 保存词向量（model.embeddings）
    save_pretrained(vocab, model.embeddings.weight.data, 'skipgram.vec')

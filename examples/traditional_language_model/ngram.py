import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

sys.path.append('../../')
from llmtoolkit.datasets.nlmdataset import NGramDataset, Word2VecDataset
from llmtoolkit.models.lm.word2vec import NGramLanguageModel
from llmtoolkit.utils.data_utils import (get_loader, load_ptb_data,
                                         save_pretrained_vector)

if __name__ == '__main__':
    embedding_dim = 64
    context_size = 2
    hidden_dim = 128
    batch_size = 1024
    num_epoch = 10

    # 读取文本数据，构建FFNNLM训练数据集（n-grams）
    ptb_data = load_ptb_data('../../data/ptb')
    word2vec = Word2VecDataset(ptb_data)
    vocab = word2vec.vocab
    print(f'Vocab size: {len(vocab)}')

    dataset = NGramDataset(ptb_data, vocab, context_size)
    data_loader = get_loader(dataset, batch_size)

    # 负对数似然损失函数
    nll_loss = nn.NLLLoss()
    # 构建NGramLanguageModel，并加载至device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NGramLanguageModel(len(vocab), embedding_dim, context_size,
                               hidden_dim)
    model.to(device)
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    total_losses = []
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
        total_losses.append(total_loss)

    # 保存词向量（model.embeddings）
    save_pretrained_vector(vocab, model.embeddings.weight.data, 'ffnnlm.vec')

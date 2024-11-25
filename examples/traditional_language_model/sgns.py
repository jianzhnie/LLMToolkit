import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

sys.path.append('../../')
from llmtoolkit.datasets.nlmdataset import (NegativeSampleingSkipGramDataset,
                                            Word2VecDataset)
from llmtoolkit.models.lm.word2vec import SkipGramNegativeSamplingModel
from llmtoolkit.utils.data_utils import (get_loader, load_ptb_data,
                                         save_pretrained)


def get_unigram_distribution(corpus, vocab_size):
    # 从给定语料中统计unigram概率分布
    token_counts = torch.tensor([0] * vocab_size)
    total_count = 0
    for sentence in corpus:
        total_count += len(sentence)
        for token in sentence:
            token_counts[token] += 1
    unigram_dist = torch.div(token_counts.float(), total_count)
    return unigram_dist


if __name__ == '__main__':
    embedding_dim = 64
    context_size = 2
    hidden_dim = 128
    batch_size = 1024
    num_epoch = 10
    n_negatives = 10

    # 读取文本数据
    ptb_data = load_ptb_data('../../data/ptb')
    word2vec = Word2VecDataset(ptb_data)
    vocab = word2vec.vocab  # 计算unigram概率分布
    unigram_dist = get_unigram_distribution(ptb_data, vocab, len(vocab))
    # 根据unigram分布计算负采样分布: p(w)**0.75
    negative_sampling_dist = unigram_dist**0.75
    negative_sampling_dist /= negative_sampling_dist.sum()
    # 构建SGNS训练数据集
    dataset = NegativeSampleingSkipGramDataset(ptb_data,
                                               vocab,
                                               context_size=context_size,
                                               n_negatives=n_negatives,
                                               ns_dist=negative_sampling_dist)
    data_loader = get_loader(dataset, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkipGramNegativeSamplingModel(len(vocab), embedding_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
            words, contexts, neg_contexts = [x.to(device) for x in batch]
            optimizer.zero_grad()
            batch_size = words.shape[0]
            # 提取batch内词、上下文以及负样本的向量表示
            word_embeds, context_embeds, neg_context_embeds = model(
                words, contexts, neg_contexts)
            word_embeds = word_embeds.unsqueeze(dim=2)
            # 正样本的分类（对数）似然
            context_loss = F.logsigmoid(
                torch.bmm(context_embeds, word_embeds).squeeze(dim=2))
            context_loss = context_loss.mean(dim=1)
            # 负样本的分类（对数）似然
            neg_context_loss = F.logsigmoid(
                torch.bmm(neg_context_embeds,
                          word_embeds).squeeze(dim=2).neg())
            neg_context_loss = neg_context_loss.view(batch_size, -1,
                                                     n_negatives).sum(dim=2)
            neg_context_loss = neg_context_loss.mean(dim=1)
            # 损失：负对数似然
            loss = -(context_loss + neg_context_loss).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss: {total_loss:.2f}')

    # 合并词嵌入矩阵与上下文嵌入矩阵，作为最终的预训练词向量
    combined_embeds = model.w_embeddings.weight + model.c_embeddings.weight
    save_pretrained(vocab, combined_embeds.data, 'sgns.vec')

"""
Author: jianzhnie
Date: 2021-12-17 12:13:36
LastEditTime: 2022-01-05 15:25:13
LastEditors: jianzhnie
Description:

"""

import os
import re
from typing import List

import torch
from torch.utils.data import DataLoader

from llmtoolkit.data.vocab import Vocab

# Constants
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def minEditDistance(source: str, target: str) -> int:
    """计算两个字符串的最小编辑距离.

    参数:
      source: 源字符串
      target: 目标字符串

    返回:
      两个字符串的最小编辑距离,即源字符串转换成目标字符串的最少编辑操作次数
    """

    n = len(source)
    m = len(target)

    # 初始化编辑距离矩阵
    dist_matrix: List[List[int]] = [[0 for _ in range(m + 1)]
                                    for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist_matrix[i][0] = i
    for j in range(1, m + 1):
        dist_matrix[0][j] = j
    # 动态规划计算最小编辑距离
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == target[j - 1]:
                dist_matrix[i][j] = dist_matrix[i - 1][j - 1]
            else:
                dist_matrix[i][j] = min(
                    dist_matrix[i - 1][j] + 1,
                    dist_matrix[i][j - 1] + 1,
                    dist_matrix[i - 1][j - 1] + 1,
                )
    return dist_matrix[n][m]


def truncate_pad(inputs: List[int],
                 max_seq_len: int,
                 padding_token_id: int = 0) -> List[int]:
    """Truncate and pad sequence to max sequence length."""
    if len(inputs) > max_seq_len:
        inputs = inputs[:max_seq_len]
    else:
        inputs = inputs + [padding_token_id] * (max_seq_len - len(inputs))
    return inputs


def load_ptb_data(data_dir, split='train'):
    """Penn Tree Bank（PTB）。该语料库取自“华尔街日报”的文章，分为训练集、验证集和测试集。
    在原始格式中，文本文件的每一行表示由空格分隔的一句话, 函数 `read_ptb_data` 将PTB数据集加载到文本行的列表中
    data_url: http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip
    """
    # Readthetrainingset.
    file_path = os.path.join(data_dir, 'ptb.{}.txt'.format(split))

    with open(file_path) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


def remove_empty_paired_punc(in_str):
    return (in_str.replace('（）', '').replace('《》',
                                             '').replace('【】',
                                                         '').replace('[]', ''))


def remove_html_tags(in_str):
    html_pattern = re.compile(r'<[^>]+>', re.S)
    return html_pattern.sub('', in_str)


def remove_control_chars(in_str):
    control_chars = ''.join(
        map(chr,
            list(range(0, 32)) + list(range(127, 160))))
    control_chars = re.compile('[%s]' % re.escape(control_chars))
    return control_chars.sub('', in_str)


def cleaning_wikidata(wiki_file):
    new_text = []
    with open(wiki_file, 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            line = line.strip()
            if re.search(r'^(<doc id)|(</doc>)', line):
                print(line)
                continue
            line = remove_empty_paired_punc(line)
            line = remove_html_tags(line)
            line = remove_control_chars(line)
            new_text.append(line)

    return new_text


def length_to_mask(lengths):
    max_len = torch.max(lengths)
    # padding: True
    # seqs: False
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.shape[0], max_len) > lengths.unsqueeze(1)
    mask = mask.type(torch.bool)
    return mask


def save_pretrained_vector(vocab: Vocab, embeds: torch.tensor, save_path: str):
    """Save pretrained token vectors in a unified format, where the first line
    specifies the `number_of_tokens` and `embedding_dim` followed with all
    token vectors, one token per line."""
    with open(save_path, 'w') as writer:
        writer.write(f'{embeds.shape[0]} {embeds.shape[1]}\n')
        for idx, token in vocab.idx_to_token.items():
            vec = ' '.join(['{:.4f}'.format(x) for x in embeds[idx]])
            writer.write(f'{token} {vec}\n')
    print(f'Pretrained embeddings saved to: {save_path}')


def get_loader(dataset, batch_size, shuffle=True):
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             collate_fn=dataset.collate_fn,
                             shuffle=shuffle)
    return data_loader

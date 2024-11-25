import os
import random
import sys

import torch

from llmtoolkit.data.vocab import Vocab, tokenize

sys.path.append('../../../')


def read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [
        line.strip().lower().split(' . ') for line in lines
        if len(line.split(' . ')) >= 2
    ]
    random.shuffle(paragraphs)
    return paragraphs


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs.

    - 将一个句子或两个句子作为输入，然后返回BERT输入序列的标记及其相应的片段索引。
    - 当输入为单个文本时，BERT输入序列是特殊类别词元“<cls>”、文本序列的标记、以及特殊分隔词元“<sep>”的连结。
    - 当输入为文本对时，BERT输入序列是“<cls>”、第一个文本序列的标记、“<sep>”、第二个文本序列标记、以及“<sep>”的连结。
    - 我们将始终如一地将术语“BERT输入序列”与其他类型的“序列”区分开来。例如，一个BERT输入序列可以包括一个文本序列或两个文本序列。
        - 为了区分文本对，根据输入序列学到的片段嵌入 𝐞𝐴 和 𝐞𝐵 分别被添加到第一序列和第二序列的词元嵌入中。对于单文本输入，仅使用 𝐞𝐴 。

    Defined in :numref:`sec_bert`
    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    # [0] is the label of sequence A
    segments = [0] * (len(tokens_a) + 2)
    # [0] is the label of sequence B
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def get_next_sentence(sentence, next_sentence, paragraphs):
    """
    - 为了帮助理解两个文本序列之间的关系，BERT在预训练中考虑了一个二元分类任务——下一句预测。
    - 在为预训练生成句子对时，有一半的时间它们确实是标签为“真”的连续句子；
    - 在另一半的时间里，第二个句子是从语料库中随机抽取的，标记为“假”.
    - 生成这样的数据用于帮助模型对下一个句子是否为相邻句子进行分类
    """
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs 是三重列表的嵌套
        # 先随机选择一个段落, 再随机选择一个句子
        paragraph = random.choice(paragraphs)
        next_sentence = random.choice(paragraph)
        is_next = False
    return sentence, next_sentence, is_next


def get_nsp_data_from_paragraph(paragraph, paragraphs, max_len):
    """
    生成<下一个句子预测任务>的数据集:
    - 接收当前句子和全部句子
    - 生成句子对:

        -1. taken_a token_b
        -2. segments : [0, 0, 0, 0, 0, 0, 1, 1, 1]
        -3. bool 值: True or False

    example:
    -   (['<cls>', 'romani', 'was', 'the', 'first', 'decisive', 'victory', '<sep>', \
        'it', 'also', 'made', 'the', 'clearing', 'of', 'his', 'troops',  '.', '<sep>'],
        [0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        True)
    """
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    """
    Inputs:
    - tokens是表示BERT输入序列的词元的列表，
    - candidate_pred_positions 是不包括特殊词元的BERT输入序列的词元索引的列表（特殊词元在遮蔽语言模型任务中不被预测），
    - num_mlm_preds 指示预测的数量（选择15%要预测的随机词元）。

    Outputs:
    - mlm_input_tokens: masked language model 掩码输入
    - pred_positions_and_labels: 发生预测的词元索引及对应的词元

    定义 MaskLanguageModel 之后:
        - 在每个预测位置，输入可以由特殊的“掩码”词元或随机词元替换，或者保持不变。
        - 最后，该函数返回可能替换后的输入词元、发生预测的词元索引和这些预测的标签。
    """
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def get_mlm_data_from_tokens(tokens, vocab):
    """通过调用前述的_replace_mlm_tokens函数.

    - 将BERT输入序列（tokens）作为输入，
    - 返回输入词元的索引、发生预测的词元索引以及这些预测的标签索引。
    """
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    (
        all_token_ids,
        all_segments,
        valid_lens,
    ) = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for token_ids, pred_positions, mlm_pred_label_ids, segments, is_next in examples:
        all_token_ids.append(
            torch.tensor(
                token_ids + [vocab['<pad>']] * (max_len - len(token_ids)),
                dtype=torch.long,
            ))
        all_segments.append(
            torch.tensor(segments + [0] * (max_len - len(segments)),
                         dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(
            torch.tensor(
                pred_positions + [0] *
                (max_num_mlm_preds - len(pred_positions)),
                dtype=torch.long,
            ))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor(
                [1.0] * len(mlm_pred_label_ids) + [0.0] *
                (max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32,
            ))
        all_mlm_labels.append(
            torch.tensor(
                mlm_pred_label_ids + [0] *
                (max_num_mlm_preds - len(mlm_pred_label_ids)),
                dtype=torch.long,
            ))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (
        all_token_ids,
        all_segments,
        valid_lens,
        all_pred_positions,
        all_mlm_weights,
        all_mlm_labels,
        nsp_labels,
    )


if __name__ == '__main__':
    data_dir = '/home/robin/jianzh/nlp-toolkit/examples/data/wikitext-2'
    paragraphs = read_wiki(data_dir)
    print(paragraphs[0])
    paragraphs = [
        tokenize(paragraph, token='word') for paragraph in paragraphs
    ]
    print('==' * 100)
    print(len(paragraphs[0]))
    print(len(paragraphs))
    print(paragraphs[0])

    sentences = [
        sentence for paragraph in paragraphs for sentence in paragraph
    ]
    print('==' * 100)
    print(sentences[0])
    vocab = Vocab(sentences,
                  min_freq=5,
                  reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])

    examples = []
    max_len = 100
    for paragraph in paragraphs:
        examples.extend(
            get_nsp_data_from_paragraph(paragraph, paragraphs, max_len))
        print(examples[0])
        break

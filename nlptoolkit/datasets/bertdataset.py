'''
Author: jianzhnie
Date: 2022-03-04 17:13:34
LastEditTime: 2022-03-04 17:16:27
LastEditors: jianzhnie
Description:

'''
import os
import random
from typing import List

from torch.utils.data import Dataset

from nlptoolkit.data.tokenizer import Tokenizer
from nlptoolkit.data.vocab import Vocab


class BertDataSet(Dataset):
    def __init__(
            self,
            data_dir,
            data_split: str = 'train',
            tokenizer: Tokenizer = Tokenizer(),
            max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data_dir = os.path.join(data_dir, data_split + '.txt')
        self.paragraphs = self.preprocess_text_data(self.data_dir)
        self.tokenized_paragraphs = [
            tokenizer.tokenize(paragraph) for paragraph in self.paragraphs
        ]
        self.tokenized_sentences = [
            sentence for paragraph in self.tokenized_paragraphs
            for sentence in paragraph
        ]
        self.vocab: Vocab = Vocab.build_vocab(self.tokenized_sentences,
                                              min_freq=1,
                                              unk_token='<unk>',
                                              pad_token='<pad>',
                                              bos_token='<bos>',
                                              eos_token='<eos>')
        self.vocab_words = self.vocab.token_to_idx.keys()

    def get_bert_data(self, paragraphs, max_seq_len):
        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(
                self.get_nsp_data_from_paragraph(paragraph, paragraphs,
                                                 max_seq_len))
        # 获取遮蔽语言模型任务的数据
        examples = [(self.get(tokens, self.vocab) + (segments, is_next))
                    for tokens, segments, is_next in examples]

    def get_next_sentence(self, sentence, next_sentence, paragraphs):
        if random.random() < 0.5:
            is_next = True
        else:
            # paragraphs是三重列表的嵌套
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = False
        return sentence, next_sentence, is_next

    def get_nsp_data_from_paragraph(self, paragraph, paragraphs, max_len):
        nsp_data_from_paragraph = []
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b, is_next = self.get_next_sentence(
                paragraph[i], paragraph[i + 1], paragraphs)
            # 考虑1个'<cls>'词元和2个'<sep>'词元
            if len(tokens_a) + len(tokens_b) + 3 > max_len:
                continue
            tokens, segments = self.get_tokens_and_segments(tokens_a, tokens_b)
            nsp_data_from_paragraph.append((tokens, segments, is_next))
        return nsp_data_from_paragraph

    def get_mlm_data_from_tokens(self, tokens):
        candidate_pred_positions = []
        # tokens是一个字符串列表
        for i, token in enumerate(tokens):
            # 在遮蔽语言模型任务中不会预测特殊词元
            if token in ['<cls>', '<sep>']:
                continue
            candidate_pred_positions.append(i)
        # 遮蔽语言模型任务中预测15%的随机词元
        num_mlm_preds = max(1, round(len(tokens) * 0.15))
        mlm_input_tokens, pred_positions_and_labels = self.replace_mskelm_tokens(
            tokens, candidate_pred_positions, num_mlm_preds)
        pred_positions_and_labels = sorted(pred_positions_and_labels,
                                           key=lambda x: x[0])
        pred_positions = [v[0] for v in pred_positions_and_labels]
        mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
        return self.vocab[mlm_input_tokens], pred_positions, self.vocab[
            mlm_pred_labels]

    def get_tokens_and_segments(self, tokens_a, tokens_b=None):
        """Get tokens of the BERT input sequence and their segment IDs.

        Defined in :numref:`sec_bert`"""
        tokens = ['<cls>'] + tokens_a + ['<sep>']
        # 0 and 1 are marking segment A and B, respectively
        segments = [0] * (len(tokens_a) + 2)
        if tokens_b is not None:
            tokens += tokens_b + ['<sep>']
            segments += [1] * (len(tokens_b) + 1)
        return tokens, segments

    def replace_mskelm_tokens(self, tokens, candidate_pred_positions,
                              num_mlm_preds, vocab_words):
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
                    masked_token = random.choice(vocab_words)
            mlm_input_tokens[mlm_pred_position] = masked_token
            pred_positions_and_labels.append(
                (mlm_pred_position, tokens[mlm_pred_position]))
        return mlm_input_tokens, pred_positions_and_labels

    def preprocess_text_data(self, path: str) -> List[List[str]]:
        """
        Args:
            path (str): The path to the text file.

        Returns:
            List[List[str]]: A list of tokenized sentences.
        """
        assert os.path.exists(path)
        paragraphs = []
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                if line.split(' . ') >= 2:
                    paragraph = line.strip().lower().split(' . ')
                    paragraphs.append(paragraph)

        return paragraphs

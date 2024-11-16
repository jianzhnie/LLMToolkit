import os
import sys

from torch.utils.data import Dataset

sys.path.append('../../')
from typing import List

import torch

from llmtoolkit.data.tokenizer import Tokenizer
from llmtoolkit.data.vocab import Vocab


class WikiTextDataset(Dataset):
    """PyTorch Dataset for processing WikiText data.

    Args:
        data_dir (str): The directory containing the data files.
        data_split (str, optional): The data split to load ('train', 'valid', or 'test'). Default is 'train'.
        tokenizer (Tokenizer, optional): An instance of Tokenizer for tokenizing text data.

    Attributes:
        tokenizer (Tokenizer): An instance of the Tokenizer used for tokenization.
        data_dir (str): The path to the data file.
        data_lst (List[List[str]]): A list of tokenized sentences.
        vocab (Vocab): The vocabulary built from the tokenized data.
    """

    def __init__(
            self,
            data_dir: str,
            data_split: str = 'train',
            tokenizer: Tokenizer = Tokenizer(),
            window_size: int = 3,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.data_dir = os.path.join(data_dir, data_split + '.txt')
        self.data_lst = self.preprocess(self.data_dir)
        self.ngram_lst = self.generate_ngram_dataset(window_size)
        self.vocab = Vocab.build_vocab(self.data_lst,
                                       min_freq=1,
                                       unk_token='<unk>',
                                       pad_token='<pad>',
                                       bos_token='<bos>',
                                       eos_token='<eos>')
        self.vocab.save_vocabulary(os.path.join(data_dir, 'vocab.txt'))

    def __len__(self) -> int:
        return len(self.data_lst)

    def __getitem__(self, index: int) -> List[str]:
        text, target = self.ngram_lst[index]
        inputs = self.vocab.to_index(text)
        targets = self.vocab.to_index(target)
        return inputs, targets

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return (inputs, targets)

    def generate_ngram_dataset(self, context_size):
        ngram_data = []
        for sentence in self.data_lst:
            ngram_data += self.get_ngram_data(sentence, context_size)
        return ngram_data

    def get_ngram_data(self, sentence, context_size):
        ngram_data = []
        for i in range(context_size, len(sentence)):
            context = [sentence[i - j - 1] for j in range(context_size)]
            context = context[::-1]
            target = sentence[i]
            ngram_data.append((context, target))
        return ngram_data

    def preprocess(self, path: str) -> List[List[str]]:
        """Tokenizes a text file and returns a list of tokenized sentences.

        Args:
            path (str): The path to the text file.

        Returns:
            List[List[str]]: A list of tokenized sentences.
        """
        assert os.path.exists(path)
        # Tokenize file content
        tokenized_sentences = []
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for sentence in lines:
                split_words = self.tokenizer.tokenize(sentence)
                if len(split_words) > 0:
                    tokenized_sentences.append(split_words)
        return tokenized_sentences

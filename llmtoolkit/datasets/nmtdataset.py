import sys

import torch

sys.path.append('../../')
from typing import List, Tuple

from torch.utils.data import Dataset

from llmtoolkit.data import Tokenizer, Vocab


class NMTDataset(Dataset):
    """Machine translation dataset.

    Args:
        file_path (str): The path to the machine translation data file.
        max_seq_len (int): Maximum sequence length for source and target sentences.

    Attributes:
        src_vocab (Vocab): Vocabulary for source language.
        tgt_vocab (Vocab): Vocabulary for target language.
    """

    def __init__(self, file_path: str = 'data', max_seq_len: int = 10):
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer()
        # Load datasets
        self.src_texts, self.tgt_texts = self.load_and_preprocess_data()
        # Tokenizer
        self.src_sentences, self.tgt_sentences = self._tokenize(
            self.src_texts, self.tgt_texts)

        # Build vocabularies
        self.src_vocab = self.build_vocab(self.src_sentences)
        self.tgt_vocab = self.build_vocab(self.tgt_sentences)

    def load_and_preprocess_data(
            self) -> Tuple[List[List[str]], List[List[str]]]:
        """Load and preprocess the machine translation dataset.

        Returns:
            Tuple[List[List[str]], List[List[str]]]: A tuple containing two lists.
                - The first list contains source sentences.
                - The second list contains target sentences.
        """
        source_texts = []
        target_texts = []

        with open(self.file_path, encoding='utf-8') as f:
            for line in f:
                line = line.replace('\u202f', ' ').replace('\xa0', ' ')
                parts = line.strip().split('\t')
                source_texts.append(parts[0])
                target_texts.append(parts[1])

        assert len(source_texts) == len(
            target_texts), 'Source and target sentence counts do not match.'

        return source_texts, target_texts

    def _tokenize(
            self, source_texts: List[str], target_texts: List[str]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Tokenize source and target texts.

        Args:
            source_texts (List[str]): List of source language texts.
            target_texts (List[str]): List of target language texts.

        Returns:
            Tuple[List[List[str]], List[List[str]]]: A tuple containing two lists.
                - The first list contains tokenized source sentences.
                - The second list contains tokenized target sentences.

        Raises:
            AssertionError: If the number of source and target sentences does not match.
        """
        src_sentences = []
        tgt_sentences = []

        for src_text, tgt_text in zip(source_texts, target_texts):
            src_tokens = self.tokenizer.tokenize(src_text)
            tgt_tokens = self.tokenizer.tokenize(tgt_text)
            src_sentences.append(src_tokens)
            tgt_sentences.append(tgt_tokens)

        assert len(src_sentences) == len(
            tgt_sentences
        ), 'The number of source and target sentences does not match.'

        return src_sentences, tgt_sentences

    def build_vocab(self, sentences: List[List[str]]) -> Vocab:
        """Build a vocabulary from a list of tokenized sentences.

        Args:
            sentences (List[List[str]]): List of tokenized sentences.

        Returns:
            Vocab: Vocabulary object.
        """
        return Vocab.build_vocab(
            sentences,
            min_freq=2,
            pad_token='<pad>',
            unk_token='<unk>',
            bos_token='<bos>',
            eos_token='<eos>',
        )

    def __getitem__(
            self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset."""
        # Convert tokens to indices
        src_txt = ['<bos>'] + self.src_sentences[idx] + ['<eos>']
        src_tokens = self.src_vocab.to_index(src_txt)
        tgt_txt = ['<bos>'] + self.tgt_sentences[idx] + ['<eos>']
        tgt_tokens = self.tgt_vocab.to_index(tgt_txt)

        # Ensure the sequences have the maximum sequence length
        src_tokens = self.truncate_pad(src_tokens, self.max_seq_len,
                                       self.src_vocab['<pad>'])
        tgt_tokens = self.truncate_pad(tgt_tokens, self.max_seq_len,
                                       self.tgt_vocab['<pad>'])
        # Convert to tensors
        src_tensor = self.to_tensor(src_tokens)
        tgt_tensor = self.to_tensor(tgt_tokens)
        # Get the valid lengths of the sequences
        src_len = src_tensor.ne(self.src_vocab['<pad>']).sum()
        tgt_len = tgt_tensor.ne(self.tgt_vocab['<pad>']).sum()
        return src_tensor, src_len, tgt_tensor, tgt_len

    def __len__(self) -> int:
        return len(self.src_sentences)

    @staticmethod
    def truncate_pad(tokens: List[int], max_length: int,
                     pad_token: int) -> List[int]:
        """Pad or truncate a list of tokens to a specified length.

        Args:
            tokens (List[int]): List of tokens.
            max_length (int): Maximum length to pad or truncate to.
            pad_token (int): Padding token.

        Returns:
            List[int]: Padded or truncated list of tokens.
        """
        if len(tokens) < max_length:
            tokens.extend([pad_token] * (max_length - len(tokens)))
        elif len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens

    @staticmethod
    def to_tensor(tokens: List[int]) -> torch.Tensor:
        """Convert a list of tokens to a PyTorch tensor.

        Args:
            tokens (List[int]): List of tokens.

        Returns:
            torch.Tensor: PyTorch tensor.
        """
        return torch.tensor(tokens, dtype=torch.long)


if __name__ == '__main__':
    root = '/home/robin/work_dir/llm/nlp-toolkit/data/nmt/fra-eng/fra.txt'
    nmtdataset = NMTDataset(file_path=root)
    for idx in range(10):
        src_tensor, tgt_tensor = nmtdataset[idx]
        print(src_tensor, tgt_tensor)

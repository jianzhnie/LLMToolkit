"""
Author: jianzhnie
Date: 2022-01-05 09:40:22
LastEditTime: 2022-03-04 17:17:59
LastEditors: jianzhnie
Description:

"""

import math
import random
import sys
from collections import defaultdict
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

sys.path.append('../../')
from llmtoolkit.data.vocab import Vocab
from llmtoolkit.utils.data_utils import (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
                                         UNK_TOKEN, load_ptb_data)


class RNNlmDataset(Dataset):

    def __init__(self, corpus, vocab):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            # 模型输入：BOS_TOKEN, w_1, w_2, ..., w_n
            input = [self.bos] + sentence
            # 模型输出：w_1, w_2, ..., w_n, EOS_TOKEN
            target = sentence + [self.eos]
            self.data.append((input, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = [torch.tensor(ex[1]) for ex in examples]
        # 对batch内的样本进行padding，使其具有相同长度
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad)
        targets = pad_sequence(targets,
                               batch_first=True,
                               padding_value=self.pad)
        return (inputs, targets)


class NGramDataset(Dataset):

    def __init__(self, corpus, vocab: Vocab, context_size=2):
        self.data = []
        self.bos = vocab.bos_token
        self.eos = vocab.eos_token
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            # 插入句首句尾符号
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size:
                continue
            for i in range(context_size, len(sentence)):
                # 模型输入：长为context_size的上文
                context = sentence[i - context_size:i]
                context = vocab.to_index(context)
                # 模型输出：当前词
                target = sentence[i]
                target = vocab.to_index(target)
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return (inputs, targets)


class CbowDataset(Dataset):

    def __init__(self, corpus, vocab: Vocab, context_size=2):
        self.data = []
        self.bos = vocab.bos_token
        self.eos = vocab.eos_token
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            if len(sentence) < context_size * 2 + 1:
                continue
            for i in range(context_size, len(sentence) - context_size):
                # 模型输入：左右分别取context_size长度的上下文
                context = (sentence[(i - context_size):i] +
                           sentence[(i + 1):(i + context_size + 1)])
                # 模型输出：当前词
                target = sentence[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples])
        targets = torch.tensor([ex[1] for ex in examples])
        return (inputs, targets)


class SkipGramDataset(Dataset):

    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                # 模型输入：当前词
                w = sentence[i]
                w = vocab.to_index(w)
                # 模型输出：一定窗口大小内的上下文
                left_context_index = max(0, i - context_size)
                right_context_index = min(len(sentence), i + context_size)
                context = (sentence[left_context_index:i] +
                           sentence[(i + 1):(right_context_index + 1)])
                context = vocab.to_index(context)
                self.data.extend([(w, c) for c in context])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples])
        targets = torch.tensor([ex[1] for ex in examples])
        return (inputs, targets)


class NegativeSampleingSkipGramDataset(Dataset):
    """Negative Sampleing for Skip-Gram Dataset."""

    def __init__(self,
                 corpus,
                 vocab: Vocab,
                 context_size=2,
                 n_negatives=5,
                 ns_dist=None):
        self.data = []
        self.bos = vocab.bos_token
        self.eos = vocab.eos_token
        self.pad = vocab.pad_token
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                # 模型输入：(w, context) ；输出为0/1，表示context是否为负样本
                w = sentence[i]
                w = vocab.to_index(w)
                left_context_index = max(0, i - context_size)
                right_context_index = min(len(sentence), i + context_size)
                context = (sentence[left_context_index:i] +
                           sentence[(i + 1):(right_context_index + 1)])
                context += [self.pad] * (2 * context_size - len(context))
                context = vocab.to_index(context)
                self.data.append((w, context))

        # 负样本数量
        self.n_negatives = n_negatives
        # 负采样分布：若参数ns_dist为None，则使用uniform分布
        self.ns_dist = ns_dist if ns_dist is not None else torch.ones(
            len(vocab))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        contexts = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        batch_size, context_size = contexts.shape
        neg_contexts = []
        # 对batch内的样本分别进行负采样
        for i in range(batch_size):
            # 保证负样本不包含当前样本中的context
            ns_dist = self.ns_dist.index_fill(0, contexts[i], 0.0)
            neg_contexts.append(
                torch.multinomial(ns_dist,
                                  self.n_negatives * context_size,
                                  replacement=True))
        neg_contexts = torch.stack(neg_contexts, dim=0)
        return words, contexts, neg_contexts


class GloveDataset(Dataset):

    def __init__(self, corpus, vocab: Vocab, context_size=2):
        # 记录词与上下文在给定语料中的共现次数
        self.cooccur_counts = defaultdict(float)
        self.bos = vocab.bos_token
        self.eos = vocab.eos_token
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                word = sentence[i]
                word = vocab.to_index(word)
                left_contexts = sentence[max(0, i - context_size):i]
                left_contexts = vocab.to_index(left_contexts)
                right_contexts = sentence[(i + 1):(
                    min(len(sentence), i + context_size) + 1)]
                right_contexts = vocab.to_index(right_contexts)
                # 共现次数随距离衰减: 1/d(w, c)
                for k, contexts_words in enumerate(left_contexts[::-1]):
                    self.cooccur_counts[(word, contexts_words)] += 1 / (k + 1)
                for k, contexts_words in enumerate(right_contexts):
                    self.cooccur_counts[(word, contexts_words)] += 1 / (k + 1)
        self.data = [
            (word, contexts_words, count)
            for (word, contexts_words), count in self.cooccur_counts.items()
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples])
        contexts = torch.tensor([ex[1] for ex in examples])
        counts = torch.tensor([ex[2] for ex in examples])
        return (words, contexts, counts)


class Word2VecDataset(object):
    """实现了 Word2Vec 训练所需的一些工具和数据处理函数的类。这个类包括了以下功能：

    1. 初始化 Word2VecToolkit，并构建词汇表（Vocab）。

    2. keep 方法用于确定是否保留一个词汇标记，根据词汇频率和阈值来决定。

    3. get_subsample_datasets 方法执行基于词汇频率和阈值的词汇子采样，返回经过子采样的句子列表。

    4. get_word_frequency 方法用于获取特定词汇标记的频率统计信息。

    5. get_skipgram_datasets 方法生成Skipgram模型的数据集，包括中心词和上下文词。

    6. get_negaitive_datasets 方法生成用于Skipgram模型的负采样数据集。

    7. get_negative_sample 方法生成Skipgram模型的负采样数据。

    8. generate_ngram_sample 生成 Ngram 模型的采样数据。

    9. generate_cbow_sample 生成CBOW模型的采样数据，包括上下文词和中心词。
    """

    def __init__(self, sentences: List[List[str]], threshold: float = 1e-4):
        """Initialize the Word2Vec Toolkit.

        Args:
            sentences (List[List[str]]): A list of lists containing tokenized sentences.
            threshold (float): The subsampling threshold.
        """
        self.sentences = sentences
        self.vocab = Vocab.build_vocab(
            sentences,
            min_freq=1,
            unk_token=UNK_TOKEN,
            pad_token=PAD_TOKEN,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
        )
        self.word_counter = self.vocab.get_token_freq()
        self.generator = self.init_generator(self.word_counter)
        self.threshold = threshold

    def should_keep_token(self, token: str, num_tokens: int) -> bool:
        """Determine whether to keep a token based on word frequencies and
        threshold.

        Args:
            token (str): The token to evaluate.
            num_tokens (int): The total number of tokens in the dataset.

        Returns:
            bool: True if the token should be kept, False otherwise.
        """
        ratio = self.word_counter[token] / num_tokens
        flag = random.uniform(0, 1) < math.sqrt(self.threshold / ratio)
        return flag

    def get_subsampled_datasets(self) -> List[List[str]]:
        """Perform word subsampling based on word frequencies and threshold.

        Returns:
            List[List[str]]: List of sentences with subsampled words.
        """
        num_tokens = sum(self.word_counter.values())
        self.subsampled_datasets = [[
            token for token in line
            if self.should_keep_token(token, num_tokens)
        ] for line in self.sentences]
        return self.subsampled_datasets

    def get_word_frequency(self, token: str) -> Tuple[int, int]:
        """Get the frequency of a specific token.

        Args:
            token (str): The token to query.

        Returns:
            Tuple[int, int]: The original and subsampled frequency counts of the token.
        """
        origin_count = sum(
            [sentence.count(token) for sentence in self.sentences])
        subsampled_count = sum(
            [sentence.count(token) for sentence in self.subsampled_datasets])

        return origin_count, subsampled_count

    def init_generator(self, word_counter: dict = None, ratio: float = 0.75):
        num_words = len(self.vocab.special_token_dict)
        total_words = len(self.vocab)
        population = list(range(num_words, total_words))
        weights = [
            word_counter[self.vocab.to_tokens(idx)]**ratio
            for idx in population
        ]
        generator = RandomGenerator(population, weights)
        return generator

    def get_negative_sample(self,
                            contexts: List[str],
                            n_negatives: int = 5) -> List[List[str]]:
        """Generate negative samples for Skipgram datasets.

        Args:
            contexts List[str]: Lists of contexts.
            n_negatives (int): Number of negative samples per context.

        Returns:
            List[str]: Lists of negative samples.
        """
        negatives = []
        while len(negatives) < n_negatives * len(contexts):
            negative_idx = self.generator.draw()
            negative_word = self.vocab.to_tokens(negative_idx)
            if negative_word not in contexts:
                negatives.append(negative_word)
        return negatives

    def get_negative_datasets(self,
                              all_contexts: List[List[str]],
                              n_negatives: int = 5) -> List[List[str]]:
        """Generate negative samples for Skipgram datasets.

        Args:
            contexts List[List[str]]: Lists of contexts.
            n_negatives (int): Number of negative samples per context.

        Returns:
            List[List[str]]: Lists of negative samples.
        """
        all_negatives = []
        for context in all_contexts:
            all_negatives.append(self.get_negative_sample(
                context, n_negatives))
        return all_negatives

    def get_skipgram_datasets(
            self,
            sentences,
            context_size: int = 2) -> Tuple[List[List[str]], List[List[str]]]:
        """Generate Skipgram datasets.

        Args:
            context_size (int): The context window size.

        Returns:
            Tuple[List[List[str]], List[List[str]]]: Lists of centers and contexts.
        """
        centers, contexts = [], []
        for sentence in sentences:
            if len(sentence) < 2:
                continue
            skipgram_data = self.generate_skipgram_sample(
                sentence, context_size)
            center, context = zip(*skipgram_data)
            centers.append(center)
            contexts.append(context)
        return centers, contexts

    def generate_ngram_sample(
            self, sentence: List[str],
            context_size: int) -> List[Tuple[List[str], str]]:
        """Generate an n-gram dataset from a given sentence.

        Args:
            sentence (List[str]): The input sentence as a list of words.
            context_size (int): The size of the context for each n-gram.

        Returns:
            List[Tuple[List[str], str]]: A list of tuples, each containing a context list and a target word.

        Example:
            >>> sentence = ["I", "love", "to", "eat", "ice", "cream"]
            >>> context_size = 2
            >>> data = generate_ngram_dataset(sentence, context_size)
            >>> print(data)
            [(['I', 'love'], 'to'), (['love', 'to'], 'eat'), (['to', 'eat'], 'ice'), (['eat', 'ice'], 'cream')]
        """
        ngram_data = []
        sentence_length = len(sentence)
        if sentence_length < context_size:
            return []

        for i in range(context_size, sentence_length):
            # Construct the context for the current n-gram
            context = sentence[i - context_size:i]
            target = sentence[i]
            ngram_data.append((context, target))

        return ngram_data

    def generate_cbow_sample(
            self,
            sentence: List[str],
            context_size: int,
            use_random_window: bool = True) -> List[Tuple[List[str], str]]:
        """Generate a CBOW (Continuous Bag of Words) dataset from a given
        sentence.

        Args:
            sentence (List[str]): The input sentence as a list of words.
            context_size (int): The context window size, determining the number of words to consider before and after the target word.

        Returns:
            List[Tuple[List[str], str]]: A list of tuples, where each tuple contains a list of context words and the target word.

        Example:
            >>> sentence = ["I", "love", "to", "eat", "ice", "cream"]
            >>> context_size = 2
            >>> data = generate_cbow_dataset(sentence, context_size)
            >>> print(data)
            [(['love', 'to'], 'I'),
            (['I', 'to', 'eat'], 'love'),
            (['I', 'love', 'eat', 'ice'], 'to'),
            (['love', 'to', 'ice', 'cream'], 'eat'),
            (['to', 'eat', 'cream'], 'ice'),
            (['eat', 'ice'], 'cream')]
        """
        cbow_data = []
        sentence_length = len(sentence)
        if sentence_length < 2:
            return []
        for i in range(sentence_length):
            window_size = (random.randint(1, context_size)
                           if use_random_window else context_size)

            # Determine the context word indices within the window
            left_idx = max(0, i - window_size)
            right_idx = min(sentence_length, i + 1 + window_size)
            context_indices = list(range(left_idx, right_idx))
            # Get the target word
            center_word = sentence[i]
            # Exclude the target word from the context words
            context_indices.remove(i)
            context_word = [sentence[idx] for idx in context_indices]

            # Append the context and target as a tuple to the dataset
            cbow_data.append((context_word, center_word))

        return cbow_data

    def generate_skipgram_sample(
            self,
            sentence: List[str],
            context_size: int,
            use_random_window: bool = True
    ) -> Tuple[List[str], List[List[str]]]:
        """Generate an Skip-gram dataset from a given sentence.
        返回跳元模型中的中心词和上下文词.

        在Word2Vec中，上下文窗口是指在训练过程中，用来确定中心词周围哪些词作为上下文进行预测的窗口大小。这个窗口的大小可以影响到模型的性能和训练效果。

        使用随机采样一个介于1到max_window_size之间的整数作为上下文窗口，而不是固定使用max_window_size，有一些原理和考虑：

        - Diversification（多样性）：随机采样窗口大小可以引入更多的多样性，使得模型不仅仅关注固定大小的上下文窗口。这样可以更好地捕捉不同距离的关系，
            避免过于集中在一个特定的上下文范围内。

        - 平衡计算复杂度：较大的上下文窗口可能会引入更多的噪声，因为与中心词距离较远的词可能与中心词的语义关联较弱。同时，较大的窗口可能会增加计算的复杂度,
            因为要考虑更多的上下文词。通过随机采样窗口大小，可以在捕捉不同距离关系的同时，平衡计算的复杂度。

        - 训练稳定性：随机窗口大小可以提高模型的训练稳定性。如果总是使用固定的 max_window_size，可能会使模型对于固定距离范围内的关系过于敏感，从而导致训练不稳定。

        - 对罕见词的处理：对于罕见词，较大的上下文窗口可能会导致模型将与中心词无关的词考虑在内，从而影响预测的准确性。随机窗口大小可以在一定程度上缓解这个问题。

        综合考虑以上因素，使用随机采样一个介于1到max_window_size之间的窗口大小可以帮助模型更好地学习单词之间的关系，提高训练的效果和稳定性。

        Args:
            sentence (List[str]): The input sentence as a list of words.
            context_size (int): The maximum window size for context words.

        Returns:
            List[Tuple[List[str], str]]: A list of tuples, where each tuple contains a list of context words and the target word.

        Example:
        ```python
            >>> sentence = ["I", "love", "to", "eat", "ice", "cream"]
            >>> context_size = 2
            >>> data = generate_skipgram_dataset(sentence, context_size)
            >>> print(data)
                [('I', ['love', 'to']),
                ('love', ['I', 'to', 'eat']),
                ('to', ['I', 'love', 'eat', 'ice']),
                ('eat', ['love', 'to', 'ice', 'cream']),
                ('ice', ['to', 'eat', 'cream']),
                ('cream', ['eat', 'ice'])]
        ```
        """
        skipgram_data = []

        sentence_length = len(sentence)
        # To form "center word - context word" pairs, a sentence should have at least 2 words.
        if sentence_length < 2:
            return []

        for i in range(sentence_length):
            window_size = (random.randint(1, context_size)
                           if use_random_window else context_size)

            # Determine the context word indices within the window
            left_idx = max(0, i - window_size)
            right_idx = min(sentence_length, i + 1 + window_size)
            context_indices = list(range(left_idx, right_idx))
            center_word = sentence[i]
            # Exclude the center word from the context words
            context_indices.remove(i)
            context_words = [sentence[idx] for idx in context_indices]
            # Append the center word and context words as a tuple to the dataset
            skipgram_data.append((center_word, context_words))

        return skipgram_data


class RandomGenerator:
    """Randomly sample integers from a population based on given weights.

    It is used for making random selections from a given sequence (list, string, or any iterable) with replacement.
    This means that items can be selected more than once, and the probability of each item being selected is uniform.

    Args:
        population: This is the sequence from which you want to make random selections.
        weights (optional): This parameter allows you to specify a list of weights for the items in the population. \n
        The weights determine the probability of each item being selected. If not provided, all items are assumed to have equal probability.
        k: This is the number of random selections you want to make. It defaults to 1 if not specified.
    Example:
        >>> generator = RandomGenerator(population = ['apple', 'banana', 'cherry', 'date'], weights = [0.2, 0.3, 0.4, 0.1])
        >>> sample = [generator.draw() for _ in range(10)]
    """

    def __init__(self,
                 population: List[int] = None,
                 weights: List[float] = None):
        self.population = population
        self.weights = weights
        self.candidates = []
        self.i = 0

    def draw(self) -> int:
        """Draw a random sample based on the provided weights.

        Returns:
            int: A randomly sampled integer.
        """
        if self.i == len(self.candidates):
            # Cache k random samples
            self.candidates = random.choices(self.population,
                                             self.weights,
                                             k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


# Example usage:
if __name__ == '__main__':
    generator = RandomGenerator(
        population=['apple', 'banana', 'cherry', 'date'],
        weights=[0.2, 0.3, 0.4, 0.1])
    sample = [generator.draw() for _ in range(10)]
    print(f'Randomly sampled value: {sample}')

    corpus = ['I', 'love', 'to', 'eat', 'ice', 'cream']
    print(corpus)
    context_size = 2

    word2vec_dataset = Word2VecDataset(corpus)

    ngrm = word2vec_dataset.generate_ngram_sample(corpus, context_size)
    print(ngrm)

    cbow = word2vec_dataset.generate_cbow_sample(corpus, context_size)
    print(cbow)

    skip = word2vec_dataset.generate_skipgram_sample(corpus, context_size)
    print(skip)

    data_dir = '/home/robin/work_dir/llm/nlp-toolkit/data/ptb'
    sentences = load_ptb_data(data_dir, split='train')

    word2vec_dataset = Word2VecDataset(sentences)
    subsample = word2vec_dataset.get_subsampled_datasets()
    a, b = word2vec_dataset.get_word_frequency('the')
    print(a, b)

    all_centers, all_contexts = word2vec_dataset.get_skipgram_datasets(
        sentences)

    print(all_centers[:5], all_contexts[:5])
    all_negativaes = word2vec_dataset.get_negative_datasets(all_contexts)
    print(all_negativaes[:5])

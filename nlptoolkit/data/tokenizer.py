'''
Author: jianzhnie
Date: 2021-12-29 16:07:11
LastEditTime: 2022-01-04 10:37:58
LastEditors: jianzhnie
Description:

'''

import re
from typing import List, Union

import jieba

from .vocab import Vocab


class BaseTokenizer(object):
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def cut(self, sentence):
        pass

    def encode(self, sentence):
        pass


class Tokenizer:
    """
    Tokenizes text lines into word or character tokens.

    Args:
        lang (str, optional): Language identifier. Default is 'en'.

    Attributes:
        lang (str): Language identifier.

    Usage:
    ```
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize("Hello, World!", token='word')
    ```

    Defined in :numref:`sec_utils`
    """
    def __init__(self, lang: str = 'en'):
        self.lang = lang

    def tokenize(self,
                 sentence: str,
                 token: str = 'word') -> Union[List[str], List[List[str]]]:
        """
        Tokenize the input sentence into word or character tokens.

        Args:
            sentence (str): The input sentence to tokenize.
            token (str, optional): Token type. Either 'word' or 'char'. Default is 'word'.

        Returns:
            Union[List[str], List[List[str]]]: A list of tokens.

        Raises:
            ValueError: If an unknown token type is provided.

        Usage:
        ```
        tokens = tokenizer.tokenize("Hello, World!", token='word')
        ```

        """
        # 将句子中的特殊字符（如星号、引号、换行符、反斜杠、加号、减号、斜杠、等号、括号、单引号、冒号、方括号、竖线、感叹号和分号）
        # 替换为一个空格。

        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", ' ',
                          str(sentence))
        # 将连续多个空格替换为一个空格。这有助于将多个连续空格合并成一个。
        sentence = re.sub(r'[ ]+', ' ', sentence)
        # 将连续多个感叹号替换为一个感叹号，类似地，后面的行也分别用于处理连续的逗号和问号。
        sentence = re.sub(r'\!+', '!', sentence)
        sentence = re.sub(r'\,+', ',', sentence)
        sentence = re.sub(r'\?+', '?', sentence)
        # 替换非字母字符（包括数字、符号和空格）为一个空格。这将确保句子中只包含字母字符。
        sentence = re.sub(r'[^A-Za-z]+', ' ', sentence)
        # 将整个句子转换为小写字母，以确保文本的一致性，因为在自然语言处理任务中通常不区分大小写。
        sentence = sentence.lower()
        # 接下来，根据指定的token类型，函数将句子分割成单词或字符，并返回结果：
        if token == 'word':
            return sentence.split()
        elif token == 'char':
            return [list(word) for word in sentence.split()]
        else:
            raise ValueError('Unknown token type: ' + token)


class JiebaTokenizer(BaseTokenizer):
    """
    Constructs a tokenizer based on `jieba <https://github.com/fxsjy/jieba>`__.
    It supports :meth:`cut` method to split the text to tokens, and :meth:`encode`
    method to covert text to token ids.

    Args:
        vocab(paddlenlp.data.Vocab): An instance of :class:`paddlenlp.data.Vocab`.
    """
    def __init__(self, vocab: Vocab):
        super(JiebaTokenizer, self).__init__(vocab)

        self.tokenizer = jieba.Tokenizer()
        # initialize tokenizer
        self.tokenizer.FREQ = {
            key: 1
            for key in self.vocab.token_to_idx.keys()
        }
        self.tokenizer.total = len(self.tokenizer.FREQ)
        self.tokenizer.initialized = True

    def cut(self, sentence, cut_all=False, use_hmm=True):
        """
        The method used to cut the text to tokens.

        Args:
            sentence(str): The text that needs to be cuted.
            cut_all(bool, optional): Whether to use the full mode. If True,
                using full mode that gets all the possible words from the
                sentence, which is fast but not accurate. If False, using
                accurate mode that attempts to cut the sentence into the most
                accurate segmentations, which is suitable for text analysis.
                Default: False.
            use_hmm(bool, optional): Whether to use the HMM model. Default: True.

        Returns:
            list[str]: A list of tokens.

        Example:
            .. code-block:: python

                from paddlenlp.data import Vocab, JiebaTokenizer
                # The vocab file. The sample file can be downloaded firstly.
                # wget https://bj.bcebos.com/paddlenlp/data/senta_word_dict.txt
                vocab_file_path = './senta_word_dict.txt'
                # Initialize the Vocab
                vocab = Vocab.load_vocabulary(
                    vocab_file_path,
                    unk_token='[UNK]',
                    pad_token='[PAD]')
                tokenizer = JiebaTokenizer(vocab)

                tokens = tokenizer.cut('我爱你中国')
                print(tokens)
                # ['我爱你', '中国']
        """
        return self.tokenizer.lcut(sentence, cut_all, use_hmm)

    def encode(self, sentence, cut_all=False, use_hmm=True):
        """
        The method used to convert the text to ids. It will firstly call
        :meth:`cut` method to cut the text to tokens. Then, convert tokens to
        ids using `vocab`.

        Args:
            sentence(str): The text that needs to be cuted.
            cut_all(bool, optional): Whether to use the full mode. If True,
                using full mode that gets all the possible words from the
                sentence, which is fast but not accurate. If False, using
                accurate mode that attempts to cut the sentence into the most
                accurate segmentations, which is suitable for text analysis.
                Default: False.
            use_hmm(bool, optional): Whether to use the HMM model. Default: True.

        Returns:
            list[int]: A list of ids.

        Example:
            .. code-block:: python

                from paddlenlp.data import Vocab, JiebaTokenizer
                # The vocab file. The sample file can be downloaded firstly.
                # wget https://bj.bcebos.com/paddlenlp/data/senta_word_dict.txt
                vocab_file_path = './senta_word_dict.txt'
                # Initialize the Vocab
                vocab = Vocab.load_vocabulary(
                    vocab_file_path,
                    unk_token='[UNK]',
                    pad_token='[PAD]')
                tokenizer = JiebaTokenizer(vocab)

                ids = tokenizer.encode('我爱你中国')
                print(ids)
                # [1170578, 575565]
        """
        words = self.cut(sentence, cut_all, use_hmm)

        return [
            self.get_idx_from_word(word, self.vocab.token_to_idx,
                                   self.vocab.unk_token) for word in words
        ]

    def get_idx_from_word(self, word, word_to_idx, unk_word):
        if word in word_to_idx:
            return word_to_idx[word]
        return word_to_idx[unk_word]

"""Text tokenization module for NLP tasks.

This module provides base and concrete implementations of tokenizers
that can process text into tokens at both word and character levels.

在自然语言处理（NLP）和文本处理领域，有许多不同的词元化方法，常用的包括以下几种：

1.空格分词（Whitespace Tokenization）：

    将文本按照空格进行分割，将每个分割后的词作为一个词元。例如，"Hello world!"会被分割为[“Hello”, “world!”]。

2.分词器（Tokenizer）：

    使用专门的分词器工具，如NLTK（Natural Language Toolkit）或spaCy，在文本中识别和分割单词。这些分词器可以处理更复杂的分割规则，例如处理标点符号、缩写词等。

3.n-gram分词：

    将文本切分为连续的n个词的组合，这些组合称为n-gram。常见的是二元（bigram）和三元（trigram）分词。例如，"Hello world!"的bigram分词为[“Hello world”, “world!”]。

4.字符级别分词（Character-level Tokenization）：

    将文本按照字符进行分割，将每个字符作为一个词元。这种方法对于处理字符级别的任务（如拼写检查、机器翻译等）很有用。

5.子词（Subword）分词：

    将单词切分为更小的单元，如词根、前缀或后缀等。这种方法可以处理未登录词（out-of-vocabulary）问题，并且对于具有复杂形态的语言（如德语、芬兰语）也很有效。常见的子词分词算法有Byte-Pair Encoding（BPE）和SentencePiece。

6.正则表达式分词（Regular Expression Tokenization）：

    使用正则表达式来定义词元的分割规则。这样可以灵活地根据需要定义分割规则，适用于特定领域的文本处理任务。

7.词干提取（Stemming）和词形还原（Lemmatization）：

    词干提取和词形还原是将单词转换为其基本形式的方法，通常用于减少词汇的复杂性。它们可以与其他词元化方法结合使用。

8.专业领域的自定义分词：

    对于特定领域的文本，有时需要开发自定义的分词方法，以处理领域特有的术语和规则。
这些词元化方法可以根据任务和语言的不同进行组合和调整，选择合适的方法有助于提高NLP应用的性能和效果。
"""

import re
from abc import ABC, abstractmethod
from typing import List, Union

import jieba

from llmtoolkit.data.vocab import Vocab


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers.

    This class defines the interface that all tokenizer implementations
    must follow, providing basic tokenization and encoding/decoding functionality.

    Attributes:
        vocab (Vocab): Vocabulary object for token-to-index mapping
    """

    def __init__(self, vocab: Vocab) -> None:
        """Initialize the tokenizer with a vocabulary.

        Args:
            vocab: Vocabulary object for token/index conversion
        """
        self.vocab = vocab

    @abstractmethod
    def cut(self, sentence: str) -> List[str]:
        """Cut a sentence into tokens.

        Args:
            sentence: Input text to tokenize

        Returns:
            List of tokens
        """
        pass

    @abstractmethod
    def encode(self, sentence: str) -> List[int]:
        """Convert a sentence into token indices.

        Args:
            sentence: Input text to encode

        Returns:
            List of token indices
        """
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert token indices back into text.

        Args:
            ids: List of token indices

        Returns:
            Reconstructed text
        """
        pass


class Tokenizer(BaseTokenizer):
    """Concrete tokenizer implementation supporting word and character-level
    tokenization.

    This tokenizer handles text preprocessing, including:
    - Special character removal
    - Whitespace normalization
    - Case normalization
    - Punctuation normalization
    - Word/character-level tokenization

    Attributes:
        vocab (Vocab): Vocabulary object for token-to-index mapping
        _special_chars_pattern: Regex pattern for special characters
        _whitespace_pattern: Regex pattern for whitespace normalization
        _punctuation_patterns: Dict of patterns for punctuation normalization
    """

    # Class-level regex patterns for better performance
    _special_chars_pattern = re.compile(r"[\*\""
                                        "\n\\…\+\-\/\=\(\)'•:\|'\!;]")
    _whitespace_pattern = re.compile(r'[ ]+')
    _punctuation_patterns = {
        'exclamation': re.compile(r'\!+'),
        'comma': re.compile(r'\,+'),
        'question': re.compile(r'\?+'),
        'non_alpha': re.compile(r'[^A-Za-z]+'),
    }

    def __init__(self, vocab: Vocab) -> None:
        """Initialize the tokenizer.

        Args:
            vocab: Vocabulary object for token/index conversion
        """
        super().__init__(vocab)

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by normalizing characters, whitespace, and case.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        # 将句子中的特殊字符（如星号、引号、换行符、反斜杠、加号、减号、斜杠、等号、括号、单引号、冒号、方括号、竖线、感叹号和分号）
        # 替换为一个空格。
        # Replace special characters with space
        text = self._special_chars_pattern.sub(' ', str(text))

        # Normalize whitespace
        # 将连续多个空格替换为一个空格。这有助于将多个连续空格合并成一个。
        text = self._whitespace_pattern.sub(' ', text)

        # Normalize punctuation
        text = self._punctuation_patterns['exclamation'].sub('!', text)
        text = self._punctuation_patterns['comma'].sub(',', text)
        text = self._punctuation_patterns['question'].sub('?', text)

        # Replace non-alphabetic characters with space
        text = self._punctuation_patterns['non_alpha'].sub(' ', text)

        # Convert to lowercase
        return text.lower().strip()

    def tokenize(
            self,
            sentence: str,
            token_type: str = 'word') -> Union[List[str], List[List[str]]]:
        """Tokenize text into word or character tokens.

        Args:
            sentence: Input text to tokenize
            token_type: Tokenization type ('word' or 'char')

        Returns:
            For token_type='word': List of word tokens
            For token_type='char': List of character token lists

        Raises:
            ValueError: If token_type is not 'word' or 'char'
        """
        if not isinstance(sentence, str):
            raise TypeError(f'Expected string input, got {type(sentence)}')

        # Preprocess the text
        processed_text = self.preprocess_text(sentence)

        # Tokenize based on specified type
        if token_type == 'word':
            return processed_text.split()
        elif token_type == 'char':
            return [list(word) for word in processed_text.split()]
        else:
            raise ValueError(
                f"Unknown token type: {token_type}. Expected 'word' or 'char'")

    def cut(self, sentence: str) -> List[str]:
        """Cut text into word tokens.

        Args:
            sentence: Input text to tokenize

        Returns:
            List of word tokens
        """
        return self.tokenize(sentence, token_type='word')

    def encode(self, sentence: str) -> List[int]:
        """Convert text into token indices.

        Args:
            sentence: Input text to encode

        Returns:
            List of token indices
        """
        tokens = self.cut(sentence)
        return self.vocab.to_index(tokens)

    def decode(self, ids: List[int]) -> str:
        """Convert token indices back into text.

        Args:
            ids: List of token indices

        Returns:
            Reconstructed text
        """
        tokens = self.vocab.to_tokens(ids)
        return ' '.join(tokens)


class JiebaTokenizer(BaseTokenizer):
    """Constructs a tokenizer based on `jieba.

    <https://github.com/fxsjy/jieba>`__. It supports :meth:`cut` method to
    split the text to tokens, and :meth:`encode` method to convert text to token
    ids.

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
        """The method used to cut the text to tokens.

        Args:
            sentence(str): The text that needs to be cut.
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
            sentence(str): The text that needs to be cut.
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
        tokens = self.cut(sentence, cut_all, use_hmm)
        return self.vocab.to_index(tokens)

    def decode(self, ids: List[int]) -> str:
        """Convert token indices back into text.

        Args:
            ids: List of token indices

        Returns:
            Reconstructed text
        """
        if not ids:
            raise ValueError('Token IDs list cannot be empty')
        tokens = self.vocab.to_tokens(ids)
        return ' '.join(tokens)

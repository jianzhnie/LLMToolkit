"""GPT-2 Tokenizer Implementation.

This module implements the GPT-2 tokenizer using Byte-Pair Encoding (BPE).
It provides functionality to encode text into tokens and decode tokens back to text,
with proper handling of Unicode characters and special cases.

Main components:
- ByteEncoder: Handles byte-to-unicode conversion
- TokenPairGenerator: Generates token pairs for BPE
- GPT2Tokenizer: Main tokenizer implementation
- VocabularyLoader: Handles loading and downloading tokenizer vocabulary
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import regex as re
import requests
from tqdm import tqdm


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""

    model_name: str
    models_dir: Union[str, Path]
    errors: str = 'replace'
    cache_size: int = 128


class ByteEncoder:
    """Handles conversion between bytes and unicode characters."""

    @staticmethod
    @lru_cache(maxsize=128)
    def bytes_to_unicode() -> Dict[int, str]:
        """Creates a mapping from bytes to unicode characters.

        Returns:
            Dict[int, str]: Mapping from byte values to unicode characters.
        """
        # Basic ASCII range (printable characters)
        bs: List[int] = list(range(ord('!'), ord('~') + 1))
        # Latin-1 supplement block
        bs.extend(list(range(ord('¡'), ord('¬') + 1)))
        bs.extend(list(range(ord('®'), ord('ÿ') + 1)))

        cs: List[int] = bs[:]
        n: int = 0

        # Handle remaining byte values
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1

        return dict(zip(bs, [chr(n) for n in cs]))


class TokenPairGenerator:
    """Generates pairs of tokens for BPE algorithm."""

    @staticmethod
    def get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """Generate adjacent pairs of characters from a word.

        Args:
            word: Tuple of characters representing a word.

        Returns:
            Set of character pairs.
        """
        pairs: Set[Tuple[str, str]] = set()
        prev_char = word[0]

        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char

        return pairs


class GPT2Tokenizer:
    """Main GPT-2 tokenizer implementation using Byte-Pair Encoding.

    Attributes:
        encoder: Dictionary mapping tokens to IDs.
        decoder: Dictionary mapping IDs to tokens.
        byte_encoder: Mapping from bytes to unicode.
        byte_decoder: Mapping from unicode to bytes.
        bpe_ranks: Dictionary of BPE merge rankings.
        cache: Cache for BPE results.
        pattern: Regex pattern for tokenization.
    """

    def __init__(self, config: TokenizerConfig):
        """Initialize the tokenizer with the given configuration.

        Args:
            config: TokenizerConfig instance with initialization parameters.

        Raises:
            TokenizationError: If vocabulary files cannot be loaded.
        """
        try:
            vocab_loader = VocabularyLoader(config.model_name,
                                            config.models_dir)
            self.encoder, bpe_merges = vocab_loader.load_vocabulary()
        except Exception as e:
            raise Exception(f'Failed to load vocabulary: {str(e)}')

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = config.errors

        # Initialize byte encoding mappings
        self.byte_encoder = ByteEncoder.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Initialize BPE rankings
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache: Dict[str, str] = {}

        # Compile regex pattern for tokenization
        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d"""  # Contractions
            r"""| ?\p{L}+"""  # Letters
            r"""| ?\p{N}+"""  # Numbers
            r"""| ?[^\s\p{L}\p{N}]+"""  # Special characters
            r"""|\s+(?!\S)|\s+"""  # Whitespace
        )

    def bpe(self, token: str) -> str:
        """Apply Byte-Pair Encoding to a token.

        Args:
            token: Input token to encode.

        Returns:
            BPE-encoded token.
        """
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = TokenPairGenerator.get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word: List[str] = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[
                        i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break

            pairs = TokenPairGenerator.get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        """Encode text into tokens.

        Args:
            text: Input text to tokenize.

        Returns:
            List of token IDs.

        Raises:
            TokenizationError: If encoding fails.
        """
        try:
            bpe_tokens: List[int] = []

            for token in re.findall(self.pattern, text):
                # Convert token to byte-level representation
                token = ''.join(self.byte_encoder[b]
                                for b in token.encode('utf-8'))

                # Apply BPE and convert to token IDs
                bpe_token_ids = [
                    self.encoder[bpe_token]
                    for bpe_token in self.bpe(token).split(' ')
                ]
                bpe_tokens.extend(bpe_token_ids)

            return bpe_tokens

        except Exception as e:
            raise Exception(f'Failed to encode text: {str(e)}')

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text.

        Args:
            tokens: List of token IDs to decode.

        Returns:
            Decoded text.

        Raises:
            TokenizationError: If decoding fails.
        """
        try:
            # Convert token IDs back to text
            text = ''.join(self.decoder[token] for token in tokens)

            # Convert byte-level representation back to UTF-8 text
            decoded_bytes = bytearray(self.byte_decoder[c] for c in text)
            return decoded_bytes.decode('utf-8', errors=self.errors)

        except Exception as e:
            raise Exception(f'Failed to decode tokens: {str(e)}')


class VocabularyLoader:
    """Handles loading and downloading of tokenizer vocabulary files."""

    def __init__(self, model_name: str, models_dir: Union[str, Path]):
        """Initialize the vocabulary loader.

        Args:
            model_name: Name of the model.
            models_dir: Directory containing model files.
        """
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.vocab_files = {
            'encoder': self.models_dir / model_name / 'encoder.json',
            'vocab': self.models_dir / model_name / 'vocab.bpe',
        }

    def load_vocabulary(self) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
        """Load vocabulary files.

        Returns:
            Tuple of encoder dictionary and BPE merges.

        Raises:
            FileNotFoundError: If vocabulary files are missing.
        """
        # Load encoder
        try:
            with open(self.vocab_files['encoder'], 'r') as f:
                encoder = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Encoder file not found at {self.vocab_files['encoder']}")

        # Load BPE vocabulary
        try:
            with open(self.vocab_files['vocab'], 'r', encoding='utf-8') as f:
                bpe_data = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabulary file not found at {self.vocab_files['vocab']}")

        # Parse BPE merges
        bpe_merges = [
            tuple(merge_str.split())
            for merge_str in bpe_data.split('\n')[1:-1]
        ]

        return encoder, bpe_merges

    def download_vocabulary(self, force: bool = False) -> None:
        """Download vocabulary files from OpenAI's servers.

        Args:
            force: If True, download files even if they exist.

        Raises:
            requests.RequestException: If download fails.
        """
        # Create directory if it doesn't exist
        vocab_dir = self.models_dir / self.model_name
        vocab_dir.mkdir(parents=True, exist_ok=True)

        base_url = 'https://openaipublic.blob.core.windows.net/gpt-2/models/117M/'

        for filename in ['encoder.json', 'vocab.bpe']:
            file_path = vocab_dir / filename

            if not force and file_path.exists():
                continue

            # Download file
            response = requests.get(base_url + filename, stream=True)
            response.raise_for_status()

            # Save file with progress bar
            file_size = int(response.headers['content-length'])
            chunk_size = 1000

            with open(file_path, 'wb') as f:
                with tqdm(
                        ncols=100,
                        desc=f'Downloading {filename}',
                        total=file_size,
                        unit_scale=True,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))


def create_tokenizer(model_name: str,
                     models_dir: Union[str, Path],
                     download_if_missing: bool = True) -> GPT2Tokenizer:
    """Factory function to create a GPT2Tokenizer instance.

    Args:
        model_name: Name of the model.
        models_dir: Directory containing model files.
        download_if_missing: Whether to download vocabulary files if missing.

    Returns:
        Configured GPT2Tokenizer instance.

    Raises:
        TokenizationError: If tokenizer creation fails.
    """
    try:
        config = TokenizerConfig(model_name=model_name, models_dir=models_dir)

        if download_if_missing:
            loader = VocabularyLoader(model_name, models_dir)
            loader.download_vocabulary()

        return GPT2Tokenizer(config)

    except Exception as e:
        raise Exception(f'Failed to create tokenizer: {str(e)}')

import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from llmtoolkit.data.tokenizers import Tokenizer
from llmtoolkit.data.vocab import Vocab
from llmtoolkit.utils.data_utils import truncate_pad

IGNORE_INDEX = -100


class TrainingInstance:
    """A single training instance (sentence pair) for masked language model
    training.

    Args:
        tokens (List[str]): List of tokens representing the masked input of the sentence pair.
        segment_ids (List[int]): List of segment IDs indicating token belongingness to sentences.
        is_next_sentence (bool): Indicates if the next sentence is the next sentence or randomly chosen.
        masked_lm_labels (List[str]): List of tokens representing the real token of the sentence pair.
        masked_lm_pred_positions (List[int]): List of positions where tokens are masked during training.
        masked_lm_pred_labels (List[str]): List of masked tokens corresponding to masked positions.
    """

    def __init__(
        self,
        tokens: List[str],
        segment_ids: List[int],
        is_next_sentence: bool,
        masked_lm_labels: List[str],
        masked_lm_pred_positions: List[int],
        masked_lm_pred_labels: List[str],
    ) -> None:
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_next_sentence = is_next_sentence
        self.masked_lm_labels = masked_lm_labels
        self.masked_lm_pred_positions = masked_lm_pred_positions
        self.masked_lm_pred_labels = masked_lm_pred_labels

    def __str__(self) -> str:
        """Returns a string representation of the TrainingInstance object.

        Returns:
            str: String representation of the TrainingInstance.
        """
        tokens_str = ' '.join(self.tokens)
        segment_ids_str = ' '.join(map(str, self.segment_ids))
        masked_positions_str = ' '.join(map(str,
                                            self.masked_lm_pred_positions))
        masked_labels_str = ' '.join(self.masked_lm_pred_labels)

        outputs = ''
        outputs += f'Tokens: {tokens_str}\n'
        outputs += f'Segment IDs: {segment_ids_str}\n'
        outputs += f'Masked Pred Positions: {masked_positions_str}\n'
        outputs += f'Masked Pred Labels: {masked_labels_str}\n'
        outputs += f'Is Next Sentence: {self.is_next_sentence}'

        return outputs

    def __repr__(self) -> str:
        """Returns a string representation of the TrainingInstance object.

        Returns:
            str: String representation of the TrainingInstance.
        """
        return self.__str__()


class BertDataset(Dataset):
    """Dataset class for BERT pretraining tasks.

    Args:
        data_dir (str): Directory containing the data files.
        data_split (str): Split of the data ('train', 'val', etc.).
        tokenizer (Tokenizer): Tokenizer object for tokenizing the text.
        max_seq_len (int): Maximum sequence length for BERT input.

    Example:
        data_dir = '~/text_data/wikitext-2/'

        bert_dataset = BertDataset(data_dir=data_dir, max_seq_len=128)
    """

    def __init__(
            self,
            data_dir: str,
            data_split: str = 'train',
            tokenizer: Tokenizer = Tokenizer(),
            max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data_dir = os.path.join(data_dir, f'{data_split}.txt')
        self.paragraphs = self.preprocess_text_data(self.data_dir)
        self.tokenized_paragraphs = self.tokenize_text(self.paragraphs)
        self.vocab: Vocab = self.build_vocab()
        self.vocab_words = list(self.vocab.token_to_idx.keys())
        self.bert_instances = self.get_bert_pretraing_instances(
            self.tokenized_paragraphs, max_seq_len)
        (
            self.all_input_ids,
            self.all_token_type_ids,
            self.valid_lens,
            self.all_masked_lm_labels,
            self.all_masked_lm_pred_positions,
            self.all_masked_lm_pred_weights,
            self.all_masked_lm_pred_labels,
            self.all_next_sentence_labels,
        ) = self.format_bert_inputs(self.bert_instances, max_seq_len)

    def tokenize_text(self,
                      paragraphs: List[List[str]]) -> List[List[List[str]]]:
        """Tokenize paragraphs and sentences in the text.

        Args:
            paragraphs (List[List[str]]): List of paragraphs.

        Returns:
            List[List[List[str]]]: Tokenized paragraphs and sentences.
        """
        tokenized_paragraphs = []
        for paragraph in paragraphs:
            tokenized_paragraph = []
            for sentence in paragraph:
                tokenized_sentences = self.tokenizer.tokenize(sentence)
                tokenized_paragraph.append(tokenized_sentences)
            tokenized_paragraphs.append(tokenized_paragraph)
        return tokenized_paragraphs

    def build_vocab(self) -> Vocab:
        """Build vocabulary from tokenized sentences.

        Returns:
            Vocab: Vocabulary object.
        """
        tokenized_sentences = [
            sentence for paragraph in self.tokenized_paragraphs
            for sentence in paragraph
        ]
        vocab = Vocab.build_vocab(
            tokenized_sentences,
            min_freq=1,
            unk_token='<unk>',
            pad_token='<pad>',
            bos_token='<bos>',
            eos_token='<eos>',
            cls_token='<cls>',
            seq_token='<sep>',
        )
        return vocab

    def format_bert_inputs(
        self,
        instances: List[TrainingInstance],
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, ]:
        """Formats BERT pretraining inputs from instances.

        Args:
            instances (List[TrainingInstance]): List of BERT pretraining instances.
            max_seq_len (int): Maximum sequence length for BERT input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Formatted BERT pretraining inputs (tokens, segments, valid lengths, MLM prediction positions, MLM weights, MLM labels, NSP labels).
        """
        max_num_mlm_preds = round(max_seq_len * 0.15)
        all_input_ids, all_token_type_ids, all_masked_lm_labels, valid_lens = (
            [],
            [],
            [],
            [],
        )
        (
            all_masked_lm_pred_positions,
            all_masked_lm_pred_weights,
            all_masked_lm_pred_labels,
        ) = [], [], []
        all_next_sentence_labels = []

        for _, instance in enumerate(tqdm(instances)):
            # Tokenize and pad tokens
            input_token_ids = self.vocab[instance.tokens]
            input_token_ids = truncate_pad(
                input_token_ids,
                max_seq_len,
                padding_token_id=self.vocab['<pad>'],
            )
            all_input_ids.append(input_token_ids)

            # Masked token labels
            label_ids = truncate_pad(
                instance.masked_lm_labels,
                max_seq_len,
                padding_token_id=self.vocab['<pad>'],
            )

            all_masked_lm_labels.append(label_ids)

            # Pad segments
            token_type_ids = truncate_pad(
                instance.segment_ids,
                max_seq_len,
                padding_token_id=self.vocab['<pad>'],
            )
            all_token_type_ids.append(token_type_ids)

            # Compute valid lengths (excluding padding tokens)
            valid_lens.append(len(instance.tokens))

            # Pad MLM prediction positions, MLM weights, and MLM labels
            masked_lm_padding_len = max_num_mlm_preds - len(
                instance.masked_lm_pred_positions)
            masked_lm_pred_positions = (instance.masked_lm_pred_positions +
                                        [0] * masked_lm_padding_len)

            masked_lm_pred_weights = [1.0] * len(
                instance.masked_lm_pred_labels) + [0.0] * masked_lm_padding_len

            masked_lm_pred_labels = (instance.masked_lm_pred_labels +
                                     [0] * masked_lm_padding_len)

            all_masked_lm_pred_positions.append(masked_lm_pred_positions)
            all_masked_lm_pred_weights.append(masked_lm_pred_weights)
            all_masked_lm_pred_labels.append(masked_lm_pred_labels)

            # NSP labels
            all_next_sentence_labels.append(instance.is_next_sentence)

        # Convert lists to tensors and return
        return (
            all_input_ids,
            all_token_type_ids,
            valid_lens,
            all_masked_lm_labels,
            all_masked_lm_pred_positions,
            all_masked_lm_pred_weights,
            all_masked_lm_pred_labels,
            all_next_sentence_labels,
        )

    def get_bert_pretraing_instances(
            self, paragraphs: List[List[str]],
            max_seq_len: int) -> List[Tuple[List[str], List[int], bool]]:
        """Get BERT pretraining data from tokenized paragraphs.

        Args:
            paragraphs (List[List[str]]): List of tokenized paragraphs.
            max_seq_len (int): Maximum sequence length for BERT input.

        Returns:
            List[Tuple[List[str], List[int], bool]]: List of BERT pretraining data tuples.
        """
        examples = []
        for paragraph in paragraphs:
            nsp_data_from_paragraph = self.get_nsp_data_from_paragraph(
                paragraph, paragraphs, max_seq_len)
            examples.extend(nsp_data_from_paragraph)

        # Get Masked Language Model (MLM) data
        bert_instances = []
        for tokens, segment_ids, is_next in examples:
            (
                masked_lm_tokens,
                masked_lm_labels,
                masked_lm_pred_positions,
                masked_lm_pred_labels,
            ) = self.get_masked_lm_data_from_tokens(tokens)

            instance = TrainingInstance(
                tokens=masked_lm_tokens,
                segment_ids=segment_ids,
                is_next_sentence=is_next,
                masked_lm_labels=masked_lm_labels,
                masked_lm_pred_positions=masked_lm_pred_positions,
                masked_lm_pred_labels=masked_lm_pred_labels,
            )
            bert_instances.append(instance)

        return bert_instances

    def get_next_sentence(
        self,
        sentence: List[str],
        next_sentence: List[str],
        paragraphs: List[List[List[str]]],
    ) -> Tuple[List[str], bool]:
        """Randomly selects the next sentence for NSP task.

        Args:
            sentence (List[str]): Current sentence.
            next_sentence: (List[str]): Next sentence.
            paragraphs (List[List[str]]): List of paragraphs.

        Returns:
            Tuple[List[str], List[str], bool]: Tokens of next sentence and whether they are consecutive sentences.
        """
        if random.random() < 0.5:
            is_next = 1
        else:
            # paragraphs 是三重列表的嵌套
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = 0
        return sentence, next_sentence, is_next

    def get_nsp_data_from_paragraph(
            self, paragraph: List[str], paragraphs: List[List[List[str]]],
            max_seq_len: int) -> List[Tuple[List[str], bool]]:
        """Generate NSP (Next Sentence Prediction) data from a paragraph.

        Args:
            paragraph (List[List[str]]): Tokenized paragraph.
            paragraphs (List[List[List[str]]]): List of paragraphs.
            max_seq_len (int): Maximum sequence length for BERT input.

        Returns:
            List[Tuple[List[str], bool]]: List of NSP data tuples.
        """
        nsp_data_from_paragraph = []
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b, is_next = self.get_next_sentence(
                paragraph[i], paragraph[i + 1], paragraphs)
            # 考虑1个'<cls>'词元和2个'<sep>'词元
            if len(tokens_a) + len(tokens_b) + 3 > max_seq_len:
                continue
            tokens, segments = self.get_tokens_and_segments(tokens_a, tokens_b)
            nsp_data_from_paragraph.append((tokens, segments, is_next))
        return nsp_data_from_paragraph

    def get_tokens_and_segments(
            self,
            tokens_a: List[str],
            tokens_b: List[str] = None) -> Tuple[List[str], List[int]]:
        """Get tokens and segments for BERT input.

        Args:
            tokens_a (List[str]): Tokens of the sentence.
            tokens_b (List[str]): Tokens of the next sentence.

        Returns:
            Tuple[List[str], List[int]]: Tokens and corresponding segments.
        """
        tokens = ['<cls>'] + tokens_a + ['<sep>']
        # 0 and 1 are marking segment A and B, respectively
        segments = [0] * (len(tokens_a) + 2)
        if tokens_b is not None:
            tokens += tokens_b + ['<sep>']
            segments += [1] * (len(tokens_b) + 1)
        return tokens, segments

    def get_masked_lm_data_from_tokens(
            self, tokens: List[str]) -> Tuple[List[int], List[int], List[int]]:
        """Generates Masked Language Model (MLM) data from a list of tokens.

        Args:
            tokens (List[str]): List of tokens in a sentence.

        Returns:
            Tuple[List[int], List[int], List[int]]: Indices of MLM input tokens, positions of masked tokens, and MLM labels.
        """
        candidate_pred_positions = []
        for i, token in enumerate(tokens):
            # Exclude special tokens from prediction
            if token not in ['<cls>', '<sep>']:
                candidate_pred_positions.append(i)

        # Predict 15% of the tokens
        num_masked_lm_preds = max(1,
                                  round(len(candidate_pred_positions) * 0.15))
        (
            masked_lm_tokens,
            masked_lm_labels,
            masked_lm_pred_positions,
            masked_lm_pred_labels,
        ) = self.replace_masked_lm_tokens(
            tokens,
            candidate_pred_positions,
            num_masked_lm_preds,
            vocab_words=self.vocab_words,
        )

        return (
            masked_lm_tokens,
            masked_lm_labels,
            masked_lm_pred_positions,
            masked_lm_pred_labels,
        )

    def replace_masked_lm_tokens(
        self,
        tokens: List[str],
        candidate_pred_positions: List[int],
        num_masked_lm_preds: int,
        vocab_words: List[str],
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Replaces tokens with <mask> or random tokens to create MLM training
        data.

        Args:
            tokens (List[str]): List of tokens in a sentence.
            candidate_pred_positions (List[int]): List of indices where predictions can be made.
            num_masked_lm_preds (int): Number of tokens to predict.
            vocab_words (List[str]): Vocab word list.

        Returns:
            Tuple[List[int], List[int], List[int]]: Indices of MLM input tokens and positions with corresponding labels.
        """
        # Create a copy of tokens for Masked Language Model (MLM) input,
        # where inputs might contain replaced '<mask>' or random tokens
        masked_lm_input_tokens = tokens.copy()
        masked_lm_pred_positions = []
        masked_lm_pred_labels = []
        masked_lm_labels = [IGNORE_INDEX] * len(tokens)

        # Shuffle for 15% of random tokens for prediction in Masked Language Model (MLM) task
        random.shuffle(candidate_pred_positions)
        for mlm_pred_position in candidate_pred_positions:
            if len(masked_lm_pred_positions) >= num_masked_lm_preds:
                break
            masked_token = None
            # 80% of the time, replace with <mask>
            if random.random() < 0.8:
                masked_token = '<mask>'
            else:
                # 10% of the time, keep the word unchanged
                if random.random() < 0.5:
                    masked_token = masked_lm_input_tokens[mlm_pred_position]
                # 10% of the time, replace with a random word from vocabulary
                else:
                    masked_token = random.choice(vocab_words)

            masked_lm_input_tokens[mlm_pred_position] = masked_token
            masked_lm_labels[mlm_pred_position] = self.vocab[masked_token]

            masked_lm_pred_positions.append(mlm_pred_position)
            masked_lm_pred_labels.append(tokens[mlm_pred_position])
            sorted_ids = sorted(
                range(len(masked_lm_pred_positions)),
                key=lambda x: masked_lm_pred_positions[x],
            )

            masked_lm_pred_positions = [
                masked_lm_pred_positions[i] for i in sorted_ids
            ]
            masked_lm_pred_labels = [
                masked_lm_pred_labels[i] for i in sorted_ids
            ]
            masked_lm_pred_labels = self.vocab[masked_lm_pred_labels]

        return (
            masked_lm_input_tokens,
            masked_lm_labels,
            masked_lm_pred_positions,
            masked_lm_pred_labels,
        )

    def preprocess_text_data(self, path: str) -> List[List[str]]:
        """Preprocesses the text data from the specified file.

        Args:
            path (str): Path to the text file.

        Returns:
            List[List[str]]: List of  paragraphs (sentence list, atleast 2 ).
        """
        assert os.path.exists(path)
        paragraphs = []
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.split(' . ')) >= 2:
                    paragraph = line.strip().lower().split(' . ')
                    paragraphs.append(paragraph)
        return paragraphs

    def __len__(self) -> int:
        """Returns the total number of paragraphs in the dataset.

        Returns:
            int: Number of paragraphs.
        """
        return len(self.all_input_ids)

    def __getitem__(self, idx: int) -> List[Tuple[List[str], List[int], bool]]:
        """Retrieves the NSP data for a specific paragraph.

        Args:
            idx (int): Index of the paragraph.

        Returns:
            List[Tuple[List[str], List[int], bool]]: List of NSP data tuples.
        """
        inputs = dict(
            input_ids=torch.tensor(self.all_input_ids[idx], dtype=torch.long),
            token_type_ids=torch.tensor(self.all_token_type_ids[idx],
                                        dtype=torch.long),
            masked_lm_labels=torch.tensor(self.all_masked_lm_labels[idx],
                                          dtype=torch.long),
            next_sentence_label=torch.tensor(
                self.all_next_sentence_labels[idx], dtype=torch.long),
        )

        return inputs


if __name__ == '__main__':
    data_dir = '/home/robin/work_dir/llm/nlp-toolkit/text_data/wikitext-2/'
    bert_dataset = BertDataset(data_dir=data_dir,
                               data_split='valid',
                               max_seq_len=128)
    for i in range(10):
        print(bert_dataset[i])

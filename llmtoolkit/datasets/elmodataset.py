import codecs

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from llmtoolkit.data.vocab import (BOS_TOKEN, BOW_TOKEN, EOS_TOKEN, EOW_TOKEN,
                                   PAD_TOKEN, Vocab)


def load_corpus(path, max_tok_len=None, max_seq_len=None):
    # Read raw text file
    # and build vocabulary for both words and chars
    text = []
    charset = {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BOW_TOKEN, EOW_TOKEN}
    print(f'Loading corpus from {path}')
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            tokens = line.rstrip().split(' ')
            if max_seq_len is not None and len(tokens) + 2 > max_seq_len:
                tokens = line[:max_seq_len - 2]
            sent = [BOS_TOKEN]
            for token in tokens:
                if max_tok_len is not None and len(token) + 2 > max_tok_len:
                    token = token[:max_tok_len - 2]
                sent.append(token)
                for ch in token:
                    charset.add(ch)
            sent.append(EOS_TOKEN)
            text.append(sent)

    # Build word and character vocabulary
    print('Building word-level vocabulary')
    vocab_w = Vocab(text,
                    min_freq=2,
                    reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    print('Building char-level vocabulary')
    vocab_c = Vocab(tokens=list(charset))

    # Construct corpus using word_voab and char_vocab
    corpus_w = [vocab_w.to_ids(sent) for sent in text]
    corpus_c = []
    bow = vocab_c[BOW_TOKEN]
    eow = vocab_c[EOW_TOKEN]
    for i, sent in enumerate(text):
        sent_c = []
        for token in sent:
            if token == BOS_TOKEN or token == EOS_TOKEN:
                token_c = [bow, vocab_c[token], eow]
            else:
                token = list(token)
                token_c = [bow] + vocab_c.to_ids(token) + [eow]
            sent_c.append(token_c)
        assert len(sent_c) == len(corpus_w[i])
        corpus_c.append(sent_c)

    assert len(corpus_w) == len(corpus_c)
    return corpus_w, corpus_c, vocab_w, vocab_c


# Dataset
class BiLMDataset(Dataset):

    def __init__(self, corpus_w, corpus_c, vocab_w, vocab_c):
        super(BiLMDataset, self).__init__()
        self.pad_w = vocab_w[PAD_TOKEN]
        self.pad_c = vocab_c[PAD_TOKEN]

        self.data = []
        for sent_w, sent_c in tqdm(zip(corpus_w, corpus_c)):
            self.data.append((sent_w, sent_c))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, examples):
        # lengths: batch_size
        seq_lens = torch.LongTensor([len(ex[0]) for ex in examples])

        # inputs_w: batch_size * seq_lens
        inputs_w = [torch.tensor(ex[0]) for ex in examples]
        inputs_w = pad_sequence(inputs_w,
                                batch_first=True,
                                padding_value=self.pad_w)

        # inputs_c: batch_size * max_seq_len * max_tok_len
        batch_size, max_seq_len = inputs_w.shape
        max_tok_len = max(
            [max([len(tok) for tok in ex[1]]) for ex in examples])

        inputs_c = torch.LongTensor(batch_size, max_seq_len,
                                    max_tok_len).fill_(self.pad_c)
        for i, (sent_w, sent_c) in enumerate(examples):
            for j, tok in enumerate(sent_c):
                inputs_c[i][j][:len(tok)] = torch.LongTensor(tok)

        # fw_input_indexes, bw_input_indexes = [], []
        # targets_fw : batch_size * seq_lens
        # target_bw  : batch_size * seq_Lens
        targets_fw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
        targets_bw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
        for i, (sent_w, sent_c) in enumerate(examples):
            targets_fw[i][:len(sent_w) - 1] = torch.LongTensor(sent_w[1:])
            targets_bw[i][1:len(sent_w)] = torch.LongTensor(
                sent_w[:len(sent_w) - 1])

        return inputs_w, inputs_c, seq_lens, targets_fw, targets_bw

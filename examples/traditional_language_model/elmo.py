'''
Author: jianzhnie
Date: 2022-01-19 17:15:05
LastEditTime: 2022-03-04 18:30:51
LastEditors: jianzhnie
Description:

'''

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from llmtoolkit.data.utils.utils import PAD_TOKEN, get_loader
from llmtoolkit.data.vocab import save_vocab
from llmtoolkit.datasets.elmodataset import BiLMDataset, load_corpus
from llmtoolkit.models.elmo.elmo_model import BiLM

sys.path.append('../../')

configs = {
    'max_tok_len':
    50,
    'train_file':
    '/home/robin/jianzh/nlp-toolkit/examples/data/fra-eng/english.txt',  # path to your training file, line-by-line and tokenized
    'model_path':
    '/home/robin/jianzh/nlp-toolkit/work_dir/elmo',
    'char_embedding_dim':
    50,
    'char_conv_filters': [[1, 32], [2, 32], [3, 32], [4, 32], [5, 32], [6, 32],
                          [7, 32]],
    'num_highways':
    2,
    'projection_dim':
    512,
    'hidden_dim':
    1024,
    'num_layers':
    2,
    'batch_size':
    32,
    'dropout_prob':
    0.1,
    'learning_rate':
    0.0004,
    'clip_grad':
    5,
    'num_epoch':
    10
}

if __name__ == '__main__':
    corpus_w, corpus_c, vocab_w, vocab_c = load_corpus(configs['train_file'])
    train_data = BiLMDataset(corpus_w, corpus_c, vocab_w, vocab_c)
    train_loader = get_loader(train_data, configs['batch_size'])

    criterion = nn.CrossEntropyLoss(ignore_index=vocab_w[PAD_TOKEN],
                                    reduction='sum')
    print('Building BiLM model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLM(configs, vocab_w, vocab_c)
    model.to(device)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad,
                                  model.parameters()),
                           lr=configs['learning_rate'])

    model.train()
    for epoch in range(configs['num_epoch']):
        total_loss = 0
        total_tags = 0  # number of valid predictions
        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch}'):
            batch = [x.to(device) for x in batch]
            inputs_w, inputs_c, seq_lens, targets_fw, targets_bw = batch

            optimizer.zero_grad()
            outputs_fw, outputs_bw = model(inputs_c, seq_lens)
            loss_fw = criterion(outputs_fw.view(-1, outputs_fw.shape[-1]),
                                targets_fw.view(-1))
            loss_bw = criterion(outputs_bw.view(-1, outputs_bw.shape[-1]),
                                targets_bw.view(-1))
            loss = (loss_fw + loss_bw) / 2.0
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           configs['clip_grad'])
            optimizer.step()

            total_loss += loss_fw.item()
            total_tags += seq_lens.sum().item()

        train_ppl = np.exp(total_loss / total_tags)
        print(f'Train PPL: {train_ppl:.2f}')

    # save BiLM encoders
    model.save_pretrained(configs['model_path'])
    # save configs
    json.dump(configs,
              open(os.path.join(configs['model_path'], 'configs.json'), 'w'))
    # save vocabularies
    save_vocab(vocab_w, os.path.join(configs['model_path'], 'word.dic'))
    save_vocab(vocab_c, os.path.join(configs['model_path'], 'char.dic'))

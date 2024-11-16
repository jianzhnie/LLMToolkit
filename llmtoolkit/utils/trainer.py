import math
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append('../../')
from llmtoolkit.data import Tokenizer, Vocab
from llmtoolkit.models.seq2seq.rnn_mt import RNNSeq2Seq
from llmtoolkit.utils.data_utils import truncate_pad
from llmtoolkit.utils.model_utils import save_model_checkpoints
from llmtoolkit.utils.train_utils import epoch_time


# Function to train the seq2seq model
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    clip_ratio: float = None,
    epoch: int = 100,
    device: str = 'cpu',
    log_interval: int = 10,
) -> float:
    """Train the seq2seq model for one epoch.

    Args:
        model (nn.Module): The seq2seq model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (optim.Optimizer): The optimizer for gradient updates.
        criterion (nn.CrossEntropyLoss): The loss criterion.
        clip_ratio (float): Gradient clipping threshold.
        epoch (int): Current epoch number.
        log_interval (int): Log interval for printing progress.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    epoch_loss = 0
    moving_loss = 0
    start_time = time.time()
    for idx, batch in enumerate(dataloader):
        src, src_len, trg, trg_len = [t.to(device) for t in batch]

        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output, trg, trg_len)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        if clip_ratio is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_ratio)

        optimizer.step()
        epoch_loss += loss.item()

        moving_loss += loss.item()
        if idx % log_interval == 0 and idx > 0:
            cur_loss = moving_loss / log_interval
            elapsed = time.time() - start_time
            print(
                'Train: | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f}'.format(epoch, idx, len(dataloader),
                                      elapsed * 1000 / log_interval, cur_loss))
            moving_loss = 0
            start_time = time.time()

    return epoch_loss / len(dataloader)


# Function to evaluate the seq2seq model
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: str = 'cpu',
    log_interval: int = 10,
) -> float:
    """Evaluate the seq2seq model on the validation or test data.

    Args:
        model (nn.Module): The seq2seq model.
        dataloader (DataLoader): DataLoader for validation or test data.
        criterion (nn.CrossEntropyLoss): The loss criterion.
        log_interval (int): Log interval for printing progress.

    Returns:
        float: Average evaluation loss.
    """
    model.eval()
    epoch_loss = 0
    moving_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            src, src_len, trg, trg_len = [t.to(device) for t in batch]
            output = model(src, trg)
            loss = criterion(output, trg, trg_len)
            epoch_loss += loss.item()
            moving_loss += loss.item()

            if idx % log_interval == 0 and idx > 0:
                cur_loss = moving_loss / log_interval
                print('Val: |{:5d}/{:5d} batches |'
                      'loss {:5.2f} '.format(idx, len(dataloader), cur_loss))
                moving_loss = 0
    return epoch_loss / len(dataloader)


def predict_seq2seq(
    model: RNNSeq2Seq,
    tokenizer: Tokenizer,
    src_sentence: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_seq_len: int = None,
    device: str = 'cpu',
) -> str:
    """Predict the target sentence for a given source sentence.

    Args:
        model (RNNSeq2Seq): The sequence-to-sequence model.
        tokenizer (Tokenizer): Tokenizer for source sentence.
        src_sentence (str): The source sentence.
        src_vocab (Vocab): Vocabulary for source language.
        tgt_vocab (Vocab): Vocabulary for target language.
        max_seq_len (int, optional): Maximum sequence length for prediction.
        device (str): Device to run prediction ('cpu' or 'cuda').

    Returns:
        str: The predicted target sentence.
    """
    model.eval()  # Set the model to evaluation mode

    # Tokenize the source sentence and add <eos> token
    src_tokens = tokenizer.tokenize(src_sentence) + ['<eos>']

    # Convert source tokens to indices using the source vocabulary
    src_indices = src_vocab.to_index(src_tokens)

    # Truncate or pad the source sequence to the specified max_seq_len
    src_indices = truncate_pad(src_indices, max_seq_len, src_vocab['<pad>'])

    # Convert source indices to a tensor and move it to the specified device
    src_tensor = torch.tensor(src_indices,
                              dtype=torch.long).unsqueeze(0).to(device)
    tgt_indices = [tgt_vocab['<bos>']]
    tgt_indices = truncate_pad(tgt_indices, max_seq_len, tgt_vocab['<pad>'])
    # Initialize the decoder input with <bos> token
    dec_inputs = torch.tensor(tgt_indices,
                              dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        # Encode the source sequence
        enc_outputs, enc_state = model.encoder(src_tensor)
        # Use the last encoder output as context for the decoder
        context = enc_outputs[-1].repeat(max_seq_len, 1, 1)
        # Initialize the decoder hidden state with the encoder hidden state
        output_seq = []

        for _ in range(max_seq_len):
            # Decode one step at a time
            outputs, dec_state = model.decoder(dec_inputs, enc_state, context)
            # Get the predicted token with the highest probability
            dec_inputs = outputs.argmax(dim=2)
            pred = dec_inputs.squeeze(dim=0).type(torch.int32).item()
            # If the predicted token is <eos>, stop predicting
            if pred == tgt_vocab['<eos>']:
                break

            output_seq.append(pred)

    # Convert the predicted token indices to text using the target vocabulary
    pred_text = ' '.join(tgt_vocab.to_tokens(output_seq))

    return pred_text


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    num_epochs: int = 100,
    clip_ratio: float = 0.25,
    device: str = 'cpu',
    log_interval: int = 10,
    save_model_path: str = 'rnn_nmt',
):
    """Train and evaluate the seq2seq model.

    Args:
        model (nn.Module): The seq2seq model.
        train_loader (DataLoader): DataLoader for the training data.
        eval_loader (DataLoader): DataLoader for the evaluation data.
        optimizer (optim.Optimizer): The optimizer for gradient updates.
        criterion (nn.CrossEntropyLoss): The loss criterion.
        num_epochs (int): Number of training epochs.
        clip_ratio (float): Gradient clipping threshold.
        log_interval (int): Log interval for printing progress.
        save_model_path (str): Path to save the best model.

    Returns:
        None
    """
    # Initialize the best validation loss
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        # Train the model
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            clip_ratio,
            epoch,
            device,
            log_interval,
        )
        # Evaluate the model on the validation set
        val_loss = evaluate(model, eval_loader, criterion, device,
                            log_interval)
        end_time = time.time()
        # Calculate elapsed time for the epoch
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # Print epoch information
        print('-' * 89)
        print(f'Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}'
        )
        print(
            f'Valid Loss: {val_loss:.3f} | Valid PPL: {math.exp(val_loss):7.3f}'
        )
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            save_model_checkpoints(model, epoch, save_model_path)
            best_val_loss = val_loss
        print(f'Best val loss: {best_val_loss:.3f}')
        print('-' * 89)
    # Evaluate the model on the test set
    test_loss = evaluate(model, eval_loader, criterion, log_interval)
    print('End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))

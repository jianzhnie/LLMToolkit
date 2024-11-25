from typing import List, Union

import torch
import torch.nn as nn


class MaskedSoftmaxCELoss(nn.Module):
    """The softmax cross-entropy loss with masks.

    This loss is suitable for sequence-to-sequence models where sequences have varying lengths.
    It applies a mask to ignore padding elements when computing the loss.

    Args:
        ignore_index (int, optional): Index to ignore when computing the loss. Defaults to -100.

    Attributes:
        ignore_index (int): Index to ignore when computing the loss.
        criterion (nn.CrossEntropyLoss): Cross-entropy loss criterion with predefined settings.
    """

    def __init__(self):
        super(MaskedSoftmaxCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred: torch.Tensor, label: torch.Tensor,
                valid_len: torch.Tensor) -> torch.Tensor:
        """Compute the masked softmax cross-entropy loss.

        Args:
            pred (torch.Tensor): Predicted scores (logits) of shape (batch_size, seq_len, vocab_size).
            label (torch.Tensor): Ground truth labels of shape (batch_size, seq_len).
            valid_len (torch.Tensor): Lengths of valid elements in each sequence of shape (batch_size,).

        Returns:
            weighted_loss (torch.Tensor): Weighted loss of shape (batch_size,).
        """
        mask = self.generate_mask(label, valid_len)
        loss = self.criterion(pred.permute(0, 2, 1), label)
        weighted_loss = torch.sum(loss * mask) / torch.sum(mask)
        return weighted_loss

    def generate_mask(
            self, inputs: torch.Tensor,
            valid_len: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """Generate a mask to ignore padding elements.

        Args:
            inputs (torch.Tensor): Ground truth inputs of shape (batch_size, seq_len).
            valid_len (torch.Tensor): Lengths of valid elements in each sequence of shape (batch_size,).

        Returns:
            mask (torch.Tensor): Mask of shape (batch_size, seq_len) where padding elements are set to 0.
        """
        batch_size, seq_len = inputs.size()
        mask = torch.arange(seq_len,
                            dtype=torch.float32, device=inputs.device).expand(
                                batch_size, seq_len) < valid_len.unsqueeze(1)
        return mask


def sequence_mask(inputs: torch.Tensor,
                  valid_len: Union[List[int], torch.Tensor],
                  value: float = 0.0) -> torch.Tensor:
    """Apply a sequence mask to the input tensor to mask out elements beyond
    the valid lengths of each sequence.

    Args:
        inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length, ...).
        valid_len (Union[List[int], torch.Tensor]): List or tensor of valid sequence lengths for each batch element.
        value (float, optional): Value to fill masked elements with (default is 0.0).

    Returns:
        torch.Tensor: Masked tensor with the same shape as the input tensor.

    Example:
        >>> inputs = torch.tensor([[1, 2, 3, 4, 5],
        ...                   [6, 7, 8, 9, 10],
        ...                   [11, 12, 13, 14, 15]], dtype=torch.float32)
        >>> valid_len = [2, 3, 4]
        >>> masked = sequence_mask(inputs, valid_len)
        >>> print(masked)
        tensor([[1., 2., 0., 0., 0.],
                [6., 7., 8., 0., 0.],
                [11., 12., 13., 14., 0.]])
    """
    # Get the maximum sequence length
    maxlen = inputs.size(1)
    # Create a mask for each sequence
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=inputs.device)[None, :] < valid_len[:, None]
    # Apply the mask to the input tensor
    inputs[~mask] = value
    return inputs


def masked_softmax(inputs: torch.Tensor,
                   valid_lens: torch.Tensor = None) -> torch.Tensor:
    """Perform softmax operation on the last axis of the input tensor while
    considering valid sequence lengths.

    Args:
        inputs (torch.Tensor): The input tensor of shape (batch_size, sequence_length, num_classes).
        valid_lens (torch.Tensor, optional): 1D or 2D tensor representing valid sequence lengths.
            If provided, it masks out elements beyond the specified lengths.

    Returns:
        torch.Tensor: The softmax output tensor of the same shape as the input.

    Note:
        If `valid_lens` is provided, elements beyond valid sequence lengths are masked with a large negative value
        to ensure that their softmax outputs are effectively 0.

    Example:
        # Compute masked softmax on a batch of sequences
        inputs = torch.rand(2, 3, 4)  # Batch size: 2, Sequence length: 3, Number of classes: 4
        valid_lens = torch.tensor([2, 3])  # Valid lengths for each sequence in the batch
        masked_probs = masked_softmax(inputs, valid_lens)
    """
    if valid_lens is None:
        # If valid_lens is not provided, perform regular softmax
        return nn.functional.softmax(inputs, dim=-1)
    else:
        # Calculate the shape of the input tensor
        shape = inputs.shape
        # Reshape valid_lens if it's a 2D tensor
        if valid_lens.dim() == 1:
            # Repeat the valid_lens to match the sequence length dimension of inputs
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        # Create a mask for elements beyond valid sequence lengths and replace them with a large negative value
        inputs = inputs.reshape(-1, shape[-1])
        inputs = sequence_mask(inputs, valid_lens, value=-1e6)
        # Reshape the masked tensor back to its original shape and perform softmax
        return nn.functional.softmax(inputs.reshape(shape), dim=-1)


if __name__ == '__main__':
    Loss = MaskedSoftmaxCELoss()
    inputs = torch.tensor(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        dtype=torch.float32)
    valid_len = torch.tensor([2, 3, 4])
    masked = Loss.generate_mask(inputs, valid_len)
    print(masked)

    inputs_matrix = torch.rand(2, 2, 4)
    valid_lens = torch.tensor([2, 3])
    masked = masked_softmax(inputs_matrix, valid_lens)
    print(masked)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

KD_loss = nn.KLDivLoss(reduce='mean')


def kd_step(teacher: nn.Module, student: nn.Module, temperature: float,
            inputs: torch.tensor, optimizer: Optimizer):

    teacher.eval()
    student.train()

    with torch.no_grad():
        logits_t = teacher(inputs=inputs)

    logits_s = student(inputs=inputs)

    loss = KD_loss(input=F.log_softmax(logits_s / temperature, dim=-1),
                   target=F.log_softmax(logits_t / temperature, dim=-1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss

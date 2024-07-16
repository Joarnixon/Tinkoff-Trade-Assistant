import torch.nn.functional as F
from torch import float32, tensor


def cross_entropy_loss(output, target, weight=tensor([1, 1, 1])):
    target = F.one_hot(target, num_classes=3).to(float32)
    return F.cross_entropy(output, target, weight)
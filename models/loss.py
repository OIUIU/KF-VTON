import torch
import torch.nn.functional as F
import torch.nn as nn

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, w, h = input.size()
    nt, wt, ht  = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(
            ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    # loss_ce = nn.CrossEntropyLoss()
    # loss = loss_ce(input,torch.squeeze(target))

    return loss
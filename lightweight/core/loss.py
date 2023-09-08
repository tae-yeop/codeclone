import torch
import torch.nn as nn


def tv_loss(img):
    # https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    # https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/image/tv.py
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = (h_variance + w_variance)
    return loss
class Loss():
    ...
import torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

__all__ = ['CrossEntropyLoss', 'FocalLoss']

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma=gamma
        self.ce = CrossEntropyLoss(reduction='none')

    def forward(self, logit, label):
        logit = logit.view(logit.size(0), -1) 
         
        #print(f'logit: {logit.size()}, label: {label.size()}')
        CE = self.ce(logit, label)
        loss = ((1.0 - torch.exp(-CE)) ** self.gamma) * CE
        return loss.mean()

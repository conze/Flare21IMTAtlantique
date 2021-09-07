#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Function

class DiceCoeff(Function):
    ''' dice coef for individual examples '''
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

def dice_coeff(input, target):
    ''' dice coef for batches '''
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
    return s / (i + 1)
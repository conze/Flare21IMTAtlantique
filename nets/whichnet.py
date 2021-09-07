#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nets import AttNet

def whichnet(net_id, n_classes):
        
    if net_id == 1:
        pretrained, vgg = False, True
        net = AttNet.uNet13DS(n_classes = n_classes, pretrained = pretrained)
        
    return net, vgg
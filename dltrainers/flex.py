# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

__all__ = """
""".split()

import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.legacy import nn as legnn
import layers

class Flex(nn.Module):
    def __init__(self, creator):
        super(Flex, self).__init__()
        self.creator = creator
        self.layer = None
    def forward(self, *args):
        if self.layer is None:
            self.layer = self.creator(*args)
        return self.layer.forward(*args)


def Linear(*args, **kw):
    def creator(x):
        assert x.ndimension()==2
        d = x.size(1)
        return nn.Linear(x.size(1), *args, **kw)
    return Flex(creator)


def Conv1d(*args, **kw):
    def creator(x):
        assert x.ndimension()==3
        d = x.size(1)
        return nn.Conv1d(x.size(1), *args, **kw)
    return Flex(creator)
        

def Conv2d(*args, **kw):
    def creator(x):
        assert x.ndimension()==4
        d = x.size(1)
        return nn.Conv2d(x.size(1), *args, **kw)
    return Flex(creator)
        

def Conv3d(*args, **kw):
    def creator(x):
        assert x.ndimension()==5
        d = x.size(1)
        return nn.Conv3d(x.size(1), *args, **kw)
    return Flex(creator)


def Lstm1(*args, **kw):
    def creator(x):
        assert x.ndimension()==3
        d = x.size(1)
        return layers.Lstm1(x.size(1), *args, **kw)
    return Flex(creator)


def Lstm1to0(*args, **kw):
    def creator(x):
        assert x.ndimension()==3
        d = x.size(1)
        return layers.Lstm1to0(x.size(1), *args, **kw)
    return Flex(creator)


def Lstm2(*args, **kw):
    def creator(x):
        assert x.ndimension()==4
        d = x.size(1)
        return layers.Lstm2(x.size(1), *args, **kw)
    return Flex(creator)


def Lstm2to1(*args, **kw):
    def creator(x):
        assert x.ndimension()==4
        d = x.size(1)
        return layers.Lstm2to1(x.size(1), *args, **kw)
    return Flex(creator)

def flex_freeze(model):
    # FIXME
    return model

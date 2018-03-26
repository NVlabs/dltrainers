# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.legacy import nn as legnn
import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.legacy import nn as legnn
import helpers

BD = "BD"
LBD = "LBD"
LDB = "LDB"
BDL = "BDL"
BLD = "BLD"
BWHD = "BWHD"
BDWH = "BDWH"
BWH = "BWH"

def deprecated(f):
    def g(*args, **kw):
        raise Exception("deprecated")
    return g

def lbd2bdl(x):
    assert len(x.size()) == 3
    return x.permute(1, 2, 0).contiguous()


def bdl2lbd(x):
    assert len(x.size()) == 3
    return x.permute(2, 0, 1).contiguous()

def data(x):
    if isinstance(x, Variable):
        return x.data
    else:
        return x

@deprecated
def typeas(x, y):
    """Make x the same type as y, for numpy, torch, torch.cuda."""
    assert not isinstance(x, Variable)
    if isinstance(y, Variable):
        y = y.data
    if isinstance(y, np.ndarray):
        return asnd(x)
    if isinstance(x, np.ndarray):
        if isinstance(y, (torch.FloatTensor, torch.cuda.FloatTensor)):
            x = torch.FloatTensor(x)
        else:
            x = torch.DoubleTensor(x)
    return x.type_as(y)

class Fun(nn.Module):
    def __init__(self, f):
        nn.Module.__init__(self)
        self.f = f
    def forward(self, x):
        return self.f(x)

class PixelsToBatch(nn.Module):
    def forward(self, x):
        b, d, h, w = x.size()
        return x.permute(0, 2, 3, 1).contiguous().view(b*h*w, d)

class WeightedGrad(autograd.Function):
    def forward(self, input, weights):
        self.weights = weights
        return input
    def backward(self, grad_output):
        return grad_output * self.weights, None

def weighted_grad(x, y):
    return WeightedGrad()(x, y)

class Info(nn.Module):
    def __init__(self, info=""):
        nn.Module.__init__(self)
        self.info = info
    def forward(self, x):
        print "Info", self.info, x.size(), x.min().data[0], x.max().data[0]
        return x

class CheckSizes(nn.Module):
    def __init__(self, *args, **kw):
        nn.Module.__init__(self)
        self.order = kw.get("order")
        self.name = kw.get("name")
        self.limits = [(x, x) if isinstance(x, int) else x for x in args]
    def forward(self, x):
        for (i, actual), (lo, hi) in zip(enumerate(tuple(x.size())), self.limits):
            if actual < lo:
                raise Exception("{} ({}): index {} too low ({} not >= {})"
                                .format(self.name, self.order,
                                        i, actual, lo))
            if actual > hi:
                raise Exception("{} ({}): index {} too high ({} not <= {})"
                                .format(self.name, self.order,
                                        i, actual, hi))
        return x


class Cuda(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.use_cuda = None
    def cpu(self):
        self.use_cuda = None
    def cuda(self, unit=0):
        self.use_cuda = unit
    def forward(self, x):
        if self.use_cuda is None:
            return x.cpu()
        else:
            return x.cuda(self.use_cuda)

class Cpu(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
    def forward(self, x):
        return x.cpu()

class Check(nn.Module):
    def __init__(self, *shape, **kw):
        nn.Module.__init__(self)
        self.expected = tuple(shape)
        self.valid = kw.get("valid", (-1e-5, 1+1e-5))
    def forward(self, x):
        expected_shape = self.expected
        actual_shape = tuple(x.size())
        assert len(actual_shape)==len(expected_shape)
        for i in range(len(actual_shape)):
            assert expected_shape[i]<0 or expected_shape[i]==actual_shape[i], \
                   (expected_shape, actual_shape, i)
        assert data(x).min() >= self.valid[0], (data(x).min(), self.valid)
        assert data(x).max() <= self.valid[1], (data(x).max(), self.valid)
        return x

class Reorder(nn.Module):
    def __init__(self, old, new):
        nn.Module.__init__(self)
        self.permutation = tuple([old.find(c) for c in new])
    def forward(self, x):
        return x.permute(*self.permutation).contiguous()

class Permute(nn.Module):
    def __init__(self, *args):
        nn.Module.__init__(self)
        self.permutation = args
    def forward(self, x):
        return x.permute(*self.permutation).contiguous()

class Viewer(nn.Module):
    def __init__(self, *args):
        nn.Module.__init__(self)
        self.shape = args

    def forward(self, x):
        return x.view(*self.shape)

    def __repr__(self):
        return "Viewer %s" % (self.shape,)

class Flat(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        rank = len(x.size())
        assert rank > 2
        new_depth = np.prod(tuple(x.size())[1:])
        return x.view(-1, new_depth)

    def __repr__(self):
        return "Flat"


class Textline2Img(nn.Module):
    input_order = BWH
    output_order = BDWH

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, seq):
        b, l, d = seq.size()
        return seq.view(b, 1, l, d)


class Img2Seq(nn.Module):
    input_order = BDWH
    output_order = BDL

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, img):
        b, d, w, h = img.size()
        perm = img.permute(0, 1, 3, 2).contiguous()
        return perm.view(b, d * h, w)


class ImgMaxSeq(nn.Module):
    input_order = BDWH
    output_order = BDL

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, img):
        # BDWH -> BDW -> BWD
        return img.max(3)[0].squeeze(3)


class ImgSumSeq(nn.Module):
    input_order = BDWH
    output_order = BDL

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, img):
        # BDWH -> BDW -> BWD
        return img.sum(3)[0].squeeze(3).permute(0, 2, 1).contiguous()


class LSTM1(nn.Module):
    """A simple bidirectional LSTM.

    All the sequence processing layers use BDL order by default to
    be consistent with 1D convolutions.
    """
    input_order = BDL
    output_order = BDL

    def __init__(self, ninput=None, noutput=None, ndir=2):
        nn.Module.__init__(self)
        assert ninput is not None
        assert noutput is not None
        self.ndir = ndir
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=self.ndir - 1)

    def forward(self, seq, volatile=False):
        seq = bdl2lbd(seq)
        l, bs, d = seq.size()
        assert d == self.ninput, seq.size()
        h0 = Variable(helpers.typeas(torch.zeros(self.ndir, bs, self.noutput), seq),
                      volatile=volatile)
        c0 = Variable(helpers.typeas(torch.zeros(self.ndir, bs, self.noutput), seq),
                      volatile=volatile)
        post_lstm, _ = self.lstm(seq, (h0, c0))
        return lbd2bdl(post_lstm)


class LSTM2to1(nn.Module):
    """An LSTM that summarizes one dimension."""
    input_order = BDWH
    output_order = BDL

    def __init__(self, ninput=None, noutput=None):
        nn.Module.__init__(self)
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=False)

    def forward(self, img, volatile=False):
        # BDWH -> HBWD -> HBsD
        b, d, w, h = img.size()
        seq = img.permute(3, 0, 2, 1).contiguous().view(h, b * w, d)
        bs = b * w
        h0 = Variable(
            helpers.typeas(torch.zeros(1, bs, self.noutput), img), volatile=volatile)
        c0 = Variable(
            helpers.typeas(torch.zeros(1, bs, self.noutput), img), volatile=volatile)
        # HBsD -> HBsD
        assert seq.size() == (h, b * w, d), (seq.size(), (h, b * w, d))
        post_lstm, _ = self.lstm(seq, (h0, c0))
        assert post_lstm.size() == (h, b * w, self.noutput), (post_lstm.size(),
                                                              (h, b * w, self.noutput))
        # HBsD -> BsD -> BWD
        final = post_lstm.select(0, h - 1).view(b, w, self.noutput)
        assert final.size() == (b, w, self.noutput), (final.size(), (b, w, self.noutput))
        # BWD -> BDW
        final = final.permute(0, 2, 1).contiguous()
        assert final.size() == (b, self.noutput, w), (final.size(),
                                                      (b, self.noutput, self.noutput))
        return final


class LSTM1to0(nn.Module):
    """An LSTM that summarizes one dimension."""
    input_order = BDL
    output_order = BD

    def __init__(self, ninput=None, noutput=None):
        nn.Module.__init__(self)
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=False)

    def forward(self, seq):
        volatile = not isinstance(seq, Variable) or seq.volatile
        seq = bdl2lbd(seq)
        l, b, d = seq.size()
        assert d == self.ninput, (d, self.ninput)
        h0 = Variable(
            helpers.typeas(torch.zeros(1, b, self.noutput), seq), volatile=volatile)
        c0 = Variable(
            helpers.typeas(torch.zeros(1, b, self.noutput), seq), volatile=volatile)
        assert seq.size() == (l, b, d)
        post_lstm, _ = self.lstm(seq, (h0, c0))
        assert post_lstm.size() == (l, b, self.noutput)
        final = post_lstm.select(0, l - 1).view(b, self.noutput)
        return final


class RowwiseLSTM(nn.Module):
    def __init__(self, ninput=None, noutput=None, ndir=2):
        nn.Module.__init__(self)
        self.ndir = ndir
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=self.ndir - 1)

    def forward(self, img):
        volatile = not isinstance(img, Variable) or img.volatile
        b, d, h, w = img.size()
        # BDHW -> WHBD -> WB'D
        seq = img.permute(3, 2, 0, 1).contiguous().view(w, h * b, d)
        # WB'D
        h0 = helpers.typeas(torch.zeros(self.ndir, h * b, self.noutput), img)
        c0 = helpers.typeas(torch.zeros(self.ndir, h * b, self.noutput), img)
        h0 = Variable(h0, volatile=volatile)
        c0 = Variable(c0, volatile=volatile)
        seqresult, _ = self.lstm(seq, (h0, c0))
        # WB'D' -> BD'HW
        result = seqresult.view(
            w, h, b, self.noutput * self.ndir).permute(2, 3, 1, 0)
        return result


class LSTM2(nn.Module):
    """A 2D LSTM module."""

    def __init__(self, ninput=None, noutput=None, nhidden=None, ndir=2):
        nn.Module.__init__(self)
        assert ndir in [1, 2]
        nhidden = nhidden or noutput
        self.hlstm = RowwiseLSTM(ninput, nhidden, ndir=ndir)
        self.vlstm = RowwiseLSTM(nhidden * ndir, noutput, ndir=ndir)

    def forward(self, img):
        horiz = self.hlstm(img)
        horizT = horiz.permute(0, 1, 3, 2).contiguous()
        vert = self.vlstm(horizT)
        vertT = vert.permute(0, 1, 3, 2).contiguous()
        return vertT


# copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

"""A set of "trainers", classes that wrap around Torch models
and provide methods for training and evaluation."""

import numpy as np
import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from scipy import ndimage
import helpers as dlh
import dlmodels as dlm

class BasicTrainer(object):
    """Trainers take care of bookkeeping for training models.

    The basic method is `train_batch(inputs, targets)`.  This converts
    to/from numpy and torch tensors, CPU/GPU tensors, and optionally
    dlh.reorders the axes of the input and output tensors to account for
    different model and data conventions. It also catches errors
    during forward propagation and reports the model and input shapes
    (shape mismatches are the most common source of errors.

    Trainers are just a temporary tool that's wrapped around a model
    for training purposes, so you can create, use, and discard them
    as convenient.
    """

    def __init__(self, model, use_cuda=True,
                 fields = ("input", "output"),
                 data_order = (None, None),
                 model_order = (None, None)):
        self.use_cuda = use_cuda
        self.model = self._cuda(model)
        if "META" not in dir(self.model):
            self.model.META = dict(ntrain=0)
        self.meta = self.model.META
        self.init_loss()
        self.input_name, self.output_name = fields
        self.data_input, self.data_output = data_order
        self.model_input, self.model_output = model_order
        self.no_display = False
        self.current_lr = None
        self.set_lr(1e-3)

    def log(self, key, **kw):
        record = dict(ntrain=self.meta["ntrain"])
        record.update(kw)
        if key not in self.meta:
            self.meta[key] = []
        self.meta[key].append(record)

    def log_retrieve(self, key, *args):
        result = []
        for record in self.meta[key]:
            if len([a for a in args if a not in record]) > 0:
                continue
            result.append(tuple((record[a] for a in args)))
        return zip(*result)

    def log_has(self, key):
        return key in self.meta and len(self.meta[key]) > 0

    def set_input_orders(self, data_input, model_input):
        assert isinstance(data_input, str)
        assert isinstance(model_input, str)
        assert len(data_input) == len(model_input)
        self.data_input = data_input
        self.model_input = model_input

    def set_output_orders(self, data_output, model_output):
        assert isinstance(model_output, str)
        assert isinstance(data_output, str)
        assert len(data_output) == len(model_output)
        self.data_output = data_output
        self.model_output = model_output

    def _cuda(self, x):
        """Convert object to CUDA if use_cuda==True."""
        if self.use_cuda:
            return x.cuda()
        else:
            return x.cpu()

    def set_training(self, mode=True):
        """Set training or prediction mode."""
        if mode:
            if not self.model.training:
                self.model.train()
            self.cuinput = autograd.Variable(
                torch.randn(1, 1, 100, 100).cuda())
            self.cutarget = autograd.Variable(torch.randn(1, 11).cuda())
        else:
            if self.model.training:
                self.model.eval()
            self.cuinput = autograd.Variable(torch.randn(1, 1, 100, 100).cuda(),
                                             volatile=True)
            self.cutarget = autograd.Variable(torch.randn(1, 11).cuda(),
                                              volatile=True)

    def set_lr(self, lr, momentum=0.9, weight_decay=0.0):
        """Set the optimizer to SGD with the given parameters."""
        self.log("rates", lr=lr)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)

    def get_outputs(self, outputs):
        """Performs any necessary transformations on the output tensor.

        May perform transformations like BDHW to BHWD.
        """
        return dlh.reorder(dlh.novar(self.cuoutput), self.model_output, self.data_output).cpu()

    def set_inputs(self, batch):
        """Sets the cuinput variable from the input data.

        May perform transformations like BHWD to BDHW.
        """
        dlh.assign(self.cuinput, batch, self.data_input, self.model_input)

    def set_targets(self, targets, outputs, weights=None):
        """Sets the cutarget variable from the given tensor.

        May perform transformations like BHWD to BDHW.
        Also optionally allows weights to be set for some kinds of trainers.
        """
        self.target_shape = dlh.shp(targets)
        assert weights is None, "weights not implemented"
        assert dlh.shp(targets) == dlh.shp(outputs)
        dlh.assign(self.cutarget, targets, self.data_output, self.model_output)

    def init_loss(self, loss=nn.MSELoss()):
        self.criterion = self._cuda(loss)

    def compute_loss(self, targets, weights=None):
        self.set_targets(targets, self.cuoutput, weights=weights)
        return self.criterion(self.cuoutput, self.cutarget)

    def forward(self):
        try:
            self.cuoutput = self.model(self.cuinput)
        except RuntimeError, err:
            print "runtime error in forward step:"
            print "batch shape", self.data_input
            print "model input", self.model_input, dlh.shp(self.cuinput)
            print "model:"
            print self.model
            raise err

    def train_batch(self, inputs, targets, weights=None, update=True):
        if update:
            self.set_training(True)
            self.optimizer.zero_grad()
        else:
            self.set_training(False)
        self.set_inputs(inputs)
        self.forward()
        culoss = self.compute_loss(targets, weights=weights)
        if update:
            culoss.backward()
            self.optimizer.step()
        ploss = dlh.novar(culoss)[0]
        if update:
            self.meta["ntrain"] += len(inputs)
            self.log("training", loss=ploss)
        return self.get_outputs(self.cuoutput), ploss

    def eval_batch(self, inputs, targets):
        return self.train_batch(inputs, targets, update=False)

    def predict_batch(self, inputs):
        self.set_training(False)
        self.set_inputs(inputs)
        self.forward()
        return self.get_outputs(self.cuoutput)

    def display_loss(self, every=100, smooth=1e-2, yscale=None):
        if self.no_display: return
        # we import these locally to avoid dependence on display
        # functions for training
        import matplotlib as mpl
        from matplotlib import pyplot
        from IPython import display
        from scipy.ndimage import filters
        if len(self.meta["losses"]) % every != 0:
            return
        pyplot.clf()
        if self.log_has("training"):
            pyplot.subplot(121)
            counts, losses = self.log_retrieve("training", "ntrain", "loss")
            pyplot.plot(counts, losses)
        if self.log_has("test"):
            pyplot.subplot(122)
            counts, losses = self.log_retrieve("test", "ntrain", "loss")
            pyplot.plot(counts, losses)
        display.clear_output(wait=True)
        display.display(pyplot.gcf())

    def set_sample_fields(self, input_name, output_name):
        self.input_name = input_name
        self.output_name = output_name

    def train_for(self, training, training_size=1e99):
        if isinstance(training, types.FunctionType):
            training = training()
        count = 0
        losses = []
        for batch in training:
            if count >= training_size: break
            input_tensor = batch[self.input_name]
            output_tensor = batch[self.output_name]
            _, loss = self.train_batch(input_tensor, output_tensor)
            count += len(input_tensor)
            losses.append(loss)
        loss = np.mean(losses)
        return loss, count

    def eval_for(self, testset, testset_size=1e99):
        if isinstance(testset, types.FunctionType):
            testset = testset()
        count = 0
        losses = []
        for batch in testset:
            if count >= testset_size: break
            input_tensor = batch[self.input_name]
            output_tensor = batch[self.output_name]
            _, loss = self.eval_batch(input_tensor, output_tensor)
            count += len(input_tensor)
            losses.append(loss)
        loss = np.mean(losses)
        self.log("test", loss=loss)
        return loss, count

class ImageClassifierTrainer(BasicTrainer):
    def __init__(self, *args, **kw):
        BasicTrainer.__init__(self, *args, **kw)
        self.set_input_orders("BHWD", "BDHW")
        self.set_output_orders("BC", "BC")

    def set_inputs(self, images, depth1=False):
        dlh.assign(self.cuinput, images, self.data_input, self.model_input)

    def set_targets(self, targets, outputs, weights=None):
        assert weights is None, "weights not implemented"
        assert dlh.rank(outputs) == 2, dlh.shp(outputs)
        if isinstance(targets, list): targets = dlh.as_nda(targets)
        targets = dlh.as_torch(targets)
        if dlh.rank(targets) == 1:
            targets = targets.unsqueeze(1)
            b, c = dlh.shp(outputs)
            onehot = torch.zeros(b, c)
            onehot.scatter_(1, targets, 1)
            targets = onehot
        assert dlh.shp(targets) == dlh.shp(outputs)
        dlh.assign(self.cutarget, targets, self.data_output, self.model_output)

def zoom_like(batch, target_shape, order=0):
    target_shape = tuple(target_shape)
    if dlh.shp(batch) == target_shape:
        return batch
    assert order >= 0
    scales = [r * 1.0 / b for r, b in zip(target_shape, batch.shape)]
    result = np.zeros(target_shape)
    ndimage.zoom(dlh.as_nda(batch), scales, order=order, output=result)
    return dlh.typeas(result, batch)


class Image2ImageTrainer(BasicTrainer):
    """Train image to image models.

    This takes images in BHWD order and performs all the necessary
    transformations internally. It also zooms targets up/down to
    the output size of the model. The Torch model still needs to
    take/return BDHW tensors.
    """
    def __init__(self, *args, **kw):
        BasicTrainer.__init__(self, *args, **kw)
        self.set_input_orders("BHWD", "BDHW")
        self.set_output_orders("BHWD", "BDHW")

    def compute_loss(self, targets, weights=None):
        self.set_targets(targets, self.cuoutput, weights=weights)
        return self.criterion(dlm.layers.pixels_to_batch(self.cuoutput),
                              dlm.layers.pixels_to_batch(self.cutarget))

    def set_inputs(self, images):
        dlh.assign(self.cuinput, images, self.data_input, self.model_input)

    def set_targets(self, targets, outputs, weights=None):
        assert weights is None
        assert dlh.rank(outputs) == 4, dlh.shp(outputs)
        assert dlh.rank(targets) == 4, dlh.shp(targets)
        targets = dlh.reorder(targets, self.data_output, self.model_output)
        assert dlh.size(outputs, 0) == dlh.size(targets, 0), "must have same batch size as output"
        assert dlh.size(outputs, 1) == dlh.size(targets, 1), "must have same depth as output"
        targets = zoom_like(targets, dlh.shp(outputs))
        dlh.assign(self.cutarget, targets)

def ctc_align(prob, target):
    """Perform CTC alignment on torch sequence batches (using ocrolstm).

    Inputs are in BDL format.
    """
    import cctc
    assert dlh.sequence_is_normalized(prob), prob
    assert dlh.sequence_is_normalized(target), target
    # inputs are BDL
    prob_ = dlh.novar(prob).permute(0, 2, 1).cpu().contiguous()
    target_ = dlh.novar(target).permute(0, 2, 1).cpu().contiguous()
    # prob_ and target_ are both BLD now
    assert prob_.size(0) == target_.size(0), (prob_.size(), target_.size())
    assert prob_.size(2) == target_.size(2), (prob_.size(), target_.size())
    assert prob_.size(1) >= target_.size(1), (prob_.size(), target_.size())
    result = torch.rand(1)
    cctc.ctc_align_targets_batch(result, prob_, target_)
    return dlh.typeas(result.permute(0, 2, 1).contiguous(), prob)

def sequence_softmax(seq):
    """Given a BDL sequence, computes the softmax for each time step."""
    b, d, l = seq.size()
    batch = seq.permute(0, 2, 1).contiguous().view(b*l, d)
    smbatch = F.softmax(batch)
    result = smbatch.view(b, l, d).permute(0, 2, 1).contiguous()
    return result

class Image2SeqTrainer(BasicTrainer):
    """Train image to sequence models using CTC.

    This takes images in BHWD order, plus output sequences
    consisting of lists of integers.
    """
    def __init__(self, *args, **kw):
        BasicTrainer.__init__(self, *args, **kw)

    def init_loss(self, loss=None):
        assert loss is None, "Image2SeqTrainer must be trained with BCELoss (default)"
        self.criterion = nn.BCELoss(size_average=False)

    def compute_loss(self, targets, weights=None):
        self.cutargets = None   # not used
        assert weights is None
        logits = self.cuoutput
        b, d, l = logits.size()
        probs = sequence_softmax(logits)
        assert dlh.sequence_is_normalized(probs), probs
        ttargets = torch.FloatTensor(targets)
        target_b, target_d, target_l = ttargets.size()
        assert b == target_b, (b, target_b)
        assert dlh.sequence_is_normalized(ttargets), ttargets
        aligned = ctc_align(probs.cpu(), ttargets.cpu())
        assert dlh.sequence_is_normalized(aligned)
        return self.criterion(probs, Variable(self._cuda(aligned)))

    def set_inputs(self, images):
        batch = dlh.bhwd2bdhw(images)
        dlh.assign(self.cuinput, batch, self.data_input, self.model_input)

    def set_targets(self, targets, outputs, weights=None):
        raise Exception("overridden by compute_loss")

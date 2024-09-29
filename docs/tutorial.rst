.. _tutorial:

Tutorial
========

``spikingtorch`` is a package to train spiking neural networks (SSN) using
Pytorch. By integrating SNNs into Pytorch we can leverage all the
tools and datasets already available in one of the most commonly
used deep learning frameworks. This tutorial will cover
the basic concepts behind ``spikingtorch`` as well as the common
challenges of training spiking neural networks.

On spiking neurons
------------------

The fundamental attribute of spiking neurons is that
they communicate through spikes. Here, spikes can be viewed
as a binary output. SNNs can be therefore built similar to
how we build artificial neural networks, with spiking neurons
occupying the role of non-linear or rectifying functions in
ANNs.

The way one can increase the amount of information encoded in
the output of a spiking neuron (SN) is by considering a sequence or
train of spikes. The longer this sequence is the more information
that one can encode in the output of a spiking neuron.

The same applies to the inputs to an SN: while one can use real valued
inputs to any spiking neuron, often inputs are encoded in a sequence
of spikes. There are multiple types of encoding that have been explored,
from time to spike to rate encodings based on either periodic or random
spike trains.

Training SNNs using stochastic gradient descent methods requires dealing
with backpropagation through time: every SNN is essentially a type of
recurrent neural
network. ``spikingtorch`` deals with this through the creation of a few
key building blocks that are designed to abstract some of this
complexity.

Implementing spiking neural networks using ``SpikingNet``
---------------------------------------------------------

The way we define a spiking neural network in ``spikingtorch`` is through
the use of a ``SpikingNet`` object::

    model = SpikingNet(net, (Nout,))

Here :code:`net` represents a Pytorch module, and
:code:`(Nout,)` is a tuple with the dimensions of :code:`net`'s output.


Spiking neuron models
---------------------

``spikingtorch`` implements a number of spiking neuron models. These
can be used as layers in more complex spiking neural networks. For
instance, here is the definition of a simple SNN using the :code:`IF` layer,
which represents an integrate and fire neuron::

    from spikingtorch import IF
    import torch.nn as nn

    class MySNN(nn.Module):

        def __init__(self):
            super(MySNN, self).__init__()
            self.Nout = Nout
            self.conv1 = nn.Conv2d(1, 4, (4,4), stride=2, padding=0) # 15x15
            self.conv2 = nn.Conv2d(4, 6, (4,4), stride=1, padding=0)
            self.l1 = nn.Linear(600, Nout, bias=None)

            self.sl1 = IF()
            self.sl2 = IF()
            self.sl3 = IF()

        def forward(self, xi, init):
            xi = self.conv1(xi)
            s1 = self.sl1(xi, init)
            xi = self.conv2(s1)
            s2 = self.sl2(xi, init)
            xi = s2.view(s2.shape[0],-1)
            xi2 = self.l1(xi)
            return self.sl3(xi2, init)

This code should be familiar to anyone who has experience working
with Pytorch. We define our network as you would create a model.
The main difference is that the ``forward`` method takes two
arguments, the input to the network and an additional ``init`` flag
that is subsequently passed to the spiking neuron layers.

Users don't have  to worry about this ``init`` flag, but this is
the way a ``SpikingNet`` object currently communicates to each spiking
layer the need to reset its internal memory.

Encoders and decoders
---------------------

Encoders and decoders comprise the third key building block
of ``spikingtorch``. Encoders take inputs and transform them so that
they can be passed into a ``SpikingNet`` object. For instance
in this example::

    from spikingtorch.spikeio import PoissonEncoder
    nsteps = 8
    enc = PoissonEncoder(nsteps, 1.0)

we create a :code:`PoissonEncoder` object that transforms inputs into
a Poisson spike train comprising 8 steps using an appropriate scale
factor. The output of a Pytorch ``DataLoader`` can be directly used
as input.

Likewise, the output of a ``SpikingNet`` is a train of spikes. These
have to be transformed so that we can apply a loss function to the output
of the SNN. For instance::

    from spikingtorch.spikeio import SumDecoder
    decoder = SumDecoder(nsteps, 1.0)

creates a :code:`SumDecoder` object that transforms the spike trains
into a single value that can then passed to a loss function.

Putting it all together
-----------------------

The bulding blocks introduced above allow the training of SNNs
on machine learning tasks using standard data loaders, loss functions,
and optimizers implemented in Pytorch::

        mdata = encoder(data)
        output = decoder(model(mdata))
        loss = F.cross_entropy(output, mtarget)
        loss.backward()

The ``spikingtorch`` GitHub repository
has a few examples that users can use as a starting
point to train their own spiking neural networks.



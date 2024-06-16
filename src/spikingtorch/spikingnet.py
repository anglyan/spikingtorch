# Copyright © 2020-2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: spikingtorch

import torch
import torch.nn as nn

class SpikingNetBase(nn.Module):
    r"""Implement a base class for spiking neural networks

    Spiking neural networks take a tensor of must run over a number of
    time steps. In order to simplify the creation of arbitrary spiking
    neural networks, a `SpikingNet` module takes

    SpikingNet-derived modules take inputs with the shape `(Nbatch, Nsteps, *Ndims)`,
    where `Nsteps` is the length of the input spike train and `Ndims` can
    comprise an arbitrary number of dimensions. 
    
    Modules inheriting from `SpikingNet` must implement a number of properties
    and class methods:

    - They should explicitly define the number of output classes `Nout`.
    - They should implement a `single_pass` method, which is the equiva
      a single pass. This method should take two arguments: the input
      to the network and a boolean `init` flag.
    """

    def __init__(self):
        super(SpikingNetBase, self).__init__()

    def single_pass(self, xi, init):
        pass

    def forward(self, x):
        n_batch, n_steps = x.shape[0], x.shape[1]
        out_size = n_batch, n_steps, *self.Nout
        sp = torch.zeros(out_size).to(x.device)
        for i in range(n_steps):
            init = True if i==0 else False
            xi = torch.index_select(x, 1, torch.tensor([i]).to(x.device))
            sp[:,i,:] = self.single_pass(xi, init)
        return sp


class SpikingNet(SpikingNetBase):

    """
    Implement a general spiking neural network from a Pytorch module

    Args:
        net: A pytorch module with spiking neural network units
        Nout: The number of output neurons in ``net``

    Inputs: input
        * **input**: tensor of shape :math:`(N, T, N_{in})` representing batch,
          number of timesteps and either a 1D, 2D or 3D input depending on the
          presence of input channels and either 1D or 2D inputs.

    Outputs: output,
        * **output**: tensor of shape :math:`(N, N_{out})`.

    Attributes:

    .. note::
        All the spiking neurons are initialized at the beginning of each sample

    Examples::

        >>> snn = SpikingNet(net, 10)
    """

    def __init__(self, net, Nout):
        super(SpikingNet, self).__init__()
        self.Nout = Nout
        self.net = net

    def single_pass(self, xi, init):
        return self.net(xi, init)


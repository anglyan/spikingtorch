# Copyright © 2020-2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: spikingtorch

import torch
import torch.nn as nn

class SpikingNet(nn.Module):

    def __init__(self):
        super(SpikingNet, self).__init__()

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


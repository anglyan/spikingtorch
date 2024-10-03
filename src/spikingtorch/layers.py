# Copyright © 2020-2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: spikingtorch

import torch
import torch.nn as nn
import math as m

class Step(torch.autograd.Function):
    """
    Implements a step function with steep sigmoid gradient propagation 

    """

    @staticmethod
    def forward(ctx, input, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        output = torch.sigmoid(constant*input)
        ctx.save_for_backward(output)
        return _H(input)

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        fakeoutput, = ctx.saved_tensors
        return grad_output * fakeoutput*(1-fakeoutput)*ctx.constant, None

def _H(x):
    """Heaviside step function"""
    return 0.5*(torch.sign(x)+1)

step = Step.apply

class NoisyStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, constant, noise=None):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        output = torch.sigmoid(constant*input)
        ctx.save_for_backward(output)
        if noise is None:
            noise = 0
        return step(input+noise)

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        fakeoutput, = ctx.saved_tensors
        return grad_output * fakeoutput*(1-fakeoutput)*ctx.constant, None, None

noisystep = NoisyStep.apply


class CoarseLIF(nn.Module):
    """
    Implements a coarse-scale implementation of a Leaky Integrate and Fire neuron

    """

    def __init__(self, tau, v0=1, beta=5):
        super(CoarseLIF, self).__init__()
        self.tau = tau
        self.a = m.exp(-1./tau)
        self.b = tau*(1 - self.a)
        self.c = tau*(1 - self.b)
        self.v0 = v0
        self.beta = beta

    def reset(self, xi):
        self.v = torch.zeros_like(xi).to(xi.device)
        self.s = torch.zeros_like(xi).to(xi.device)

    def forward(self, xi, init=False):
        if init:
            v = torch.zeros_like(xi).to(xi.device)
            s = torch.zeros_like(xi).to(xi.device)
        else:
            v = self.v
            s = self.s
        self.v = (1-s) * (self.a * v + self.b * xi) + s * self.c * xi
        self.s = step(self.v-self.v0, self.beta)
        return self.s


class SpikingLayer(CoarseLIF):
    pass


class LIF(nn.Module):
    """
    Implements a Leaky Integrate and Fire neuron

    """

    def __init__(self, tau=8, v0=1, beta=5, decay=None):
        super(LIF, self).__init__()
        if decay is None:
            self.tau = tau
            self.a = m.exp(-1./tau)
            self.b = tau*(1 - self.a)
        else:
            self.a = decay
            self.b = 1

        self.v0 = v0
        self.beta = beta

    def reset(self, xi):
        self.v = torch.zeros_like(xi).to(xi.device)
        self.s = torch.zeros_like(xi).to(xi.device)

    def forward(self, xi, init=False):
        if init:
            self.reset(xi)    
        self.v = (1-self.s) * self.a * self.v + self.b * xi
        self.s = step(self.v-self.v0, self.beta)
        return self.s


class IF(nn.Module):
    """
    Implements an Integrate and Fire neuron

    """
    def __init__(self, v0=1, beta=5):
        super(IF, self).__init__()
        self.v0 = v0
        self.beta = beta

    def reset(self, xi):
        self.v = torch.zeros_like(xi).to(xi.device)
        self.s = torch.zeros_like(xi).to(xi.device)

    def forward(self, xi, init=False):
        if init:
            self.reset(xi)
        self.v = (1-self.s) * self.v +  xi
        self.s = step(self.v-self.v0, self.beta)
        return self.s


class McPitts(nn.Module):

    def __init__(self, thr, beta=3):
        super(McPitts, self).__init__()
        self.thr = thr
        self.beta = beta

    def forward(self, x):
        return step(x-self.thr, self.beta)


class BioLIF(nn.Module):

    def __init__(self, nudt, v0=0.5, beta=5, track_act=False):
        super(BioLIF, self).__init__()
        self.nu0 = nudt
        self.v0 = v0
        self.beta = beta
        self.track_act = track_act

    def reset(self, xi):
        self.v = torch.zeros_like(xi).to(xi.device)
        self.s = torch.zeros_like(xi).to(xi.device)
        self.act = torch.zeros_like(xi).to(xi.device)

    def forward(self, xe, xi, init):
        if init:
            self.reset(xe)
        nu = (1+xe+xi)
        a = torch.exp(-self.nu0*nu).detach()
        de = xe/nu
        self.v = (1-self.s) * self.v * a + de*(1-a)
        self.s = step(self.v-self.v0, self.beta)
        if self.track_act:
            self.act += self.s
        return self.s

class SpikingDecay(nn.Module):

    def __init__(self, tau, v0=1, beta=5):
        super(SpikingDecay, self).__init__()
        self.tau = tau
        self.a = 1/tau
        self.v0 = v0
        self.beta = beta

    def forward(self, xi, init=False):
        if init:
            v = torch.zeros_like(xi).to(xi.device)
            s = torch.zeros_like(xi).to(xi.device)
        else:
            v = self.v
            s = self.s
        self.v = (1-s) * self.a * v + xi
        self.s = step(self.v-self.v0, self.beta)
        return s


class SpikingLayer2(nn.Module):

    def __init__(self, tau, v0=1, beta=5):
        super(SpikingLayer2, self).__init__()
        self.tau = tau
        self.a = m.exp(-1./tau)
        self.b = tau*(1 - self.a)
        self.v0 = v0
        self.beta = beta

    def forward(self, xi, s, v):
        v = (1-s) * (self.a * v + self.b * xi)
        s = step(v-self.v0, self.beta)
        return s, v


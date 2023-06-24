# Copyright © 2020-2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: spikingtorch

import torch

class PoissonEncoder:

    def __init__(self, nsteps, scale=1.0):
        self.nsteps = nsteps
        self.scale = scale

    def __call__(self, x):

        slist = []
        for _ in range(self.nsteps):
            base = torch.rand_like(x)
            base[self.scale*x<base] = 0.0
            base[base>0] = 1.0
            slist.append(base)

        return torch.concat(slist, dim=1).detach()


class PeriodicEncoder:

    def __init__(self, nsteps, scale=1.0):
        self.nsteps = nsteps
        self.scale = scale

    def __call__(self, x):

        slist = []
        memory = torch.zeros_like(x)
        for _ in range(self.nsteps):
            memory += self.scale*x
            out = torch.clone(memory)
            out[out<1] = 0
            out[out>0] = 1
            memory[out > 0] = 0
            slist.append(out)

        return torch.concat(slist, dim=1).detach()


class RateDecoder:
    
    def __call__(self, x):
        return torch.mean(x, dim=1)


class SumDecoder:

    def __init__(self, nsteps, scale, offset=True):
        self.scale = scale
        self.nsteps = nsteps
        self.offset = offset

    def __call__(self, x):
        if self.offset:
            return self.scale*(torch.sum(x, dim=1) - 0.5*self.nsteps)
        else:
            return self.scale*torch.sum(x, dim=1)



class Poisson2D:

    def __init__(self, nsteps, scale=1.0):
        self.nsteps = nsteps
        self.scale = scale

    def __call__(self, x):
        return poisson_spikes2d(x, self.nsteps, self.scale)
    


def poisson_spikes(x, nsteps, scale):
    batch_size, nin = x.shape
    xout = torch.rand((batch_size, nsteps, nin))
    x = torch.unsqueeze(x,1)
    xout[scale*x.expand_as(xout)<xout] = 0.0
    xout[xout>0] = 1.0
    return xout

def poisson_spikes2d(x, nsteps, scale):
    batch_size, _, Nx, Ny = x.shape
    xout = torch.rand((batch_size, nsteps, Nx, Ny))
#    x = torch.unsqueeze(x,1)
    xout[scale*x.expand_as(xout)<xout] = 0.0
    xout[xout>0] = 1.0
    return xout

def markov_spikes(x, nsteps):
    batch_size, nin = x.shape
    xout = torch.zeros((batch_size, nsteps, nin))

    xr = torch.rand_like(x)

    xr[xr>x] = 0
    xr[xr>0] = 1.0
    xout[:,0,:] = xr
    p0 = 1-x

    for i in range(1, nsteps):
        xi = torch.clone(xr)
        xr2 = torch.rand_like(x)
        xr2[xr2>x] = 0
        xr2[xr2>0] = 1.0
        xi[xr==0] = xr2[xr==0]

        xr2 = torch.rand_like(x)
        xr2[xr2<p0] = 0.0
        xr2[xr2>0] = 1.0

        xi[xr>0] = xr2[xr>0]
        xout[:,i,:] = xi
        xr = torch.clone(xi)

    return xout

if __name__ == "__main__":

    x = torch.rand((1,12))
    print(x)
    xl = markov_spikes(x, 10)
    print(xl)
    print(torch.mean(xl, dim=1))

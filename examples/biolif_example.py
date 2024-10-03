# Copyright © 2020-2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: spikingtorch

import torch
import torch.nn as nn
import numpy as np

from torchvision import datasets, transforms

from spikingtorch import SpikingNet
from spikingtorch.layers import BioLIF
from spikingtorch.spikeio import PoissonEncoder, SumDecoder
from spikingtorch.utils import ClippingNet, Clipper

import torch.nn.functional as F
import torch.optim as optim


class SpikingConvNetwork(ClippingNet):

    def __init__(self, Nout, nu1, nu2, beta=5, scale=1):
        super(SpikingConvNetwork, self).__init__()
        self.Nout = Nout
        self.size2 = 10
        self.Nhid2 = self.size2*self.size2*6
        self.scale = scale
        self.conv1e = nn.Conv2d(1, 4, (4,4), stride=2, padding=0, bias=None) # 15x15
        self.conv1i = nn.Conv2d(1, 4, (4,4), stride=2, padding=0, bias=None) # 15x15

        self.conv2e = nn.Conv2d(4, 6, (4,4), stride=1, padding=0, bias=None)
        self.conv2i = nn.Conv2d(4, 6, (4,4), stride=1, padding=0, bias=None)

        self.l1e = nn.Linear(self.Nhid2, self.Nout, bias=None)
        self.l1i = nn.Linear(self.Nhid2, self.Nout, bias=None)

        self.exc_weights = [self.conv1e, self.conv2e, self.l1e]
        self.inh_weights = [self.conv1i, self.conv2i, self.l1i]

        self.sl1 = BioLIF(nu1, beta=beta)
        self.sl2 = BioLIF(nu1, beta=beta)
        self.sl3 = BioLIF(nu2, beta=beta)

    def forward(self, xi, init):
        xe = self.conv1e(xi)
        xi = self.conv1i(xi)
        s1 = self.sl1(xe, xi, init)
        xe = self.conv2e(s1)
        xi = self.conv2i(s1)
        s2 = self.sl2(xe, xi, init)
        s = s2.view(s2.shape[0],-1)
        xe = self.l1e(s)
        xi = self.l1i(s)
        return self.sl3(xe, xi, init)




def train(model, encoder, decoder, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        mtarget = target
        mdata = encoder(data)
        data, mtarget = mdata.to(device), mtarget.to(device)
        output = decoder(model(data))
        loss = F.cross_entropy(output, mtarget)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print(torch.max(output).detach(), torch.min(output).detach())
        losses.append(loss.item())
    return losses


def test(model, encoder, decoder, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            mtarget = target
            mdata = data
            mdata = encoder(data)
            data, mtarget = mdata.to(device), mtarget.to(device)
            output = decoder(model(data))
            test_loss += F.cross_entropy(output, mtarget).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).to(device)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)



if __name__ == "__main__":

    train_dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('~/DATA/FMNIST', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
        batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('~/DATA/FMNIST', train=False, transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
        batch_size=1000, shuffle=True)
    
    device = torch.device("cpu")
    net = SpikingConvNetwork(10, 0.25, 0.25, beta=3, scale=1).to(device)
    print(net.clip_max)
    model = SpikingNet(net, (10,)).to(device)
    optimizer = Clipper(model, optim.Adam(model.parameters(), lr=0.01))

    data = []
    losses = []
    nsteps = 8
    encoder = PoissonEncoder(nsteps, 1.0)
    decoder = SumDecoder(nsteps, 1.0)

    for epoch in range(1, 10 + 1):
        losses.extend(train(model, encoder, decoder, device, train_dataloader, optimizer, epoch))
        result = test(model, encoder, decoder, device, test_dataloader)
        data.append([epoch, result])

    data = np.array(data)
    filename = "training_data.npy"
    filemode = "model_state.pt"
    np.save(filename, data)
    np.save("loss.npy", np.array(losses))
    torch.save(model.state_dict(), filemode)

    




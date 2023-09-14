# Copyright © 2020-2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: spikingtorch

import torch
import torch.nn as nn
import numpy as np

from torchvision import datasets, transforms

from spikingtorch.layers import IF
from spikingtorch import SpikingNet
from spikingtorch.spikeio import PoissonEncoder, SumDecoder

import torch.nn.functional as F
import torch.optim as optim


class SpikingConvNetwork(SpikingNet):

    def __init__(self, Nout, t1, t2, beta=5, scale=1):
        super(SpikingConvNetwork, self).__init__()
        self.Nout = (Nout,)
        self.size2 = 10
        self.Nhid2 = self.size2*self.size2*6
        self.scale = scale
        self.conv1 = nn.Conv2d(1, 4, (4,4), stride=2, padding=0) # 15x15
        self.l1 = nn.Linear(self.Nhid2, self.Nout[0], bias=None)
        self.conv2 = nn.Conv2d(4, 6, (4,4), stride=1, padding=0)
        self.sl1 = IF(beta=beta)
        self.sl2 = IF(beta=beta)
        self.sl3 = IF(beta=beta)

    def single_pass(self, xi, init):
        xi = self.conv1(xi)
        s1 = self.sl1(xi, init)
        xi = self.conv2(s1)
        s2 = self.sl2(xi, init)
        xi = s2.view(s2.shape[0],-1)
        xi2 = self.l1(xi)
        return self.sl3(xi2, init)


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
#            print(output)
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
        datasets.FashionMNIST('FMNIST', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
        batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('FMNIST', train=False, transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
        batch_size=1000, shuffle=True)

    device = torch.device("cpu")
    model = SpikingConvNetwork(10, 4, 4, beta=3, scale=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    




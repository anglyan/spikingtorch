import torch.nn as nn

class Clipper:

    def __init__(self, snn, optimizer):
        self.snn = snn
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        s = self.optimizer.step()
        self.snn.net.clip()
        return s


class ClippingNet(nn.Module):

    def __init__(self):
        super(ClippingNet, self).__init__()
        self.wmax = 2.0
        self.clip_max = 2.0
        self.exc_weights = []
        self.inh_weights = []

    def iterate_weights(self):
        for module in self.modules():
            if hasattr(module, 'weight'):
                yield module

    def init_weights(self):
        for module in self.modules():
            if hasattr(module, 'weight'):
                if module in self.exc_weights:
                    module.weight.data.abs_()
                else:
                    module.weight.data.abs_().mul_(0.01)
                if module.bias is not None:
                    module.bias.data.abs_()

    def clip(self):
        for module in self.iterate_weights():
            module.weight.data.clip_(0, self.wmax)
            if module.bias is not None:
                module.bias.data.clip_(0, 1)

    def get_weights(self):
        out = []
        for module in self.iterate_weights():
            out.append(module.weight.data.numpy())
        return out


